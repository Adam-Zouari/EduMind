"""Frozen execution inventories and sealed, evaluation-only recovery inputs."""

import json
import os
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash

EXECUTION_SCHEMA = "generation-execution-v2"


def retain_schedule(candidate, manifest, model_lock, plan, protocol, questions):
    return {
        "schema": EXECUTION_SCHEMA,
        "execution_id": uuid4().hex,
        "candidate": candidate,
        "mode": protocol.mode(candidate),
        "model_id": protocol.model_id(candidate),
        "model_identity": stable_hash(model_lock.get(protocol.model_id(candidate), {})),
        "manifest_identity": manifest.fingerprint,
        "protocol_checksum": protocol.meta.checksum,
        "profile": plan.profile,
        "device": plan.settings["device"],
        "dtype": plan.settings["dtype"],
        "seeds": list(protocol.measured_seeds[: plan.repetitions]),
        "questions": deepcopy(questions),
        "evaluator": evaluator_identity(protocol),
    }


def evaluator_identity(protocol):
    from .judge import LABEL_SCHEMA, load_judge_contract

    if protocol.judge_contract_path is None:
        return {"label_schema": LABEL_SCHEMA, "contract_sha256": None}
    contract = load_judge_contract(protocol.judge_contract_path)
    return {
        "label_schema": LABEL_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "contract_checksum": contract["checksum"],
    }


def write_new_json(path, value):
    """Reserve immutable identity/evidence files, flush before any inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def validate_schedule(raw, protocol, profile):
    """Never infer expected questions or seeds from surviving prediction rows."""
    schedule = raw.get("schedule")
    if not isinstance(schedule, dict) or schedule.get("schema") != EXECUTION_SCHEMA:
        raise ValueError(
            "Missing/unsupported independently retained generation schedule"
        )
    if raw.get("schedule_checksum") != stable_hash(schedule):
        raise ValueError("Generation schedule checksum mismatch")
    if (
        schedule.get("profile") != profile
        or schedule.get("protocol_checksum") != protocol.meta.checksum
    ):
        raise ValueError("Generation schedule profile/protocol mismatch")
    seeds = list(protocol.measured_seeds[: protocol.profile(profile).repetitions])
    if schedule.get("seeds") != seeds:
        raise ValueError("Generation schedule seed inventory mismatch")
    for key in ("candidate", "mode", "model_id", "device", "dtype"):
        if raw.get(key) != schedule.get(key):
            raise ValueError(f"Generation schedule {key} identity mismatch")
    candidate = schedule["candidate"]
    if (schedule["model_id"], schedule["mode"]) != (
        protocol.model_id(candidate),
        protocol.mode(candidate),
    ):
        raise ValueError("Generation candidate/model identity mismatch")
    if schedule.get("evaluator") != evaluator_identity(protocol):
        raise ValueError(
            "Generation evaluator identity mismatch; use a newly versioned evaluation"
        )
    questions = schedule.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Generation frozen question inventory is empty")
    expected = {q["id"]: q for q in questions}
    if len(expected) != len(questions):
        raise ValueError("Duplicate question in frozen schedule")
    supplied = {}
    for record in raw["records"]:
        q = record["question"]
        if q["id"] not in expected or q["id"] in supplied:
            raise ValueError("Duplicate/unexpected retained question ID")
        if stable_hash(q) != stable_hash(expected[q["id"]]):
            raise ValueError("Retained question/reference identity changed")
        attempts = {}
        for attempt in record["attempts"]:
            seed = attempt["seed"]
            if type(seed) is not int or seed not in seeds or seed in attempts:
                raise ValueError("Duplicate/unexpected retained seed ID")
            attempts[seed] = attempt
        supplied[q["id"]] = attempts
    # Missing slots stay explicit and incomplete; they are not executed failures.
    from .evaluate import _empty_attempt

    return [
        {
            "question": q,
            "attempts": [
                supplied.get(q["id"], {}).get(seed)
                or {
                    **_empty_attempt(seed, "missing_retained_outcome"),
                    "outcome": "missing_outcome",
                }
                for seed in seeds
            ],
        }
        for q in questions
    ]


def validate_execution_context(raw, candidate, manifest, model_lock, plan, protocol):
    from .validation import questions_from_manifest

    expected = retain_schedule(
        candidate,
        manifest,
        model_lock,
        plan,
        protocol,
        questions_from_manifest(manifest),
    )
    actual = raw["schedule"]
    expected["execution_id"] = actual["execution_id"]
    if expected != actual:
        raise ValueError(
            "Worker schedule does not match requested manifest/model/protocol execution"
        )


def seal_execution(raw, directory):
    """Publish once after resource release; later evaluation never rewrites it."""
    directory = Path(directory)
    path = directory / "generation_execution.json"
    write_new_json(path, raw)
    binding = {
        "schema": EXECUTION_SCHEMA,
        "execution_id": raw["schedule"]["execution_id"],
        "schedule_checksum": raw["schedule_checksum"],
        "prediction_sha256": sha256_file(path),
    }
    write_new_json(directory / "execution_binding.json", binding)
    return path


def load_execution(path, protocol):
    path = Path(path).resolve()
    binding = json.loads(
        path.with_name("execution_binding.json").read_text(encoding="utf-8")
    )
    if not isinstance(binding, dict) or set(binding) != {
        "schema",
        "execution_id",
        "schedule_checksum",
        "prediction_sha256",
    }:
        raise ValueError("Unsupported saved generation execution binding")
    if binding.get("schema") != EXECUTION_SCHEMA or binding.get(
        "prediction_sha256"
    ) != sha256_file(path):
        raise ValueError("Saved generation prediction checksum mismatch")
    raw = json.loads(path.read_text(encoding="utf-8"))
    schedule = raw["schedule"]
    if binding.get("schedule_checksum") != raw.get("schedule_checksum") or binding.get(
        "execution_id"
    ) != schedule.get("execution_id"):
        raise ValueError("Saved generation execution identity mismatch")
    retained = json.loads(
        path.with_name("generation_schedule.json").read_text(encoding="utf-8")
    )
    if retained != schedule:
        raise ValueError("Independently retained schedule mismatch")
    validate_schedule(raw, protocol, schedule["profile"])
    return raw, {**binding, "prediction_path": str(path)}


def save_evaluation(path, result=None, error=None, *, provenance):
    from dataclasses import asdict

    if error is not None:
        payload = {
            "status": "incomplete",
            "error": str(error),
            **{
                key: getattr(error, key)
                for key in (
                    "metrics",
                    "operational",
                    "parameters",
                    "intervals",
                    "artifacts",
                )
            },
            "samples": [asdict(row) for row in error.samples],
        }
    else:
        payload = dict(
            zip(
                (
                    "samples",
                    "operational",
                    "metrics",
                    "parameters",
                    "intervals",
                    "artifacts",
                ),
                result,
                strict=True,
            )
        )
        payload.update(status="complete", samples=[asdict(row) for row in result[0]])
    atomic_write_json(Path(path), {"evaluation_provenance": provenance, **payload})
