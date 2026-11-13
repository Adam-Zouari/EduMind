"""Evaluation-only recovery from immutable vector execution snapshots.

There is deliberately no model, adapter, Docker, preparation or retry dispatch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml

from edumind.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    local_file_lock,
    sha256_file,
    stable_hash,
)
from experiments.benchmarks.common.contracts import CandidateExecutionError

from .adapters import Hit
from .protocol import load_protocol
from .validation import load_inputs
from .workload import validate_hits


def _plain(value):
    return json.loads(
        json.dumps(
            value,
            default=lambda item: (
                item.tolist()
                if hasattr(item, "tolist")
                else item.item()
                if hasattr(item, "item")
                else repr(item)
            ),
        )
    )


def write_seal(path, kind, payload):
    path = Path(path)
    envelope = _plain({"schema_version": 1, "kind": kind, "payload": payload})
    envelope["checksum"] = stable_hash(envelope)
    path.parent.mkdir(parents=True, exist_ok=True)
    with local_file_lock(path.with_suffix(".lock")):
        if path.exists():
            raise ValueError(
                f"Retained execution evidence already exists; refusing overwrite: {path}"
            )
        atomic_write_json(path, envelope)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "checksum": envelope["checksum"],
    }


def read_seal(path, kind=None):
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        set(envelope) != {"schema_version", "kind", "payload", "checksum"}
        or type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != 1
        or kind is not None
        and envelope["kind"] != kind
        or envelope["checksum"]
        != stable_hash(
            {key: value for key, value in envelope.items() if key != "checksum"}
        )
    ):
        raise ValueError("Retained vector evidence seal mismatch/unsupported schema")
    return envelope


def snapshot_writer(directory, context):
    directory = Path(directory)
    prefix = "retained-" + stable_hash(context["candidate"])[:16]
    context_file = directory / (prefix + "-context.json")
    context_ref = write_seal(context_file, "context", context)
    references = {"context": context_ref, "stages": {}}

    def save(stage, evidence):
        references["stages"][stage] = write_seal(
            directory / (prefix + "-" + stage + ".json"),
            stage,
            {
                "context_sha256": context_ref["sha256"],
                "candidate": context["candidate"],
                "evidence": evidence,
            },
        )

    return save, references


def load_retained(directory, *, verify_inputs=True):
    directory = Path(directory).resolve()
    paths = list(directory.glob("retained-*-context.json"))
    if len(paths) != 1:
        raise ValueError("Recovery requires exactly one retained candidate context")
    context_envelope = read_seal(paths[0], "context")
    context = context_envelope["payload"]
    report = context["input_report"]
    if (
        report.get("status") != "passed"
        or report.get("profile") != context["profile"]
        or report.get("fingerprint") != context["input_validation"]
        or report.get("report_checksum")
        != stable_hash(
            {key: value for key, value in report.items() if key != "report_checksum"}
        )
    ):
        raise ValueError("Original input validation report seal/identity mismatch")
    identity = {
        key: value
        for key, value in report.items()
        if key not in {"fingerprint", "status", "errors", "metadata", "report_checksum"}
    }
    if stable_hash(identity) != report["fingerprint"]:
        raise ValueError(
            "Original input validation fingerprint does not cover its identity"
        )
    protocol = context["protocol"]
    if protocol.get("checksum") != stable_hash(protocol["resolved"]):
        raise ValueError("Retained protocol checksum mismatch")
    if (
        report.get("requirements") != protocol["resolved"]
        or str(Path(report["manifest_path"]).resolve()) != context["manifest"]
    ):
        raise ValueError("Retained input requirements/manifest differ from execution")
    if verify_inputs:
        for file, checksum in report["files"].items():
            if sha256_file(Path(file)) != checksum:
                raise ValueError(f"Original vector input bytes changed: {file}")
        if context["profile"] == "locked":
            reservation = context.get("locked_reservation")
            if (
                not reservation
                or sha256_file(Path(reservation["path"])) != reservation["sha256"]
            ):
                raise ValueError("Original locked reservation is missing/changed")
            marker = json.loads(Path(reservation["path"]).read_text(encoding="utf-8"))
            if marker != {
                "input": context["input_validation"],
                "candidate": context["candidate"],
            }:
                raise ValueError(
                    "Original locked reservation names different inputs/candidate"
                )
    stages, references = (
        {},
        {
            "context": {"path": str(paths[0]), "sha256": sha256_file(paths[0])},
            "stages": {},
        },
    )
    for kind in ("server", "integration"):
        snapshots = list(directory.glob("retained-*-" + kind + ".json"))
        if len(snapshots) > 1:
            raise ValueError("Duplicate retained execution stages")
        if snapshots:
            envelope = read_seal(snapshots[0], kind)
            payload = envelope["payload"]
            if (
                payload["context_sha256"] != references["context"]["sha256"]
                or payload["candidate"] != context["candidate"]
            ):
                raise ValueError(
                    "Retained execution belongs to a different context/candidate"
                )
            stages[kind] = payload["evidence"]
            references["stages"][kind] = {
                "path": str(snapshots[0]),
                "sha256": sha256_file(snapshots[0]),
                "checksum": envelope["checksum"],
            }
    if "server" not in stages:
        raise ValueError("No retained server execution; recovery cannot rerun it")
    return context, stages, references


def validate_server_ledger(raw, workloads, corpora, protocol, profile):
    expected = {
        (condition["id"], window, query, attempt)
        for condition in workloads["conditions"]
        for window in range(condition["windows"])
        for query in condition["query_ids"]
        for attempt in range(1, protocol.profile(profile).repetitions + 1)
    }
    observed = set()
    conditions = {row["id"]: row for row in workloads["conditions"]}
    for row in raw["requests"]:
        if row["status"] not in {"success", "failed", "not_executed"}:
            raise ValueError("Unknown retained logical-operation status")
        if row["status"] == "success" and (
            not isinstance(row["latency_seconds"], (int, float))
            or not math.isfinite(row["latency_seconds"])
            or row["latency_seconds"] < 0
        ):
            raise ValueError("Invalid retained successful request timing")
        if row["operation"] != "search":
            continue
        key = (row["condition"], row["window"], row["query_id"], row["attempt"])
        if key in observed or key not in expected:
            raise ValueError("Duplicate/unscheduled retained search identity")
        observed.add(key)
        if row["quality_designated"] != (row["window"] == 0 and row["attempt"] == 1):
            raise ValueError("Retained designated-first identity changed")
        condition = conditions[row["condition"]]
        corpus = corpora[condition["corpus"]]
        if row["status"] == "success":
            if len(row["ids"]) != len(row["scores"]):
                raise ValueError("Retained ranking ID/score mismatch")
            violation = validate_hits(
                [
                    Hit(identifier, score, {})
                    for identifier, score in zip(row["ids"], row["scores"], strict=True)
                ],
                dict(zip(corpus.ids, corpus.metadata, strict=True)),
                20,
                condition["filters"],
            )
            if condition["filters"] and bool(violation) != row["filter_violation"]:
                raise ValueError(
                    "Retained filter outcome differs from trusted metadata"
                )
    if observed != expected:
        raise ValueError(
            "Missing retained scheduled search outcomes; recovery cannot manufacture attempts"
        )
    for row in raw["windows"]:
        rows = [
            request
            for request in raw["requests"]
            if request["operation"] == "search"
            and request["condition"] == row["condition"]
            and request["window"] == row["window"]
        ]
        if (
            not math.isfinite(row["wall_seconds"])
            or row["wall_seconds"] < 0
            or row["successful"]
            != sum(request["status"] == "success" for request in rows)
            or row["submitted"]
            != sum(request["status"] != "not_executed" for request in rows)
        ):
            raise ValueError(
                "Retained load window does not reconcile to its logical requests"
            )


def validate_integration_ledger(raw, workloads, corpora, protocol, profile):
    """Validate saved live pools without querying, embedding or re-ranking."""
    from dataclasses import asdict

    from experiments.benchmarks.common.datasets import load_manifest

    from .selection import candidate_config

    spec = workloads["integration"]
    corpus = corpora[spec["corpus"]]
    if raw["spec"] != spec or [row["identifier"] for row in raw["chunks"]] != list(
        corpus.ids
    ):
        raise ValueError("Retained integration corpus/spec identity mismatch")
    references = load_manifest(
        Path(workloads["manifest_path"]).parent / spec["manifest"]
    )
    expected_questions = {
        str(row["id"]): row
        for row in references.samples
        if row.get("kind") == "question" and row.get("answerable")
    }
    if {str(row["id"]) for row in raw["questions"]} != set(corpus.query_ids) or len(
        raw["questions"]
    ) != len(corpus.query_ids):
        raise ValueError("Retained integration question identity mismatch")
    for question in raw["questions"]:
        expected = expected_questions[str(question["id"])]
        if any(
            question.get(key) != expected.get(key)
            for key in ("question", "document_id", "evidence", "answerable")
        ):
            raise ValueError("Retained integration reference content changed")
    _, config = candidate_config(
        protocol, raw["candidate"], int(corpus.vectors.shape[1])
    )
    records = []
    for position, (chunk, metadata) in enumerate(
        zip(raw["chunks"], raw["metadata"], strict=True)
    ):
        trusted = corpus.metadata[position]
        if (
            chunk["text"] != corpus.texts[position]
            or chunk["document_id"] != trusted["source_id"]
            or chunk["start"] != trusted["start"]
            or chunk["end"] != trusted["end"]
            or metadata["source_id"] != trusted["source_id"]
        ):
            raise ValueError("Retained integration chunk spans/text/source differ")
        records.append(
            {
                "id": chunk["identifier"],
                "text": chunk["text"],
                "metadata": metadata,
                "vector": list(map(float, corpus.vectors[position])),
            }
        )
    identity = stable_hash({"configuration": asdict(config), "records": records})
    expected = {
        (query_id, attempt)
        for query_id in corpus.query_ids
        for attempt in range(1, protocol.profile(profile).repetitions + 1)
    }
    requests = {(row["query_id"], row["attempt"]): row for row in raw["requests"]}
    rankings = {(row["query_id"], row["attempt"]): row for row in raw["rankings"]}
    if (
        len(requests) != len(raw["requests"])
        or len(rankings) != len(raw["rankings"])
        or set(requests) != expected
        or set(rankings) != expected
    ):
        raise ValueError("Missing/duplicate retained integration attempts")
    for key, row in rankings.items():
        if {
            name: value for name, value in row.items() if name != "evidence"
        } != requests[key]:
            raise ValueError(
                "Retained integration request and ranking ledgers disagree"
            )
        evidence = row["evidence"]
        pool, ranking = evidence.get("pool"), evidence.get("ranking")
        if pool is None:
            if ranking is not None or row["status"] == "success":
                raise ValueError("Retained success/ranking has no verified pool")
            continue
        if (
            evidence.get("identity") != identity
            or evidence.get("configuration") != asdict(config)
            or any(
                type(position) is not int or not 0 <= position < len(corpus.ids)
                for position in pool
            )
            or len(pool) != len(set(pool))
            or evidence.get("pool_ids") != [corpus.ids[position] for position in pool]
        ):
            raise ValueError("Retained integration pool identity mismatch")
        if ranking is not None and (
            len(ranking) != len(pool) or sorted(ranking) != sorted(pool)
        ):
            raise ValueError(
                "Retained reranker output is not the actual pool permutation"
            )
        allowed = [
            i
            for i, metadata in enumerate(raw["metadata"])
            if all(
                metadata.get(name) == value for name, value in spec["filters"].items()
            )
        ]
        if len(pool) != min(20, len(allowed)) or not set(pool) <= set(allowed):
            raise ValueError(
                "Retained integration pool is short or contains forbidden records"
            )
        if evidence.get("database_executed"):
            ids, scores = (
                evidence.get("database_ids", []),
                evidence.get("database_scores", []),
            )
            if len(ids) != len(scores) or validate_hits(
                [
                    Hit(identifier, score, {})
                    for identifier, score in zip(ids, scores, strict=True)
                ],
                dict(zip(corpus.ids, raw["metadata"], strict=True)),
                20,
                spec["filters"],
            ):
                raise ValueError("Retained integration database pool is invalid")
        if ranking is not None and evidence.get("ranking_ids") != [
            corpus.ids[position] for position in ranking
        ]:
            raise ValueError("Retained integration ranking IDs disagree with positions")
        question = expected_questions[key[0]]
        checksum = stable_hash(
            {
                "identity": identity,
                "query": str(question["question"]),
                "filters": spec.get("filters"),
                "database_ids": evidence.get("database_ids"),
                "database_scores": evidence.get("database_scores"),
                "pool_ids": evidence["pool_ids"],
            }
        )
        if checksum != evidence.get("pool_checksum"):
            raise ValueError("Retained integration observed-pool checksum mismatch")


def recover(candidate_directory, output_directory, *, tokenizer=None):
    """Write a new evaluation, preserving original predictions and run/lock state."""
    context, stages, references = load_retained(candidate_directory)
    output = Path(output_directory).resolve()
    source = Path(candidate_directory).resolve()
    if output == source or source in output.parents or output.exists():
        raise ValueError(
            "Recovery needs a new output directory outside the original candidate"
        )
    manifest = Path(context["manifest"])
    workloads, corpora = load_inputs(manifest)
    if workloads["split"] != (
        "locked-test" if context["profile"] == "locked" else context["profile"]
    ):
        raise ValueError(
            "Original manifest split differs from retained execution profile"
        )
    if {key: value.content_checksum for key, value in corpora.items()} != context[
        "corpus_checksums"
    ]:
        raise ValueError("Retained corpus content identity mismatch")
    output.mkdir(parents=True)
    frozen_protocol = output / "protocol.yaml"
    atomic_write_text(frozen_protocol, yaml.safe_dump(context["protocol"]["resolved"]))
    protocol = load_protocol(frozen_protocol)
    validate_server_ledger(
        stages["server"], workloads, corpora, protocol, context["profile"]
    )
    from .benchmark import score_server_evidence
    from .retrieval_run import score_integration_evidence
    from .run import _merge, _table_cells

    results, problems = ([], {}, {}, {}, {}, {}), []

    def evaluate(call):
        nonlocal results
        try:
            extra = call()
        except CandidateExecutionError as exc:
            problems.append(str(exc))
            extra = (
                list(exc.samples),
                exc.operational,
                exc.metrics,
                exc.parameters,
                exc.intervals,
                exc.artifacts,
            )
        results = _merge(results, extra)

    evaluate(
        lambda: score_server_evidence(
            workloads, corpora, protocol, context["profile"], stages["server"]
        )
    )
    if context["integration_required"]:
        if "integration" not in stages:
            problems.append(
                "missing_retained_integration_evidence; no inference retry is permitted"
            )
        else:
            validate_integration_ledger(
                stages["integration"],
                {**workloads, "manifest_path": str(manifest)},
                corpora,
                protocol,
                context["profile"],
            )
            if tokenizer is None:
                from edumind.rag.tokenizers import TiktokenOffsetTokenizer

                tokenizer = TiktokenOffsetTokenizer()
            if stages["integration"]["tokenizer"] != getattr(
                tokenizer, "name", "tiktoken:cl100k_base"
            ):
                raise ValueError(
                    "Recovery tokenizer differs from retained scoring identity"
                )
            evaluate(
                lambda: score_integration_evidence(
                    stages["integration"],
                    protocol,
                    context["profile"],
                    tokenizer=tokenizer,
                )
            )
    artifacts = _table_cells(results[5])
    for name, payload in artifacts.items():
        if isinstance(payload, dict):
            atomic_write_json(output / (name + ".json"), payload)
        else:
            frame = pd.DataFrame(payload)
            frame.to_parquet(output / (name + ".parquet"), index=False)
            frame.to_csv(output / (name + ".csv"), index=False)
            pd.testing.assert_frame_equal(
                frame.fillna("").astype(str),
                pd.read_csv(output / (name + ".csv"), dtype=str, keep_default_na=False),
            )
    report = {
        "schema_version": 1,
        "evaluation_version": "vector-evaluation-recovery-v1",
        "evaluation_id": str(uuid4()),
        "candidate": context["candidate"],
        "source_run_id": context["source_run_id"],
        "profile": context["profile"],
        "prediction_identity": stable_hash(references),
        "original_execution": references,
        "protocol_checksum": protocol.meta.checksum,
        "integration_provenance": context.get("integration_provenance", {}),
        "complete": not problems,
        "problems": problems,
        "metrics": results[2],
        "intervals": results[4],
        "inference_executed": False,
        "evaluator_source_sha256": {
            str(path.resolve()): sha256_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("benchmark.py"),
                Path(__file__).with_name("metrics.py"),
                Path(__file__).with_name("retrieval_run.py"),
                Path(__file__).parents[1] / "rag/chunking_embedding/metrics.py",
                Path(__file__).parents[1] / "common/datasets.py",
                Path(__file__).parents[3] / "src/edumind/rag/tokenizers.py",
            )
        },
    }
    report["report_checksum"] = stable_hash(report)
    atomic_write_json(output / "evaluation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = recover(args.candidate_directory, args.output)
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "evaluation_id": report["evaluation_id"],
                "inference_executed": False,
            }
        )
    )
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
