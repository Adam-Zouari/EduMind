"""Evaluation-only recovery from immutable vector execution snapshots.

There is deliberately no model, adapter, Docker, preparation or retry dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml

from benchmarks.common.contracts import CandidateExecutionError
from edumind.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    local_file_lock,
    sha256_file,
    stable_hash,
)

from .adapters import Hit
from .protocol import load_protocol
from .validation import load_inputs
from .workload import (
    SELECTED_FILES,
    classify_hits,
    global_inventory,
    search_rows,
)


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
    envelope = _plain({"schema_version": 2, "kind": kind, "payload": payload})
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
        or envelope["schema_version"] != 2
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
    if protocol["resolved"].get(
        "protocol_version"
    ) != "vector-database-v6" or protocol.get("checksum") != stable_hash(
        protocol["resolved"]
    ):
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
    from dataclasses import asdict

    from .metrics import usable_duration
    from .selection import candidate_config

    known = global_inventory(corpora)
    expected = []
    for session in workloads["sessions"]:
        corpus = corpora[session["id"]]
        session_ids = frozenset(corpus.ids)
        _, config = candidate_config(
            protocol, raw["candidate"], int(corpus.vectors.shape[1])
        )
        for identity, _, allowed in search_rows(session, corpus):
            expected.append(
                (
                    {**identity, "configuration_checksum": stable_hash(asdict(config))},
                    session_ids,
                    allowed,
                )
            )
    observed = [row for row in raw["requests"] if row["operation"] == "search"]
    if len(observed) != len(expected):
        raise ValueError(
            "Missing/duplicate retained scheduled search outcomes; recovery cannot manufacture attempts"
        )
    for row in raw["requests"]:
        if row["status"] not in {"success", "failed", "not_executed"}:
            raise ValueError("Unknown retained logical-operation status")
        timing = row["latency_seconds"]
        if timing is not None and (
            row["status"] != "success" or not usable_duration(timing)
        ):
            raise ValueError("Invalid retained successful request timing")
    for row, (identity, session_ids, allowed) in zip(observed, expected, strict=True):
        if any(row.get(key) != value for key, value in identity.items()):
            raise ValueError(
                "Retained search identity differs from the sealed ordered schedule"
            )
        if row["status"] == "success":
            if (
                not isinstance(row["ids"], list)
                or not isinstance(row["scores"], list)
                or len(row["ids"]) != len(row["scores"])
            ):
                raise ValueError("Retained ranking ID/score mismatch")
            checks = classify_hits(
                [
                    Hit(identifier, score, {})
                    for identifier, score in zip(row["ids"], row["scores"], strict=True)
                ],
                known,
                session_ids,
                allowed,
                filtered=row["condition"] == SELECTED_FILES,
            )
            if any(row.get(name) != value for name, value in checks.items()):
                raise ValueError(
                    "Retained session/filter outcome differs from trusted inventory"
                )
        elif (
            row["filter_violation"] is not None
            or row["session_isolation_violation"] is not None
            or row["outside_session_ids"]
            or row["outside_selection_ids"]
        ):
            raise ValueError(
                "Retained unsuccessful request claims unavailable integrity checks"
            )
        if row["status"] == "not_executed" and (
            row["ids"] is not None or row["scores"] is not None
        ):
            raise ValueError("An unexecuted search cannot retain an output")


def validate_integration_ledger(raw, workloads, corpora, protocol, profile):
    """Reconcile all saved session scopes and pools; never call a live runtime."""
    from dataclasses import asdict

    from benchmarks.common.datasets import load_manifest, validate_evidence

    from .metrics import usable_duration
    from .selection import candidate_config

    spec = workloads["integration"]
    if raw["spec"] != spec or len(raw["sessions"]) != len(spec["session_ids"]):
        raise ValueError("Retained integration session/spec identity mismatch")
    sessions = [
        session
        for session in workloads["sessions"]
        if session["id"] in spec["session_ids"]
    ]
    references = load_manifest(
        Path(workloads["manifest_path"]).parent / spec["manifest"]
    )
    sources = validate_evidence(references)
    expected_questions = {
        row["id"]: {**row, **sources[row["id"]].metadata()}
        for row in references.samples
        if row["kind"] == "question" and row["answerable"]
    }
    globally_known = global_inventory(corpora)
    for session, bundle in zip(sessions, raw["sessions"], strict=True):
        sid = session["id"]
        corpus = corpora[sid]
        session_ids = frozenset(corpus.ids)
        if (
            bundle["session_id"] != sid
            or [row["identifier"] for row in bundle["chunks"]] != list(corpus.ids)
            or bundle["questions"]
            != [expected_questions[qid] for qid in corpus.query_ids]
        ):
            raise ValueError(
                "Retained integration session/chunk/question identity mismatch"
            )
        _, config = candidate_config(
            protocol, raw["candidate"], int(corpus.vectors.shape[1])
        )
        configuration = asdict(config)
        records = []
        for position, (chunk, metadata) in enumerate(
            zip(bundle["chunks"], bundle["metadata"], strict=True)
        ):
            trusted = corpus.metadata[position]
            if (
                chunk["text"] != corpus.texts[position]
                or chunk["document_id"] != trusted["source_id"]
                or chunk["start"] != trusted["start"]
                or chunk["end"] != trusted["end"]
                or metadata
                != {
                    "source_id": chunk["document_id"],
                    "document_id": chunk["document_id"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "token_count": chunk["tokens"],
                }
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
        identity = stable_hash({"configuration": configuration, "records": records})
        expected = list(search_rows(session, corpus))
        if len(bundle["requests"]) != len(expected) or len(bundle["rankings"]) != len(
            expected
        ):
            raise ValueError("Missing/duplicate retained integration attempts")
        for request, row, (scheduled, filters, allowed) in zip(
            bundle["requests"], bundle["rankings"], expected, strict=True
        ):
            frozen = {
                **scheduled,
                "operation": "integration",
                "configuration_checksum": stable_hash(configuration),
            }
            if (
                any(request.get(key) != value for key, value in frozen.items())
                or {name: value for name, value in row.items() if name != "evidence"}
                != request
            ):
                raise ValueError(
                    "Retained integration request order, scope or ranking ledger differs"
                )
            if request["status"] not in {"success", "failed", "not_executed"}:
                raise ValueError("Unknown retained integration outcome")
            if request["latency_seconds"] is not None and (
                request["status"] != "success"
                or not usable_duration(request["latency_seconds"])
            ):
                raise ValueError("Invalid retained integration timing")
            observed = row["evidence"]
            if request["status"] == "not_executed":
                if observed:
                    raise ValueError(
                        "Unexecuted integration request has observed evidence"
                    )
                continue
            if (
                observed.get("identity") != identity
                or observed.get("configuration") != configuration
            ):
                raise ValueError("Retained integration runtime identity mismatch")
            for field, evidence_field in (
                ("ids", "database_ids"),
                ("filter_violation", "filter_violation"),
                ("session_isolation_violation", "session_isolation_violation"),
                ("database_latency_seconds", "database_latency_seconds"),
            ):
                if row.get(field) != observed.get(evidence_field):
                    raise ValueError("Retained database stage evidence disagrees")
            if row["database_executed"] != observed["database_executed"]:
                raise ValueError("Retained database stage outcome disagrees")
            if observed.get("session_isolation_violation") is not None:
                ids, scores = observed["database_ids"], observed["database_scores"]
                checks = classify_hits(
                    [
                        Hit(identifier, score, {})
                        for identifier, score in zip(ids, scores, strict=True)
                    ],
                    globally_known,
                    session_ids,
                    allowed,
                    filtered=bool(filters),
                )
                if any(observed.get(key) != value for key, value in checks.items()):
                    raise ValueError(
                        "Retained database stage differs from trusted session inventory"
                    )
                if (
                    checks["session_isolation_violation"] or checks["filter_violation"]
                ) and observed.get("pool") is not None:
                    raise ValueError(
                        "Retained forbidden database response has a downstream pool"
                    )
            pool, ranking = observed.get("pool"), observed.get("ranking")
            if pool is None:
                if ranking is not None or row["status"] == "success":
                    raise ValueError("Retained success/ranking has no verified pool")
                continue
            if (
                observed.get("expected_count") != min(20, len(allowed))
                or len(pool) != min(20, len(allowed))
                or any(
                    type(position) is not int or not 0 <= position < len(corpus.ids)
                    for position in pool
                )
                or len(pool) != len(set(pool))
                or observed.get("pool_ids")
                != [corpus.ids[position] for position in pool]
                or not set(observed["pool_ids"]) <= allowed
            ):
                raise ValueError(
                    "Retained integration pool is short or contains forbidden records"
                )
            if observed["database_executed"] and (
                observed.get("session_isolation_violation") is not False
                or len(observed["database_ids"]) != min(20, len(allowed))
            ):
                raise ValueError(
                    "Retained downstream pool has no verified complete database source"
                )
            if (row["status"] == "success") != (ranking is not None) or (
                ranking is not None
                and (
                    any(type(position) is not int for position in ranking)
                    or sorted(ranking) != sorted(pool)
                    or observed.get("ranking_ids")
                    != [corpus.ids[position] for position in ranking]
                )
            ):
                raise ValueError(
                    "Retained reranker output is not the actual pool permutation"
                )
            checksum = stable_hash(
                {
                    "identity": identity,
                    "query": str(expected_questions[scheduled["query_id"]]["question"]),
                    "filters": filters,
                    "database_ids": observed.get("database_ids"),
                    "database_scores": observed.get("database_scores"),
                    "pool_ids": observed["pool_ids"],
                }
            )
            if checksum != observed.get("pool_checksum"):
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
    if context["workloads"] != {
        name: workloads[name] for name in ("sessions", "operations")
    }:
        raise ValueError(
            "Retained session and operation schedules differ from original inputs"
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
        "schema_version": 2,
        "evaluation_version": "vector-evaluation-recovery-v2",
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
                Path(__file__).parents[2] / "src/edumind/rag/tokenizers.py",
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
