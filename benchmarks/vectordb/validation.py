"""Standalone vector input validation and lightweight sealed-report verification.

Prepared vectors are input artifacts: this module never embeds or downloads.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmarks.common.protocol import (
    strict_object,
    string,
)
from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.paths import PROJECT_ROOT

from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol
from .workload import SCOPES, Corpus, freeze_session_search, global_inventory

REPORT_ROOT = PROJECT_ROOT / "artifacts/benchmarks/data-validation"


def manifest_path(profile):
    return PROJECT_ROOT / f"data/benchmarks/vectordb/{profile}.json"


def _array(value, parent):
    return (
        np.load(parent / value, allow_pickle=False)
        if isinstance(value, str)
        else np.asarray(value)
    )


def load_inputs(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("Unsupported vector session manifest schema")
    corpora = {}
    for row in payload["sessions"]:
        records = row["records"]
        vectors = _array(row["vectors"], path.parent)
        queries = _array(row["queries"], path.parent)
        corpus = Corpus(
            vectors,
            queries,
            np.zeros(len(queries), dtype=np.int64),
            tuple(item["metadata"] for item in records),
            "",
            tuple(item["id"] for item in records),
            tuple(item["text"] for item in records),
            tuple(row["query_ids"]),
            tuple(row["query_sources"]),
            row["provenance"],
        )
        if row["id"] in corpora:
            raise ValueError("Duplicate corpus identity")
        corpora[row["id"]] = replace(corpus, fingerprint=corpus.content_checksum)
    return payload, corpora


def _identity(path, profile, protocol, inventory):
    path = Path(path).resolve()
    sources = tuple(
        Path(item).resolve()
        for item in (
            inventory
            or (
                (path,)
                if profile == "smoke"
                else tuple(
                    path if name == profile else manifest_path(name)
                    for name in ("development", "validation", "locked")
                )
            )
        )
    )
    if path not in sources:
        raise ValueError("Inventory must contain invocation manifest")
    files = {}
    for source in sources:
        files[str(source)] = sha256_file(source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        for corpus in payload["sessions"]:
            for field in ("vectors", "queries"):
                if isinstance(corpus[field], str):
                    asset = (source.parent / corpus[field]).resolve()
                    files[str(asset)] = sha256_file(asset)
        integration = payload.get("integration", {})
        related = list(integration.get("files", []))
        for field in (
            "manifest",
            "embedding_selection",
            "retrieval_selection",
            "chunking_protocol",
            "retrieval_protocol",
            "model_lock",
        ):
            if integration:
                related.append(integration[field])
                if field.endswith("selection"):
                    decision_path = (source.parent / integration[field]).resolve()
                    decision = json.loads(decision_path.read_text(encoding="utf-8"))
                    summary = (
                        decision_path.parent / decision["source_summary"]
                    ).resolve()
                    files[str(summary)] = sha256_file(summary)
        for item in related:
            asset = (source.parent / item).resolve()
            files[str(asset)] = sha256_file(asset)
    identity = {
        "schema_version": 1,
        "validator_version": "vector-session-data-v2",
        "benchmark": "vectordb",
        "profile": profile,
        "manifest_path": str(path),
        "inventory": list(map(str, sources)),
        "files": files,
        "requirements": protocol.meta.resolved,
        "implementation": {
            str(p): sha256_file(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("workload.py"),
                Path(__file__).with_name("protocol.py"),
                PROJECT_ROOT / "benchmarks/common/datasets.py",
                PROJECT_ROOT / "benchmarks/rag/chunking_embedding/validation.py",
                PROJECT_ROOT / "benchmarks/rag/retrieval_reranking/validation.py",
                PROJECT_ROOT / "src/edumind/rag/tokenizers.py",
            )
        },
    }
    return stable_hash(identity), identity


def _validate(path, profile, protocol, inventory):
    payload, corpora = load_inputs(path)
    strict_object(
        payload,
        "vector manifest",
        {
            "schema_version",
            "name",
            "split",
            "sessions",
            "operations",
            "integration",
            "review",
        },
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 2
        or payload["split"] != ("locked-test" if profile == "locked" else profile)
    ):
        raise ValueError("Unsupported vector manifest schema/split")
    if not corpora:
        raise ValueError("Nonempty study session inventory is required")
    seen = set(SCOPES)
    by_query = set()
    by_family = {}
    pair_offset = 0
    for name, corpus in corpora.items():
        session = next(row for row in payload["sessions"] if row["id"] == name)
        strict_object(
            session,
            "study session",
            {
                "id",
                "session_group_id",
                "records",
                "vectors",
                "queries",
                "query_ids",
                "query_sources",
                "provenance",
                "selected_sources",
                "schedule",
                "exact_references",
            },
        )
        string(name, "session.id")
        group = string(session["session_group_id"], "session.session_group_id")
        for record in session["records"]:
            strict_object(record, "canonical chunk", {"id", "text", "metadata"})
        for field, array in (("vectors", corpus.vectors), ("queries", corpus.queries)):
            if (
                array.ndim != 2
                or not len(array)
                or array.dtype.kind != "f"
                or not np.isfinite(array).all()
                or np.any(np.linalg.norm(array, axis=1) <= 0)
            ):
                raise ValueError(
                    f"{name}.{field}: require finite nonzero floating vectors"
                )
        if corpus.vectors.shape[1] != corpus.queries.shape[1] or len(
            corpus.metadata
        ) != len(corpus.vectors):
            raise ValueError("Corpus/query shape mismatch")
        if len(corpus.ids) != len(set(corpus.ids)) or not all(
            isinstance(value, str) and value for value in corpus.ids
        ):
            raise ValueError("Canonical chunk IDs must be nonempty and unique")
        if (
            len(corpus.query_ids) != len(corpus.queries)
            or len(set(corpus.query_ids)) != len(corpus.queries)
            or len(corpus.source_groups) != len(corpus.queries)
        ):
            raise ValueError("Query identity/source allocation mismatch")
        for row in corpus.metadata:
            string(row.get("source_id"), "canonical source_id")
        if profile != "smoke":
            for field in (
                "source_dataset",
                "source_revision",
                "source_license",
                "kind",
            ):
                string(corpus.provenance.get(field), "provenance." + field)
            if corpus.provenance["kind"] != "representative":
                raise ValueError(
                    "Authoritative primary sessions must be representative; synthetic stress is separate"
                )
            for group in corpus.source_groups:
                string(group, "independent query source")
            for row in corpus.metadata:
                string(
                    row.get("independent_source_id") or row.get("source_group_id"),
                    "reviewed record source group",
                )
        if any(not isinstance(text, str) or not text.strip() for text in corpus.texts):
            raise ValueError("Canonical record texts must be strings")
        for qid in corpus.query_ids:
            string(qid, "session query ID")
            if qid in by_query:
                raise ValueError(f"Query {qid} is scheduled in more than one session")
            by_query.add(qid)
        files = {row["source_id"] for row in corpus.metadata}
        families = {("file", source) for source in files}
        for item in corpus.metadata:
            for field in (
                "independent_source_id",
                "source_group_id",
                "duplicate_group_id",
            ):
                if field in item:
                    families.add(
                        (
                            "duplicate" if field == "duplicate_group_id" else "family",
                            string(item[field], field),
                        )
                    )
        families.update(
            ("family", string(value, "query source group"))
            for value in corpus.source_groups
        )
        for family in families:
            if by_family.setdefault(family, group) != group:
                raise ValueError(
                    "Sessions sharing reviewed sources must share a session_group_id"
                )
        selected = session["selected_sources"]
        if not isinstance(selected, dict) or not set(selected) <= set(corpus.query_ids):
            raise ValueError(
                "Selected files must be keyed by a scheduled session query"
            )
        for sources in selected.values():
            if (
                not isinstance(sources, list)
                or not sources
                or any(not isinstance(source, str) for source in sources)
                or sources != sorted(set(sources))
                or not set(sources) < files
            ):
                raise ValueError(
                    "Selected files must be a canonical nonempty proper subset of session files"
                )
        schedule, references = freeze_session_search(
            session,
            corpus,
            repetitions=protocol.profile(profile).repetitions,
            pair_offset=pair_offset,
        )
        pair_offset += len(selected)
        if stable_hash(schedule) != stable_hash(session["schedule"]):
            raise ValueError(
                "Session schedule differs from the frozen counterbalanced order"
            )
        if stable_hash(references) != stable_hash(session["exact_references"]):
            raise ValueError(
                "Session exact references differ from the scoped exhaustive oracle"
            )
    global_inventory(corpora)
    op_ids = set()
    operation_corpora = {}
    for operation in payload["operations"]:
        strict_object(
            operation,
            "document operation",
            {"id", "corpus", "source_id", "replacement", "insert"}
            | (set(operation) & {"condition_id", "source_group"}),
        )
        if operation["id"] in op_ids:
            raise ValueError("Duplicate operation identity")
        op_ids.add(operation["id"])
        for field in ("condition_id", "source_group"):
            if field in operation:
                string(operation[field], "operation." + field)
        if operation.get("condition_id", operation["id"]) in seen:
            raise ValueError(
                "Operation and search condition identities must be disjoint"
            )
        condition_id = operation.get("condition_id", operation["id"])
        if (
            condition_id in operation_corpora
            and operation_corpora[condition_id] != operation["corpus"]
        ):
            raise ValueError(
                "A document-operation condition cannot pool different corpus snapshots"
            )
        operation_corpora[condition_id] = operation["corpus"]
        corpus = corpora[operation["corpus"]]
        source = string(operation["source_id"], "operation.source_id")
        if source not in {row["source_id"] for row in corpus.metadata}:
            raise ValueError("Update/delete must target an existing document")
        for field in ("replacement", "insert"):
            rows = operation[field]
            if not rows or len({row["id"] for row in rows}) != len(rows):
                raise ValueError(
                    "Prepared document operations need unique nonempty records"
                )
            for row in rows:
                strict_object(
                    row, "prepared record", {"id", "vector", "text", "metadata"}
                )
                vector = np.asarray(row["vector"])
                if (
                    vector.shape != (corpus.vectors.shape[1],)
                    or not np.isfinite(vector).all()
                    or np.linalg.norm(vector) <= 0
                ):
                    raise ValueError("Invalid prepared mutation vector")
                if (
                    field == "replacement"
                    and row["metadata"].get("source_id") != source
                ):
                    raise ValueError(
                        "Replacement must preserve logical document identity"
                    )
                if field == "insert" and (
                    row["id"] in corpus.ids
                    or row["metadata"].get("source_id")
                    in {item["source_id"] for item in corpus.metadata}
                ):
                    raise ValueError("Insertion must identify a new document/record")
                if (
                    field == "replacement"
                    and row["id"] in corpus.ids
                    and corpus.metadata[corpus.ids.index(row["id"])]["source_id"]
                    != source
                ):
                    raise ValueError(
                        "Replacement cannot overwrite a different document"
                    )
    integration = payload["integration"]
    if integration:
        required = {
            "session_ids",
            "manifest",
            "embedding_selection",
            "retrieval_selection",
            "chunking_protocol",
            "retrieval_protocol",
            "model_lock",
            "files",
        }
        strict_object(integration, "integration", required)
        session_ids = integration["session_ids"]
        if (
            not isinstance(session_ids, list)
            or not session_ids
            or len(set(session_ids)) != len(session_ids)
            or not set(session_ids) <= corpora.keys()
        ):
            raise ValueError("Integration must identify its sealed study sessions")
        from benchmarks.common.datasets import evidence_units, load_manifest
        from benchmarks.rag.chunking_embedding.protocol import (
            load_protocol as load_chunking,
        )
        from benchmarks.rag.retrieval_reranking.protocol import (
            load_protocol as load_retrieval,
        )
        from benchmarks.rag.retrieval_reranking.validation import (
            validate as validate_retrieval,
        )

        parent = Path(path).parent
        chunking = load_chunking(parent / integration["chunking_protocol"])
        retrieval = load_retrieval(parent / integration["retrieval_protocol"])
        validate_retrieval(
            parent / integration["manifest"],
            profile,
            retrieval,
            chunking_protocol=chunking,
        )
        manifest = load_manifest(parent / integration["manifest"])
        documents = {
            str(row["id"]): row
            for row in manifest.samples
            if row.get("kind") == "document"
        }
        questions = {
            str(row["id"]): row
            for row in manifest.samples
            if row.get("kind") == "question" and row["answerable"]
        }
        if set(questions) != {
            qid for name in session_ids for qid in corpora[name].query_ids
        }:
            raise ValueError("Integration/vector query allocation mismatch")
        for name in session_ids:
            corpus = corpora[name]
            for record_id, text, metadata in zip(
                corpus.ids, corpus.texts, corpus.metadata, strict=True
            ):
                document = documents.get(metadata["source_id"])
                start, end = metadata.get("start"), metadata.get("end")
                if (
                    document is None
                    or type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(document["text"])
                    or text != document["text"][start:end]
                ):
                    raise ValueError(
                        f"Canonical integration span/text mismatch: {record_id}"
                    )
            eligible_documents = {row["source_id"] for row in corpus.metadata}
            if any(
                unit.document_id not in eligible_documents
                for qid in corpus.query_ids
                for unit in evidence_units(questions[qid])
            ):
                raise ValueError(
                    "Gold evidence lies outside the question's all-session scope"
                )
    elif profile != "smoke":
        raise ValueError(
            "Authoritative vector inputs require frozen integration evidence"
        )
    if profile != "smoke":
        strict_object(
            payload["review"], "review", {"reviewed_by", "reviewed_date", "rationale"}
        )
        for value in payload["review"].values():
            string(value, "review")
        observed, splits = {}, set()
        for source in inventory:
            other, groups = load_inputs(source)
            splits.add(other["split"])
            for session in other["sessions"]:
                corpus = groups[session["id"]]
                record_groups = {
                    row[field]
                    for row in corpus.metadata
                    for field in ("independent_source_id", "source_group_id")
                    if field in row
                }
                duplicate_groups = {
                    row["duplicate_group_id"]
                    for row in corpus.metadata
                    if "duplicate_group_id" in row
                }
                for key in [
                    "session:" + session["id"],
                    "session-group:" + session["session_group_id"],
                    *("query:" + qid for qid in corpus.query_ids),
                    *("source:" + group for group in record_groups if group),
                    *("duplicate:" + group for group in duplicate_groups),
                    *("document:" + row["source_id"] for row in corpus.metadata),
                    *("source:" + group for group in corpus.source_groups),
                    *("chunk:" + value for value in corpus.ids),
                    *("text:" + stable_hash(text) for text in corpus.texts),
                ]:
                    if key in observed and observed[key] != other["split"]:
                        raise ValueError(f"Cross-split vector source leakage: {key}")
                    observed[key] = other["split"]
        if splits != {"development", "validation", "locked-test"}:
            raise ValueError("Complete three-split vector inventory is required")
    return {
        "corpus_checksums": {
            name: corpus.content_checksum for name, corpus in corpora.items()
        },
        "session_count": len(corpora),
        "query_count": len(by_query),
        "paired_query_count": pair_offset,
        "ci_support": "reviewed" if protocol.ci_support else "unfrozen",
    }


def prepare_report(path, profile, protocol, *, inventory=None, report_root=REPORT_ROOT):
    try:
        fingerprint, identity = _identity(path, profile, protocol, inventory)
        report = {
            **identity,
            "fingerprint": fingerprint,
            "status": "passed",
            "errors": [],
        }
        try:
            report["metadata"] = _validate(
                path, profile, protocol, identity["inventory"]
            )
        except (ValueError, OSError, TypeError, KeyError) as exc:
            report.update(status="failed", errors=[f"{type(exc).__name__}: {exc}"])
    except (ValueError, OSError, TypeError, KeyError) as exc:
        report = {
            "schema_version": 1,
            "status": "failed",
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
        fingerprint = stable_hash(report)
    report["report_checksum"] = stable_hash(report)
    destination = Path(report_root) / "vectordb" / (fingerprint + ".json")
    atomic_write_json(destination, report)
    return destination, report


def verify_inputs(path, profile, protocol, *, inventory=None, report_root=REPORT_ROOT):
    fingerprint, identity = _identity(path, profile, protocol, inventory)
    destination = Path(report_root) / "vectordb" / (fingerprint + ".json")
    if not destination.is_file():
        raise ValueError(
            "Missing/stale vector validation report; run standalone vectordb validation"
        )
    report = json.loads(destination.read_text(encoding="utf-8"))
    if report.get("report_checksum") != stable_hash(
        {key: value for key, value in report.items() if key != "report_checksum"}
    ):
        raise ValueError("Vector validation report seal mismatch")
    if (
        report.get("status") != "passed"
        or report.get("fingerprint") != fingerprint
        or any(report.get(key) != value for key, value in identity.items())
    ):
        raise ValueError("Vector validation report identity mismatch")
    return destination, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "development", "validation", "locked"),
        default="smoke",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--inventory", type=Path, action="append")
    args = parser.parse_args(argv)
    path, report = prepare_report(
        args.manifest or manifest_path(args.profile),
        args.profile,
        load_protocol(args.protocol),
        inventory=args.inventory,
    )
    print(
        json.dumps(
            {
                "report": str(path),
                "status": report["status"],
                "errors": report["errors"],
            }
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
