"""Standalone vector input validation and lightweight sealed-report verification.

Prepared vectors are input artifacts: this module never embeds or downloads.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.protocol import (
    integer,
    strict_object,
    string,
)

from .protocol import DEFAULT_PROTOCOL_PATH, FILTER_FIELDS, load_protocol
from .workload import Corpus

REPORT_ROOT = PROJECT_ROOT / "artifacts/benchmarks/data-validation"


def manifest_path(profile):
    return (
        Path(__file__).with_name("smoke.json")
        if profile == "smoke"
        else PROJECT_ROOT / f"data/benchmarks/vectordb/{profile}.json"
    )


def _array(value, parent):
    return (
        np.load(parent / value, allow_pickle=False)
        if isinstance(value, str)
        else np.asarray(value)
    )


def load_inputs(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    corpora = {}
    for row in payload["corpora"]:
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
        for corpus in payload["corpora"]:
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
        "validator_version": "vector-data-v1",
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
                PROJECT_ROOT / "experiments/benchmarks/common/datasets.py",
                PROJECT_ROOT
                / "experiments/benchmarks/rag/chunking_embedding/validation.py",
                PROJECT_ROOT
                / "experiments/benchmarks/rag/retrieval_reranking/validation.py",
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
            "corpora",
            "conditions",
            "operations",
            "integration",
            "review",
        },
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["split"] != ("locked-test" if profile == "locked" else profile)
    ):
        raise ValueError("Unsupported vector manifest schema/split")
    if not corpora or not payload["conditions"]:
        raise ValueError("Nonempty vector corpus and conditions are required")
    seen = set()
    for name, corpus in corpora.items():
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
            if corpus.provenance["kind"] not in {"representative", "synthetic-stress"}:
                raise ValueError("Synthetic stress provenance must be explicit")
            for group in corpus.source_groups:
                string(group, "independent query source")
            for row in corpus.metadata:
                string(
                    row.get("independent_source_id") or row.get("source_group_id"),
                    "reviewed record source group",
                )
        if any(not isinstance(text, str) for text in corpus.texts):
            raise ValueError("Canonical record texts must be strings")
    for condition in payload["conditions"]:
        strict_object(
            condition,
            "condition",
            {
                "id",
                "corpus",
                "mode",
                "filters",
                "query_ids",
                "concurrency",
                "windows",
                "warmup_requests",
            }
            | (
                {"independent_window_groups"}
                if "independent_window_groups" in condition
                else set()
            ),
        )
        name = string(condition["id"], "condition.id")
        if name in seen or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in name
        ):
            raise ValueError("Duplicate/unsafe condition identity")
        seen.add(name)
        corpus = corpora[condition["corpus"]]
        if (
            condition["mode"] not in {"controlled", "load"}
            or not set(condition["filters"]) <= FILTER_FIELDS
        ):
            raise ValueError("Unsupported condition mode/trusted filter field")
        if (
            not condition["query_ids"]
            or len(set(condition["query_ids"])) != len(condition["query_ids"])
            or not set(condition["query_ids"]) <= set(corpus.query_ids)
        ):
            raise ValueError("Unknown/duplicate condition query identity")
        integer(condition["concurrency"], "condition.concurrency", minimum=1)
        integer(condition["windows"], "condition.windows", minimum=1)
        integer(condition["warmup_requests"], "condition.warmup_requests", minimum=1)
        if "independent_window_groups" in condition:
            groups = condition["independent_window_groups"]
            if (
                not isinstance(groups, list)
                or len(groups) != condition["windows"]
                or any(
                    not isinstance(group, str) or not group.strip() for group in groups
                )
            ):
                raise ValueError(
                    "Reviewed independent window groups must cover every scheduled window"
                )
        if condition["mode"] == "controlled" and (
            condition["concurrency"] != 1 or condition["windows"] != 1
        ):
            raise ValueError("Controlled conditions require one serial window")
        for key, value in condition["filters"].items():
            if (
                not isinstance(value, (str, int, float, bool))
                or isinstance(value, float)
                and not np.isfinite(value)
            ):
                raise ValueError(f"Invalid trusted filter value: {key}")
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
            "corpus",
            "manifest",
            "embedding_selection",
            "retrieval_selection",
            "chunking_protocol",
            "retrieval_protocol",
            "model_lock",
            "filters",
            "files",
        }
        strict_object(integration, "integration", required)
        corpus = corpora[integration["corpus"]]
        if not set(integration["filters"]) <= FILTER_FIELDS:
            raise ValueError("Unsupported integration filter")
        from experiments.benchmarks.common.datasets import evidence_units, load_manifest
        from experiments.benchmarks.rag.chunking_embedding.protocol import (
            load_protocol as load_chunking,
        )
        from experiments.benchmarks.rag.retrieval_reranking.protocol import (
            load_protocol as load_retrieval,
        )
        from experiments.benchmarks.rag.retrieval_reranking.validation import (
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
        for record_id, text, metadata in zip(
            corpus.ids, corpus.texts, corpus.metadata, strict=True
        ):
            document = documents[metadata["source_id"]]
            start, end = metadata.get("start"), metadata.get("end")
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(document["text"])
                or text != document["text"][start:end]
            ):
                raise ValueError(
                    f"Canonical integration span/text mismatch: {record_id}"
                )
        eligible_documents = {
            row["source_id"]
            for row in corpus.metadata
            if all(
                row.get(key) == value for key, value in integration["filters"].items()
            )
        }
        questions = [
            row
            for row in manifest.samples
            if row.get("kind") == "question" and row.get("answerable")
        ]
        if {str(row["id"]) for row in questions} != set(corpus.query_ids):
            raise ValueError("Integration/vector query allocation mismatch")
        for question in questions:
            if any(
                interval.document_id not in eligible_documents
                for unit in evidence_units(question)
                for interval in unit.intervals
            ):
                raise ValueError("Gold evidence lies outside integration filter scope")
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
            for corpus in groups.values():
                record_groups = {
                    row.get("independent_source_id") or row.get("source_group_id")
                    for row in corpus.metadata
                }
                for key in [
                    *("source:" + group for group in record_groups if group),
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
        "condition_count": len(seen),
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
