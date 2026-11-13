"""Frozen full-stack retrieval within the database/configuration comparison.

The comparison imports this module; there is no separate retrieval-selection CLI.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from edumind.common.artifacts import sha256_file
from experiments.benchmarks.common.contracts import (
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.datasets import evidence_units, load_manifest
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.rag.chunking_embedding.metrics import (
    covered_unit_ids,
    score_question,
)

from .docker_metrics import DockerMonitor
from .integration import SelectedServer
from .metrics import interval, mean, percentile, repeatability
from .selection import candidate_config


def prepare_integration(spec, parent, profile, corpora):
    from experiments.benchmarks.preparation.models import load_selected_model_lock
    from experiments.benchmarks.rag.chunking_embedding.protocol import (
        load_protocol as chunking,
    )
    from experiments.benchmarks.rag.evaluation import build_index, reranker_for
    from experiments.benchmarks.rag.retrieval_reranking.protocol import (
        load_protocol as retrieval,
    )
    from experiments.benchmarks.rag.retrieval_reranking.validation import verify_inputs

    chunking_protocol = chunking(parent / spec["chunking_protocol"])
    retrieval_protocol = retrieval(parent / spec["retrieval_protocol"])
    manifest_file = parent / spec["manifest"]
    verify_inputs(
        manifest_file, profile, retrieval_protocol, chunking_protocol=chunking_protocol
    )
    embedding_decision = load_engineer_decision(
        parent / spec["embedding_selection"],
        exact=1,
        expected_source=("rag", "chunking-embedding", "validation"),
        expected_protocols={"chunking_embedding": chunking_protocol.meta.checksum},
    )
    embedding = embedding_decision.selected_candidates[0]
    retrieval_decision = load_engineer_decision(
        parent / spec["retrieval_selection"],
        exact=1,
        expected_source=("rag", "retrieval-reranking", "validation"),
        expected_protocols={"retrieval": retrieval_protocol.meta.checksum},
    )
    selected = retrieval_decision.selected_candidates[0]
    provenance = {
        "chunking_embedding_candidate": embedding,
        "retrieval_candidate": selected,
        "model_lock_sha256": sha256_file(parent / spec["model_lock"]),
        "chunking_protocol_checksum": chunking_protocol.meta.checksum,
        "retrieval_protocol_checksum": retrieval_protocol.meta.checksum,
        "embedding_source_run_id": embedding_decision.source_run_id,
        "retrieval_source_run_id": retrieval_decision.source_run_id,
        "embedding_decision_sha256": sha256_file(parent / spec["embedding_selection"]),
        "retrieval_decision_sha256": sha256_file(parent / spec["retrieval_selection"]),
    }
    model_lock = load_selected_model_lock(parent / spec["model_lock"])
    chunker, embedding_name = embedding.split("|", 1)
    execution = retrieval_protocol.profile(profile)
    manifest = load_manifest(manifest_file)
    index = build_index(
        manifest,
        chunker,
        embedding_name,
        model_lock,
        with_bm25=selected.split("|", 1)[0] in {"bm25", "rrf"},
        device=execution.device,
        dtype=execution.dtype,
        chunking_protocol=chunking_protocol,
        retrieval_protocol=retrieval_protocol,
    )
    frozen = corpora[spec["corpus"]]
    if (
        tuple(chunk.identifier for chunk in index.chunks) != frozen.ids
        or tuple(chunk.text for chunk in index.chunks) != frozen.texts
        or index.vectors.shape != frozen.vectors.shape
        or not np.array_equal(index.vectors, frozen.vectors)
    ):
        raise ValueError(
            "Integration corpus/vectors differ from the sealed selected-model inputs"
        )
    questions = [
        row
        for row in manifest.samples
        if row.get("kind") == "question" and row.get("answerable")
    ]
    if not questions or {str(row["id"]) for row in questions} != set(frozen.query_ids):
        raise ValueError(
            "Integration question allocation differs from sealed vector queries"
        )
    return {
        "index": index,
        "questions": questions,
        "retrieval": selected,
        "model_lock": model_lock,
        "retrieval_protocol": retrieval_protocol,
        "reranker_factory": lambda: reranker_for(
            selected,
            model_lock,
            device=execution.device,
            dtype=execution.dtype,
            retrieval_protocol=retrieval_protocol,
        ),
        "spec": spec,
        "source_groups": dict(zip(frozen.query_ids, frozen.source_groups, strict=True)),
        "integration_provenance": provenance,
    }


def evaluate_integration(candidate, prepared, protocol, profile, *, evidence_sink=None):
    index, questions, spec = prepared["index"], prepared["questions"], prepared["spec"]
    database, config = candidate_config(
        protocol, candidate, int(index.vectors.shape[1])
    )
    server = SelectedServer(
        database,
        config,
        index,
        prepared["retrieval"],
        prepared["retrieval_protocol"],
        reranker=prepared["reranker_factory"](),
    )
    cid = "integration-" + spec["corpus"]
    try:
        requests, rankings, problems, monitoring = _execute_integration(
            server, questions, spec, protocol, profile, cid, database
        )
    finally:
        server.close()
    evidence = {
        "candidate": candidate,
        "requests": requests,
        "rankings": rankings,
        "problems": problems,
        "monitoring": monitoring,
        "questions": questions,
        "spec": spec,
        "retrieval": prepared["retrieval"],
        "source_groups": prepared.get("source_groups", {}),
        "integration_provenance": prepared.get("integration_provenance", {}),
        "chunks": [
            {
                key: getattr(chunk, key)
                for key in (
                    "identifier",
                    "document_id",
                    "text",
                    "start",
                    "end",
                    "tokens",
                )
            }
            for chunk in index.chunks
        ],
        "metadata": [dict(row.metadata) for row in server.records],
        "tokenizer": getattr(index.tokenizer, "name", "tiktoken:cl100k_base"),
    }
    if evidence_sink is not None:
        evidence_sink(evidence)
    return score_integration_evidence(
        evidence, protocol, profile, tokenizer=index.tokenizer
    )


def score_integration_evidence(evidence, protocol, profile, *, tokenizer):
    """Re-evaluate original pools/rankings only; never regenerate retrieval."""
    from types import SimpleNamespace

    prepared = evidence
    spec, questions = evidence["spec"], evidence["questions"]
    index = SimpleNamespace(
        chunks=[SimpleNamespace(**row) for row in evidence["chunks"]],
        tokenizer=tokenizer,
    )
    server = SimpleNamespace(
        records=[SimpleNamespace(metadata=row) for row in evidence["metadata"]],
        close=lambda: None,
    )
    requests, rankings, monitoring = (
        evidence["requests"],
        evidence["rankings"],
        evidence["monitoring"],
    )
    query_rows, pair_rows, samples, problems = [], [], [], list(evidence["problems"])
    execution = protocol.profile(profile)
    cid = "integration-" + spec["corpus"]
    try:
        for question in questions:
            attempts = [
                row for row in requests if row["query_id"] == str(question["id"])
            ]
            first = next(
                row["evidence"]
                for row in rankings
                if row["query_id"] == str(question["id"]) and row["attempt"] == 1
            )
            metrics = {
                f"quality.{name}_at_{k}": None
                for name in ("ndcg", "evidence_unit_recall", "evidence_token_precision")
                for k in (3, 5)
            }
            metrics["diagnostic.candidate_pool_evidence_unit_recall_at_20"] = None
            units = evidence_units(question)
            if first.get("pool") is not None:
                pool = [index.chunks[i] for i in first["pool"]]
                metrics["diagnostic.candidate_pool_evidence_unit_recall_at_20"] = len(
                    set().union(*(covered_unit_ids(chunk, units) for chunk in pool))
                ) / len(units)
            if first.get("ranking") is not None:
                eligible = [
                    chunk
                    for chunk, record in zip(index.chunks, server.records, strict=True)
                    if all(
                        record.metadata.get(key) == value
                        for key, value in spec.get("filters", {}).items()
                    )
                ]
                try:
                    score = score_question(
                        question,
                        [index.chunks[i] for i in first["ranking"]],
                        eligible,
                        index.tokenizer,
                        cutoffs=(3, 5),
                        alpha=0.5,
                    )
                    metrics.update(
                        {
                            "quality." + key: value
                            for key, value in score.metrics.items()
                            if not key.startswith("alpha_")
                        }
                    )
                except (ValueError, TypeError, KeyError) as exc:
                    problems.append(
                        f"Integration scoring integrity {question['id']}: {type(exc).__name__}: {exc}"
                    )
            repeated = repeatability(
                [
                    {
                        **row,
                        "status": "success"
                        if row["ids"] is not None and not row["filter_violation"]
                        else row["status"]
                        if row["status"] == "not_executed"
                        else "failed",
                    }
                    for row in attempts
                ],
                execution.repetitions,
            )
            if prepared["retrieval"].split("|", 1)[0] != "bm25":
                metrics["diagnostic.ranking_repeatability"] = repeated["value"]
                pair_rows.extend(
                    {"query_id": str(question["id"]), **pair}
                    for pair in repeated["pairs"]
                )
            latency = percentile(
                [
                    row["latency_seconds"]
                    for row in attempts
                    if row["status"] == "success"
                ],
                0.5,
            )
            group = prepared.get("source_groups", {}).get(str(question["id"])) or str(
                question.get("independent_source_id")
                or question.get("source_group_id")
                or question["document_id"]
            )
            query_rows.append(
                {
                    "query_id": str(question["id"]),
                    "document_id": str(question["document_id"]),
                    "source_group": group,
                    "metrics": metrics,
                    "latency_seconds": latency,
                }
            )
            samples.append(
                SampleResult(
                    cid + ":" + str(question["id"]),
                    {
                        key + ".condition." + cid: value
                        for key, value in metrics.items()
                    },
                    latency,
                    {
                        "source_group": group,
                        "document_id": str(question["document_id"]),
                        "condition": cid,
                    },
                )
            )
            if any(
                value is None
                for key, value in metrics.items()
                if key.startswith("quality.")
            ):
                problems.append(
                    f"Incomplete first integration quality: {question['id']}"
                )
    finally:
        server.close()
    grouped = defaultdict(list)
    for row in query_rows:
        grouped[row["source_group"]].append(row)
    groups = list(grouped.values())
    aggregates, intervals, statuses = {}, {}, {}
    names = {key for row in query_rows for key in row["metrics"]}
    for name in names:

        def statistic(draw, name=name):
            documents = []
            # Retain duplicate draw copies as distinct sources/documents.
            for source in draw:
                docs = defaultdict(list)
                for row in source:
                    docs[row["document_id"]].append(row["metrics"].get(name))
                documents.extend(mean(values) for values in docs.values())
            return mean(documents)

        key = name + ".condition." + cid
        bounds = interval(
            groups,
            statistic,
            protocol=protocol,
            profile=profile,
            metric=key,
            unit="reviewed_source_group",
        )
        aggregates[key], intervals[key] = bounds["estimate"], bounds
        aggregates[key + ".sample_count"] = len(
            {
                row["document_id"]
                for row in query_rows
                if row["metrics"].get(name) is not None
            }
        )
        aggregates[key + ".independent_source_count"] = bounds[
            "contributing_source_count"
        ]
        statuses[key] = {
            "status": "incomplete"
            if any(row["metrics"].get(name) is None for row in query_rows)
            else "scored",
            "ci": bounds,
        }
    for q in (0.5, 0.95):
        name = f"operational.full_stack_retrieval_latency_ms_p{int(q * 100)}"
        key = name + ".condition." + cid
        bounds = interval(
            groups,
            lambda draw, q=q: percentile(
                [
                    row["latency_seconds"] * 1000
                    for group in draw
                    for row in group
                    if row["latency_seconds"] is not None
                ],
                q,
            ),
            protocol=protocol,
            profile=profile,
            metric=key,
            unit="reviewed_source_group",
        )
        aggregates[key], intervals[key] = bounds["estimate"], bounds
        aggregates[key + ".sample_count"] = sum(
            row["latency_seconds"] is not None for row in query_rows
        )
        aggregates[key + ".independent_source_count"] = bounds[
            "contributing_source_count"
        ]
    artifacts = {
        "integration_requests": requests,
        "integration_rankings": rankings,
        "integration_query_metrics": query_rows,
        "integration_repeatability": pair_rows,
        "integration_metric_statuses": statuses,
        "integration_validation": {
            "status": "failed" if problems else "passed",
            "problems": problems,
            "monitoring": monitoring,
        },
        "integration_resources": monitoring["samples"] + monitoring["checkpoints"],
    }
    usable = [row["bytes"] for row in monitoring["samples"] if row["bytes"] is not None]
    aggregates["operational.peak_server_ram_bytes.scope.server-lifecycle"] = (
        max(usable) if usable else None
    )
    if problems:
        raise CandidateExecutionError(
            "; ".join(problems),
            samples=tuple(samples),
            metrics=aggregates,
            intervals=intervals,
            artifacts=artifacts,
        )
    return (
        samples,
        {},
        aggregates,
        {
            "retrieval_stack": prepared["retrieval"],
            "integration_provenance": prepared.get("integration_provenance", {}),
        },
        intervals,
        artifacts,
    )


def _execute_integration(server, questions, spec, protocol, profile, cid, database):
    """Finish the resource window before evidence scoring or bootstrap."""
    monitor = DockerMonitor(
        database,
        interval_seconds=protocol.docker_monitor_interval_seconds,
        stats_timeout_seconds=protocol.docker_stats_timeout_seconds,
        storage_timeout_seconds=protocol.docker_storage_timeout_seconds,
        helper_timeout_seconds=protocol.docker_helper_timeout_seconds,
        monitor_memory=protocol.monitor_server_memory,
        monitor_storage=protocol.monitor_storage,
    )
    requests, rankings, problems = [], [], []
    ready = False
    with monitor:
        try:
            server.prepare()
            checkpoint = monitor.checkpoint("integration-" + spec["corpus"])
            if protocol.monitor_storage and not checkpoint["complete"]:
                problems.append(
                    "Missing integration completed-build storage checkpoint"
                )
            server.retrieve(questions[0], filters=spec.get("filters"))
            ready = True
        except Exception as exc:  # noqa: BLE001 - preserve explicit unexecuted schedules
            problems.append(
                f"Integration preparation/warmup: {type(exc).__name__}: {exc}"
            )
        for question in questions:
            for attempt in range(1, protocol.profile(profile).repetitions + 1):
                evidence = {}
                row = {
                    "condition": cid,
                    "query_id": str(question["id"]),
                    "attempt": attempt,
                    "status": "not_executed",
                    "error": "preparation_failed" if not ready else None,
                    "latency_seconds": None,
                    "filter_violation": None,
                    "ids": None,
                }
                if ready:
                    started = time.perf_counter()
                    try:
                        result = server.retrieve(
                            question, filters=spec.get("filters"), evidence=evidence
                        )
                        row.update(
                            status="success",
                            latency_seconds=result["latency_seconds"],
                            error=None,
                        )
                    except Exception as exc:  # noqa: BLE001 - retain completed upstream stages
                        row.update(
                            status="failed",
                            error=f"{type(exc).__name__}: {exc}",
                            elapsed_to_failure_seconds=time.perf_counter() - started,
                        )
                row.update(
                    ids=evidence.get("database_ids"),
                    filter_violation=evidence.get("filter_violation"),
                    database_executed=evidence.get("database_executed", False),
                    database_latency_seconds=evidence.get("database_latency_seconds"),
                )
                requests.append(row)
                rankings.append({**row, "evidence": evidence})
    monitoring = monitor.evidence()
    if protocol.monitor_server_memory and monitoring["coverage"] != "complete":
        problems.append("Incomplete integration server telemetry")
    return requests, rankings, problems, monitoring


if __name__ == "__main__":
    raise SystemExit(
        "The separate retrieval CLI is retired; use experiments.benchmarks.vectordb.run"
    )
