"""Frozen full-stack retrieval within the database/configuration comparison.

The comparison imports this module; there is no separate retrieval-selection CLI.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace

import numpy as np

from benchmarks.common.contracts import (
    CandidateExecutionError,
    SampleResult,
)
from benchmarks.common.datasets import evidence_units, load_manifest, validate_evidence
from benchmarks.common.decisions import load_engineer_decision
from benchmarks.rag.chunking_embedding.metrics import (
    covered_unit_ids,
    score_question,
)
from benchmarks.rag.evaluation import InputCompatibilityError, OutputIntegrityError
from benchmarks.rag.methods import BM25
from benchmarks.rag.statistics import (
    SOURCE_KEYS,
    SOURCE_SCOPES,
    question_documents,
    question_mean,
    source_groups,
    source_ids,
    source_scope,
)
from edumind.common.artifacts import sha256_file, stable_hash
from edumind.rag.errors import RAGConfigurationError

from .docker_metrics import DockerMonitor
from .integration import SelectedServer
from .metrics import interval, percentile, usable_duration
from .selection import candidate_config
from .workload import SCOPES, global_inventory, search_rows


def prepare_integration(spec, parent, profile, corpora, sessions):
    from benchmarks.preparation.models import load_selected_model_lock
    from benchmarks.rag.chunking_embedding.protocol import (
        load_protocol as chunking,
    )
    from benchmarks.rag.evaluation import build_index, reranker_for
    from benchmarks.rag.retrieval_reranking.protocol import (
        load_protocol as retrieval,
    )
    from benchmarks.rag.retrieval_reranking.validation import verify_inputs

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
    sources = validate_evidence(manifest)
    questions = {
        row["id"]: {**row, **sources[row["id"]].metadata()}
        for row in manifest.samples
        if row["kind"] == "question" and row["answerable"]
    }
    selected_sessions = [
        session for session in sessions if session["id"] in spec["session_ids"]
    ]
    documents = {
        row["source_id"]
        for session in selected_sessions
        for row in corpora[session["id"]].metadata
    }
    local = replace(
        manifest,
        samples=tuple(
            row
            for row in manifest.samples
            if row["kind"] == "question" or row["id"] in documents
        ),
    )
    # Prepare the selected embedding runtime exactly once, then project the
    # frozen canonical build into each workspace. Lexical statistics stay local.
    index = build_index(
        local,
        chunker,
        embedding_name,
        model_lock,
        with_bm25=False,
        device=execution.device,
        dtype=execution.dtype,
        chunking_protocol=chunking_protocol,
        retrieval_protocol=retrieval_protocol,
    )
    bundles = _session_indexes(
        index, selected_sessions, corpora, questions, selected, retrieval_protocol
    )
    if set(questions) != {q["id"] for bundle in bundles for q in bundle["questions"]}:
        raise ValueError(
            "Integration question allocation differs from sealed vector queries"
        )
    return {
        "sessions": bundles,
        "global_ids": set(global_inventory(corpora)),
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
        "integration_provenance": provenance,
    }


def _session_indexes(index, sessions, corpora, questions, selected, protocol):
    by_id = {chunk.identifier: i for i, chunk in enumerate(index.chunks)}
    bundles = []
    for session in sessions:
        frozen = corpora[session["id"]]
        documents = {item["source_id"] for item in frozen.metadata}
        expected_ids = {
            chunk.identifier for chunk in index.chunks if chunk.document_id in documents
        }
        if set(frozen.ids) != expected_ids:
            raise ValueError(
                "Integration workspace must contain all canonical chunks of its documents"
            )
        positions = [by_id[identifier] for identifier in frozen.ids]
        chunks = [index.chunks[position] for position in positions]
        if tuple(chunk.text for chunk in chunks) != frozen.texts or not np.array_equal(
            index.vectors[positions], frozen.vectors
        ):
            raise ValueError(
                "Integration session corpus/vectors differ from frozen selected-model inputs"
            )
        lexical = (
            BM25(
                [chunk.text for chunk in chunks],
                k1=protocol.bm25_k1,
                b=protocol.bm25_b,
                epsilon=protocol.bm25_epsilon,
            )
            if selected.split("|", 1)[0] in {"bm25", "rrf"}
            else None
        )
        bundles.append(
            {
                "index": replace(
                    index, chunks=chunks, vectors=frozen.vectors, bm25=lexical
                ),
                "session": session,
                "corpus": frozen,
                "questions": [questions[qid] for qid in frozen.query_ids],
            }
        )
    return bundles


def evaluate_integration(candidate, prepared, protocol, profile, *, evidence_sink=None):
    retained = []
    for bundle in prepared["sessions"]:
        index = bundle["index"]
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
            global_ids=prepared["global_ids"],
        )
        try:
            requests, rankings, problems, monitoring = _execute_integration(
                server, bundle, protocol, profile, database
            )
        finally:
            server.close()
        retained.append(
            {
                "session_id": bundle["session"]["id"],
                "requests": requests,
                "questions": bundle["questions"],
                "rankings": rankings,
                "problems": problems,
                "monitoring": monitoring,
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
            }
        )
    tokenizer = prepared["sessions"][0]["index"].tokenizer
    evidence = {
        "candidate": candidate,
        "sessions": retained,
        "spec": prepared["spec"],
        "retrieval": prepared["retrieval"],
        "integration_provenance": prepared.get("integration_provenance", {}),
        "tokenizer": getattr(tokenizer, "name", "tiktoken:cl100k_base"),
    }
    if evidence_sink is not None:
        evidence_sink(evidence)
    return score_integration_evidence(evidence, protocol, profile, tokenizer=tokenizer)


POOL_RECALL = "diagnostic.candidate_pool_evidence_unit_recall_at_20"
APPLICATION_METRICS = (
    *(
        f"quality.{name}_at_{k}"
        for name in ("ndcg", "evidence_unit_recall", "evidence_token_precision")
        for k in (3, 5)
    ),
    POOL_RECALL,
)


def score_integration_evidence(evidence, protocol, profile, *, tokenizer):
    """Rescore original verified first outputs across sessions, by question."""
    from types import SimpleNamespace

    query_rows, requests, rankings, resources, problems = [], [], [], [], []
    for bundle in evidence["sessions"]:
        chunks = [SimpleNamespace(**row) for row in bundle["chunks"]]
        questions = {question["id"]: question for question in bundle["questions"]}
        requests.extend(bundle["requests"])
        rankings.extend(bundle["rankings"])
        resources.extend(
            {**row, "session_id": bundle["session_id"]}
            for row in bundle["monitoring"]["samples"]
            + bundle["monitoring"]["checkpoints"]
        )
        problems.extend(bundle["problems"])
        by_query = {}
        for row in bundle["rankings"]:
            by_query.setdefault((row["query_id"], row["condition"]), []).append(row)
        for (qid, scope), rows in by_query.items():
            question = questions[qid]
            first = next(row for row in rows if row["quality_designated"])
            selected = first["selected_source_ids"]
            units = evidence_units(question)
            applicable = selected is None or all(
                unit.document_id in selected for unit in units
            )
            metrics = dict.fromkeys(APPLICATION_METRICS) if applicable else {}
            if applicable:
                observed = first["evidence"]
                if observed.get("pool") is not None:
                    pool = [chunks[i] for i in observed["pool"]]
                    metrics[POOL_RECALL] = len(
                        set().union(*(covered_unit_ids(chunk, units) for chunk in pool))
                    ) / len(units)
                if observed.get("ranking") is not None:
                    eligible = [
                        chunk
                        for chunk in chunks
                        if selected is None or chunk.document_id in selected
                    ]
                    try:
                        score = score_question(
                            question,
                            [chunks[i] for i in observed["ranking"]],
                            eligible,
                            tokenizer,
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
                            f"Integration scoring integrity {qid}: {type(exc).__name__}: {exc}"
                        )
                if any(value is None for value in metrics.values()):
                    problems.append(
                        f"Incomplete first integration quality: {bundle['session_id']}/{scope}/{qid}"
                    )
            for row in rows:
                if row["status"] == "not_executed" or row.get("integrity_error"):
                    problems.append(
                        f"Integration execution/integrity incomplete: {bundle['session_id']}/{scope}/{qid}/{row['attempt']}"
                    )
            query_rows.append(
                {
                    "query_id": qid,
                    "session_id": bundle["session_id"],
                    "condition": scope,
                    "selected_source_ids": selected,
                    "paired": first["paired"],
                    **{key: question[key] for key in SOURCE_KEYS},
                    "source_scope": source_scope(question),
                    "quality_eligible": applicable,
                    "quality_status": "inapplicable"
                    if not applicable
                    else "incomplete"
                    if any(value is None for value in metrics.values())
                    else "scored",
                    "quality_reason": "selected_scope_excludes_required_evidence"
                    if not applicable
                    else None,
                    "metrics": metrics,
                    "latency_seconds": percentile(
                        [
                            row["latency_seconds"]
                            for row in rows
                            if row["status"] == "success"
                            and usable_duration(row["latency_seconds"])
                        ],
                        0.5,
                    ),
                }
            )

    aggregates, intervals, statuses, samples = {}, {}, {}, []

    def metadata(row):
        return row

    for scope in SCOPES:
        local = [row for row in query_rows if row["condition"] == scope]
        cid = "integration-" + scope
        for row in local:
            sample_metrics = {}
            for name, value in row["metrics"].items():
                sample_metrics[name + ".condition." + cid] = value
                sample_metrics[
                    f"slice.{row['source_scope']}.{name}.condition.{cid}"
                ] = value
            samples.append(
                SampleResult(
                    f"{cid}:{row['session_id']}:{row['query_id']}",
                    sample_metrics,
                    row["latency_seconds"],
                    {key: row[key] for key in SOURCE_KEYS},
                )
            )
        for name in APPLICATION_METRICS:
            for question_scope in ("overall", *SOURCE_SCOPES):
                eligible = [
                    row
                    for row in local
                    if row["quality_eligible"]
                    and (
                        question_scope == "overall"
                        or row["source_scope"] == question_scope
                    )
                ]
                base = (
                    name
                    if question_scope == "overall"
                    else f"slice.{question_scope}.{name}"
                )
                key = base + ".condition." + cid

                def statistic(draw, name=name):
                    return question_mean(
                        row["metrics"][name] for group in draw for row in group
                    )

                _integration_metric(
                    eligible,
                    key,
                    statistic,
                    metadata,
                    aggregates,
                    intervals,
                    statuses,
                    protocol,
                    profile,
                    quality=True,
                )
        for q in (0.5, 0.95):
            key = f"operational.full_stack_retrieval_latency_ms_p{int(q * 100)}.condition.{cid}"

            def statistic(draw, q=q):
                return percentile(
                    [
                        row["latency_seconds"] * 1000
                        for group in draw
                        for row in group
                        if row["latency_seconds"] is not None
                    ],
                    q,
                )

            _integration_metric(
                local,
                key,
                statistic,
                metadata,
                aggregates,
                intervals,
                statuses,
                protocol,
                profile,
            )
    artifacts = {
        "integration_requests": requests,
        "integration_rankings": rankings,
        "integration_query_metrics": query_rows,
        "integration_metric_statuses": statuses,
        "integration_validation": {
            "status": "failed" if problems else "passed",
            "problems": problems,
            "monitoring": {
                row["session_id"]: row["monitoring"] for row in evidence["sessions"]
            },
        },
        "integration_resources": resources,
    }
    usable = [row["bytes"] for row in resources if row.get("bytes") is not None]
    aggregates["operational.peak_server_ram_bytes.scope.server-lifecycle"] = (
        max(usable) if usable else None
    )
    parameters = {
        "retrieval_stack": evidence["retrieval"],
        "integration_provenance": evidence["integration_provenance"],
    }
    if problems:
        raise CandidateExecutionError(
            "; ".join(problems),
            samples=tuple(samples),
            metrics=aggregates,
            parameters=parameters,
            intervals=intervals,
            artifacts=artifacts,
        )
    return samples, {}, aggregates, parameters, intervals, artifacts


def _integration_metric(
    cohort,
    key,
    statistic,
    metadata,
    aggregates,
    intervals,
    statuses,
    protocol,
    profile,
    *,
    quality=False,
):
    contributing = [row for row in cohort if statistic([[row]]) is not None]
    incomplete = quality and len(cohort) != len(contributing)
    reason = (
        "inapplicable"
        if not cohort
        else "required_quality_coverage_incomplete"
        if incomplete
        else "unknown_independent_source_grouping"
        if profile != "smoke"
        and any(row["independent_source_id"] is None for row in cohort)
        else None
    )
    bounds = interval(
        source_groups(cohort, metadata),
        statistic,
        protocol=protocol,
        profile=profile,
        metric=key,
        unit="reviewed_source_group",
        blocked_reason=reason,
    )
    bounds.update(
        independent_source_count=len(source_ids(cohort)),
        contributing_source_count=len(source_ids(contributing)),
    )
    aggregates[key] = None if incomplete else bounds["estimate"]
    intervals[key] = bounds
    aggregates[key + ".sample_count"] = len(contributing)
    aggregates[key + ".independent_source_count"] = len(source_ids(contributing))
    statuses[key] = {
        "status": "inapplicable"
        if not cohort
        else "incomplete"
        if incomplete
        else "scored"
        if contributing
        else "unavailable",
        "eligible_questions": len(cohort),
        "contributing_questions": len(contributing),
        "eligible_documents": len(question_documents(cohort)),
        "contributing_documents": len(question_documents(contributing)),
        "ci": bounds,
    }


def _execute_integration(server, bundle, protocol, profile, database):
    """Finish the resource window before evidence scoring or bootstrap."""
    session, corpus = bundle["session"], bundle["corpus"]
    questions = {q["id"]: q for q in bundle["questions"]}
    schedule = list(search_rows(session, corpus))
    warmups = protocol.profile(profile).warmups
    checkpoint_id = "integration-" + session["id"]
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
    blocked = set()
    with monitor:
        try:
            server.prepare()
            checkpoint = monitor.checkpoint(checkpoint_id)
        except Exception as exc:  # noqa: BLE001 - retain backend/telemetry setup failures and unexecuted rows
            problems.append(f"Integration preparation: {type(exc).__name__}: {exc}")
        else:
            if protocol.monitor_storage and not checkpoint["complete"]:
                problems.append(
                    "Missing integration completed-build storage checkpoint"
                )
            ready = True
        if ready:
            for scope in SCOPES:
                representative = next(
                    (item for item in schedule if item[0]["condition"] == scope), None
                )
                if representative is None:
                    continue
                identity, filters, _ = representative
                question = questions[identity["query_id"]]
                try:
                    for _ in range(warmups):
                        server.retrieve(question, filters=filters)
                except Exception as exc:  # noqa: BLE001 - any model/client warmup failure blocks only this scope
                    blocked.add(scope)
                    problems.append(
                        f"Integration warmup {scope}: {type(exc).__name__}: {exc}"
                    )
        for identity, filters, _ in schedule:
            question = questions[identity["query_id"]]
            evidence = {}
            row = {
                **identity,
                "operation": "integration",
                "status": "not_executed",
                "error": "preparation_or_warmup_failed",
                "latency_seconds": None,
                "configuration_checksum": stable_hash(asdict(server.config)),
            }
            if ready and identity["condition"] not in blocked:
                started = time.perf_counter()
                try:
                    result = server.retrieve(
                        question,
                        filters=filters,
                        evidence=evidence,
                    )
                    latency = result["latency_seconds"]
                except Exception as exc:  # noqa: BLE001 - retain partial model/client evidence and continue the schedule
                    row.update(
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        integrity_error=isinstance(
                            exc,
                            (
                                ValueError,
                                InputCompatibilityError,
                                OutputIntegrityError,
                                RAGConfigurationError,
                            ),
                        )
                        or bool(
                            evidence.get("filter_violation")
                            or evidence.get("session_isolation_violation")
                        ),
                        elapsed_to_failure_seconds=time.perf_counter() - started,
                    )
                else:
                    row.update(status="success", latency_seconds=latency, error=None)
            row.update(
                ids=evidence.get("database_ids"),
                filter_violation=evidence.get("filter_violation"),
                session_isolation_violation=evidence.get("session_isolation_violation"),
                database_executed=evidence.get("database_executed", False),
                database_latency_seconds=evidence.get("database_latency_seconds"),
            )
            requests.append(row)
            rankings.append({**row, "evidence": evidence})
    monitoring = monitor.evidence()
    if protocol.monitor_server_memory and monitoring["coverage"] != "complete":
        problems.append("Incomplete integration server telemetry")
    return requests, rankings, problems, monitoring
