"""Frozen database/configuration comparison; never performs hidden tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from edumind.common.artifacts import sha256_file, stable_hash
from experiments.benchmarks.common.arguments import default_decision_path
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
)
from experiments.benchmarks.common.runner import run_benchmark

from .benchmark import evaluate_candidate
from .docker_metrics import image_lock, verify_image
from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol
from .selection import STAGE, SUITE, candidate_config, selected_candidates
from .validation import REPORT_ROOT, load_inputs, manifest_path, verify_inputs

DIRECTORY = Path(__file__).parent
COMPOSE = DIRECTORY / "compose.yml"
CSV_TABLES = (
    "requests",
    "rankings",
    "query_metrics",
    "repeatability",
    "timings",
    "resources",
    "samples",
    "integration_requests",
    "integration_rankings",
    "integration_query_metrics",
    "integration_repeatability",
    "integration_resources",
)


def _table_cells(artifacts):
    """Canonicalize nested cells before child upload, not after persistence."""
    import numpy as np

    def cell(value):
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            return json.dumps(value, sort_keys=True, default=lambda item: item.tolist())
        return value

    return {
        name: [{key: cell(value) for key, value in row.items()} for row in payload]
        if isinstance(payload, list)
        else payload
        for name, payload in artifacts.items()
        if not isinstance(payload, list) or payload
    }


def parent_artifacts(directory, results, plan, *, protocol=None):
    """Equivalent CSV/Parquet values: complex cells use canonical JSON in both."""
    paths = []
    rows = [
        {
            "candidate": result.candidate,
            "status": result.status,
            "metric": name,
            "value": value,
        }
        for result in results
        for name, value in result.metrics.items()
    ]
    frame = pd.DataFrame(rows)
    for suffix in ("csv", "parquet"):
        path = directory / f"comparison.{suffix}"
        if suffix == "csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_parquet(path, index=False)
        paths.append(path)
    retained = []
    for child in (directory / "candidates").iterdir():
        if not child.is_dir() or not list(child.glob("retained-*-context.json")):
            continue
        from .recover import load_retained

        if not list(child.glob("retained-*-server.json")):
            paths.extend(child.glob("retained-*.json"))
            continue  # Pre-execution failure is retained, never treated as a paired observation.
        context, stages, _ = load_retained(child, verify_inputs=False)
        retained.append((context, stages["server"]))
        paths.extend(child.glob("retained-*.json"))
    if protocol is not None:
        from itertools import combinations

        from .comparisons import paired_operational, paired_quality

        paired = []
        for (left_context, left), (right_context, right) in combinations(retained, 2):
            if (
                left_context["input_validation"] != right_context["input_validation"]
                or left_context["workloads"] != right_context["workloads"]
                or left_context.get("integration_provenance")
                != right_context.get("integration_provenance")
            ):
                raise ValueError(
                    "Paired candidates used different frozen inputs/conditions/upstream components"
                )
            paired.extend(
                paired_operational(
                    left, right, left_context["workloads"], protocol, plan.profile
                )
            )
            by_candidate = {result.candidate: result for result in results}
            paired.extend(
                paired_quality(
                    by_candidate[left["candidate"]],
                    by_candidate[right["candidate"]],
                    protocol,
                    plan.profile,
                )
            )
        if paired:
            frame = pd.DataFrame(paired)
            pq, csv = (
                directory / "paired_operational.parquet",
                directory / "paired_operational.csv",
            )
            frame.to_parquet(pq, index=False)
            frame.to_csv(csv, index=False)
            pd.testing.assert_frame_equal(
                frame.fillna("").astype(str),
                pd.read_csv(csv, dtype=str, keep_default_na=False),
            )
            paths.extend((pq, csv))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "development", "validation", "locked"),
        default="smoke",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--inventory", type=Path, action="append")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--shortlist", "--database-selection", type=Path)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/benchmarks")
    )
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    execution = protocol.profile(args.profile)
    if args.profile != "smoke" and not (
        protocol.monitor_server_memory and protocol.monitor_storage
    ):
        raise ValueError(
            "Authoritative vector execution requires lifecycle memory and storage monitoring"
        )
    path = args.manifest or manifest_path(args.profile)
    report_path, report = verify_inputs(
        path,
        args.profile,
        protocol,
        inventory=args.inventory,
        report_root=args.report_root,
    )
    payload, corpora = load_inputs(path)
    decision_path = None
    if args.profile in {"validation", "locked"}:
        decision_path = args.shortlist or default_decision_path(
            "vector-database", args.profile
        )
        candidates = selected_candidates(
            decision_path, args.profile, protocol
        ).selected_candidates
    else:
        candidates = tuple(row["id"] for row in protocol.candidates[args.profile])
    if not candidates:
        raise ValueError(
            "No reviewed database/configuration roster; freeze development candidates before execution"
        )
    for candidate in candidates:
        for corpus in corpora.values():
            candidate_config(protocol, candidate, int(corpus.vectors.shape[1]))
    revisions = image_lock()
    locked_reservation = None
    # Reserve before any held-out corpus/model preparation, not merely requests.
    if args.profile == "locked":
        marker = (
            args.artifact_root
            / "locked-consumption"
            / (stable_hash(report["files"]) + ".json")
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        try:
            with marker.open("x", encoding="utf-8") as handle:
                json.dump(
                    {"input": report["fingerprint"], "candidate": candidates[0]}, handle
                )
        except FileExistsError as exc:
            raise ValueError(
                "Vector locked inputs already reserved/consumed; do not rerun inference"
            ) from exc
        locked_reservation = {
            "path": str(marker.resolve()),
            "sha256": sha256_file(marker),
        }
    # Build/prepare the selected model-backed bundle before server resource windows.
    integration = None
    if payload["integration"]:
        from .retrieval_run import prepare_integration

        integration = prepare_integration(
            payload["integration"], Path(path).parent, args.profile, corpora
        )
    elif args.profile != "smoke":
        raise ValueError(
            "Authoritative comparison requires the frozen full-stack integration inputs"
        )
    plan = BenchmarkPlan(
        SUITE,
        STAGE,
        args.profile,
        payload["name"],
        candidates,
        protocol.meta.seed,
        execution.repetitions,
        execution.bootstrap_resamples,
        execution.warmups,
        {
            "device": "cpu",
            "dtype": execution.dtype,
            "input_validation": report["fingerprint"],
            "integration_provenance": integration["integration_provenance"]
            if integration
            else {},
        },
    )

    def evaluate(candidate, execution_context):
        from .recover import snapshot_writer

        save, retained = snapshot_writer(
            execution_context["artifact_directory"],
            {
                "candidate": candidate,
                "profile": args.profile,
                "source_run_id": execution_context["mlflow_run_id"],
                "input_validation": report["fingerprint"],
                "input_report": report,
                "manifest": str(Path(path).resolve()),
                "protocol": protocol.meta.artifact_payload(),
                "corpus_checksums": {
                    name: corpus.content_checksum for name, corpus in corpora.items()
                },
                "integration_required": integration is not None,
                "integration_provenance": integration["integration_provenance"]
                if integration
                else {},
                "locked_reservation": locked_reservation,
                "workloads": {
                    name: payload[name] for name in ("conditions", "operations")
                },
            },
        )
        database, _ = candidate_config(
            protocol, candidate, int(next(iter(corpora.values())).vectors.shape[1])
        )
        verify_image(
            database,
            revisions[f"image:{database}"],
            timeout_seconds=protocol.docker_storage_timeout_seconds,
        )
        output = ([], {}, {}, {}, {}, {})
        try:
            output = evaluate_candidate(
                candidate,
                payload,
                corpora,
                protocol,
                args.profile,
                COMPOSE,
                evidence_sink=lambda evidence: save("server", evidence),
            )
            if integration is not None:
                from .retrieval_run import evaluate_integration

                extra = evaluate_integration(
                    candidate,
                    integration,
                    protocol,
                    args.profile,
                    evidence_sink=lambda evidence: save("integration", evidence),
                )
                output = _merge(output, extra)
        except CandidateExecutionError as exc:
            output = _merge(
                output,
                (
                    list(exc.samples),
                    exc.operational,
                    exc.metrics,
                    exc.parameters,
                    exc.intervals,
                    exc.artifacts,
                ),
            )
            raise CandidateExecutionError(
                str(exc),
                samples=tuple(output[0]),
                operational=output[1],
                metrics=output[2],
                parameters=output[3],
                intervals=output[4],
                artifacts={**_table_cells(output[5]), "execution_evidence": retained},
            ) from exc
        except Exception as exc:
            raise CandidateExecutionError(
                f"{type(exc).__name__}: {exc}",
                samples=tuple(output[0]),
                operational=output[1],
                metrics=output[2],
                parameters=output[3],
                intervals=output[4],
                artifacts={**_table_cells(output[5]), "execution_evidence": retained},
            ) from exc
        samples, operational, metrics, parameters, intervals, artifacts = output
        artifacts["execution_evidence"] = retained
        artifacts.setdefault(
            "samples",
            [
                {
                    "sample_id": row.sample_id,
                    "metrics": dict(row.metrics),
                    "latency_seconds": row.latency_seconds,
                    **dict(row.metadata),
                }
                for row in samples
            ],
        )
        return (
            samples,
            operational,
            metrics,
            parameters,
            intervals,
            _table_cells(artifacts),
        )

    primary = tuple(
        (
            "quality.filtered_ann_recall_at_20"
            if row["filters"]
            else "quality.ann_recall_at_20"
        )
        + ".condition."
        + row["id"]
        for row in payload["conditions"]
        if any(
            all(metadata.get(key) == value for key, value in row["filters"].items())
            for metadata in corpora[row["corpus"]].metadata
        )
    )
    if not primary:
        raise ValueError("At least one nonempty ANN condition is required")
    result = run_benchmark(
        plan,
        evaluate,
        dataset_checksum=stable_hash(
            {name: corpus.content_checksum for name, corpus in corpora.items()}
        ),
        directions={key: "max" for key in primary},
        primary_metric=primary,
        required_metrics=primary,
        revisions=revisions,
        decision_files={"database": decision_path} if decision_path else {},
        protocols={"vector_database": protocol.meta},
        input_artifacts={"data_validation": report_path, "manifest": Path(path)},
        monitor_resources=False,
        paired_comparisons=False,
        candidate_artifact_name="candidate.json",
        parent_artifact_builder=lambda directory, results, plan: parent_artifacts(
            directory, results, plan, protocol=protocol
        ),
        evaluator_receives_context=True,
        no_mlflow=args.no_mlflow,
        artifact_root=args.artifact_root,
        csv_artifact_tables=CSV_TABLES,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "artifacts": str(result.artifact_directory),
                "complete": result.complete,
            }
        )
    )
    return 0 if result.complete else 2


def _merge(left, right):
    result = (left[0] + right[0], *({**left[i], **right[i]} for i in range(1, 6)))
    key = "operational.peak_server_ram_bytes.scope.server-lifecycle"
    observations = [
        side[2].get(key) for side in (left, right) if side[2].get(key) is not None
    ]
    if observations:
        result[2][key] = max(observations)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
