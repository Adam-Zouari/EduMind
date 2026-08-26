"""Measure degradation from verified text to extracted text with one frozen RAG system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from edumind.common.artifacts import stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import load_selected_model_lock, model_revisions
from experiments.benchmarks.rag.evaluation import RETRIEVAL_QUALITY_DIRECTIONS, build_index
from experiments.benchmarks.rag.generation.evaluate import GENERATION_DIRECTIONS, evaluate_candidate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare verified and extracted text with one frozen complete RAG system"
    )
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--extracted-manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        required=True,
        help="chunker@@embedding@@retrieval@@generator@@top_k=N",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    arguments = parser.parse_args()

    reference = load_manifest(arguments.reference_manifest)
    extracted = load_manifest(arguments.extracted_manifest)
    _validate_pair(reference.samples, extracted.samples)
    chunker, embedding, retrieval, generator, top_k_value = arguments.candidate.split("@@", 4)
    top_k = int(top_k_value.removeprefix("top_k="))
    model_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json"
    )
    revisions = model_revisions(model_lock)
    manifests = {"verified-reference": reference, "selected-extraction": extracted}
    indexes = {
        name: build_index(manifest, chunker, embedding, model_lock, with_bm25=True)
        for name, manifest in manifests.items()
    }
    plan = BenchmarkPlan(
        "rag",
        "extraction-confirmation",
        "standard",
        f"{reference.name}+{extracted.name}",
        tuple(manifests),
        repetitions=3,
        bootstrap_resamples=10_000,
        warmups=2,
    )

    result = run_benchmark(
        plan,
        lambda name: evaluate_candidate(
            generator,
            manifests[name],
            model_lock,
            final_index=indexes[name],
            retrieval_method=retrieval,
            top_k=top_k,
            repetitions=plan.repetitions,
            device=arguments.device,
            bootstrap_resamples=plan.bootstrap_resamples,
            bootstrap_seed=plan.seed,
        ),
        dataset_checksum=stable_hash(
            {"reference": reference.fingerprint, "extracted": extracted.fingerprint}
        ),
        directions={**GENERATION_DIRECTIONS, **RETRIEVAL_QUALITY_DIRECTIONS},
        primary_metric="citation_f1",
        revisions={**revisions, "frozen_system": arguments.candidate},
        no_mlflow=arguments.no_mlflow,
    )
    print(
        json.dumps(
            {"run_id": result.run_id, "artifacts": str(result.artifact_directory)},
            indent=2,
        )
    )
    return 0 if result.complete else 2


def _validate_pair(reference_samples, extracted_samples) -> None:
    def questions(samples):
        return {
            str(row["id"]): (str(row.get("document_id")), str(row.get("question")))
            for row in samples
            if row.get("kind") == "question"
        }

    if questions(reference_samples) != questions(extracted_samples):
        raise ValueError(
            "Reference and extracted manifests must contain identical question IDs, "
            "document IDs, and question text; each manifest keeps its own evidence offsets."
        )


if __name__ == "__main__":
    raise SystemExit(main())
