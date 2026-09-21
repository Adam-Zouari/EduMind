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
from experiments.benchmarks.rag.evaluation import build_index, retrieval_quality_directions
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_CHUNKING_PROTOCOL_PATH,
    load_protocol as load_chunking_protocol,
)
from experiments.benchmarks.rag.generation.evaluate import GENERATION_DIRECTIONS, evaluate_candidate
from experiments.benchmarks.rag.generation.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_GENERATION_PROTOCOL_PATH,
    load_protocol as load_generation_protocol,
)
from experiments.benchmarks.rag.retrieval.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_RETRIEVAL_PROTOCOL_PATH,
    load_protocol as load_retrieval_protocol,
)
from experiments.benchmarks.rag.retrieval.profiles import parse_candidate
from experiments.benchmarks.rag.final.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_FINAL_PROTOCOL_PATH,
    load_protocol as load_final_protocol,
)


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
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16", "auto"), required=True
    )
    parser.add_argument("--final-protocol", type=Path, default=DEFAULT_FINAL_PROTOCOL_PATH)
    parser.add_argument("--generation-protocol", type=Path, default=DEFAULT_GENERATION_PROTOCOL_PATH)
    parser.add_argument("--retrieval-protocol", type=Path, default=DEFAULT_RETRIEVAL_PROTOCOL_PATH)
    parser.add_argument("--chunking-protocol", type=Path, default=DEFAULT_CHUNKING_PROTOCOL_PATH)
    arguments = parser.parse_args()

    final_protocol = load_final_protocol(arguments.final_protocol)
    generation_protocol = load_generation_protocol(arguments.generation_protocol)
    retrieval_protocol = load_retrieval_protocol(arguments.retrieval_protocol)
    chunking_protocol = load_chunking_protocol(arguments.chunking_protocol)
    execution = final_protocol.profile("validation")
    if execution.hardware_required and (arguments.device, arguments.dtype) != (
        execution.device,
        execution.dtype,
    ):
        parser.error(
            "extraction confirmation requires "
            f"--device {execution.device} --dtype {execution.dtype}"
        )

    reference = load_manifest(arguments.reference_manifest)
    extracted = load_manifest(arguments.extracted_manifest)
    _validate_pair(reference.samples, extracted.samples)
    chunker, embedding, retrieval, generator, top_k_value = arguments.candidate.split("@@", 4)
    top_k = int(top_k_value.removeprefix("top_k="))
    if top_k not in final_protocol.top_k:
        raise ValueError(f"top_k must be one of {final_protocol.top_k}")
    chunking_protocol.strategy(chunker)
    parsed_retrieval = parse_candidate(retrieval)
    generator_model = generation_protocol.model_id(generator)
    required_models = {
        embedding,
        generator_model,
        generation_protocol.faithfulness_model,
    }
    if parsed_retrieval.reranker_model is not None:
        required_models.add(parsed_retrieval.reranker_model)
    model_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=tuple(sorted(required_models)),
    )
    revisions = model_revisions(model_lock)
    manifests = {"verified-reference": reference, "selected-extraction": extracted}
    indexes = {
        name: build_index(
            manifest,
            chunker,
            embedding,
            model_lock,
            with_bm25=True,
            device=arguments.device,
            dtype=arguments.dtype,
            chunking_protocol=chunking_protocol,
            retrieval_protocol=retrieval_protocol,
        )
        for name, manifest in manifests.items()
    }
    plan = BenchmarkPlan(
        "rag",
        "extraction-confirmation",
        "development",
        f"{reference.name}+{extracted.name}",
        tuple(manifests),
        seed=final_protocol.meta.seed,
        repetitions=execution.repetitions,
        bootstrap_resamples=execution.bootstrap_resamples,
        warmups=execution.warmups,
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
            dtype=arguments.dtype,
            bootstrap_resamples=plan.bootstrap_resamples,
            bootstrap_seed=plan.seed,
            protocol=generation_protocol,
            retrieval_protocol=retrieval_protocol,
            warmups=plan.warmups,
            question_scope="all",
        ),
        dataset_checksum=stable_hash(
            {"reference": reference.fingerprint, "extracted": extracted.fingerprint}
        ),
        directions={
            **GENERATION_DIRECTIONS,
            **retrieval_quality_directions(retrieval_protocol.quality_cutoffs),
        },
        primary_metric="citation_f1",
        revisions={**revisions, "frozen_system": arguments.candidate},
        input_artifacts={
            "reference_manifest": arguments.reference_manifest,
            "extracted_manifest": arguments.extracted_manifest,
        },
        protocols={
            "chunking_embedding": chunking_protocol.meta,
            "retrieval": retrieval_protocol.metadata(arguments.retrieval_protocol),
            "generation": generation_protocol.meta,
            "final_rag": final_protocol.meta,
        },
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
