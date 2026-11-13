"""Fresh selected-system worker; server preparation and live request execution."""

from pathlib import Path

from experiments.benchmarks.common.process import (
    benchmark_objects_from_payload,
    json_worker_main,
)
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    load_protocol as chunking_protocol,
)
from experiments.benchmarks.rag.evaluation import build_index, reranker_for
from experiments.benchmarks.rag.final.benchmark import _unload_retrieval, execute
from experiments.benchmarks.rag.final.composition import load_composition
from experiments.benchmarks.rag.final.protocol import load_protocol as final_protocol
from experiments.benchmarks.rag.generation.protocol import (
    load_protocol as generation_protocol,
)
from experiments.benchmarks.rag.retrieval_reranking.protocol import (
    load_protocol as retrieval_protocol,
)
from experiments.benchmarks.vectordb.integration import SelectedServer
from experiments.benchmarks.vectordb.protocol import load_protocol as vector_protocol
from experiments.benchmarks.vectordb.selection import candidate_config

LOADERS = {
    "final_rag": final_protocol,
    "chunking_embedding": chunking_protocol,
    "retrieval": retrieval_protocol,
    "generation": generation_protocol,
    "vector_database": vector_protocol,
}


def execute_payload(payload):
    _, manifest, model_lock, plan = benchmark_objects_from_payload(payload)
    protocols = {
        key: loader(Path(payload["protocol_paths"][key]))
        for key, loader in LOADERS.items()
    }
    for key, protocol in protocols.items():
        protocol.meta.validate_worker_payload(payload["protocols"][key])
    execution = (
        protocols["final_rag"].preflight
        if payload.get("preflight", False)
        else protocols["final_rag"].profile(plan.profile)
    )
    if (plan.seed, plan.warmups, plan.repetitions, plan.bootstrap_resamples) != (
        protocols["final_rag"].meta.seed,
        execution.warmups,
        execution.repetitions,
        execution.bootstrap_resamples,
    ):
        raise ValueError("Final worker plan differs from its frozen execution settings")
    expected_device = plan.settings.get("device")
    if expected_device not in (
        ("cpu", "cuda") if plan.profile == "smoke" else ("cuda",)
    ):
        raise ValueError("Final worker device violates profile")
    if plan.settings.get("dtype") != (
        "float32" if expected_device == "cpu" else "float16"
    ):
        raise ValueError("Final worker dtype violates device contract")
    composition = load_composition(
        Path(payload["composition_path"]), {key: p.meta for key, p in protocols.items()}
    )
    if composition.checksum != payload["composition_checksum"]:
        raise ValueError("Final composition changed before worker execution")
    chunker, embedding = composition.system["chunking_embedding"].split("|", 1)
    retrieval = composition.system["retrieval"]
    protocols["retrieval"].validate_composition(protocols["chunking_embedding"])
    index = build_index(
        manifest,
        chunker,
        embedding,
        model_lock,
        with_bm25=retrieval.split("|")[0] in {"bm25", "rrf"},
        with_dense=True,
        device=plan.settings["device"],
        dtype=plan.settings["dtype"],
        chunking_protocol=protocols["chunking_embedding"],
        retrieval_protocol=protocols["retrieval"],
    )
    database, config = candidate_config(
        protocols["vector_database"],
        composition.system["database"],
        int(index.vectors.shape[1]),
    )
    reranker = reranker_for(
        retrieval,
        model_lock,
        device=plan.settings["device"],
        dtype=plan.settings["dtype"],
        retrieval_protocol=protocols["retrieval"],
    )
    server = SelectedServer(
        database, config, index, retrieval, protocols["retrieval"], reranker=reranker
    )
    try:
        prepared = server.prepare()
        _unload_retrieval(server)
        raw = execute(
            composition.system["generator"],
            manifest,
            model_lock,
            plan,
            protocols["generation"],
            composition,
            server,
            evidence_directory=payload["evidence_directory"],
            preflight=payload.get("preflight", False),
        )
        raw["server_preparation"] = prepared
        return raw
    finally:
        server.close()


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute_payload))
