"""Strict execution contract for one reviewed, frozen Final RAG system."""

from dataclasses import dataclass
from pathlib import Path

from experiments.benchmarks.common.protocol import (
    PreflightSettings,
    ProtocolMetadata,
    execution_profiles,
    increasing_integers,
    load_yaml,
    metadata,
    preflight_settings,
    strict_object,
    string,
)

DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")


@dataclass(frozen=True)
class FinalRAGProtocol:
    meta: ProtocolMetadata
    top_k: tuple[int, ...]
    locked_marker_version: str
    preflight: PreflightSettings

    def profile(self, name):
        return self.meta.profile(name)


def load_protocol(path=DEFAULT_PROTOCOL_PATH):
    root = strict_object(
        load_yaml(path, "final-rag"),
        "final-rag",
        {
            "schema_version",
            "protocol_version",
            "seed",
            "selection",
            "locked_test",
            "profiles",
            "preflight",
        },
    )
    selection = strict_object(root["selection"], "selection", {"top_k"})
    top_k = increasing_integers(selection["top_k"], "selection.top_k")
    if top_k != (3, 5):
        raise ValueError("Final RAG permits one frozen top-K from [3, 5]")
    locked = strict_object(root["locked_test"], "locked_test", {"marker_version"})
    profiles = execution_profiles(root["profiles"], names=("smoke", "locked"))
    if any(p.batch_size != 1 or p.warmups != 1 for p in profiles.values()):
        raise ValueError("Final RAG requires one warmup and inference batch size one")
    if profiles["smoke"].repetitions != 1 or profiles["locked"].repetitions != 3:
        raise ValueError("Final RAG schedules one smoke or three locked attempts")
    if (profiles["locked"].device, profiles["locked"].dtype) != ("cuda", "float16"):
        raise ValueError("Locked Final RAG requires CUDA/FP16")
    return FinalRAGProtocol(
        metadata("final_rag", Path(path), root, profiles=profiles),
        top_k,
        string(locked["marker_version"], "locked_test.marker_version"),
        preflight_settings(root["preflight"]),
    )
