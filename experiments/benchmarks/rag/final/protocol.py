"""Strict protocol for final end-to-end RAG validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.benchmarks.common.protocol import (
    ProtocolMetadata,
    execution_profiles,
    increasing_integers,
    integer,
    load_yaml,
    metadata,
    strict_object,
    string,
)


DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")


@dataclass(frozen=True)
class FinalRAGProtocol:
    meta: ProtocolMetadata
    top_k: tuple[int, ...]
    maximum_retrieval_finalists: int
    maximum_generation_finalists: int
    maximum_final_systems: int
    human_review_sample_count: int
    human_review_system_count: int
    required_judgment_count: int
    locked_marker_version: str

    def profile(self, profile: str):
        return self.meta.profile(profile)


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> FinalRAGProtocol:
    root = strict_object(
        load_yaml(path, "final-rag"),
        "final-rag protocol root",
        {"schema_version", "protocol_version", "seed", "selection", "human_review", "locked_test", "profiles"},
    )
    selection = strict_object(root["selection"], "selection", {"top_k", "maximum_retrieval_finalists", "maximum_generation_finalists", "maximum_final_systems"})
    top_k = increasing_integers(selection["top_k"], "selection.top_k")
    retrieval = integer(selection["maximum_retrieval_finalists"], "selection.maximum_retrieval_finalists", minimum=1)
    generation = integer(selection["maximum_generation_finalists"], "selection.maximum_generation_finalists", minimum=1)
    systems = integer(selection["maximum_final_systems"], "selection.maximum_final_systems", minimum=1)
    if systems < retrieval * generation * len(top_k):
        raise ValueError("maximum_final_systems cannot be smaller than the declared matrix")
    review = strict_object(root["human_review"], "human_review", {"sample_count", "system_count", "required_judgment_count"})
    samples = integer(review["sample_count"], "human_review.sample_count", minimum=1)
    review_systems = integer(review["system_count"], "human_review.system_count", minimum=1)
    judgments = integer(review["required_judgment_count"], "human_review.required_judgment_count", minimum=1)
    if judgments != samples * review_systems:
        raise ValueError("required_judgment_count must equal sample_count times system_count")
    locked = strict_object(root["locked_test"], "locked_test", {"marker_version"})
    marker = string(locked["marker_version"], "locked_test.marker_version")
    profiles = execution_profiles(
        root["profiles"],
        names=("smoke", "development", "validation", "locked"),
    )
    if {profile.batch_size for profile in profiles.values()} != {1}:
        raise ValueError("Final RAG profiles must use batch size one")
    return FinalRAGProtocol(
        metadata("final_rag", path, root, profiles=profiles), top_k, retrieval,
        generation, systems, samples, review_systems, judgments, marker,
    )
