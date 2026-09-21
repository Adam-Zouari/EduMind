"""Strict protocol for document extraction benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from experiments.benchmarks.extraction.document.profiles import DOCUMENT_LOCK_CANDIDATES
from experiments.benchmarks.common.protocol import (
    ProtocolMetadata,
    boolean,
    choice,
    execution_profiles,
    integer,
    load_yaml,
    mapping,
    metadata,
    number,
    sequence,
    strict_object,
    string,
)


DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")


@dataclass(frozen=True)
class DocumentProtocol:
    meta: ProtocolMetadata
    ocr_engines: tuple[str, ...]
    ocr_modes: tuple[str, ...]
    table_modes: tuple[str, ...]
    formula_enrichment: tuple[bool, ...]
    smoke_configuration: Mapping[str, object]
    parser: Mapping[str, object]
    element_matching_threshold: float
    duplicate_content_threshold: float
    sample_counts: Mapping[str, Mapping[str, int]]
    confidence_level: float
    evaluator_timeout_seconds: int
    maximum_architecture_finalists: int
    backend_devices: Mapping[str, tuple[str, ...]]

    def profile(self, name: str):
        return self.meta.profile(name)

    def parser_options(self, engine: str) -> dict[str, object]:
        options = {
            key: value
            for key, value in self.parser.items()
            if key not in {"granite", "paddle"}
        }
        if engine == "docling-standard":
            options.update(self.smoke_configuration)
        elif engine == "docling-vlm-granite-258m":
            options.update(mapping(self.parser["granite"], "parser.granite"))
        elif engine == "paddleocr-vl-1.6":
            options.update(mapping(self.parser["paddle"], "parser.paddle"))
        return options

    def validate_candidate_factors(self, factors: Mapping[str, str]) -> None:
        if not factors:
            return
        expected = {"ocr", "mode", "table", "formula"}
        if set(factors) != expected:
            raise ValueError("Document configuration candidates require all four factors")
        allowed = {
            "ocr": set(self.ocr_engines),
            "mode": set(self.ocr_modes),
            "table": set(self.table_modes),
            "formula": {"off", "on"},
        }
        for name, choices in allowed.items():
            if factors[name] not in choices:
                raise ValueError(
                    f"Document candidate {name}={factors[name]!r} is outside protocol.yaml"
                )

    def configuration_candidates(self, profile: str, *, image: bool = False) -> tuple[str, ...]:
        if profile == "smoke":
            rows = (self.smoke_configuration,)
        else:
            rows = tuple(
                {
                    "ocr_engine": ocr,
                    "ocr_mode": mode,
                    "table_mode": table,
                    "formula_enrichment": formula,
                }
                for ocr, mode, table, formula in product(
                    self.ocr_engines,
                    self.ocr_modes,
                    self.table_modes,
                    self.formula_enrichment,
                )
            )
        candidates = []
        for row in rows:
            mode = "full_page" if image else str(row["ocr_mode"])
            candidate = (
                f"docling-standard|ocr={row['ocr_engine']}|mode={mode}|"
                f"table={row['table_mode']}|"
                f"formula={'on' if row['formula_enrichment'] else 'off'}"
            )
            if candidate not in candidates:
                candidates.append(candidate)
        return tuple(candidates)

    def minimum_samples(self, profile: str, kind: str | None) -> int:
        if profile not in {"development", "validation"}:
            return 0
        counts = self.sample_counts[profile]
        return counts[kind] if kind else sum(counts.values())


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> DocumentProtocol:
    return protocol_from_mapping(load_yaml(path, "document"), source_path=path)


def protocol_from_mapping(
    value: object, *, source_path: Path = DEFAULT_PROTOCOL_PATH
) -> DocumentProtocol:
    root = strict_object(
        value,
        "document protocol root",
        {
            "schema_version",
            "protocol_version",
            "seed",
            "candidate_factors",
            "parser",
            "matching",
            "datasets",
            "statistics",
            "evaluators",
            "selection",
            "backend_devices",
            "profiles",
        },
    )
    factors = strict_object(
        root["candidate_factors"],
        "candidate_factors",
        {"ocr_engines", "ocr_modes", "table_modes", "formula_enrichment", "smoke_configuration"},
    )
    ocr_engines = _strings(factors["ocr_engines"], "candidate_factors.ocr_engines")
    if set(ocr_engines) != {"rapidocr", "tesseract", "easyocr"}:
        raise ValueError("Document protocol must declare the three reviewed OCR engines")
    ocr_modes = _strings(factors["ocr_modes"], "candidate_factors.ocr_modes")
    if set(ocr_modes) != {"pdf_aware_layout_regions", "full_page"}:
        raise ValueError("Document protocol must declare both reviewed OCR modes")
    table_modes = _strings(factors["table_modes"], "candidate_factors.table_modes")
    if set(table_modes) != {"fast", "accurate"}:
        raise ValueError("Document protocol must declare both TableFormer modes")
    formulas = tuple(
        boolean(value, f"candidate_factors.formula_enrichment[{index}]")
        for index, value in enumerate(sequence(factors["formula_enrichment"], "formula_enrichment"))
    )
    if len(formulas) != 2 or set(formulas) != {False, True}:
        raise ValueError("Document formula factor must contain false and true")
    smoke = strict_object(
        factors["smoke_configuration"],
        "candidate_factors.smoke_configuration",
        {"ocr_engine", "ocr_mode", "table_mode", "formula_enrichment"},
    )
    smoke_configuration = {
        "ocr_engine": choice(smoke["ocr_engine"], "smoke.ocr_engine", set(ocr_engines)),
        "ocr_mode": choice(smoke["ocr_mode"], "smoke.ocr_mode", set(ocr_modes)),
        "table_mode": choice(smoke["table_mode"], "smoke.table_mode", set(table_modes)),
        "formula_enrichment": boolean(smoke["formula_enrichment"], "smoke.formula_enrichment"),
    }
    parser = strict_object(
        root["parser"],
        "parser",
        {
            "language",
            "image_scale",
            "do_ocr",
            "do_table_structure",
            "do_cell_matching",
            "do_code_enrichment",
            "granite",
            "paddle",
        },
    )
    if choice(parser["language"], "parser.language", {"english"}) != "english":
        raise AssertionError
    number(parser["image_scale"], "parser.image_scale", minimum=0, minimum_exclusive=True)
    for name in ("do_ocr", "do_table_structure", "do_cell_matching", "do_code_enrichment"):
        boolean(parser[name], f"parser.{name}")
    granite = strict_object(parser["granite"], "parser.granite", {"load_in_8bit"})
    boolean(granite["load_in_8bit"], "parser.granite.load_in_8bit")
    paddle = strict_object(
        parser["paddle"], "parser.paddle", {"pipeline_version", "recognition_backend"}
    )
    choice(paddle["pipeline_version"], "parser.paddle.pipeline_version", {"v1.6"})
    choice(paddle["recognition_backend"], "parser.paddle.recognition_backend", {"native"})
    matching = strict_object(
        root["matching"], "matching", {"element_threshold", "duplicate_content_threshold"}
    )
    element_threshold = number(matching["element_threshold"], "matching.element_threshold", minimum=0, maximum=1)
    duplicate_threshold = number(matching["duplicate_content_threshold"], "matching.duplicate_content_threshold", minimum=0, maximum=1)
    datasets = strict_object(root["datasets"], "datasets", {"development", "validation"})
    sample_counts = {
        split: _counts(datasets[split], f"datasets.{split}")
        for split in ("development", "validation")
    }
    statistics = strict_object(root["statistics"], "statistics", {"confidence_level"})
    confidence = number(
        statistics["confidence_level"], "statistics.confidence_level", minimum=0, maximum=1,
        minimum_exclusive=True, maximum_exclusive=True,
    )
    evaluators = strict_object(root["evaluators"], "evaluators", {"timeout_seconds"})
    timeout = integer(evaluators["timeout_seconds"], "evaluators.timeout_seconds", minimum=1)
    selection = strict_object(root["selection"], "selection", {"maximum_architecture_finalists"})
    maximum_finalists = integer(selection["maximum_architecture_finalists"], "selection.maximum_architecture_finalists", minimum=1)
    candidate_aliases = set(DOCUMENT_LOCK_CANDIDATES)
    raw_devices = strict_object(
        root["backend_devices"],
        "backend_devices",
        candidate_aliases,
    )
    devices = {}
    for name in raw_devices:
        values = tuple(
            choice(value, f"backend_devices.{name}[{index}]", {"cpu", "cuda"})
            for index, value in enumerate(sequence(raw_devices[name], f"backend_devices.{name}"))
        )
        if not values or len(values) != len(set(values)):
            raise ValueError(
                f"backend_devices.{name} must contain unique supported devices"
            )
        devices[name] = values
    profiles = execution_profiles(
        root["profiles"], names=("smoke", "development", "validation")
    )
    if {profile.batch_size for profile in profiles.values()} != {1}:
        raise ValueError("Document parser profiles must process one sample at a time")
    meta = metadata("document", source_path, root, profiles=profiles)
    return DocumentProtocol(
        meta,
        ocr_engines,
        ocr_modes,
        table_modes,
        formulas,
        smoke_configuration,
        parser,
        element_threshold,
        duplicate_threshold,
        sample_counts,
        confidence,
        timeout,
        maximum_finalists,
        devices,
    )


def protocol_from_worker(value: object) -> DocumentProtocol:
    payload = mapping(value, "document worker protocol")
    root = mapping(payload.get("resolved"), "document resolved protocol")
    protocol = protocol_from_mapping(root)
    protocol.meta.validate_worker_payload(payload)
    return protocol


def _strings(value: object, label: str) -> tuple[str, ...]:
    values = tuple(string(item, f"{label}[{index}]") for index, item in enumerate(sequence(value, label)))
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique values")
    return values


def _counts(value: object, label: str) -> dict[str, int]:
    payload = strict_object(value, label, {"image", "pdf", "docx"})
    return {name: integer(payload[name], f"{label}.{name}", minimum=1) for name in payload}
