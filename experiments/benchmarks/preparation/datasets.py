"""Prepare licensed benchmark datasets and frozen manifests."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import requests

from edumind.common.artifacts import atomic_write_json, sha256_file
from edumind.extraction.normalization import normalize_text

from experiments.benchmarks.common.datasets import (
    assert_no_split_leakage,
    load_manifest,
    manifest_content_checksum,
)

QASPER_DATASET = "allenai/qasper"
QASPER_REVISION = "3065362e337ded696bbb0171b073c73e513c9410"


def prepare_qasper(output_directory: Path, *, seed: int = 42) -> list[Path]:
    """Download pinned QASPER and create paper-isolated benchmark manifests."""
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("datasets is required; install requirements/benchmarks.lock") from exc
    dataset = load_dataset(QASPER_DATASET, "qasper", revision=QASPER_REVISION)
    plans = (
        ("dev", "train", 100),
        ("validation", "validation", 40),
        ("locked-test", "test", 40),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    selected_ids: set[str] = set()
    for manifest_split, source_split, count in plans:
        papers = [dict(row) for row in dataset[source_split]]
        selected = _stratified_papers(papers, count, seed)
        paper_ids = {str(paper["id"]) for paper in selected}
        if selected_ids & paper_ids:
            raise ValueError("QASPER paper leakage detected across prepared splits")
        selected_ids.update(paper_ids)
        samples = _qasper_samples(selected)
        payload = {
            "name": f"qasper-{manifest_split}",
            "version": "1.0.0",
            "task": "rag",
            "split": manifest_split,
            "source": QASPER_DATASET,
            "license": "CC-BY-4.0",
            "revision": QASPER_REVISION,
            "checksum": manifest_content_checksum(samples),
            "preprocessing_version": "qasper-normalized-text-v1",
            "split_seed": seed,
            "selected_ids": sorted(paper_ids),
            "samples": samples,
        }
        output = output_directory / f"qasper-{manifest_split}.json"
        atomic_write_json(output, payload)
        outputs.append(output)
    assert_no_split_leakage([load_manifest(path) for path in outputs])
    return outputs


def prepare_rag_selection_manifest(
    qasper_path: Path, structured_path: Path, output_path: Path
) -> Path:
    """Combine one QASPER split with verified table/formula/mixed RAG samples."""
    qasper = load_manifest(qasper_path)
    structured = load_manifest(structured_path)
    if qasper.split != structured.split:
        raise ValueError(
            f"RAG source splits differ: {qasper.split!r} versus {structured.split!r}"
        )
    structured_questions = [
        sample for sample in structured.samples if sample.get("kind") == "question"
    ]
    required_types = {"table", "formula", "mixed"}
    invalid_types = sorted(
        {
            str(sample.get("evidence_type", ""))
            for sample in structured_questions
            if str(sample.get("evidence_type", "")) not in required_types | {"text"}
        }
    )
    if invalid_types:
        raise ValueError(
            "Structured RAG questions use unsupported evidence_type values: "
            + ", ".join(invalid_types)
        )
    evidence_counts = Counter(
        str(sample.get("evidence_type"))
        for sample in structured_questions
        if sample.get("answerable") and sample.get("evidence")
    )
    insufficient = {
        evidence_type: evidence_counts[evidence_type]
        for evidence_type in required_types
        if evidence_counts[evidence_type] < 10
    }
    if insufficient:
        raise ValueError(
            "Structured RAG manifests require at least 10 answerable questions with "
            "verified spans for each structural evidence type; received "
            + ", ".join(f"{name}={count}" for name, count in sorted(insufficient.items()))
        )
    combined = [*qasper.samples, *structured.samples]
    identifiers = [str(sample.get("id", "")) for sample in combined]
    duplicates = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "RAG manifests contain duplicate IDs; namespace the structured corpus: "
            + ", ".join(duplicates[:10])
        )
    payload = {
        "name": f"edumind-rag-selection-{qasper.split}",
        "version": "2.0.0",
        "task": "rag",
        "split": qasper.split,
        "source": f"{qasper.source}+{structured.source}",
        "license": f"{qasper.license};{structured.license}",
        "revision": f"{qasper.revision}+{structured.revision}",
        "checksum": manifest_content_checksum(combined),
        "preprocessing_version": "structured-rag-markdown-v1",
        "split_seed": 42,
        "samples": combined,
    }
    atomic_write_json(output_path, payload)
    resolved = load_manifest(output_path)
    assert_no_split_leakage([resolved])
    return output_path


def prepare_public_assets(plan_path: Path, output_directory: Path) -> list[Path]:
    """Download a licensed asset plan and reject checksum or license omissions."""
    import json

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assets = payload.get("assets", [])
    if not isinstance(assets, list) or not assets:
        raise ValueError("Asset plan must contain a non-empty assets list")
    output_directory.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for asset in assets:
        if not isinstance(asset, Mapping):
            raise ValueError("Every asset entry must be an object")
        url = str(asset.get("url", ""))
        filename = Path(str(asset.get("filename", ""))).name
        expected = str(asset.get("sha256", "")).casefold()
        license_name = str(asset.get("license", ""))
        if not url.startswith("https://") or not filename or len(expected) != 64 or not license_name:
            raise ValueError("Asset entries require HTTPS URL, filename, SHA-256, and license")
        destination = output_directory / filename
        temporary = destination.with_suffix(destination.suffix + ".partial")
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            if sha256_file(temporary) != expected:
                raise ValueError(f"Checksum mismatch for {filename}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        downloaded.append(destination)
    return downloaded


def _stratified_papers(
    papers: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[Mapping[str, Any]]:
    if len(papers) < count:
        raise ValueError(f"Requested {count} papers from a split containing {len(papers)}")
    buckets: dict[tuple[int, bool], list[Mapping[str, Any]]] = {}
    for paper in papers:
        qas = _records(paper.get("qas", []))
        bucket = (min(3, len(qas) // 4), any(_question_unanswerable(qa) for qa in qas))
        buckets.setdefault(bucket, []).append(paper)
    random_state = random.Random(seed)
    for values in buckets.values():
        values.sort(key=lambda item: str(item["id"]))
        random_state.shuffle(values)
    ordered: list[Mapping[str, Any]] = []
    keys = sorted(buckets)
    while len(ordered) < count:
        progressed = False
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop())
                progressed = True
                if len(ordered) == count:
                    break
        if not progressed:
            break
    return ordered


def _qasper_samples(papers: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for paper in papers:
        paper_id = str(paper["id"])
        text = _paper_text(paper)
        samples.append(
            {
                "id": paper_id,
                "kind": "document",
                "text": text,
                "representation": "markdown-sections",
            }
        )
        for qa in _records(paper.get("qas", [])):
            answers, evidence, answer_type, answerable = _answers_and_evidence(qa, text, paper_id)
            samples.append(
                {
                    "id": str(qa.get("question_id", "")),
                    "kind": "question",
                    "document_id": paper_id,
                    "question": str(qa.get("question", "")),
                    "answer": answers[0] if answers else "",
                    "accepted_answers": answers,
                    "answer_type": answer_type,
                    "answerable": answerable,
                    "evidence_type": "text",
                    "evidence": evidence,
                }
            )
    return samples


def _paper_text(paper: Mapping[str, Any]) -> str:
    sections = _records(paper.get("full_text", []))
    title = str(paper.get("title", "")).strip()
    abstract = str(paper.get("abstract", "")).strip()
    blocks = [f"# {title}" if title else "", "## Abstract", abstract]
    for section in sections:
        section_name = str(section.get("section_name", "")).strip()
        if section_name:
            blocks.append(f"## {section_name}")
        paragraphs = section.get("paragraphs", [])
        if isinstance(paragraphs, Sequence) and not isinstance(paragraphs, (str, bytes)):
            blocks.extend(str(paragraph) for paragraph in paragraphs)
    return normalize_text(
        "\n\n".join(block for block in blocks if block.strip()), "conservative"
    )


def _answers_and_evidence(
    qa: Mapping[str, Any], document: str, paper_id: str
) -> tuple[list[str], list[dict[str, object]], str, bool]:
    accepted: list[str] = []
    evidence: list[dict[str, object]] = []
    answer_type = "unanswerable"
    answerable = False
    for annotation in _records(qa.get("answers", [])):
        raw_answer = annotation.get("answer", annotation)
        for answer in _records(raw_answer) or (
            [dict(raw_answer)] if isinstance(raw_answer, Mapping) else []
        ):
            if bool(answer.get("unanswerable")):
                continue
            value, current_type = _answer_value(answer)
            if value:
                accepted.append(value)
                answer_type = current_type
                answerable = True
            raw_evidence = answer.get("highlighted_evidence") or answer.get("evidence") or []
            if isinstance(raw_evidence, Sequence) and not isinstance(raw_evidence, (str, bytes)):
                for evidence_text in raw_evidence:
                    normalized = normalize_text(str(evidence_text), "conservative")
                    if not normalized or normalized.startswith("FLOAT SELECTED"):
                        continue
                    start = document.find(normalized)
                    if start < 0:
                        raise ValueError(
                            "QASPER evidence offset validation failed for "
                            f"{paper_id}: {normalized[:80]}"
                        )
                    item = {
                        "id": f"{paper_id}:{start}:{start + len(normalized)}",
                        "document_id": paper_id,
                        "start": start,
                        "end": start + len(normalized),
                    }
                    if item not in evidence:
                        evidence.append(item)
    return list(dict.fromkeys(accepted)), evidence, answer_type, answerable


def _answer_value(answer: Mapping[str, Any]) -> tuple[str, str]:
    free_form = str(answer.get("free_form_answer", "")).strip()
    if free_form:
        return free_form, "free_form"
    spans = answer.get("extractive_spans", [])
    if isinstance(spans, Sequence) and not isinstance(spans, (str, bytes)):
        joined = " ".join(str(value).strip() for value in spans if str(value).strip())
        if joined:
            return joined, "extractive"
    if "yes_no" in answer:
        return ("Yes" if bool(answer["yes_no"]) else "No"), "yes_no"
    return "", "unanswerable"


def _question_unanswerable(qa: Mapping[str, Any]) -> bool:
    for annotation in _records(qa.get("answers", [])):
        raw_answer = annotation.get("answer", annotation)
        answers = _records(raw_answer) or (
            [dict(raw_answer)] if isinstance(raw_answer, Mapping) else []
        )
        if any(bool(answer.get("unanswerable")) for answer in answers):
            return True
    return False


def _records(value: object) -> list[dict[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        return []
    if any(
        not isinstance(item, Sequence) or isinstance(item, (str, bytes))
        for item in value.values()
    ):
        return [dict(value)]
    lengths = [
        len(item)
        for item in value.values()
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes))
    ]
    if not lengths:
        return [dict(value)]
    count = min(lengths)
    return [
        {
            key: item[index]
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes))
            else item
            for key, item in value.items()
        }
        for index in range(count)
    ]

