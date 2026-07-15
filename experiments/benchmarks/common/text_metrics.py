"""Text and structure metrics used by extraction experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .metrics import normalized_tokens


def content_scores(reference: str, hypothesis: str) -> dict[str, float]:
    reference_tokens = Counter(normalized_tokens(reference))
    hypothesis_tokens = Counter(normalized_tokens(hypothesis))
    overlap = sum((reference_tokens & hypothesis_tokens).values())
    reference_total = sum(reference_tokens.values())
    hypothesis_total = sum(hypothesis_tokens.values())
    precision = overlap / hypothesis_total if hypothesis_total else 0.0
    recall = overlap / reference_total if reference_total else float(not hypothesis_total)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    missing = sum((reference_tokens - hypothesis_tokens).values()) / max(1, reference_total)
    hallucinated = sum((hypothesis_tokens - reference_tokens).values()) / max(1, hypothesis_total)
    return {
        "content_precision": precision,
        "content_recall": recall,
        "content_f1": f1,
        "missing_text_rate": missing,
        "hallucinated_text_rate": hallucinated,
    }


def reading_order_accuracy(reference: str, hypothesis: str) -> float:
    """Pairwise order accuracy over tokens occurring once in both texts."""
    reference_tokens = normalized_tokens(reference)
    hypothesis_tokens = normalized_tokens(hypothesis)
    reference_counts, hypothesis_counts = Counter(reference_tokens), Counter(hypothesis_tokens)
    shared = {
        token
        for token in reference_counts
        if reference_counts[token] == 1 and hypothesis_counts[token] == 1
    }
    ordered = [token for token in reference_tokens if token in shared]
    if len(ordered) < 2:
        return float(reference_tokens == hypothesis_tokens)
    positions = {token: index for index, token in enumerate(hypothesis_tokens) if token in shared}
    correct = 0
    pairs = 0
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            pairs += 1
            correct += positions[ordered[left]] < positions[ordered[right]]
    return correct / pairs


def block_scores(reference: str, hypothesis: str) -> dict[str, float]:
    """Exact normalized-line precision/recall/F1."""
    reference_blocks = Counter(_blocks(reference))
    hypothesis_blocks = Counter(_blocks(hypothesis))
    overlap = sum((reference_blocks & hypothesis_blocks).values())
    precision = overlap / sum(hypothesis_blocks.values()) if hypothesis_blocks else 0.0
    recall = overlap / sum(reference_blocks.values()) if reference_blocks else float(not hypothesis_blocks)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "block_precision": precision,
        "block_recall": recall,
        "block_structure_f1": f1,
    }


def timestamp_mae(reference: Sequence[float], hypothesis: Sequence[float]) -> float:
    if len(reference) != len(hypothesis) or not reference:
        raise ValueError("Timestamp arrays must be non-empty and have equal length")
    return sum(abs(left - right) for left, right in zip(reference, hypothesis)) / len(reference)


def _blocks(text: str) -> list[str]:
    return [" ".join(line.casefold().split()) for line in text.splitlines() if line.strip()]
