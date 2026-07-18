"""Small deterministic evaluation helpers for repeatable notebook checks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Return binary Recall@K for a single query."""

    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return float(bool(set(ranked_ids[:k]) & relevant))


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Return reciprocal rank for one ranked result list."""

    relevant = set(relevant_ids)
    for rank, candidate in enumerate(ranked_ids, start=1):
        if candidate in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def sequence_order_accuracy(predicted_timestamps: Sequence[float], expected_count: int) -> float:
    """Score an ordered sequence without unobserved ground-truth timestamps."""

    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if len(predicted_timestamps) != expected_count:
        return 0.0
    return float(all(left < right for left, right in zip(predicted_timestamps, predicted_timestamps[1:])))
