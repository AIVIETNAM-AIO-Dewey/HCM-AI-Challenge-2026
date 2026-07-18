"""Similarity-weighted reciprocal-rank fusion for visual encoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import TypeAlias


RankedScores: TypeAlias = Mapping[str, float] | Sequence[tuple[str, float]]


def _coerce_ranking(ranking: RankedScores) -> list[tuple[str, float]]:
    """Deduplicate and place one model's candidates in stable rank order."""

    pairs = ranking.items() if isinstance(ranking, Mapping) else ranking
    best_by_id: dict[str, float] = {}
    for pair in pairs:
        try:
            candidate_id, raw_score = pair
        except (TypeError, ValueError) as exc:
            raise ValueError("ranking entries must be (candidate_id, score) pairs") from exc
        if isinstance(raw_score, bool):
            raise ValueError("similarity scores must be numeric, not boolean")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score for {candidate_id!r} must be numeric") from exc
        if not isfinite(score):
            raise ValueError(f"score for {candidate_id!r} must be finite")
        key = str(candidate_id)
        if key not in best_by_id or score > best_by_id[key]:
            best_by_id[key] = score
    return sorted(best_by_id.items(), key=lambda item: (-item[1], item[0]))


def similarity_weighted_rrf(
    ranked_lists: Mapping[str, RankedScores],
    *,
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Fuse ranked lists with original similarity divided by ``rrf_k + rank``.

    This score-reflected RRF variant preserves each encoder's original
    similarity signal, rather than only its rank, while still rewarding
    agreement near the head of several rankings.
    """

    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    configured_weights = dict(weights or {})
    fused: dict[str, float] = {}
    for source, raw_ranking in sorted(ranked_lists.items()):
        raw_weight = configured_weights.get(source, 1.0)
        if isinstance(raw_weight, bool):
            raise ValueError("source weights must be numeric, not boolean")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weight for {source!r} must be numeric") from exc
        if not isfinite(weight) or weight < 0:
            raise ValueError(f"weight for {source!r} must be finite and non-negative")
        if weight == 0:
            continue

        ranking = _coerce_ranking(raw_ranking)
        for rank, (candidate_id, similarity) in enumerate(ranking, start=1):
            fused[candidate_id] = fused.get(candidate_id, 0.0) + (
                weight * similarity / (rrf_k + rank)
            )

    return dict(sorted(fused.items(), key=lambda item: (-item[1], item[0])))
