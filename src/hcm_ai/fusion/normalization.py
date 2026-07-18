"""Deterministic normalization and adaptive modality-score fusion."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import TypeAlias


ScoreMap: TypeAlias = Mapping[str, float]


def _finite_scores(scores: ScoreMap) -> dict[str, float]:
    """Return a deterministic, finite copy of a score mapping."""

    cleaned: dict[str, float] = {}
    for candidate_id, raw_score in scores.items():
        if isinstance(raw_score, bool):
            raise ValueError("scores must be numeric, not boolean")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score for {candidate_id!r} must be numeric") from exc
        if not isfinite(score):
            raise ValueError(f"score for {candidate_id!r} must be finite")
        cleaned[str(candidate_id)] = score
    return dict(sorted(cleaned.items()))


def min_max_normalize(scores: ScoreMap, *, epsilon: float = 1e-12) -> dict[str, float]:
    """Normalize one modality's scores to ``[0, 1]``.

    Empty mappings remain empty.  When every value is equal there is no
    evidence for an intra-modality ordering, so every normalized score is
    deliberately ``0.0``.  This follows the paper's min-max equation with
    a non-zero denominator and avoids inventing a preference.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = _finite_scores(scores)
    if not values:
        return {}

    lower = min(values.values())
    upper = max(values.values())
    spread = upper - lower
    if spread <= epsilon:
        return {candidate_id: 0.0 for candidate_id in values}
    return {
        candidate_id: (score - lower) / (spread + epsilon)
        for candidate_id, score in values.items()
    }


def normalize_modalities(modality_scores: Mapping[str, ScoreMap]) -> dict[str, dict[str, float]]:
    """Min-max normalize each modality independently."""

    return {
        str(modality): min_max_normalize(scores)
        for modality, scores in sorted(modality_scores.items())
    }


def fuse_modalities(
    modality_scores: Mapping[str, ScoreMap],
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Fuse independently normalized modality scores.

    Weights are renormalized across non-empty, positive-weight modalities.
    Thus an unavailable OCR or ASR backend does not lower visual candidates
    merely because its requested weight was non-zero.  Missing candidates in
    an active modality contribute zero for that modality.
    """

    normalized = normalize_modalities(modality_scores)
    configured_weights = dict(weights or {})

    active_weights: dict[str, float] = {}
    for modality, scores in normalized.items():
        raw_weight = configured_weights.get(modality, 1.0)
        if isinstance(raw_weight, bool):
            raise ValueError("modality weights must be numeric, not boolean")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weight for {modality!r} must be numeric") from exc
        if not isfinite(weight) or weight < 0:
            raise ValueError(f"weight for {modality!r} must be finite and non-negative")
        if scores and weight > 0:
            active_weights[modality] = weight

    total_weight = sum(active_weights.values())
    if total_weight == 0:
        return {}

    fused: dict[str, float] = {}
    for modality, weight in active_weights.items():
        normalized_weight = weight / total_weight
        for candidate_id, score in normalized[modality].items():
            fused[candidate_id] = fused.get(candidate_id, 0.0) + normalized_weight * score
    return dict(sorted(fused.items()))


def rank_scores(scores: ScoreMap) -> list[tuple[str, float]]:
    """Return score pairs in stable descending-score order."""

    return sorted(_finite_scores(scores).items(), key=lambda item: (-item[1], item[0]))
