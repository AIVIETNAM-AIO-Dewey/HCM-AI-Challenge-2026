"""Deterministic temporal beam search for TRAKE-style queries.

Candidate lists are supplied one list per decomposed event.  The search keeps
only same-video, strictly chronological paths and applies the paper's
exponential time-gap decay at every extension.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite
from typing import Any

from hcm_ai.contracts import MomentResult, SequenceEvent, SequenceResult


def _field(candidate: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or mapping key from a result-like candidate."""

    if isinstance(candidate, Mapping):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _required_text(candidate: Any, field: str) -> str:
    value = _field(candidate, field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"temporal candidate {field} must be a non-empty string")
    return value.strip()


def _finite_number(candidate: Any, field: str) -> float:
    raw_value = _field(candidate, field)
    if isinstance(raw_value, bool):
        raise ValueError(f"temporal candidate {field} must be numeric")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"temporal candidate {field} must be numeric") from exc
    if not isfinite(value):
        raise ValueError(f"temporal candidate {field} must be finite")
    if field == "timestamp" and value < 0:
        raise ValueError("temporal candidate timestamp must be non-negative")
    return value


def _provenance(candidate: Any) -> list[str]:
    raw_value = _field(candidate, "provenance", [])
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        return []
    return [str(value) for value in raw_value if isinstance(value, str) and value]


def _optional_text(candidate: Any, field: str) -> str | None:
    value = _field(candidate, field)
    return value.strip() if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True, slots=True)
class _Candidate:
    """Validated candidate fields used while constructing a temporal path."""

    video_id: str
    frame_id: str
    timestamp: float
    score: float
    image_path: str | None
    shot_id: str | None
    provenance: tuple[str, ...]


def _coerce_candidate(candidate: Any) -> _Candidate:
    return _Candidate(
        video_id=_required_text(candidate, "video_id"),
        frame_id=_required_text(candidate, "frame_id"),
        timestamp=_finite_number(candidate, "timestamp"),
        score=_finite_number(candidate, "score"),
        image_path=_optional_text(candidate, "image_path"),
        shot_id=_optional_text(candidate, "shot_id"),
        provenance=tuple(_provenance(candidate)),
    )


def _deduplicate_event(candidates: Sequence[Any]) -> list[_Candidate]:
    """Keep the best duplicate candidate per stable video/frame identity."""

    best: dict[tuple[str, str], _Candidate] = {}
    for raw_candidate in candidates:
        candidate = _coerce_candidate(raw_candidate)
        identity = (candidate.video_id, candidate.frame_id)
        current = best.get(identity)
        if current is None or candidate.score > current.score:
            best[identity] = candidate
    return sorted(
        best.values(),
        key=lambda candidate: (
            -candidate.score,
            candidate.video_id,
            candidate.timestamp,
            candidate.frame_id,
        ),
    )


@dataclass(frozen=True, slots=True)
class _Beam:
    """One partial temporal path and its already-decayed score."""

    video_id: str
    candidates: tuple[_Candidate, ...]
    decays: tuple[float, ...]
    score: float

    @property
    def last(self) -> _Candidate:
        return self.candidates[-1]

    @property
    def frame_ids(self) -> frozenset[str]:
        return frozenset(candidate.frame_id for candidate in self.candidates)

    @property
    def key(self) -> tuple[str, tuple[tuple[float, str], ...]]:
        return (
            self.video_id,
            tuple((candidate.timestamp, candidate.frame_id) for candidate in self.candidates),
        )


def _rank_beams(beams: Sequence[_Beam]) -> list[_Beam]:
    return sorted(
        beams,
        key=lambda beam: (
            -beam.score,
            beam.video_id,
            tuple(candidate.timestamp for candidate in beam.candidates),
            tuple(candidate.frame_id for candidate in beam.candidates),
        ),
    )


def _event_descriptions(
    count: int,
    supplied: Sequence[str] | None,
) -> list[str]:
    if isinstance(supplied, (str, bytes)):
        raise ValueError("event_descriptions must be a sequence of strings")
    if supplied is not None and len(supplied) != count:
        raise ValueError("event_descriptions must have one description per event candidate list")
    descriptions: list[str] = []
    for index in range(count):
        raw_description = supplied[index] if supplied is not None else None
        if isinstance(raw_description, str) and raw_description.strip():
            descriptions.append(raw_description.strip())
        else:
            # SequenceEvent deliberately requires a non-empty description.
            descriptions.append(f"event {index + 1}")
    return descriptions


def _to_sequence_result(
    beam: _Beam,
    *,
    descriptions: Sequence[str],
    alpha: float,
    beam_width: int,
    rank: int,
) -> SequenceResult:
    events = [
        SequenceEvent(
            event_index=index,
            video_id=candidate.video_id,
            frame_id=candidate.frame_id,
            timestamp=candidate.timestamp,
            description=descriptions[index],
            score=candidate.score,
            image_path=candidate.image_path,
            shot_id=candidate.shot_id,
            provenance=list(candidate.provenance),
        )
        for index, candidate in enumerate(beam.candidates)
    ]
    duration = events[-1].timestamp - events[0].timestamp
    return SequenceResult(
        video_id=beam.video_id,
        events=events,
        score=beam.score,
        rank=rank,
        duration=duration,
        metadata={
            "alpha": alpha,
            "beam_width": beam_width,
            "temporal_decays": list(beam.decays),
        },
    )


def temporal_beam_search(
    event_candidates: Sequence[Sequence[MomentResult | Mapping[str, Any]]],
    *,
    event_descriptions: Sequence[str] | None = None,
    alpha: float = 0.01,
    beam_width: int = 8,
    top_k: int = 10,
) -> list[SequenceResult]:
    """Return highest scoring, same-video chronological event sequences.

    The first event has :math:`lambda_1 = 1`.  Every subsequent candidate is
    added with :math:`exp(-alpha * (t_i - t_{i-1}))`.  A path is rejected if it
    changes video, fails strict timestamp ordering, or repeats a frame.
    """

    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not isfinite(alpha)
        or alpha < 0
    ):
        raise ValueError("alpha must be a finite non-negative number")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not event_candidates:
        return []

    descriptions = _event_descriptions(len(event_candidates), event_descriptions)
    normalized_events = [_deduplicate_event(candidates) for candidates in event_candidates]
    if any(not candidates for candidates in normalized_events):
        return []

    beams = [
        _Beam(
            video_id=candidate.video_id,
            candidates=(candidate,),
            decays=(1.0,),
            score=candidate.score,
        )
        for candidate in normalized_events[0]
    ]
    beams = _rank_beams(beams)[:beam_width]

    for candidates in normalized_events[1:]:
        expanded: dict[tuple[str, tuple[tuple[float, str], ...]], _Beam] = {}
        for beam in beams:
            for candidate in candidates:
                if candidate.video_id != beam.video_id:
                    continue
                if candidate.timestamp <= beam.last.timestamp:
                    continue
                if candidate.frame_id in beam.frame_ids:
                    continue
                decay = exp(-float(alpha) * (candidate.timestamp - beam.last.timestamp))
                extended = _Beam(
                    video_id=beam.video_id,
                    candidates=(*beam.candidates, candidate),
                    decays=(*beam.decays, decay),
                    score=beam.score + candidate.score * decay,
                )
                current = expanded.get(extended.key)
                if current is None or extended.score > current.score:
                    expanded[extended.key] = extended
        if not expanded:
            return []
        beams = _rank_beams(list(expanded.values()))[:beam_width]

    return [
        _to_sequence_result(
            beam,
            descriptions=descriptions,
            alpha=float(alpha),
            beam_width=beam_width,
            rank=rank,
        )
        for rank, beam in enumerate(_rank_beams(beams)[:top_k], start=1)
    ]
