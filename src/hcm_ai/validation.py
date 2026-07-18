"""Validation for stable moment, sequence, and answer result boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from .exporting import to_primitive


class ValidationError(ValueError):
    """Raised when a pipeline boundary loses required competition metadata."""


def _require_string(data: Mapping[str, Any], field: str) -> None:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")


def _require_timestamp(data: Mapping[str, Any], field: str = "timestamp") -> float:
    value = data.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or value < 0
    ):
        raise ValidationError(f"{field} must be a non-negative number")
    return float(value)


def validate_moment(value: Any) -> dict[str, Any]:
    """Validate a single renderable retrieval moment."""

    data = to_primitive(value)
    if not isinstance(data, Mapping):
        raise ValidationError("moment must be an object")
    _require_string(data, "video_id")
    _require_string(data, "frame_id")
    _require_string(data, "image_path")
    _require_timestamp(data)
    score = data.get("score")
    if score is not None and (
        isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(float(score))
    ):
        raise ValidationError("score must be numeric when supplied")
    return dict(data)


def validate_sequence(value: Any) -> dict[str, Any]:
    """Validate a temporally ordered, single-video sequence."""

    data = to_primitive(value)
    if not isinstance(data, Mapping):
        raise ValidationError("sequence must be an object")
    _require_string(data, "video_id")
    events = data.get("events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)) or not events:
        raise ValidationError("events must be a non-empty list")

    previous = -1.0
    seen_frames: set[str] = set()
    for event in events:
        event_data = to_primitive(event)
        if not isinstance(event_data, Mapping):
            raise ValidationError("sequence event must be an object")
        _require_string(event_data, "video_id")
        _require_string(event_data, "frame_id")
        # A TRAKE sequence can be exported without a local thumbnail.  When a
        # path is present it must still be usable, but the core temporal
        # contract only requires identity and time.
        if "image_path" in event_data and event_data["image_path"] is not None:
            _require_string(event_data, "image_path")
        if event_data["video_id"] != data["video_id"]:
            raise ValidationError("all sequence events must belong to the sequence video")
        timestamp = _require_timestamp(event_data)
        if timestamp <= previous:
            raise ValidationError("sequence timestamps must be strictly increasing")
        if event_data["frame_id"] in seen_frames:
            raise ValidationError("sequence cannot repeat a frame")
        previous = timestamp
        seen_frames.add(event_data["frame_id"])
    return dict(data)


def validate_answer(value: Any) -> dict[str, Any]:
    """Ensure an answer is grounded or intentionally left unanswered."""

    data = to_primitive(value)
    if not isinstance(data, Mapping):
        raise ValidationError("answer result must be an object")
    _require_string(data, "query_id")
    answer = data.get("answer")
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        raise ValidationError("answer must be a non-empty string or null")
    evidence = data.get("evidence", [])
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise ValidationError("evidence must be a list")
    for moment in evidence:
        validate_moment(moment)
    if answer is not None and not evidence:
        raise ValidationError("a non-null answer requires evidence")
    return dict(data)
