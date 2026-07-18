"""Stable, serializable data contracts shared by pipeline boundaries.

The retrieval system deliberately carries enough identity and timing information
on every record to render a result and to produce a competition submission.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskType(str, Enum):
    """Supported challenge task families."""

    KIS = "KIS"
    QA = "QA"
    TRAKE = "TRAKE"


class ContractModel(BaseModel):
    """Common Pydantic settings for public pipeline records."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


def _non_empty(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _finite(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


class VideoRecord(ContractModel):
    """Source video metadata used during preprocessing and result rendering."""

    video_id: str
    source_path: str | None = None
    duration: float | None = Field(default=None, ge=0.0)
    fps: float | None = Field(default=None, gt=0.0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str) -> str:
        return _non_empty(value, "video_id")

    @field_validator("duration", "fps")
    @classmethod
    def validate_numeric_metadata(cls, value: float | None, info: Any) -> float | None:
        return None if value is None else _finite(value, info.field_name)


class ShotRecord(ContractModel):
    """A contiguous visual shot in one video."""

    shot_id: str
    video_id: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    keyframe_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("shot_id", "video_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("start", "end")
    @classmethod
    def validate_timestamps(cls, value: float, info: Any) -> float:
        return _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_interval(self) -> "ShotRecord":
        if self.end < self.start:
            raise ValueError("shot end must be greater than or equal to start")
        return self


class FrameRecord(ContractModel):
    """A renderable keyframe with its stable source identifiers."""

    frame_id: str
    video_id: str
    timestamp: float = Field(ge=0.0)
    image_path: str
    shot_id: str | None = None
    frame_number: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("frame_id", "video_id", "image_path")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: float) -> float:
        return _finite(value, "timestamp")


class OcrRecord(ContractModel):
    """Text extracted from a frame and linked back to that frame."""

    frame_id: str
    video_id: str
    timestamp: float = Field(ge=0.0)
    text: str
    language: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str = "ocr"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("frame_id", "video_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: float) -> float:
        return _finite(value, "timestamp")


class AsrRecord(ContractModel):
    """A timestamped speech segment linked to a nearby frame or shot when known."""

    segment_id: str
    video_id: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str
    nearest_frame_id: str | None = None
    shot_id: str | None = None
    language: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("segment_id", "video_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("start", "end")
    @classmethod
    def validate_timestamps(cls, value: float, info: Any) -> float:
        return _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_interval(self) -> "AsrRecord":
        if self.end < self.start:
            raise ValueError("ASR end must be greater than or equal to start")
        return self


class QueryRecord(ContractModel):
    """An input query from a challenge query set."""

    query_id: str
    text: str
    task: TaskType
    translated_text: str | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query_id", "text")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)


class QueryPlan(ContractModel):
    """Auditable decomposition of one query into modality-specific requests.

    The three modality weights are always non-negative and normalized to one.
    ``raw_weights`` retains the planner's unnormalized response for debugging.
    If a planner emits all zero weights, the safe visual-only fallback is used.
    """

    original_query: str
    query_id: str | None = None
    translated_query: str | None = None
    visual_queries: list[str] = Field(default_factory=list)
    ocr_query: str | None = None
    asr_query: str | None = None
    temporal_events: list[str] = Field(default_factory=list)
    visual_weight: float = 1.0
    ocr_weight: float = 0.0
    asr_weight: float = 0.0
    raw_weights: dict[str, float] = Field(default_factory=dict)
    planner: str = "heuristic"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("original_query")
    @classmethod
    def validate_original_query(cls, value: str) -> str:
        return _non_empty(value, "original_query")

    @field_validator("visual_queries", "temporal_events")
    @classmethod
    def clean_query_lists(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for value in values:
            text = value.strip()
            if text and text not in seen:
                cleaned.append(text)
                seen.add(text)
        return cleaned

    @model_validator(mode="before")
    @classmethod
    def normalize_modality_weights(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        data = dict(values)
        supplied_raw = data.get("raw_weights")
        if supplied_raw:
            if not isinstance(supplied_raw, dict):
                raise ValueError("raw_weights must be a mapping")
            raw = {
                "visual": supplied_raw.get("visual", data.get("visual_weight", 0.0)),
                "ocr": supplied_raw.get("ocr", data.get("ocr_weight", 0.0)),
                "asr": supplied_raw.get("asr", data.get("asr_weight", 0.0)),
            }
        else:
            raw = {
                "visual": data.get("visual_weight", 1.0),
                "ocr": data.get("ocr_weight", 0.0),
                "asr": data.get("asr_weight", 0.0),
            }

        numeric_raw: dict[str, float] = {}
        for modality, weight in raw.items():
            try:
                numeric = float(weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{modality} weight must be numeric") from exc
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{modality} weight must be finite and non-negative")
            numeric_raw[modality] = numeric

        total = sum(numeric_raw.values())
        if total == 0.0:
            normalized = {"visual": 1.0, "ocr": 0.0, "asr": 0.0}
        else:
            normalized = {name: weight / total for name, weight in numeric_raw.items()}

        data["raw_weights"] = numeric_raw
        data["visual_weight"] = normalized["visual"]
        data["ocr_weight"] = normalized["ocr"]
        data["asr_weight"] = normalized["asr"]
        return data


class MomentResult(ContractModel):
    """A ranked, renderable moment returned by one or more retrieval branches."""

    video_id: str
    frame_id: str
    timestamp: float = Field(ge=0.0)
    image_path: str
    shot_id: str | None = None
    score: float = 0.0
    rank: int | None = Field(default=None, ge=1)
    modality_scores: dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("modality_scores", "scores"),
    )
    fused_score: float | None = None
    reranker_score: float | None = None
    provenance: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("video_id", "frame_id", "image_path")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("timestamp", "score", "fused_score", "reranker_score")
    @classmethod
    def validate_scores_and_timestamp(cls, value: float | None, info: Any) -> float | None:
        return None if value is None else _finite(value, info.field_name)

    @field_validator("modality_scores")
    @classmethod
    def validate_modality_scores(cls, scores: dict[str, float]) -> dict[str, float]:
        return {name: _finite(float(score), f"modality_scores[{name}]") for name, score in scores.items()}


class SequenceEvent(ContractModel):
    """One ordered event in a temporal retrieval result."""

    event_index: int = Field(ge=0)
    video_id: str
    frame_id: str
    timestamp: float = Field(ge=0.0)
    description: str
    score: float = 0.0
    image_path: str | None = None
    shot_id: str | None = None
    provenance: list[str] = Field(default_factory=list)

    @field_validator("video_id", "frame_id", "description")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("timestamp", "score")
    @classmethod
    def validate_numeric_fields(cls, value: float, info: Any) -> float:
        return _finite(value, info.field_name)


class SequenceResult(ContractModel):
    """An ordered, same-video temporal answer."""

    video_id: str
    events: list[SequenceEvent] = Field(min_length=1)
    score: float = 0.0
    rank: int | None = Field(default=None, ge=1)
    reranker_score: float | None = None
    duration: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str) -> str:
        return _non_empty(value, "video_id")

    @field_validator("score", "reranker_score", "duration")
    @classmethod
    def validate_numeric_fields(cls, value: float | None, info: Any) -> float | None:
        return None if value is None else _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_temporal_sequence(self) -> "SequenceResult":
        frame_ids: set[str] = set()
        previous_timestamp: float | None = None
        for expected_index, event in enumerate(self.events):
            if event.video_id != self.video_id:
                raise ValueError("all sequence events must belong to the result video")
            if event.event_index != expected_index:
                raise ValueError("event_index values must be contiguous and start at zero")
            if event.frame_id in frame_ids:
                raise ValueError("sequence events must not repeat a frame")
            if previous_timestamp is not None and event.timestamp <= previous_timestamp:
                raise ValueError("sequence event timestamps must be strictly increasing")
            frame_ids.add(event.frame_id)
            previous_timestamp = event.timestamp

        inferred_duration = self.events[-1].timestamp - self.events[0].timestamp
        if self.duration is None:
            # ``validate_assignment=True`` would re-enter this model validator
            # if normal assignment were used here.
            object.__setattr__(self, "duration", inferred_duration)
        elif not math.isclose(self.duration, inferred_duration, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("duration must equal the first-to-last event timestamp span")
        return self


class AnswerResult(ContractModel):
    """A grounded QA response with the retrieved evidence used to produce it."""

    query_id: str
    answer: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[MomentResult] = Field(default_factory=list)
    provider: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, value: str) -> str:
        return _non_empty(value, "query_id")

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value or None

    @model_validator(mode="after")
    def validate_grounding(self) -> "AnswerResult":
        if self.answer is not None and not self.evidence:
            raise ValueError("a non-null answer requires at least one evidence moment")
        if self.answer is None:
            # Avoid recursive assignment validation from this after-validator.
            object.__setattr__(self, "confidence", 0.0)
        return self
