"""Reusable components for HCM AI Challenge multimodal moment retrieval."""

from .contracts import (
    AnswerResult,
    AsrRecord,
    FrameRecord,
    MomentResult,
    OcrRecord,
    QueryPlan,
    QueryRecord,
    SequenceEvent,
    SequenceResult,
    ShotRecord,
    TaskType,
    VideoRecord,
)
from .retrieval import SearchService, SearchSettings, SearchTrace

__all__ = [
    "AnswerResult",
    "AsrRecord",
    "FrameRecord",
    "MomentResult",
    "OcrRecord",
    "QueryPlan",
    "QueryRecord",
    "SequenceEvent",
    "SequenceResult",
    "SearchService",
    "SearchSettings",
    "SearchTrace",
    "ShotRecord",
    "TaskType",
    "VideoRecord",
]
