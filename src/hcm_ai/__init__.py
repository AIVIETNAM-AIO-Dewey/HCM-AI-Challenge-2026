"""Reusable components for HCM AI Challenge multimodal moment retrieval."""

from .environment import load_environment, resolve_env_file

# Load a checkout-local .env before modules inspect API keys or paths. Existing
# process variables (including Colab Secrets) keep precedence.
load_environment()

from .contracts import (  # noqa: E402
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
from .retrieval import SearchService, SearchSettings, SearchTrace  # noqa: E402

__all__ = [
    "AnswerResult",
    "AsrRecord",
    "FrameRecord",
    "load_environment",
    "MomentResult",
    "OcrRecord",
    "QueryPlan",
    "QueryRecord",
    "resolve_env_file",
    "SequenceEvent",
    "SequenceResult",
    "SearchService",
    "SearchSettings",
    "SearchTrace",
    "ShotRecord",
    "TaskType",
    "VideoRecord",
]
