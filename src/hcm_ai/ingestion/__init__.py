"""Dataset adapters and ingestion helpers."""

from .aic2025 import (
    AIC2025Adapter,
    OptionalDependencyError,
    build_keyframe_manifest,
    infer_task_type,
    load_aic2025_queries,
    parse_excel_queries,
    parse_txt_query,
)
from .alignment import (
    asr_records_from_segments,
    frame_records_from_keyframes,
    nearest_frame,
    ocr_records_from_extractions,
    shot_for_timestamp,
    text_rows_for_asr,
    text_rows_for_ocr,
)

__all__ = [
    "AIC2025Adapter",
    "OptionalDependencyError",
    "asr_records_from_segments",
    "build_keyframe_manifest",
    "frame_records_from_keyframes",
    "infer_task_type",
    "load_aic2025_queries",
    "parse_excel_queries",
    "parse_txt_query",
    "nearest_frame",
    "ocr_records_from_extractions",
    "shot_for_timestamp",
    "text_rows_for_asr",
    "text_rows_for_ocr",
]
