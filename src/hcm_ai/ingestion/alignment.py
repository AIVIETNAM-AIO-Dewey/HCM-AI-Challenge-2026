"""Convert preprocessing/OCR/ASR outputs into retrieval-ready data contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from hcm_ai.asr import AsrSegment
from hcm_ai.contracts import AsrRecord, FrameRecord, OcrRecord, ShotRecord
from hcm_ai.ocr import OcrExtraction
from hcm_ai.preprocessing.keyframes import ExtractedKeyframe


def frame_records_from_keyframes(
    keyframes: Iterable[ExtractedKeyframe],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> list[FrameRecord]:
    """Make renderable frame contracts from filesystem-level keyframe records."""

    shared_metadata = dict(metadata or {})
    return [
        FrameRecord(
            frame_id=item.frame_id,
            video_id=item.video_id,
            timestamp=item.timestamp,
            image_path=str(item.image_path),
            shot_id=item.shot_id,
            metadata={**shared_metadata, "source": "keyframe_extraction"},
        )
        for item in keyframes
    ]


def nearest_frame(
    frames: Sequence[FrameRecord],
    *,
    video_id: str,
    timestamp: float,
) -> FrameRecord | None:
    """Return the closest same-video frame, resolving ties by earlier time/ID."""

    candidates = [frame for frame in frames if frame.video_id == video_id]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda frame: (abs(frame.timestamp - timestamp), frame.timestamp, frame.frame_id),
    )


def shot_for_timestamp(
    shots: Sequence[ShotRecord],
    *,
    video_id: str,
    timestamp: float,
) -> ShotRecord | None:
    """Find a containing shot, including the final endpoint for convenience."""

    matches = [
        shot
        for shot in shots
        if shot.video_id == video_id and shot.start <= timestamp <= shot.end
    ]
    if not matches:
        return None
    return min(matches, key=lambda shot: (shot.end - shot.start, shot.start, shot.shot_id))


def ocr_records_from_extractions(
    frames: Sequence[FrameRecord],
    extractions: Sequence[OcrExtraction],
    *,
    source: str = "paddleocr",
) -> list[OcrRecord]:
    """Align one OCR extraction to each indexed keyframe."""

    if len(frames) != len(extractions):
        raise ValueError("frames and OCR extractions must have the same length")
    return [
        OcrRecord(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            timestamp=frame.timestamp,
            text=extraction.text,
            confidence=extraction.confidence,
            language=extraction.language,
            source=source,
        )
        for frame, extraction in zip(frames, extractions, strict=True)
    ]


def asr_records_from_segments(
    *,
    video_id: str,
    segments: Sequence[AsrSegment],
    frames: Sequence[FrameRecord],
    shots: Sequence[ShotRecord] = (),
    source: str = "faster_whisper",
) -> list[AsrRecord]:
    """Map timestamped ASR segments to their closest available keyframe/shot."""

    records: list[AsrRecord] = []
    for index, segment in enumerate(segments):
        midpoint = (segment.start + segment.end) / 2.0
        frame = nearest_frame(frames, video_id=video_id, timestamp=midpoint)
        shot = shot_for_timestamp(shots, video_id=video_id, timestamp=midpoint)
        records.append(
            AsrRecord(
                segment_id=f"{video_id}_ASR_{index:06d}",
                video_id=video_id,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                nearest_frame_id=frame.frame_id if frame else None,
                shot_id=shot.shot_id if shot else (frame.shot_id if frame else None),
                language=segment.language,
                metadata={"source": source, "midpoint": midpoint},
            )
        )
    return records


def text_rows_for_ocr(
    records: Sequence[OcrRecord],
    frames_by_id: Mapping[str, FrameRecord],
) -> tuple[list[FrameRecord], list[str]]:
    """Return aligned frame/text rows suitable for an OCR :class:`BM25Store`."""

    frames: list[FrameRecord] = []
    texts: list[str] = []
    for record in records:
        frame = frames_by_id.get(record.frame_id)
        if frame is not None and record.text.strip():
            frames.append(frame)
            texts.append(record.text)
    return frames, texts


def text_rows_for_asr(
    records: Sequence[AsrRecord],
    frames_by_id: Mapping[str, FrameRecord],
) -> tuple[list[FrameRecord], list[str]]:
    """Return aligned frame/text rows for ASR search; skip unrenderable segments."""

    frames: list[FrameRecord] = []
    texts: list[str] = []
    for record in records:
        frame = frames_by_id.get(record.nearest_frame_id or "")
        if frame is not None and record.text.strip():
            frames.append(frame)
            texts.append(record.text)
    return frames, texts


__all__ = [
    "asr_records_from_segments",
    "frame_records_from_keyframes",
    "nearest_frame",
    "ocr_records_from_extractions",
    "shot_for_timestamp",
    "text_rows_for_asr",
    "text_rows_for_ocr",
]
