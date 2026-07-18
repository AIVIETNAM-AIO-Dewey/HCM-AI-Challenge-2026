"""Thin wrappers around FFmpeg/ffprobe with clear, testable failure modes."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class MediaToolError(RuntimeError):
    """Raised when a required external media tool is missing or fails."""


@dataclass(frozen=True, slots=True)
class VideoProbe:
    """Minimal metadata needed to resume video preprocessing."""

    path: Path
    duration_seconds: float
    width: int | None
    height: int | None
    fps: float | None
    frame_count: int | None
    has_audio: bool


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as error:
        raise MediaToolError(f"Media executable not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip()
        raise MediaToolError(f"Media command failed: {' '.join(command)}\n{details}") from error


def _parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        numerator, denominator = value.split("/", maxsplit=1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else None
    except (AttributeError, ValueError, ZeroDivisionError):
        return None


def probe_video(path: str | Path, *, ffprobe_bin: str = "ffprobe") -> VideoProbe:
    """Read metadata without decoding the video stream."""

    source = Path(path)
    result = _run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(source),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    duration = payload.get("format", {}).get("duration")
    try:
        duration_seconds = max(0.0, float(duration))
    except (TypeError, ValueError):
        duration_seconds = 0.0
    frame_count = video_stream.get("nb_frames")
    try:
        parsed_frame_count = int(frame_count) if frame_count not in {None, "N/A"} else None
    except (TypeError, ValueError):
        parsed_frame_count = None
    return VideoProbe(
        path=source,
        duration_seconds=duration_seconds,
        width=_as_int(video_stream.get("width")),
        height=_as_int(video_stream.get("height")),
        fps=_parse_rate(video_stream.get("avg_frame_rate")),
        frame_count=parsed_frame_count,
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def _as_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def extract_audio(
    video_path: str | Path,
    audio_path: str | Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 16_000,
    overwrite: bool = False,
) -> Path:
    """Extract mono PCM WAV suitable for timestamped speech recognition."""

    source = Path(video_path)
    destination = Path(audio_path)
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg_bin,
            "-y" if overwrite else "-n",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    return destination
