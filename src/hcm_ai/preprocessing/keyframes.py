"""Keyframe discovery and fixed-interval extraction helpers."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .media import MediaToolError

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True, slots=True)
class ExtractedKeyframe:
    """Filesystem-level keyframe metadata before conversion to FrameRecord."""

    video_id: str
    frame_id: str
    timestamp: float
    image_path: Path
    shot_id: str | None = None


def discover_keyframes(root: str | Path, *, video_id: str | None = None, fps: float | None = None) -> list[ExtractedKeyframe]:
    """Discover supplied keyframes without assuming a legacy archive layout."""

    directory = Path(root)
    inferred_video_id = video_id or directory.name
    results: list[ExtractedKeyframe] = []
    for index, path in enumerate(sorted(candidate for candidate in directory.rglob("*") if candidate.suffix.lower() in _IMAGE_SUFFIXES)):
        timestamp = _timestamp_from_name(path.stem)
        if timestamp is None:
            timestamp = index / fps if fps and fps > 0 else float(index)
        results.append(
            ExtractedKeyframe(
                video_id=inferred_video_id,
                frame_id=f"{inferred_video_id}_{path.stem}",
                timestamp=timestamp,
                image_path=path,
            )
        )
    return results


def _timestamp_from_name(stem: str) -> float | None:
    """Interpret common timestamp names such as ``12.34`` or ``t_001234ms``."""

    seconds_match = re.search(r"(?:^|[_-])t?(\d+(?:\.\d+)?)s?(?:$|[_-])", stem, re.IGNORECASE)
    if seconds_match:
        return float(seconds_match.group(1))
    milliseconds_match = re.search(r"(\d+)ms", stem, re.IGNORECASE)
    return float(milliseconds_match.group(1)) / 1000.0 if milliseconds_match else None


def extract_interval_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    video_id: str,
    interval_seconds: float = 2.0,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
) -> list[ExtractedKeyframe]:
    """Extract one JPEG every ``interval_seconds`` with stable frame IDs."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pattern = destination / "frame_%06d.jpg"
    existing = list(destination.glob("frame_*.jpg"))
    if not existing or overwrite:
        command = [
            ffmpeg_bin,
            "-y" if overwrite else "-n",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval_seconds}",
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(pattern),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        except FileNotFoundError as error:
            raise MediaToolError(f"Media executable not found: {ffmpeg_bin}") from error
        except subprocess.CalledProcessError as error:
            raise MediaToolError(error.stderr.strip() or "ffmpeg keyframe extraction failed") from error
    return [
        ExtractedKeyframe(
            video_id=video_id,
            frame_id=f"{video_id}_F{index:06d}",
            timestamp=index * interval_seconds,
            image_path=path,
        )
        for index, path in enumerate(sorted(destination.glob("frame_*.jpg")))
    ]
