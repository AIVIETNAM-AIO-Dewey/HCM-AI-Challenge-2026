"""Shot-boundary interfaces with a deterministic interval fallback."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShotBoundary:
    """A contiguous time range in one video."""

    index: int
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("shot index must be non-negative")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("shot boundaries must have a positive duration")


class FixedIntervalShotDetector:
    """CPU-safe fallback used when TransNetV2 is unavailable."""

    def __init__(self, interval_seconds: float = 10.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds

    def detect(self, duration_seconds: float) -> list[ShotBoundary]:
        if duration_seconds <= 0:
            return []
        boundaries: list[ShotBoundary] = []
        start = 0.0
        index = 0
        while start < duration_seconds:
            end = min(duration_seconds, start + self.interval_seconds)
            boundaries.append(ShotBoundary(index=index, start_seconds=start, end_seconds=end))
            start = end
            index += 1
        return boundaries


class TransNetV2ShotDetector:
    """Adapter around an injected TransNetV2-compatible prediction function.

    The repository does not import a heavy TensorFlow model at package import
    time.  A Colab notebook can provide a function that returns cut timestamps.
    """

    def __init__(self, predictor: Callable[[str], Sequence[float]]) -> None:
        self._predictor = predictor

    def detect(self, video_path: str, duration_seconds: float) -> list[ShotBoundary]:
        cuts = sorted({float(cut) for cut in self._predictor(video_path) if 0 < float(cut) < duration_seconds})
        starts = [0.0, *cuts]
        ends = [*cuts, duration_seconds]
        return [
            ShotBoundary(index=index, start_seconds=start, end_seconds=end)
            for index, (start, end) in enumerate(zip(starts, ends))
            if end > start
        ]
