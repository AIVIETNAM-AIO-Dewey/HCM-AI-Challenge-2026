"""Lazy faster-whisper adapter with a deterministic no-op fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hcm_ai.embeddings.providers import ProviderUnavailableError


@dataclass(frozen=True, slots=True)
class AsrSegment:
    start: float
    end: float
    text: str
    language: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("ASR timestamps must be non-negative and ordered")


class AsrProvider(Protocol):
    def transcribe(self, audio_path: str | Path) -> list[AsrSegment]: ...


class NullAsrProvider:
    def transcribe(self, audio_path: str | Path) -> list[AsrSegment]:
        return []


class FasterWhisperProvider:
    """Load CTranslate2/Whisper weights only at transcription time."""

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "auto",
        compute_type: str = "int8",
        vad_filter: bool = True,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self._model: Any | None = None

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise ProviderUnavailableError("Install hcm-ai[asr] to enable faster-whisper") from error
        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio_path: str | Path) -> list[AsrSegment]:
        segments, info = self._ensure_loaded().transcribe(str(audio_path), vad_filter=self.vad_filter)
        language = getattr(info, "language", None)
        return [
            AsrSegment(start=float(segment.start), end=float(segment.end), text=str(segment.text).strip(), language=language)
            for segment in segments
            if str(segment.text).strip()
        ]
