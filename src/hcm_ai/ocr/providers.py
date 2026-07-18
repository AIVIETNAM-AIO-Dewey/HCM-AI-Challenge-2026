"""Local OCR with a no-op fallback for deterministic tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hcm_ai.embeddings.providers import ProviderUnavailableError


@dataclass(frozen=True, slots=True)
class OcrExtraction:
    text: str
    confidence: float | None = None
    language: str | None = None


class OcrProvider(Protocol):
    def extract(self, image_paths: Sequence[str | Path]) -> list[OcrExtraction]: ...


class NullOcrProvider:
    """CPU-safe provider used when OCR is disabled or unavailable."""

    def extract(self, image_paths: Sequence[str | Path]) -> list[OcrExtraction]:
        return [OcrExtraction(text="", confidence=None, language=None) for _ in image_paths]


class PaddleOcrProvider:
    """Lazy PaddleOCR provider; package import never downloads OCR models."""

    def __init__(self, *, language: str = "vi", use_gpu: bool | None = None) -> None:
        self.language = language
        self.use_gpu = use_gpu
        self._engine: Any | None = None

    def _ensure_loaded(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise ProviderUnavailableError("Install hcm-ai[ocr] to enable PaddleOCR") from error
        kwargs: dict[str, Any] = {"lang": self.language, "use_angle_cls": True}
        if self.use_gpu is not None:
            kwargs["use_gpu"] = self.use_gpu
        self._engine = PaddleOCR(**kwargs)
        return self._engine

    def extract(self, image_paths: Sequence[str | Path]) -> list[OcrExtraction]:
        engine = self._ensure_loaded()
        outputs: list[OcrExtraction] = []
        for image_path in image_paths:
            result = engine.ocr(str(image_path), cls=True)
            parts: list[str] = []
            confidences: list[float] = []
            for page in result or []:
                for line in page or []:
                    if len(line) >= 2 and isinstance(line[1], (tuple, list)):
                        parts.append(str(line[1][0]))
                        try:
                            confidences.append(float(line[1][1]))
                        except (TypeError, ValueError):
                            pass
            outputs.append(
                OcrExtraction(
                    text=" ".join(parts).strip(),
                    confidence=sum(confidences) / len(confidences) if confidences else None,
                    language=self.language,
                )
            )
        return outputs
