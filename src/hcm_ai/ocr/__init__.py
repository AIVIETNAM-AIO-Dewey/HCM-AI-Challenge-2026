"""OCR providers that keep external/cloud work optional."""

from .gemini import GeminiOcrRefiner
from .providers import NullOcrProvider, OcrExtraction, OcrProvider, PaddleOcrProvider

__all__ = ["GeminiOcrRefiner", "NullOcrProvider", "OcrExtraction", "OcrProvider", "PaddleOcrProvider"]
