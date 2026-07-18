"""Selective, cached Gemini OCR refinement layered over local bulk OCR."""

from __future__ import annotations

import json
import mimetypes
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from hcm_ai.cache import JsonCache

from .providers import OcrExtraction, OcrProvider


class GeminiOcrRefiner:
    """Refine only text-rich/candidate images; retain local OCR on failures.

    Bulk indexing should use PaddleOCR alone. This wrapper is opt-in for a
    bounded subset of likely text-heavy or result-candidate frames, and caches
    successful free-tier Gemini calls when ``cache_dir`` is supplied.
    """

    def __init__(
        self,
        base: OcrProvider,
        *,
        api_key_env: str = "GOOGLE_API_KEY",
        model: str = "gemini-2.5-flash-lite",
        cache_dir: str | Path | None = None,
        text_rich_min_chars: int = 8,
        max_refinements: int = 16,
        min_interval_seconds: float = 0.5,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if min(text_rich_min_chars, max_refinements, min_interval_seconds, max_retries) < 0:
            raise ValueError("Gemini OCR limits must be non-negative")
        self.base = base
        self.api_key_env = api_key_env
        self.model = model
        self.text_rich_min_chars = text_rich_min_chars
        self.max_refinements = max_refinements
        self.min_interval_seconds = min_interval_seconds
        self.max_retries = max_retries
        self.client = client
        self.cache = JsonCache(cache_dir, namespace="gemini_ocr") if cache_dir else None
        self._last_request_at = 0.0

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        api_key = os.environ.get(self.api_key_env) or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(f"missing Gemini key in {self.api_key_env}")
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError("Install hcm-ai[gemini] to enable Gemini OCR refinement") from error
        self.client = genai.Client(api_key=api_key)
        return self.client

    @staticmethod
    def _cache_key(image_path: Path, base: OcrExtraction, model: str) -> dict[str, Any]:
        stat = image_path.stat()
        return {
            "kind": "ocr_refinement",
            "model": model,
            "path": image_path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "base_text": base.text,
        }

    @staticmethod
    def _prompt(base: OcrExtraction) -> str:
        return (
            "Read all visible text in this video keyframe. Return JSON only: "
            '{"text": string, "confidence": number|null, "language": string|null}. '
            f"Keep exact spelling, numbers, diacritics, and line order. Local OCR draft: {base.text}"
        )

    def _request(self, image_path: Path, base: OcrExtraction) -> Mapping[str, Any]:
        delay = self.min_interval_seconds - (time.monotonic() - self._last_request_at)
        if delay > 0:
            time.sleep(delay)
        raw = image_path.read_bytes()
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        try:
            from google.genai import types

            image_part: Any = types.Part.from_bytes(data=raw, mime_type=mime_type)
        except Exception:
            image_part = {"mime_type": mime_type, "data": raw}
        response = self._client().models.generate_content(
            model=self.model,
            contents=[self._prompt(base), image_part],
            config={"response_mime_type": "application/json"},
        )
        self._last_request_at = time.monotonic()
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemini returned no OCR text")
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].removesuffix("```").strip()
        value = json.loads(clean)
        if not isinstance(value, Mapping):
            raise ValueError("Gemini OCR response must be a JSON object")
        return value

    @staticmethod
    def _coerce(raw: Mapping[str, Any], fallback: OcrExtraction) -> OcrExtraction:
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            return fallback
        confidence = raw.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else fallback.confidence
        except (TypeError, ValueError):
            confidence = fallback.confidence
        if confidence is not None:
            confidence = min(1.0, max(0.0, confidence))
        language = raw.get("language")
        return OcrExtraction(
            text=text.strip(),
            confidence=confidence,
            language=language.strip() if isinstance(language, str) and language.strip() else fallback.language,
        )

    def refine(
        self,
        image_paths: Sequence[str | Path],
        base_extractions: Sequence[OcrExtraction],
        *,
        candidate_paths: Iterable[str | Path] = (),
    ) -> list[OcrExtraction]:
        """Refine only text-rich/candidate paths and never lose local OCR."""

        if len(image_paths) != len(base_extractions):
            raise ValueError("image_paths and base_extractions must have the same length")
        approved = {str(Path(path)) for path in candidate_paths}
        refined = list(base_extractions)
        used = 0
        for index, (raw_path, base) in enumerate(zip(image_paths, base_extractions, strict=True)):
            path = Path(raw_path)
            if used >= self.max_refinements or not (
                str(path) in approved or len(base.text.strip()) >= self.text_rich_min_chars
            ):
                continue
            try:
                key = self._cache_key(path, base, self.model)
                response = self.cache.get(key) if self.cache is not None else None
                if not isinstance(response, Mapping):
                    last_error: Exception | None = None
                    for attempt in range(self.max_retries + 1):
                        try:
                            response = self._request(path, base)
                            if self.cache is not None:
                                self.cache.set(key, dict(response))
                            break
                        except Exception as error:
                            last_error = error
                            if attempt < self.max_retries:
                                time.sleep(0.25 * (attempt + 1))
                    if not isinstance(response, Mapping):
                        raise last_error or RuntimeError("Gemini OCR refinement failed")
                refined[index] = self._coerce(response, base)
                used += 1
            except Exception:
                continue
        return refined

    def extract(self, image_paths: Sequence[str | Path]) -> list[OcrExtraction]:
        """Run local OCR first, then selectively refine text-rich frames."""

        base = self.base.extract(image_paths)
        return self.refine(image_paths, base)


__all__ = ["GeminiOcrRefiner"]
