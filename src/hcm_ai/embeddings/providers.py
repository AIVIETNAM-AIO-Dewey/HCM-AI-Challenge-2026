"""Optional model providers with deterministic CPU-safe fallbacks."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol


class ProviderUnavailableError(RuntimeError):
    """An optional model is not installed, configured, or usable on this host."""


class EmbeddingProvider(Protocol):
    """Common surface used by visual retrieval indexing and query search."""

    @property
    def dimension(self) -> int: ...

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_images(self, images: Sequence[str | Path]) -> list[list[float]]: ...


class Translator(Protocol):
    def translate(self, texts: Sequence[str]) -> list[str]: ...


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class HashEmbeddingProvider:
    """Deterministic embeddings for unit tests and no-model smoke runs.

    This is intentionally not a semantic model.  It lets the full indexing and
    retrieval pipeline run without downloading weights, while production
    profiles replace it with ``LazyTransformersEmbeddingProvider``.
    """

    def __init__(self, dimension: int = 64, *, namespace: str = "hcm-ai") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self.namespace = namespace

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, value: str) -> list[float]:
        digest = hashlib.sha512(f"{self.namespace}:{value}".encode("utf-8")).digest()
        values = [0.0] * self.dimension
        for index in range(self.dimension):
            byte = digest[index % len(digest)]
            values[index] = (byte / 127.5) - 1.0
        return _normalise(values)

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def encode_images(self, images: Sequence[str | Path]) -> list[list[float]]:
        return [self._encode(Path(image).as_posix()) for image in images]


class IdentityTranslator:
    """Safe fallback which preserves the original query for auditability."""

    def translate(self, texts: Sequence[str]) -> list[str]:
        return list(texts)


class MarianTranslator:
    """Lazy local Vietnamese-to-English Marian translation provider."""

    def __init__(
        self,
        model_id: str = "Helsinki-NLP/opus-mt-vi-en",
        *,
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise ProviderUnavailableError("Install hcm-ai[models] to enable local translation") from error
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        if self.device:
            self._model.to(self.device)
        self._model.eval()
        return self._tokenizer, self._model

    def translate(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            return []
        tokenizer, model = self._ensure_loaded()
        encoded = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
        if self.device:
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
        generated = model.generate(**encoded, max_new_tokens=self.max_new_tokens)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)


class LazyTransformersEmbeddingProvider:
    """Load a Hugging Face image-text encoder only when inference is requested."""

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        *,
        device: str | None = None,
        batch_size: int = 16,
        dimension: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._dimension = dimension
        self._processor: Any | None = None
        self._model: Any | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise ProviderUnavailableError("Model dimension is unknown until the provider has loaded")
        return self._dimension

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise ProviderUnavailableError("Install hcm-ai[models] to enable transformer embeddings") from error
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id)
        if self.device:
            self._model.to(self.device)
        self._model.eval()
        config = getattr(self._model, "config", None)
        self._dimension = self._dimension or getattr(config, "projection_dim", None) or getattr(config, "hidden_size", None)
        if not self._dimension:
            raise ProviderUnavailableError(f"Could not infer embedding dimension for {self.model_id}")
        return self._processor, self._model

    def _encode(self, *, texts: Sequence[str] | None = None, images: Sequence[str | Path] | None = None) -> list[list[float]]:
        if bool(texts) == bool(images):
            raise ValueError("provide exactly one of texts or images")
        processor, model = self._ensure_loaded()
        try:
            import torch
            from PIL import Image
        except ImportError as error:
            raise ProviderUnavailableError("Install hcm-ai[models] for image/text encoding") from error
        values = list(texts or images or [])
        embeddings: list[list[float]] = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start : start + self.batch_size]
            if texts:
                inputs = processor(text=list(batch), padding=True, return_tensors="pt")
                getter = getattr(model, "get_text_features", None)
                if getter is None:
                    raise ProviderUnavailableError(f"{self.model_id} does not expose get_text_features")
            else:
                opened = [Image.open(Path(path)).convert("RGB") for path in batch]
                inputs = processor(images=opened, return_tensors="pt")
                getter = getattr(model, "get_image_features", None)
                if getter is None:
                    raise ProviderUnavailableError(f"{self.model_id} does not expose get_image_features")
            if self.device:
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                output = getter(**inputs)
                output = torch.nn.functional.normalize(output, p=2, dim=-1)
            embeddings.extend(output.detach().float().cpu().tolist())
        return embeddings

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts=texts) if texts else []

    def encode_images(self, images: Sequence[str | Path]) -> list[list[float]]:
        return self._encode(images=images) if images else []
