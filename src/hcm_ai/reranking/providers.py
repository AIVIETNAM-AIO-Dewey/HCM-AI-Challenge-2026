"""Lazy ITM reranking adapters; never make tests download model weights."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from hcm_ai.embeddings.providers import ProviderUnavailableError


class Reranker(Protocol):
    def score(self, query: str, image_paths: Sequence[str | Path]) -> list[float]: ...


class NullReranker:
    """Produces neutral gates when a GPU cross-encoder is unavailable."""

    def score(self, query: str, image_paths: Sequence[str | Path]) -> list[float]:
        return [1.0 for _ in image_paths]


class TransformersItmReranker:
    """Best-effort BLIP/BLIP-2 ITM reranker for small top-K candidate sets."""

    def __init__(self, model_id: str = "Salesforce/blip-itm-base-coco", *, device: str | None = None, batch_size: int = 4) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._processor: Any | None = None
        self._model: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            from transformers import AutoProcessor
        except ImportError as error:
            raise ProviderUnavailableError("Install hcm-ai[models] to enable ITM reranking") from error
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        try:
            from transformers import BlipForImageTextRetrieval

            self._model = BlipForImageTextRetrieval.from_pretrained(self.model_id)
        except Exception as error:
            raise ProviderUnavailableError(f"{self.model_id} is not supported as an ITM reranker") from error
        if self.device:
            self._model.to(self.device)
        self._model.eval()
        return self._processor, self._model

    def score(self, query: str, image_paths: Sequence[str | Path]) -> list[float]:
        if not image_paths:
            return []
        processor, model = self._ensure_loaded()
        try:
            import torch
            from PIL import Image
        except ImportError as error:
            raise ProviderUnavailableError("Install hcm-ai[models] to enable ITM reranking") from error
        scores: list[float] = []
        for start in range(0, len(image_paths), self.batch_size):
            paths = image_paths[start : start + self.batch_size]
            images = [Image.open(Path(path)).convert("RGB") for path in paths]
            inputs = processor(images=images, text=[query] * len(images), return_tensors="pt", padding=True)
            if self.device:
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                output = model(**inputs, use_itm_head=True)
                logits = getattr(output, "itm_score", None)
                if logits is None:
                    logits = getattr(output, "logits", None)
                if logits is None:
                    raise ProviderUnavailableError("ITM model did not return scores")
                if logits.ndim == 2 and logits.shape[-1] > 1:
                    batch_scores = torch.softmax(logits, dim=-1)[:, 1]
                else:
                    batch_scores = torch.sigmoid(logits.reshape(-1))
            scores.extend(float(value) for value in batch_scores.detach().cpu().tolist())
        return scores


class Blip2ItmReranker:
    """BLIP-2 yes/no relevance gate used only for paper-profile top-K reranking.

    Public BLIP-2 checkpoints do not expose the same ITM head as BLIP-1, so
    this adapter asks a constrained visual question and converts an affirmative
    answer into a bounded score.  It is intentionally lazy and is never part
    of the cheap retrieval stage.
    """

    def __init__(
        self,
        model_id: str = "Salesforce/blip2-opt-2.7b",
        *,
        device: str | None = None,
        batch_size: int = 1,
        max_new_tokens: int = 3,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._processor: Any | None = None
        self._model: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
        except ImportError as error:
            raise ProviderUnavailableError("Install hcm-ai[models] to enable BLIP-2 reranking") from error
        try:
            self._processor = Blip2Processor.from_pretrained(self.model_id)
            self._model = Blip2ForConditionalGeneration.from_pretrained(self.model_id)
        except Exception as error:
            raise ProviderUnavailableError(f"could not load BLIP-2 reranker {self.model_id}") from error
        if self.device:
            self._model.to(self.device)
        self._model.eval()
        return self._processor, self._model

    def score(self, query: str, image_paths: Sequence[str | Path]) -> list[float]:
        if not image_paths:
            return []
        processor, model = self._ensure_loaded()
        try:
            import torch
            from PIL import Image
        except ImportError as error:
            raise ProviderUnavailableError("Install hcm-ai[models] to enable BLIP-2 reranking") from error
        scores: list[float] = []
        prompt = f"Question: Does this image match the query '{query}'? Answer yes or no:"
        for start in range(0, len(image_paths), self.batch_size):
            paths = image_paths[start : start + self.batch_size]
            images = []
            for path in paths:
                with Image.open(Path(path)) as image:
                    images.append(image.convert("RGB").copy())
            inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True)
            if self.device:
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            answers = processor.batch_decode(generated, skip_special_tokens=True)
            for answer in answers:
                normalized = answer.strip().casefold()
                scores.append(1.0 if normalized.startswith(("yes", "có", "co ")) else 0.0)
        return scores
