"""Construct profile-selected components with deterministic safe downgrade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hcm_ai.asr import FasterWhisperProvider
from hcm_ai.config import Settings
from hcm_ai.embeddings import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    IdentityTranslator,
    LazyTransformersEmbeddingProvider,
    MarianTranslator,
    Translator,
)
from hcm_ai.ocr import PaddleOcrProvider
from hcm_ai.reranking import Blip2ItmReranker, NullReranker, Reranker, TransformersItmReranker
from hcm_ai.runtime import RuntimeCapabilities, detect_capabilities, resolve_profile


@dataclass(frozen=True, slots=True)
class ProfileComponents:
    """Lazy providers selected for one requested/effective YAML profile."""

    requested_profile: str
    effective_profile: str
    visual_embeddings: dict[str, EmbeddingProvider]
    reranker: Reranker
    translator: Translator
    ocr_provider: PaddleOcrProvider
    asr_provider: FasterWhisperProvider
    diagnostics: tuple[str, ...] = ()


def _model_id(settings: Settings, name: str, fallback: str) -> str:
    catalog = settings.models.catalog.get(name, {})
    value = catalog.get("model_id", fallback) if isinstance(catalog, dict) else fallback
    return str(value)


def _make_components(
    settings: Settings,
    *,
    requested_profile: str,
    effective_profile: str,
    capabilities: RuntimeCapabilities,
) -> ProfileComponents:
    device = "cuda" if effective_profile != "cpu" and capabilities.cuda_available else "cpu"
    encoder_names = ["siglip"] if effective_profile != "paper_gpu" else ["siglip", "beit3"]
    encoders: dict[str, EmbeddingProvider] = {
        name: LazyTransformersEmbeddingProvider(
            _model_id(
                settings,
                name,
                "google/siglip-base-patch16-224" if name == "siglip" else "microsoft/beit-3-base",
            ),
            device=device,
        )
        for name in encoder_names
    }
    if effective_profile == "balanced_gpu":
        reranker: Reranker = TransformersItmReranker(
            _model_id(settings, "blip_itm", "Salesforce/blip-itm-base-coco"), device=device
        )
    elif effective_profile == "paper_gpu":
        reranker = Blip2ItmReranker(
            _model_id(settings, "blip2_itm", "Salesforce/blip2-opt-2.7b"), device=device
        )
    else:
        reranker = NullReranker()
    return ProfileComponents(
        requested_profile=requested_profile,
        effective_profile=effective_profile,
        visual_embeddings=encoders,
        reranker=reranker,
        translator=MarianTranslator(_model_id(settings, "translation", "Helsinki-NLP/opus-mt-vi-en"), device=device),
        ocr_provider=PaddleOcrProvider(use_gpu=capabilities.cuda_available and effective_profile != "cpu"),
        asr_provider=FasterWhisperProvider(
            model_size="large-v3" if effective_profile == "paper_gpu" else settings.models.asr_model,
            device="cuda" if capabilities.cuda_available and effective_profile != "cpu" else "cpu",
            compute_type="float16" if capabilities.cuda_available and effective_profile != "cpu" else "int8",
        ),
    )


def _offline_fallback_components(
    settings: Settings,
    *,
    requested_profile: str,
    diagnostics: list[str],
) -> ProfileComponents:
    """Return a dependency-free CPU surface when Transformers is absent."""

    return ProfileComponents(
        requested_profile=requested_profile,
        effective_profile="cpu",
        visual_embeddings={"siglip": HashEmbeddingProvider(namespace="profile-fallback")},
        reranker=NullReranker(),
        translator=IdentityTranslator(),
        ocr_provider=PaddleOcrProvider(use_gpu=False),
        asr_provider=FasterWhisperProvider(model_size="small", device="cpu", compute_type="int8"),
        diagnostics=tuple([*diagnostics, "transformers unavailable; using deterministic hash embedding fallback"]),
    )


def build_profile_components(
    settings: Settings,
    *,
    requested_profile: str | None = None,
    capabilities: RuntimeCapabilities | None = None,
    self_test: bool = False,
) -> ProfileComponents:
    """Select providers and downgrade safely when hardware/dependencies fail.

    ``self_test=True`` intentionally loads only text encoders and model classes
    to catch Colab VRAM/dependency failures before a long indexing job.  It may
    download model weights, so ordinary library imports leave it false.
    """

    capabilities = capabilities or detect_capabilities()
    requested = requested_profile or settings.profile
    effective = resolve_profile(requested, capabilities)
    diagnostics: list[str] = []
    if effective != requested and requested != "auto":
        diagnostics.append(f"downgraded {requested} to {effective} from runtime capabilities")
    if not capabilities.transformers_available:
        return _offline_fallback_components(
            settings,
            requested_profile=requested,
            diagnostics=diagnostics,
        )
    components = _make_components(
        settings,
        requested_profile=requested,
        effective_profile=effective,
        capabilities=capabilities,
    )
    if not self_test:
        return components

    # The two profile-specific models are optional precision improvements.  If
    # either fails to load, retry the next lower profile rather than failing a
    # Colab cell after a lengthy job has begun.
    profiles = [effective]
    if effective == "paper_gpu":
        profiles.extend(["balanced_gpu", "cpu"])
    elif effective == "balanced_gpu":
        profiles.append("cpu")
    for candidate_profile in profiles:
        candidate = _make_components(
            settings,
            requested_profile=requested,
            effective_profile=candidate_profile,
            capabilities=capabilities,
        )
        try:
            for provider in candidate.visual_embeddings.values():
                provider.encode_texts(["profile self test"])
            ensure_reranker = getattr(candidate.reranker, "_ensure_loaded", None)
            if callable(ensure_reranker):
                ensure_reranker()
            return ProfileComponents(
                requested_profile=requested,
                effective_profile=candidate_profile,
                visual_embeddings=candidate.visual_embeddings,
                reranker=candidate.reranker,
                translator=candidate.translator,
                ocr_provider=candidate.ocr_provider,
                asr_provider=candidate.asr_provider,
                diagnostics=tuple(diagnostics),
            )
        except Exception as error:
            diagnostics.append(f"{candidate_profile} self-test failed: {type(error).__name__}: {error}")

    # Even an offline or dependency-less notebook can still use the full data
    # contracts and retrieval service with deterministic vectors.
    return _offline_fallback_components(
        settings,
        requested_profile=requested,
        diagnostics=[*diagnostics, "all requested profile self-tests failed"],
    )


__all__ = ["ProfileComponents", "build_profile_components"]
