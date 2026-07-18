from __future__ import annotations

from hcm_ai.config import load_settings
from hcm_ai.profiles import build_profile_components
from hcm_ai.runtime import RuntimeCapabilities, resolve_profile


def _cpu_capabilities() -> RuntimeCapabilities:
    return RuntimeCapabilities(
        cuda_available=False,
        gpu_name=None,
        vram_gib=0.0,
        transformers_available=False,
        faiss_available=False,
    )


def test_profile_resolution_downgrades_without_gpu_or_transformers() -> None:
    capabilities = _cpu_capabilities()
    assert resolve_profile("paper_gpu", capabilities) == "cpu"
    assert resolve_profile("balanced_gpu", capabilities) == "cpu"
    assert resolve_profile("auto", capabilities) == "cpu"


def test_yaml_cpu_profile_constructs_lazy_siglip_components() -> None:
    settings = load_settings(profile="cpu", environ={})
    components = build_profile_components(settings, capabilities=_cpu_capabilities())
    assert components.effective_profile == "cpu"
    assert set(components.visual_embeddings) == {"siglip"}
    assert components.reranker.score("query", []) == []
