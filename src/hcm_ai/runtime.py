"""Runtime capability detection and deterministic profile downgrade logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    cuda_available: bool
    gpu_name: str | None
    vram_gib: float
    transformers_available: bool
    faiss_available: bool


def detect_capabilities() -> RuntimeCapabilities:
    """Inspect optional packages without requiring a GPU-enabled environment."""

    cuda_available = False
    gpu_name: str | None = None
    vram_gib = 0.0
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device)
            vram_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    except ImportError:
        pass

    try:
        import transformers  # noqa: F401

        transformers_available = True
    except ImportError:
        transformers_available = False
    try:
        import faiss  # noqa: F401

        faiss_available = True
    except ImportError:
        faiss_available = False
    return RuntimeCapabilities(
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        vram_gib=vram_gib,
        transformers_available=transformers_available,
        faiss_available=faiss_available,
    )


def resolve_profile(requested: str = "auto", capabilities: RuntimeCapabilities | None = None) -> str:
    """Return the safest profile that can run on the current host."""

    capabilities = capabilities or detect_capabilities()
    if requested not in {"auto", "cpu", "balanced_gpu", "paper_gpu"}:
        raise ValueError(f"Unknown profile: {requested}")
    if requested == "cpu":
        return "cpu"
    if requested == "paper_gpu":
        if capabilities.cuda_available and capabilities.vram_gib >= 14.0 and capabilities.transformers_available:
            return "paper_gpu"
        return "balanced_gpu" if capabilities.cuda_available and capabilities.transformers_available else "cpu"
    if requested == "balanced_gpu":
        return "balanced_gpu" if capabilities.cuda_available and capabilities.transformers_available else "cpu"
    if capabilities.cuda_available and capabilities.vram_gib >= 14.0 and capabilities.transformers_available:
        return "paper_gpu"
    if capabilities.cuda_available and capabilities.transformers_available:
        return "balanced_gpu"
    return "cpu"
