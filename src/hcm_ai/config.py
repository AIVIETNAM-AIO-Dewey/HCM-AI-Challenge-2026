"""Configuration loading with profile selection and environment substitution."""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .environment import load_environment


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)


class PathSettings(_ConfigModel):
    data_path: str = "data"
    data_root: str = "data"
    artifact_root: str = "artifacts"
    output_root: str = "outputs"
    model_cache: str = "models"


class RuntimeSettings(_ConfigModel):
    device: str = "auto"
    seed: int = 2026
    max_workers: int = Field(default=4, ge=1)


class ModelSettings(_ConfigModel):
    catalog: dict[str, dict[str, Any]] = Field(default_factory=dict)
    active_visual_encoders: list[str] = Field(default_factory=lambda: ["siglip"])
    reranker: str | None = None
    ocr_engine: str = "paddleocr"
    asr_model: str = "small"


class VectorIndexSettings(_ConfigModel):
    backend: str = "faiss"
    metric: str = "inner_product"
    normalize_embeddings: bool = True
    index_type: str = "flat"


class TextIndexSettings(_ConfigModel):
    backend: str = "bm25"
    k1: float = 1.5
    b: float = 0.75
    fuzzy_rerank: bool = True


class IndexSettings(_ConfigModel):
    visual: VectorIndexSettings = Field(default_factory=VectorIndexSettings)
    ocr: TextIndexSettings = Field(default_factory=TextIndexSettings)
    asr: TextIndexSettings = Field(default_factory=TextIndexSettings)


class RetrievalSettings(_ConfigModel):
    visual_top_k: int = Field(default=100, ge=1)
    text_top_k: int = Field(default=100, ge=1)
    rerank_top_k: int = Field(default=32, ge=1)
    query_expansion_n: int = Field(default=4, ge=1)
    rrf_k: int = Field(default=60, ge=1)


class FusionSettings(_ConfigModel):
    normalization: str = "min_max"
    missing_score: float = 0.0


class RerankingSettings(_ConfigModel):
    enabled: bool = True
    candidate_top_k: int = Field(default=32, ge=1)
    fused_weight: float = Field(default=0.7, ge=0.0)
    reranker_weight: float = Field(default=0.3, ge=0.0)


class TemporalSettings(_ConfigModel):
    event_top_k: int = Field(default=20, ge=1)
    beam_width: int = Field(default=8, ge=1)
    alpha: float = Field(default=0.01, ge=0.0)


class GeminiSettings(_ConfigModel):
    api_key_env: str = "GOOGLE_API_KEY"
    model: str = "gemini-2.5-flash-lite"
    cache_enabled: bool = True
    max_retries: int = Field(default=2, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class Settings(_ConfigModel):
    """Resolved, validated application configuration."""

    profile: str = "balanced_gpu"
    paths: PathSettings = Field(default_factory=PathSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    indexes: IndexSettings = Field(default_factory=IndexSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    fusion: FusionSettings = Field(default_factory=FusionSettings)
    reranking: RerankingSettings = Field(default_factory=RerankingSettings)
    temporal: TemporalSettings = Field(default_factory=TemporalSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
_CONFIG_FILES = ("default.yaml", "models.yaml", "indexes.yaml")


def default_config_dir() -> Path:
    """Return the repository's versioned configuration directory."""

    return Path(__file__).resolve().parents[2] / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without importing optional pipeline dependencies."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - covered by package dependency
        raise RuntimeError("PyYAML is required to load HCM AI configuration") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration file {config_path} must contain a mapping")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive, non-mutating merge where ``override`` wins."""

    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _substitute_environment(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            # Match shell ``${NAME:-fallback}``: an empty value is also unset.
            lambda match: environ.get(match.group(1)) or match.group(2) or "",
            value,
        )
    if isinstance(value, Mapping):
        return {str(key): _substitute_environment(item, environ) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_environment(item, environ) for item in value]
    return value


def _read_config_layers(config_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    missing = [name for name in _CONFIG_FILES if not (config_dir / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing required configuration file(s) in {config_dir}: {joined}")
    return tuple(load_yaml(config_dir / name) for name in _CONFIG_FILES)  # type: ignore[return-value]


def resolve_profile(
    config_dir: str | Path | None = None,
    profile: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Merge versioned config layers and apply one named hardware profile."""

    if environ is None:
        # This is intentionally idempotent and preserves values already present
        # in the process environment.
        load_environment()
        environment: Mapping[str, str] = os.environ
    else:
        # An explicitly supplied mapping is a deterministic test/API boundary;
        # do not mix it with process or dotenv state.
        environment = environ
    directory = default_config_dir() if config_dir is None else Path(config_dir)
    default_data, models_data, indexes_data = _read_config_layers(directory)

    models_profiles = models_data.pop("profiles", {})
    indexes_profiles = indexes_data.pop("profiles", {})
    if not isinstance(models_profiles, Mapping) or not isinstance(indexes_profiles, Mapping):
        raise ValueError("profiles in models.yaml and indexes.yaml must be mappings")

    selected = profile or environment.get("HCM_AI_PROFILE") or default_data.get("profile", "balanced_gpu")
    available_profiles = set(models_profiles) | set(indexes_profiles)
    if selected not in available_profiles:
        choices = ", ".join(sorted(available_profiles)) or "none"
        raise ValueError(f"unknown profile {selected!r}; available profiles: {choices}")

    resolved = deep_merge(default_data, models_data)
    resolved = deep_merge(resolved, indexes_data)
    model_override = models_profiles.get(selected, {})
    index_override = indexes_profiles.get(selected, {})
    if not isinstance(model_override, Mapping) or not isinstance(index_override, Mapping):
        raise ValueError(f"profile {selected!r} must be a mapping")
    resolved = deep_merge(resolved, model_override)
    resolved = deep_merge(resolved, index_override)
    resolved["profile"] = selected
    resolved = _substitute_environment(resolved, environment)

    # Explicit environment variables are ergonomic in Colab and intentionally
    # take precedence over versioned defaults and profile files.
    path_overrides = {
        "DATA_PATH": "data_path",
        "DATA_ROOT": "data_root",
        "ARTIFACT_ROOT": "artifact_root",
        "OUTPUT_ROOT": "output_root",
        "MODEL_CACHE": "model_cache",
    }
    paths = dict(resolved.get("paths", {}))
    # AIC2025_ROOT is retained as a benchmark-specific compatibility alias;
    # the generic DATA_PATH is the preferred input dataset location.
    if environment.get("AIC2025_ROOT") and not environment.get("DATA_PATH"):
        paths["data_path"] = environment["AIC2025_ROOT"]
    for env_name, setting_name in path_overrides.items():
        if environment.get(env_name):
            paths[setting_name] = environment[env_name]
    resolved["paths"] = paths
    return resolved


def load_settings(
    config_dir: str | Path | None = None,
    profile: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load and validate the current settings used by application components."""

    return Settings.model_validate(resolve_profile(config_dir, profile, environ))


# ``load_config`` is retained as a concise public alias for notebook callers.
load_config = load_settings


__all__ = [
    "FusionSettings",
    "GeminiSettings",
    "IndexSettings",
    "ModelSettings",
    "PathSettings",
    "RetrievalSettings",
    "RerankingSettings",
    "RuntimeSettings",
    "Settings",
    "TemporalSettings",
    "deep_merge",
    "default_config_dir",
    "load_config",
    "load_settings",
    "load_yaml",
    "resolve_profile",
]
