"""Small shared helpers for the repository's Colab-oriented CLI entry points.

The production retrieval logic deliberately lives in :mod:`hcm_ai`.  This file
only handles command-line concerns: finding the editable ``src`` checkout,
serialising one-line machine-readable status responses, and restoring the
simple JSONL artifacts produced by the command line tools.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def ensure_package_importable() -> None:
    """Use the checkout's ``src`` tree when a script runs without installation."""

    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


ensure_package_importable()


def ensure_environment_loaded() -> None:
    """Load the checkout's ``.env`` before CLI defaults inspect ``os.environ``."""

    from hcm_ai.environment import load_environment

    load_environment()


ensure_environment_loaded()


T = TypeVar("T")


def json_ready(value: Any) -> Any:
    """Convert a public Pydantic value and paths into a JSON-safe structure."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def emit_json(value: Mapping[str, Any]) -> None:
    """Print exactly one JSON object so notebooks can consume CLI output."""

    print(json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read non-empty JSONL object rows with a useful location on failure."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {source}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {source}:{line_number}")
            yield row


def load_models(path: str | Path, model_type: type[T]) -> list[T]:
    """Load JSONL rows through a Pydantic model type."""

    validator = getattr(model_type, "model_validate", None)
    if validator is None:
        raise TypeError(f"{model_type!r} does not expose model_validate")
    return [validator(row) for row in iter_jsonl(path)]


def load_json_or_jsonl_models(path: str | Path, model_type: type[T]) -> list[T]:
    """Load a list/``records`` JSON payload or JSONL through a Pydantic model."""

    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return load_models(source, model_type)
    if source.suffix.lower() != ".json":
        raise ValueError(f"expected JSON or JSONL records, got {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("records", payload.get("items", []))
    if not isinstance(payload, list):
        raise ValueError(f"JSON record file {source} must contain a list or records field")
    validator = getattr(model_type, "model_validate", None)
    if validator is None:
        raise TypeError(f"{model_type!r} does not expose model_validate")
    return [validator(row) for row in payload]


def batches(values: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """Yield bounded batches without pulling in a data-loader dependency."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def default_path_from_env(name: str, fallback: str) -> str:
    """Use a Drive-friendly environment override when the caller supplied one."""

    return os.environ.get(name) or fallback


def dataset_path_from_env(fallback: str = "data") -> str:
    """Return the benchmark input root without relying on shell expansion."""

    return os.environ.get("DATA_PATH") or os.environ.get("AIC2025_ROOT") or fallback


def configure_model_cache(path: str | Path | None) -> None:
    """Point Hugging Face's cache at Drive without storing credentials there."""

    if path is None:
        return
    cache = Path(path)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "huggingface" / "hub"))


def resolve_runtime_settings(requested_profile: str | None = None) -> tuple[str, Any]:
    """Resolve a hardware-safe profile before loading its YAML settings."""

    from hcm_ai.config import load_settings
    from hcm_ai.runtime import resolve_profile

    requested = requested_profile or os.environ.get("HCM_AI_PROFILE") or "auto"
    resolved = resolve_profile(requested)
    return resolved, load_settings(profile=resolved)


def source_signature(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Cheap, deterministic file state used in batch artifact fingerprints.

    Hashing every Drive video on every notebook restart is prohibitively slow.
    The signature intentionally tracks relative filename, size, and mtime so a
    changed or replaced source triggers a new artifact while ordinary resume is
    fast.  The resulting manifest records this provenance explicitly.
    """

    signature: list[dict[str, Any]] = []
    for item in sorted((Path(path) for path in paths), key=lambda path: path.as_posix()):
        stat = item.stat()
        signature.append(
            {
                "path": str(item),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return signature
