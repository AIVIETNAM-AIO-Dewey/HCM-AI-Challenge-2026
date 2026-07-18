"""Load local environment variables without overriding runtime credentials."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


ENV_FILE_VARIABLE = "HCM_AI_ENV_FILE"


def _repository_env_file() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def resolve_env_file(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve the dotenv file selected by the caller, environment, or checkout.

    Explicit paths fail loudly so a misspelled Colab/Drive path does not silently
    fall back to defaults. When no path is configured, absence of ``.env`` is
    valid because production and Colab Secrets may provide every value directly.
    """

    environment = os.environ if environ is None else environ
    configured = path if path is not None else environment.get(ENV_FILE_VARIABLE)
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"configured environment file does not exist: {candidate}")
        return candidate

    repository_candidate = _repository_env_file()
    if repository_candidate.is_file():
        return repository_candidate

    discovered = find_dotenv(filename=".env", usecwd=True)
    return Path(discovered).resolve() if discovered else None


def load_environment(
    path: str | Path | None = None,
    *,
    override: bool = False,
) -> Path | None:
    """Load project settings from ``.env`` and return the loaded file path.

    ``override`` defaults to ``False`` deliberately: process environment values
    and Colab Secrets have higher precedence than a checked-out local file.
    """

    env_file = resolve_env_file(path)
    if env_file is None:
        return None
    load_dotenv(dotenv_path=env_file, override=override, encoding="utf-8")
    return env_file


__all__ = ["ENV_FILE_VARIABLE", "load_environment", "resolve_env_file"]
