"""Print a secret-safe view of dotenv/config paths for Colab CLI runs."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

if __package__:
    from ._cli_common import dataset_path_from_env, emit_json, resolve_runtime_settings
else:
    from _cli_common import dataset_path_from_env, emit_json, resolve_runtime_settings

from hcm_ai.environment import resolve_env_file


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    profile, settings = resolve_runtime_settings()
    dataset_path = Path(dataset_path_from_env(settings.paths.data_path))
    state_path = Path(settings.paths.artifact_root) / "pipeline_state.json"
    emit_json(
        {
            "env_file": resolve_env_file(),
            "profile": profile,
            "paths": settings.paths,
            "dataset_path": dataset_path,
            "dataset_exists": dataset_path.exists(),
            "pipeline_state": state_path,
            "pipeline_state_exists": state_path.is_file(),
            "gemini_key_available": bool(os.environ.get(settings.gemini.api_key_env)),
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by Colab/CLI use
    raise SystemExit(main())
