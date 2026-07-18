from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hcm_ai.config import load_settings
from hcm_ai.environment import ENV_FILE_VARIABLE, load_environment, resolve_env_file
from scripts._cli_common import (
    dataset_path_from_env,
    default_path_from_env,
    resolve_runtime_settings,
)


_CONFIG_ENV_NAMES = (
    "DATA_PATH",
    "AIC2025_ROOT",
    "DATA_ROOT",
    "ARTIFACT_ROOT",
    "OUTPUT_ROOT",
    "MODEL_CACHE",
    "HCM_AI_PROFILE",
)


def _clear_environment(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Temporarily clear names while ensuring pytest restores/removes them."""

    for name in names:
        monkeypatch.setenv(name, "__pytest_restore_marker__")
        monkeypatch.delenv(name)


def test_load_environment_preserves_runtime_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATA_ROOT=from-dotenv\nARTIFACT_ROOT=dotenv-artifacts\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", "from-runtime")
    _clear_environment(monkeypatch, "ARTIFACT_ROOT")

    loaded = load_environment(env_file)

    assert loaded == env_file.resolve()
    assert os.environ["DATA_ROOT"] == "from-runtime"
    assert os.environ["ARTIFACT_ROOT"] == "dotenv-artifacts"


def test_load_settings_uses_configured_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / "colab.env"
    env_file.write_text(
        "\n".join(
            (
                "HCM_AI_PROFILE=cpu",
                "DATA_PATH=/drive/aic2025",
                "DATA_ROOT=/drive/data",
                "ARTIFACT_ROOT=/drive/artifacts",
                "OUTPUT_ROOT=/drive/outputs",
                "MODEL_CACHE=/drive/models",
            )
        ),
        encoding="utf-8",
    )
    _clear_environment(monkeypatch, *_CONFIG_ENV_NAMES)
    monkeypatch.setenv(ENV_FILE_VARIABLE, str(env_file))

    settings = load_settings()

    assert settings.profile == "cpu"
    assert settings.paths.data_path == "/drive/aic2025"
    assert settings.paths.data_root == "/drive/data"
    assert settings.paths.artifact_root == "/drive/artifacts"
    assert settings.paths.output_root == "/drive/outputs"
    assert settings.paths.model_cache == "/drive/models"


def test_explicit_environment_mapping_does_not_mix_with_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HCM_AI_PROFILE=paper_gpu\nDATA_ROOT=wrong-source\n", encoding="utf-8")
    monkeypatch.setenv(ENV_FILE_VARIABLE, str(env_file))

    settings = load_settings(environ={"HCM_AI_PROFILE": "cpu", "DATA_ROOT": "mapping-data"})

    assert settings.profile == "cpu"
    assert settings.paths.data_root == "mapping-data"


def test_empty_environment_value_uses_yaml_fallback() -> None:
    settings = load_settings(
        environ={"HCM_AI_PROFILE": "cpu", "DATA_PATH": "", "DATA_ROOT": ""}
    )

    assert settings.paths.data_path == "data"
    assert settings.paths.data_root == "data"


def test_data_path_and_legacy_alias_precedence() -> None:
    generic = load_settings(environ={"HCM_AI_PROFILE": "cpu", "DATA_PATH": "/generic"})
    legacy = load_settings(environ={"HCM_AI_PROFILE": "cpu", "AIC2025_ROOT": "/legacy"})
    both = load_settings(
        environ={
            "HCM_AI_PROFILE": "cpu",
            "DATA_PATH": "/generic",
            "AIC2025_ROOT": "/legacy",
        }
    )

    assert generic.paths.data_path == "/generic"
    assert legacy.paths.data_path == "/legacy"
    assert both.paths.data_path == "/generic"


def test_dataset_path_helper_does_not_need_shell_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_PATH", "/generic")
    monkeypatch.setenv("AIC2025_ROOT", "/legacy")

    assert dataset_path_from_env() == "/generic"


@pytest.mark.parametrize(
    ("data_path", "legacy", "expected"),
    [
        ("", "/legacy", "/legacy"),
        ("/generic", "", "/generic"),
        ("", "", "fallback-data"),
    ],
)
def test_dataset_path_helper_fallbacks(
    data_path: str,
    legacy: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_PATH", data_path)
    monkeypatch.setenv("AIC2025_ROOT", legacy)

    assert dataset_path_from_env("fallback-data") == expected


def test_empty_cli_path_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFACT_ROOT", "")

    assert default_path_from_env("ARTIFACT_ROOT", "artifacts") == "artifacts"


def test_cli_runtime_profile_uses_environment_unless_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcm_ai.config as config_module
    import hcm_ai.runtime as runtime_module

    requested: list[str] = []

    def fake_resolve(profile: str) -> str:
        requested.append(profile)
        return profile

    monkeypatch.setattr(runtime_module, "resolve_profile", fake_resolve)
    monkeypatch.setattr(config_module, "load_settings", lambda profile: profile)
    monkeypatch.setenv("HCM_AI_PROFILE", "cpu")

    assert resolve_runtime_settings() == ("cpu", "cpu")
    assert resolve_runtime_settings("paper_gpu") == ("paper_gpu", "paper_gpu")
    assert requested == ["cpu", "paper_gpu"]


def test_missing_explicit_environment_file_fails_loudly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.env"

    with pytest.raises(FileNotFoundError, match="configured environment file does not exist"):
        resolve_env_file(missing)


def test_package_import_loads_selected_environment_file(tmp_path: Path) -> None:
    env_file = tmp_path / "import.env"
    env_file.write_text("DATA_ROOT=import-time-data\n", encoding="utf-8")
    repository_root = Path(__file__).resolve().parents[1]
    child_environment = os.environ.copy()
    child_environment[ENV_FILE_VARIABLE] = str(env_file)
    child_environment.pop("DATA_ROOT", None)
    child_environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repository_root / "src"), child_environment.get("PYTHONPATH")))
    )

    completed = subprocess.run(
        [sys.executable, "-c", "import os, hcm_ai; print(os.environ['DATA_ROOT'])"],
        cwd=tmp_path,
        env=child_environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "import-time-data"
