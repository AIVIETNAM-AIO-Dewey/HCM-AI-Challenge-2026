from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hcm_ai.contracts import QueryRecord, TaskType
from scripts import build_pipeline
from scripts.build_pipeline import _json_status
from scripts.preprocess_videos import _resolve_source, build_parser as build_preprocess_parser
from scripts.run_retrieval import (
    _default_output_path,
    _resolve_artifact_root,
    _visual_indexes_from_state,
)


def test_preprocess_can_discover_data_path_without_source_flags(tmp_path: Path) -> None:
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir()
    (keyframes / "000001.jpg").write_bytes(b"image")
    args = build_preprocess_parser().parse_args(["--data-path", str(tmp_path)])

    _resolve_source(args)

    assert args.keyframes_root == keyframes
    assert args.frame_metadata is None
    assert args.videos_root is None


def test_build_pipeline_writes_reusable_state(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest = tmp_path / "frames.jsonl"
    statuses = iter(
        (
            {"artifact": {"path": str(manifest)}, "frame_count": 2},
            {
                "profile": "cpu",
                "indexes": {"siglip": {"fingerprint": "a" * 64}},
            },
        )
    )
    monkeypatch.setattr(build_pipeline, "_run_script", lambda *_: next(statuses))

    result = build_pipeline.main(
        [
            "--data-path",
            str(tmp_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--provider",
            "hash",
        ]
    )

    assert result == 0
    state_path = tmp_path / "artifacts" / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["manifest"] == str(manifest)
    assert state["visual_indexes"] == {"siglip": "a" * 64}
    assert json.loads(capsys.readouterr().out)["state_path"] == str(state_path)


def test_retrieval_state_and_default_output_use_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    query = QueryRecord(query_id="KIS 01/02", text="a person", task=TaskType.KIS)

    specifications = _visual_indexes_from_state({"visual_indexes": {"siglip": "b" * 64}})
    output = _default_output_path(query)

    assert specifications == [f"siglip={'b' * 64}"]
    assert output == tmp_path / "KIS_01_02_kis.jsonl"


def test_json_status_uses_last_machine_readable_line() -> None:
    status = _json_status('download progress\n{"ok": true}\n', ["python", "stage.py"])

    assert status == {"ok": True}


def test_retrieval_uses_artifact_root_from_state_when_cli_omits_it(tmp_path: Path) -> None:
    state_root = tmp_path / "state-artifacts"
    fallback = tmp_path / "environment-artifacts"

    assert _resolve_artifact_root(None, {"artifact_root": str(state_root)}, fallback) == state_root
    assert _resolve_artifact_root(tmp_path / "explicit", {}, fallback) == tmp_path / "explicit"


def _run_cli(repository: Path, script: str, *arguments: str, env: dict[str, str]) -> dict:
    completed = subprocess.run(
        [sys.executable, str(repository / "scripts" / script), *arguments],
        cwd=repository,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_file_based_pipeline_smoke_and_resume(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    dataset = tmp_path / "dataset"
    keyframe = dataset / "keyframes" / "V1" / "000025.jpg"
    keyframe.parent.mkdir(parents=True)
    keyframe.write_bytes(b"hash-provider-does-not-decode-images")
    artifacts = tmp_path / "artifacts"
    outputs = tmp_path / "outputs"
    child_env = os.environ.copy()
    child_env.pop("HCM_AI_ENV_FILE", None)
    child_env.update(
        {
            "DATA_PATH": str(dataset),
            "DATA_ROOT": str(tmp_path / "work-data"),
            "ARTIFACT_ROOT": str(artifacts),
            "OUTPUT_ROOT": str(outputs),
            "MODEL_CACHE": str(tmp_path / "models"),
            "HCM_AI_PROFILE": "cpu",
        }
    )

    build_arguments = ("--provider", "hash", "--profile", "cpu", "--hash-dimension", "8")
    first = _run_cli(repository, "build_pipeline.py", *build_arguments, env=child_env)
    second = _run_cli(repository, "build_pipeline.py", *build_arguments, env=child_env)

    assert Path(first["state_path"]).is_file()
    assert second["preprocess"]["resumed"] is True
    assert second["visual"]["indexes"]["siglip"]["reused"] is True

    search = _run_cli(
        repository,
        "run_retrieval.py",
        "--query-id",
        "smoke",
        "--query",
        "a person",
        "--task",
        "KIS",
        "--profile",
        "cpu",
        "--reranker",
        "none",
        "--top-k",
        "1",
        env=child_env,
    )
    result_path = Path(search["jsonl"])
    assert result_path == outputs / "smoke_kis.jsonl"
    assert result_path.is_file()

    validation = _run_cli(
        repository,
        "validate_submission.py",
        "--input",
        str(result_path),
        "--task",
        "KIS",
        env=child_env,
    )
    assert validation["valid"] is True
