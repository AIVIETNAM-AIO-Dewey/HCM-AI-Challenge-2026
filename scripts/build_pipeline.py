"""Build resumable retrieval artifacts from DATA_PATH and persist CLI state.

This command is the file-based Colab alternative to the ingest notebook. It
delegates each stage to the existing thin scripts, captures their JSON status,
and writes ``pipeline_state.json`` so search commands need no copied
fingerprints.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__:
    from ._cli_common import (
        REPOSITORY_ROOT,
        dataset_path_from_env,
        default_path_from_env,
        emit_json,
    )
else:
    from _cli_common import (
        REPOSITORY_ROOT,
        dataset_path_from_env,
        default_path_from_env,
        emit_json,
    )

from hcm_ai.artifacts import ArtifactStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(dataset_path_from_env()),
        help="Mounted dataset root (defaults to DATA_PATH or AIC2025_ROOT)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--keyframes-root", type=Path)
    source.add_argument("--frame-metadata", type=Path)
    source.add_argument("--videos-root", type=Path)
    parser.add_argument("--default-fps", type=float, default=25.0)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument(
        "--profile",
        choices=["auto", "cpu", "balanced_gpu", "paper_gpu"],
        default=None,
    )
    parser.add_argument("--encoder", action="append", help="Repeat for multiple visual indexes")
    parser.add_argument("--provider", choices=["auto", "transformers", "hash"], default="auto")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hash-dimension", type=int, default=64)
    parser.add_argument("--ocr-records", type=Path)
    parser.add_argument("--run-ocr", action="store_true")
    parser.add_argument("--asr-records", type=Path)
    parser.add_argument("--asr-audio-root", type=Path)
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(default_path_from_env("ARTIFACT_ROOT", "artifacts")),
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="Pipeline state path (default: ARTIFACT_ROOT/pipeline_state.json)",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _json_status(stdout: str, command: Sequence[str]) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"command did not emit a JSON status object: {' '.join(command)}")


def _run_script(name: str, arguments: Sequence[str | Path]) -> dict[str, Any]:
    command = [sys.executable, str(REPOSITORY_ROOT / "scripts" / name), *map(str, arguments)]
    print(f"[pipeline] running {name}...", file=sys.stderr, flush=True)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        # Inherit stderr so model downloads and batch progress remain visible
        # in Colab instead of accumulating in memory until the stage finishes.
        stderr=None,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{' '.join(command)} failed with exit code {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSee the stage stderr above for details."
        )
    status = _json_status(completed.stdout, command)
    print(f"[pipeline] completed {name}", file=sys.stderr, flush=True)
    return status


def _add_optional(arguments: list[str | Path], flag: str, value: str | Path | None) -> None:
    if value is not None:
        arguments.extend((flag, value))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.hash_dimension <= 0:
        raise ValueError("--batch-size and --hash-dimension must be positive")
    if args.ocr_records is not None and args.run_ocr:
        raise ValueError("choose either --ocr-records or --run-ocr")
    if args.asr_records is not None and args.asr_audio_root is not None:
        raise ValueError("choose either --asr-records or --asr-audio-root")

    preprocess_arguments: list[str | Path] = [
        "--artifact-root",
        args.artifact_root,
        "--data-path",
        args.data_path,
        "--default-fps",
        str(args.default_fps),
        "--interval-seconds",
        str(args.interval_seconds),
    ]
    _add_optional(preprocess_arguments, "--keyframes-root", args.keyframes_root)
    _add_optional(preprocess_arguments, "--frame-metadata", args.frame_metadata)
    _add_optional(preprocess_arguments, "--videos-root", args.videos_root)
    if args.force:
        preprocess_arguments.append("--force")
    preprocess_status = _run_script("preprocess_videos.py", preprocess_arguments)
    manifest = Path(preprocess_status["artifact"]["path"])

    visual_arguments: list[str | Path] = [
        "--manifest",
        manifest,
        "--artifact-root",
        args.artifact_root,
        "--provider",
        args.provider,
        "--batch-size",
        str(args.batch_size),
        "--hash-dimension",
        str(args.hash_dimension),
    ]
    _add_optional(visual_arguments, "--profile", args.profile)
    _add_optional(visual_arguments, "--device", args.device)
    for encoder in args.encoder or ():
        visual_arguments.extend(("--encoder", encoder))
    if args.force:
        visual_arguments.append("--force")
    visual_status = _run_script("build_visual_index.py", visual_arguments)
    visual_indexes = {
        name: details["fingerprint"]
        for name, details in visual_status.get("indexes", {}).items()
    }
    if not visual_indexes:
        raise RuntimeError(f"no visual index was created: {visual_status}")

    state_path = args.state or args.artifact_root / "pipeline_state.json"
    previous_state = ArtifactStore.read_json(state_path) if state_path.is_file() else {}
    previous_is_compatible = previous_state.get("manifest") == str(manifest)
    text_requested = any((args.ocr_records, args.run_ocr, args.asr_records, args.asr_audio_root))
    skip_reason = (
        "disabled by --skip-text" if args.skip_text and text_requested else "not requested"
    )
    text_status: dict[str, Any] = {"modalities": {}, "skipped": {"text": skip_reason}}
    if text_requested and not args.skip_text:
        text_arguments: list[str | Path] = [
            "--manifest",
            manifest,
            "--artifact-root",
            args.artifact_root,
        ]
        _add_optional(text_arguments, "--profile", args.profile)
        _add_optional(text_arguments, "--ocr-records", args.ocr_records)
        _add_optional(text_arguments, "--asr-records", args.asr_records)
        _add_optional(text_arguments, "--asr-audio-root", args.asr_audio_root)
        if args.run_ocr:
            text_arguments.append("--run-ocr")
        if args.force:
            text_arguments.append("--force")
        text_status = _run_script("build_text_index.py", text_arguments)

    modalities = text_status.get("modalities", {})
    ocr_index = modalities.get("ocr", {}).get("index", {}).get("fingerprint")
    asr_index = modalities.get("asr", {}).get("index", {}).get("fingerprint")
    ocr_stage_ran = not args.skip_text and bool(args.ocr_records or args.run_ocr)
    asr_stage_ran = not args.skip_text and bool(args.asr_records or args.asr_audio_root)
    if previous_is_compatible and not ocr_stage_ran:
        ocr_index = previous_state.get("ocr_index")
    if previous_is_compatible and not asr_stage_ran:
        asr_index = previous_state.get("asr_index")
    state = {
        "schema_version": 1,
        "profile": visual_status.get("profile", args.profile or "auto"),
        "data_path": str(args.data_path),
        "source": preprocess_status.get("source"),
        "source_type": preprocess_status.get("source_type"),
        "artifact_root": str(args.artifact_root),
        "manifest": str(manifest),
        "visual_indexes": visual_indexes,
        "ocr_index": ocr_index,
        "asr_index": asr_index,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    ArtifactStore.write_json(state_path, state)
    emit_json(
        {
            "state_path": state_path,
            "state": state,
            "preprocess": preprocess_status,
            "visual": visual_status,
            "text": text_status,
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by Colab/CLI use
    raise SystemExit(main())
