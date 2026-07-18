"""Create a resumable renderable-frame manifest for a Drive-mounted corpus.

Supplied AIC keyframes or legacy frame metadata are intentionally preferred.
When only source videos are available, the same command can make a conservative
fixed-interval keyframe fallback through FFmpeg.  It does not overwrite
existing extracted images unless ``--overwrite-keyframes`` is explicit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _cli_common import default_path_from_env, emit_json, source_signature

from hcm_ai.artifacts import ArtifactStore, fingerprint_inputs
from hcm_ai.contracts import FrameRecord
from hcm_ai.ingestion import build_keyframe_manifest, frame_records_from_keyframes
from hcm_ai.ingestion.aic2025 import load_frame_records
from hcm_ai.preprocessing import extract_interval_keyframes


_VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--keyframes-root", type=Path, help="Existing AIC-style keyframe tree")
    source.add_argument("--frame-metadata", type=Path, help="Existing JSON or JSONL FrameRecord metadata")
    source.add_argument("--videos-root", type=Path, help="Raw videos; use only when keyframes are unavailable")
    parser.add_argument(
        "--generated-keyframes-root",
        type=Path,
        help="Destination for raw-video fallback keyframes (defaults under DATA_ROOT)",
    )
    parser.add_argument("--default-fps", type=float, default=25.0)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument(
        "--overwrite-keyframes",
        action="store_true",
        help="Explicitly allow FFmpeg to replace generated fallback keyframes",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path(default_path_from_env("ARTIFACT_ROOT", "artifacts")))
    parser.add_argument("--artifact-name", default="frames", help="Single artifact JSONL basename")
    parser.add_argument("--force", action="store_true", help="Replace an existing completed manifest artifact")
    return parser


def _video_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES),
        key=lambda path: path.as_posix(),
    )


def _frames_from_videos(args: argparse.Namespace) -> tuple[list[FrameRecord], list[Path], Path]:
    videos = _video_files(args.videos_root)
    if not videos:
        raise FileNotFoundError(f"no supported videos found under {args.videos_root}")
    generated_root = args.generated_keyframes_root or (
        Path(default_path_from_env("DATA_ROOT", "data")) / "processed" / "keyframes"
    )
    extracted = []
    for video in videos:
        video_id = video.stem
        extracted.extend(
            extract_interval_keyframes(
                video,
                generated_root / video_id,
                video_id=video_id,
                interval_seconds=args.interval_seconds,
                ffmpeg_bin=args.ffmpeg_bin,
                overwrite=args.overwrite_keyframes,
            )
        )
    return (
        frame_records_from_keyframes(
            extracted,
            metadata={
                "manifest_source": "ffmpeg_fixed_interval",
                "interval_seconds": args.interval_seconds,
            },
        ),
        videos,
        generated_root,
    )


def _load_frames(args: argparse.Namespace) -> tuple[list[FrameRecord], str, list[Path], str]:
    if args.frame_metadata is not None:
        source = args.frame_metadata
        return load_frame_records(source), "legacy_frame_metadata", [source], str(source)
    if args.keyframes_root is not None:
        source = args.keyframes_root
        frames = build_keyframe_manifest(source, default_fps=args.default_fps)
        images = [Path(frame.image_path) for frame in frames]
        return frames, "supplied_keyframes", images, str(source)

    frames, videos, generated_root = _frames_from_videos(args)
    return frames, "ffmpeg_fixed_interval", videos, str(generated_root)


def _validate_unique_frames(frames: Sequence[FrameRecord]) -> None:
    frame_ids = [frame.frame_id for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("frame manifest contains duplicate frame_id values")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.default_fps <= 0:
        raise ValueError("--default-fps must be positive")
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")

    frames, source_type, source_paths, source_label = _load_frames(args)
    _validate_unique_frames(frames)
    artifact_fingerprint = fingerprint_inputs(
        {
            "stage": "frame_manifest",
            "source_type": source_type,
            "source_state": source_signature(source_paths),
            "default_fps": args.default_fps,
            "interval_seconds": args.interval_seconds if args.videos_root else None,
            "frames": [frame.model_dump(mode="json") for frame in frames],
        }
    )
    store = ArtifactStore(args.artifact_root)
    resumed = store.is_complete("manifests/aic2025_frames", artifact_fingerprint, args.artifact_name)
    reference = store.write_jsonl(
        "manifests/aic2025_frames",
        artifact_fingerprint,
        frames,
        name=args.artifact_name,
        overwrite=args.force,
        metadata={
            "source_type": source_type,
            "source": source_label,
            "default_fps": args.default_fps,
            "frame_contract": "hcm_ai.contracts.FrameRecord",
        },
    )
    emit_json(
        {
            "artifact": reference,
            "frame_count": len(frames),
            "resumed": resumed and not args.force,
            "source_type": source_type,
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by notebook/CLI use
    raise SystemExit(main())

