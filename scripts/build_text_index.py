"""Create resumable OCR/ASR records and their Drive-backed BM25 corpora.

The command accepts already-extracted JSON(L) records, which is the preferred
path for a shared AIC2025 corpus.  ``--run-ocr`` and ``--asr-audio-root`` are
deliberately opt-in local fallbacks; optional OCR/ASR packages are not imported
unless those flags are used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from _cli_common import (
    batches,
    configure_model_cache,
    default_path_from_env,
    emit_json,
    load_json_or_jsonl_models,
    load_models,
    resolve_runtime_settings,
    source_signature,
)

from hcm_ai.artifacts import ArtifactRef, ArtifactStore, fingerprint_inputs
from hcm_ai.asr import FasterWhisperProvider
from hcm_ai.contracts import AsrRecord, FrameRecord, OcrRecord
from hcm_ai.indexing import build_text_index
from hcm_ai.ingestion import asr_records_from_segments, ocr_records_from_extractions, text_rows_for_asr, text_rows_for_ocr
from hcm_ai.ocr import PaddleOcrProvider


R = TypeVar("R", OcrRecord, AsrRecord)
_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="FrameRecord JSONL manifest")
    parser.add_argument(
        "--profile",
        default=None,
        choices=["auto", "cpu", "balanced_gpu", "paper_gpu"],
    )
    parser.add_argument(
        "--modality",
        action="append",
        choices=["ocr", "asr"],
        help="Build only this modality; repeat for both (default: infer supplied sources)",
    )
    parser.add_argument("--ocr-records", type=Path, help="Pre-extracted OcrRecord JSON/JSONL")
    parser.add_argument("--run-ocr", action="store_true", help="Run PaddleOCR over manifest images")
    parser.add_argument("--ocr-language", default="vi")
    parser.add_argument("--ocr-batch-size", type=int, default=16)
    parser.add_argument("--asr-records", type=Path, help="Pre-extracted AsrRecord JSON/JSONL")
    parser.add_argument("--asr-audio-root", type=Path, help="Audio files named after video_id for faster-whisper")
    parser.add_argument("--asr-model", help="faster-whisper model override")
    parser.add_argument("--asr-device", help="faster-whisper device override")
    parser.add_argument("--asr-compute-type", default="int8")
    parser.add_argument("--artifact-root", type=Path, default=Path(default_path_from_env("ARTIFACT_ROOT", "artifacts")))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create fresh content-addressed record/index artifacts without deleting completed ones",
    )
    return parser


def _record_fingerprint(
    *,
    modality: str,
    frames: Sequence[FrameRecord],
    source: str,
    source_state: list[dict[str, Any]],
    options: dict[str, Any],
    force_nonce: str | None,
) -> str:
    return fingerprint_inputs(
        {
            "stage": f"{modality}_records",
            "frames": [frame.model_dump(mode="json") for frame in frames],
            "source": source,
            "source_state": source_state,
            "options": options,
            "force_nonce": force_nonce,
        }
    )


def _materialize_records(
    *,
    artifacts: ArtifactStore,
    modality: str,
    artifact_fingerprint: str,
    record_type: type[R],
    create: Callable[[], list[R]],
    metadata: dict[str, Any],
) -> tuple[list[R], ArtifactRef, bool]:
    kind = f"records/{modality}"
    if artifacts.is_complete(kind, artifact_fingerprint, "records"):
        reference = artifacts.get_ref(kind, artifact_fingerprint, "records")
        return load_models(reference.path, record_type), reference, True
    records = create()
    reference = artifacts.write_jsonl(
        kind,
        artifact_fingerprint,
        records,
        name="records",
        metadata=metadata,
    )
    return records, reference, False


def _run_ocr(frames: Sequence[FrameRecord], *, language: str, batch_size: int) -> list[OcrRecord]:
    provider = PaddleOcrProvider(language=language)
    output: list[OcrRecord] = []
    for frame_batch in batches(frames, batch_size):
        extractions = provider.extract([frame.image_path for frame in frame_batch])
        output.extend(ocr_records_from_extractions(frame_batch, extractions, source="paddleocr"))
    return output


def _audio_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES),
        key=lambda path: path.as_posix(),
    )


def _run_asr(
    frames: Sequence[FrameRecord],
    *,
    audio_root: Path,
    model_size: str,
    device: str,
    compute_type: str,
) -> list[AsrRecord]:
    audio_by_stem: dict[str, Path] = {}
    for audio in _audio_files(audio_root):
        # A duplicate basename is ambiguous; preserve the first deterministic
        # path and ask users to pass a non-ambiguous corpus tree if needed.
        audio_by_stem.setdefault(audio.stem, audio)
    frames_by_video: dict[str, list[FrameRecord]] = defaultdict(list)
    for frame in frames:
        frames_by_video[frame.video_id].append(frame)
    missing = sorted(video_id for video_id in frames_by_video if video_id not in audio_by_stem)
    if len(missing) == len(frames_by_video):
        raise FileNotFoundError(f"no audio files in {audio_root} match manifest video_id values")

    provider = FasterWhisperProvider(model_size=model_size, device=device, compute_type=compute_type)
    output: list[AsrRecord] = []
    for video_id in sorted(frames_by_video):
        audio = audio_by_stem.get(video_id)
        if audio is None:
            continue
        segments = provider.transcribe(audio)
        output.extend(
            asr_records_from_segments(
                video_id=video_id,
                segments=segments,
                frames=frames_by_video[video_id],
                source="faster_whisper",
            )
        )
    return output


def _index_payload(result: Any) -> dict[str, Any]:
    return {
        "fingerprint": result.fingerprint,
        "artifact": result.artifact,
        "reused": result.reused,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ocr_batch_size <= 0:
        raise ValueError("--ocr-batch-size must be positive")
    if args.ocr_records is not None and args.run_ocr:
        raise ValueError("choose either --ocr-records or --run-ocr, not both")
    if args.asr_records is not None and args.asr_audio_root is not None:
        raise ValueError("choose either --asr-records or --asr-audio-root, not both")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    profile, settings = resolve_runtime_settings(args.profile)
    configure_model_cache(settings.paths.model_cache)
    frames = load_models(args.manifest, FrameRecord)
    frames_by_id = {frame.frame_id: frame for frame in frames}
    artifacts = ArtifactStore(args.artifact_root)
    force_nonce = uuid4().hex if args.force else None
    requested = set(args.modality or [])
    if not requested:
        if args.ocr_records is not None or args.run_ocr:
            requested.add("ocr")
        if args.asr_records is not None or args.asr_audio_root is not None:
            requested.add("asr")

    output: dict[str, Any] = {"profile": profile, "frame_count": len(frames), "modalities": {}, "skipped": {}}

    if "ocr" in requested:
        if args.ocr_records is not None:
            source = "provided_ocr_records"
            source_state = source_signature([args.ocr_records])
            create_ocr = lambda: load_json_or_jsonl_models(args.ocr_records, OcrRecord)
            options = {"record_path": str(args.ocr_records)}
        elif args.run_ocr:
            source = "paddleocr"
            source_state = source_signature([args.manifest])
            create_ocr = lambda: _run_ocr(frames, language=args.ocr_language, batch_size=args.ocr_batch_size)
            options = {"language": args.ocr_language, "batch_size": args.ocr_batch_size}
        else:
            output["skipped"]["ocr"] = "no OCR records supplied; pass --ocr-records or --run-ocr"
            create_ocr = None
            source = ""
            source_state = []
            options = {}

        if create_ocr is not None:
            record_fingerprint = _record_fingerprint(
                modality="ocr",
                frames=frames,
                source=source,
                source_state=source_state,
                options=options,
                force_nonce=force_nonce,
            )
            ocr_records, record_artifact, records_reused = _materialize_records(
                artifacts=artifacts,
                modality="ocr",
                artifact_fingerprint=record_fingerprint,
                record_type=OcrRecord,
                create=create_ocr,
                metadata={"source": source, "options": options, "frame_manifest": str(args.manifest)},
            )
            ocr_frames, ocr_texts = text_rows_for_ocr(ocr_records, frames_by_id)
            if ocr_texts:
                _, result = build_text_index(
                    ocr_frames,
                    ocr_texts,
                    modality="ocr",
                    artifacts=artifacts,
                    config={
                        "profile": profile,
                        "record_fingerprint": record_fingerprint,
                        "force_nonce": force_nonce,
                    },
                    k1=settings.indexes.ocr.k1,
                    b=settings.indexes.ocr.b,
                )
                output["modalities"]["ocr"] = {
                    "record_artifact": record_artifact,
                    "records_reused": records_reused,
                    "record_count": len(ocr_records),
                    "indexed_documents": len(ocr_texts),
                    "index": _index_payload(result),
                }
            else:
                output["skipped"]["ocr"] = "OCR source produced no non-empty text linked to manifest frames"

    if "asr" in requested:
        if args.asr_records is not None:
            source = "provided_asr_records"
            source_state = source_signature([args.asr_records])
            create_asr = lambda: load_json_or_jsonl_models(args.asr_records, AsrRecord)
            options = {"record_path": str(args.asr_records)}
        elif args.asr_audio_root is not None:
            audio_paths = _audio_files(args.asr_audio_root)
            source = "faster_whisper"
            source_state = source_signature(audio_paths)
            model_size = args.asr_model or settings.models.asr_model
            device = args.asr_device or ("cuda" if profile != "cpu" else "cpu")
            create_asr = lambda: _run_asr(
                frames,
                audio_root=args.asr_audio_root,
                model_size=model_size,
                device=device,
                compute_type=args.asr_compute_type,
            )
            options = {"model": model_size, "device": device, "compute_type": args.asr_compute_type}
        else:
            output["skipped"]["asr"] = "no ASR records supplied; pass --asr-records or --asr-audio-root"
            create_asr = None
            source = ""
            source_state = []
            options = {}

        if create_asr is not None:
            record_fingerprint = _record_fingerprint(
                modality="asr",
                frames=frames,
                source=source,
                source_state=source_state,
                options=options,
                force_nonce=force_nonce,
            )
            asr_records, record_artifact, records_reused = _materialize_records(
                artifacts=artifacts,
                modality="asr",
                artifact_fingerprint=record_fingerprint,
                record_type=AsrRecord,
                create=create_asr,
                metadata={"source": source, "options": options, "frame_manifest": str(args.manifest)},
            )
            asr_frames, asr_texts = text_rows_for_asr(asr_records, frames_by_id)
            if asr_texts:
                _, result = build_text_index(
                    asr_frames,
                    asr_texts,
                    modality="asr",
                    artifacts=artifacts,
                    config={
                        "profile": profile,
                        "record_fingerprint": record_fingerprint,
                        "force_nonce": force_nonce,
                    },
                    k1=settings.indexes.asr.k1,
                    b=settings.indexes.asr.b,
                )
                output["modalities"]["asr"] = {
                    "record_artifact": record_artifact,
                    "records_reused": records_reused,
                    "record_count": len(asr_records),
                    "indexed_documents": len(asr_texts),
                    "index": _index_payload(result),
                }
            else:
                output["skipped"]["asr"] = "ASR source produced no non-empty text linked to manifest frames"

    emit_json(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by notebook/CLI use
    raise SystemExit(main())
