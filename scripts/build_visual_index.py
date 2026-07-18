"""Encode a frame manifest into resumable FAISS-ready vector JSONL artifacts.

``InMemoryVectorStore`` uses FAISS automatically when it is installed; the
portable artifact itself is a normalized vector/FrameRecord JSONL pair so it
can be resumed from Google Drive on a fresh Colab runtime without relying on a
version-specific FAISS binary format.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from _cli_common import (
    configure_model_cache,
    default_path_from_env,
    emit_json,
    load_models,
    resolve_runtime_settings,
)

from hcm_ai.artifacts import ArtifactStore
from hcm_ai.contracts import FrameRecord
from hcm_ai.embeddings import HashEmbeddingProvider, LazyTransformersEmbeddingProvider
from hcm_ai.indexing import build_visual_index
from hcm_ai.runtime import detect_capabilities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="FrameRecord JSONL manifest")
    parser.add_argument(
        "--profile",
        default=None,
        choices=["auto", "cpu", "balanced_gpu", "paper_gpu"],
    )
    parser.add_argument(
        "--encoder",
        action="append",
        help="Encoder catalog name; repeat for multiple indexes (defaults to profile YAML)",
    )
    parser.add_argument("--provider", choices=["auto", "transformers", "hash"], default="auto")
    parser.add_argument("--device", help="Torch device override, e.g. cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hash-dimension", type=int, default=64)
    parser.add_argument(
        "--allow-hash-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When automatic SigLIP loading fails, preserve a no-GPU smoke path with hash embeddings",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path(default_path_from_env("ARTIFACT_ROOT", "artifacts")))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create a fresh content-addressed build; existing Drive artifacts remain untouched",
    )
    return parser


def _model_id(settings: Any, encoder: str) -> str:
    catalog = settings.models.catalog
    entry = catalog.get(encoder)
    if not isinstance(entry, dict) or not isinstance(entry.get("model_id"), str):
        raise ValueError(f"unknown visual encoder {encoder!r}; add it to configs/models.yaml")
    return entry["model_id"]


def _initial_provider_kind(requested: str) -> str:
    if requested != "auto":
        return requested
    return "transformers" if detect_capabilities().transformers_available else "hash"


def _provider(
    kind: str,
    *,
    encoder: str,
    model_id: str,
    device: str | None,
    batch_size: int,
    hash_dimension: int,
) -> tuple[Any, str, str | None]:
    if kind == "hash":
        namespace = f"{encoder}:{model_id}"
        return HashEmbeddingProvider(dimension=hash_dimension, namespace=namespace), "hash", namespace
    if kind == "transformers":
        return (
            LazyTransformersEmbeddingProvider(model_id=model_id, device=device, batch_size=batch_size),
            "transformers",
            None,
        )
    raise ValueError(f"unsupported provider kind {kind!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.hash_dimension <= 0:
        raise ValueError("--batch-size and --hash-dimension must be positive")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    profile, settings = resolve_runtime_settings(args.profile)
    configure_model_cache(settings.paths.model_cache)
    frames = load_models(args.manifest, FrameRecord)
    encoders = args.encoder or list(settings.models.active_visual_encoders)
    if not encoders:
        raise ValueError("the selected profile has no active visual encoders")

    artifact_store = ArtifactStore(args.artifact_root)
    indexes: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    initial_kind = _initial_provider_kind(args.provider)
    device = args.device or ("cuda" if profile != "cpu" else None)
    force_nonce = uuid4().hex if args.force else None

    for encoder in dict.fromkeys(encoders):
        model_id = _model_id(settings, encoder)
        provider_kind = initial_kind
        provider, actual_kind, namespace = _provider(
            provider_kind,
            encoder=encoder,
            model_id=model_id,
            device=device,
            batch_size=args.batch_size,
            hash_dimension=args.hash_dimension,
        )
        build_config = {
            "profile": profile,
            "encoder": encoder,
            "model_id": model_id,
            "provider": actual_kind,
            "namespace": namespace,
            "force_nonce": force_nonce,
        }

        try:
            _, result = build_visual_index(
                frames,
                provider,
                batch_size=args.batch_size,
                artifacts=artifact_store,
                config=build_config,
            )
        except Exception as error:
            # The required baseline is SigLIP.  On a new/no-GPU Colab runtime,
            # automatic model failure degrades to a deterministic smoke index.
            # Optional BEiT-3 is skipped rather than poisoning the whole run.
            if (
                encoder == "siglip"
                and args.provider == "auto"
                and actual_kind == "transformers"
                and args.allow_hash_fallback
            ):
                provider, actual_kind, namespace = _provider(
                    "hash",
                    encoder=encoder,
                    model_id=model_id,
                    device=None,
                    batch_size=args.batch_size,
                    hash_dimension=args.hash_dimension,
                )
                build_config.update({"provider": actual_kind, "namespace": namespace})
                _, result = build_visual_index(
                    frames,
                    provider,
                    batch_size=args.batch_size,
                    artifacts=artifact_store,
                    config=build_config,
                )
            else:
                skipped[encoder] = f"{type(error).__name__}: {error}"
                continue

        indexes[encoder] = {
            "fingerprint": result.fingerprint,
            "artifact": result.artifact,
            "reused": result.reused,
            "provider": actual_kind,
            "model_id": model_id,
        }

    emit_json(
        {
            "profile": profile,
            "frame_count": len(frames),
            "indexes": indexes,
            "skipped": skipped,
        }
    )
    return 0 if indexes else 2


if __name__ == "__main__":  # pragma: no cover - exercised by notebook/CLI use
    raise SystemExit(main())
