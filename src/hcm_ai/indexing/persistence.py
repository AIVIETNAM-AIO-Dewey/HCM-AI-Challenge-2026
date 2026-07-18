"""Resumable Drive-friendly serialization for the local FAISS/BM25 adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hcm_ai.artifacts import ArtifactRef, ArtifactStore, fingerprint_inputs
from hcm_ai.contracts import FrameRecord
from hcm_ai.embeddings import EmbeddingProvider

from .bm25_store import BM25Store, TextDocument
from .in_memory import InMemoryVectorStore


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    """A built/reused local index plus its content-addressed artifact reference."""

    fingerprint: str
    artifact: ArtifactRef | None
    reused: bool


def _provider_identity(provider: EmbeddingProvider) -> dict[str, Any]:
    try:
        dimension = provider.dimension
    except Exception:
        # Lazy transformer providers intentionally do not know their projection
        # dimension until weights load.  The model ID still makes the artifact
        # identity deterministic before the expensive encode step.
        dimension = getattr(provider, "_dimension", None)
    return {
        "class": type(provider).__name__,
        "model_id": getattr(provider, "model_id", None),
        "namespace": getattr(provider, "namespace", None),
        "dimension": dimension,
    }


def visual_index_fingerprint(
    frames: Sequence[FrameRecord],
    provider: EmbeddingProvider,
    *,
    config: Mapping[str, Any] | None = None,
) -> str:
    """Fingerprint data/model/config inputs before encoding any images."""

    return fingerprint_inputs(
        [frame.model_dump(mode="json") for frame in frames],
        _provider_identity(provider),
        dict(config or {}),
    )


def save_vector_store(
    store: InMemoryVectorStore,
    artifacts: ArtifactStore,
    artifact_fingerprint: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    """Persist normalized vectors and renderable frame payloads as JSONL."""

    rows = [
        {"frame": frame.model_dump(mode="json"), "vector": list(vector)}
        for frame, vector in store.entries()
    ]
    return artifacts.write_jsonl(
        "indexes/visual",
        artifact_fingerprint,
        rows,
        name="vectors",
        metadata={"backend": "faiss_or_python_cosine", "dimension": store.dimension, **dict(metadata or {})},
    )


def load_vector_store(
    artifacts: ArtifactStore,
    artifact_fingerprint: str,
    *,
    prefer_faiss: bool = True,
) -> InMemoryVectorStore:
    """Reconstruct an index without re-encoding images after a Colab restart."""

    reference = artifacts.get_ref("indexes/visual", artifact_fingerprint, name="vectors")
    rows = list(artifacts.iter_jsonl(reference.path))
    frames = [FrameRecord.model_validate(row["frame"]) for row in rows]
    vectors = [row["vector"] for row in rows]
    store = InMemoryVectorStore(prefer_faiss=prefer_faiss)
    store.add(frames, vectors)
    return store


def build_visual_index(
    frames: Sequence[FrameRecord],
    provider: EmbeddingProvider,
    *,
    batch_size: int = 32,
    artifacts: ArtifactStore | None = None,
    config: Mapping[str, Any] | None = None,
    prefer_faiss: bool = True,
) -> tuple[InMemoryVectorStore, IndexBuildResult]:
    """Build or resume a visual index, encoding only when its fingerprint is new."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    frame_list = list(frames)
    artifact_fingerprint = visual_index_fingerprint(frame_list, provider, config=config)
    if artifacts is not None and artifacts.is_complete("indexes/visual", artifact_fingerprint, name="vectors"):
        store = load_vector_store(artifacts, artifact_fingerprint, prefer_faiss=prefer_faiss)
        return store, IndexBuildResult(
            fingerprint=artifact_fingerprint,
            artifact=artifacts.get_ref("indexes/visual", artifact_fingerprint, name="vectors"),
            reused=True,
        )

    store = InMemoryVectorStore(prefer_faiss=prefer_faiss)
    for start in range(0, len(frame_list), batch_size):
        batch = frame_list[start : start + batch_size]
        vectors = provider.encode_images([frame.image_path for frame in batch])
        if len(vectors) != len(batch):
            raise ValueError("embedding provider returned a different number of image vectors")
        store.add(batch, vectors)
        if artifacts is not None:
            artifacts.write_checkpoint(
                "visual_index_progress",
                artifact_fingerprint,
                {"completed_frames": start + len(batch), "total_frames": len(frame_list)},
            )
    reference = (
        save_vector_store(
            store,
            artifacts,
            artifact_fingerprint,
            metadata={"provider": _provider_identity(provider), "config": dict(config or {})},
        )
        if artifacts is not None
        else None
    )
    return store, IndexBuildResult(artifact_fingerprint, reference, reused=False)


def text_index_fingerprint(
    frames: Sequence[FrameRecord],
    texts: Sequence[str],
    *,
    modality: str,
    config: Mapping[str, Any] | None = None,
) -> str:
    if len(frames) != len(texts):
        raise ValueError("frames and texts must have the same length")
    return fingerprint_inputs(
        [frame.model_dump(mode="json") for frame in frames],
        list(texts),
        {"modality": modality, **dict(config or {})},
    )


def save_text_store(
    store: BM25Store,
    artifacts: ArtifactStore,
    artifact_fingerprint: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    """Persist lexical documents so BM25 can be rebuilt without OCR/ASR work."""

    rows = [
        {
            "frame": document.record.model_dump(mode="json"),
            "text": document.text,
            "document_id": document.document_id,
        }
        for document in store.documents()
    ]
    return artifacts.write_jsonl(
        f"indexes/{store.modality}",
        artifact_fingerprint,
        rows,
        name="documents",
        metadata={"backend": "bm25", "modality": store.modality, **dict(metadata or {})},
    )


def load_text_store(
    artifacts: ArtifactStore,
    artifact_fingerprint: str,
    *,
    modality: str,
    **kwargs: Any,
) -> BM25Store:
    """Reload an OCR or ASR BM25 corpus from its completed artifact."""

    reference = artifacts.get_ref(f"indexes/{modality}", artifact_fingerprint, name="documents")
    documents = [
        TextDocument(
            record=FrameRecord.model_validate(row["frame"]),
            text=str(row["text"]),
            document_id=row.get("document_id"),
        )
        for row in artifacts.iter_jsonl(reference.path)
    ]
    store = BM25Store(modality=modality, **kwargs)
    store.add_documents(documents)
    return store


def build_text_index(
    frames: Sequence[FrameRecord],
    texts: Sequence[str],
    *,
    modality: str,
    artifacts: ArtifactStore | None = None,
    config: Mapping[str, Any] | None = None,
    **store_kwargs: Any,
) -> tuple[BM25Store, IndexBuildResult]:
    """Build or resume an OCR/ASR BM25 corpus with artifact fingerprints."""

    frame_list = list(frames)
    text_list = list(texts)
    artifact_fingerprint = text_index_fingerprint(
        frame_list,
        text_list,
        modality=modality,
        config=config,
    )
    if artifacts is not None and artifacts.is_complete(
        f"indexes/{modality}", artifact_fingerprint, name="documents"
    ):
        store = load_text_store(artifacts, artifact_fingerprint, modality=modality, **store_kwargs)
        return store, IndexBuildResult(
            artifact_fingerprint,
            artifacts.get_ref(f"indexes/{modality}", artifact_fingerprint, name="documents"),
            reused=True,
        )
    store = BM25Store(modality=modality, **store_kwargs)
    store.add(frame_list, text_list)
    reference = (
        save_text_store(store, artifacts, artifact_fingerprint, metadata={"config": dict(config or {})})
        if artifacts is not None
        else None
    )
    return store, IndexBuildResult(artifact_fingerprint, reference, reused=False)


__all__ = [
    "IndexBuildResult",
    "build_text_index",
    "build_visual_index",
    "load_text_store",
    "load_vector_store",
    "save_text_store",
    "save_vector_store",
    "text_index_fingerprint",
    "visual_index_fingerprint",
]
