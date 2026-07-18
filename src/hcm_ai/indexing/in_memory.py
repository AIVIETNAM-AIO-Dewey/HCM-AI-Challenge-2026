"""A deterministic cosine vector store with an optional FAISS fast path."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, TypeAlias

try:  # Contracts are created by the core package; keep this module importable alone.
    from hcm_ai.contracts import FrameRecord, MomentResult
except ModuleNotFoundError:  # pragma: no cover - only used during partial installs.
    FrameRecord = Any  # type: ignore[misc,assignment]
    MomentResult = Any  # type: ignore[misc,assignment]


Vector: TypeAlias = Sequence[float]


@dataclass(frozen=True, slots=True)
class _FallbackMomentResult:
    """Small compatibility result used only before the contracts module exists."""

    video_id: str
    frame_id: str
    timestamp: float
    image_path: str
    shot_id: str | None
    score: float
    rank: int
    modality_scores: dict[str, float]


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _moment_from_frame(record: Any, score: float, rank: int, source: str) -> Any:
    values = {
        "video_id": str(_field(record, "video_id")),
        "frame_id": str(_field(record, "frame_id")),
        "timestamp": float(_field(record, "timestamp")),
        "image_path": str(_field(record, "image_path")),
        "shot_id": _field(record, "shot_id"),
        "score": float(score),
        "rank": rank,
        "modality_scores": {"visual": float(score)},
        "provenance": [source],
    }
    if MomentResult is Any:
        return _FallbackMomentResult(
            video_id=values["video_id"],
            frame_id=values["frame_id"],
            timestamp=values["timestamp"],
            image_path=values["image_path"],
            shot_id=values["shot_id"],
            score=values["score"],
            rank=rank,
            modality_scores=values["modality_scores"],
        )
    return MomentResult(**values)


def _normalize(vector: Vector) -> tuple[float, ...]:
    values: list[float] = []
    for raw_value in vector:
        if isinstance(raw_value, bool):
            raise ValueError("vector values must be numeric, not boolean")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("vector values must be numeric") from exc
        if not isfinite(value):
            raise ValueError("vector values must be finite")
        values.append(value)
    if not values:
        raise ValueError("vectors must not be empty")
    norm = sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("zero vectors cannot be cosine-normalized")
    return tuple(value / norm for value in values)


class InMemoryVectorStore:
    """Store normalized vectors and retrieve frames by cosine similarity.

    The pure-Python path is intentionally compact and deterministic for unit
    tests.  If both ``faiss`` and ``numpy`` are installed, a temporary
    ``IndexFlatIP`` is used at query time for faster large-index retrieval.
    """

    def __init__(self, *, prefer_faiss: bool = True) -> None:
        self._prefer_faiss = prefer_faiss
        self._records: list[Any] = []
        self._vectors: list[tuple[float, ...]] = []
        self._frame_ids: set[str] = set()
        self._dimension: int | None = None

    @property
    def dimension(self) -> int | None:
        """Return the embedding dimensionality, if any vectors were added."""

        return self._dimension

    def __len__(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        """Remove all vectors and associated frame records."""

        self._records.clear()
        self._vectors.clear()
        self._frame_ids.clear()
        self._dimension = None

    def entries(self) -> list[tuple[FrameRecord, tuple[float, ...]]]:
        """Return a deterministic copy for Drive-backed index persistence."""

        return list(zip(self._records, self._vectors, strict=True))

    def add(self, records: Iterable[FrameRecord], vectors: Iterable[Vector]) -> None:
        """Add frame records and same-length embedding vectors atomically."""

        record_list = list(records)
        vector_list = [_normalize(vector) for vector in vectors]
        if len(record_list) != len(vector_list):
            raise ValueError("records and vectors must have the same length")
        if not record_list:
            return

        dimensions = {len(vector) for vector in vector_list}
        if len(dimensions) != 1:
            raise ValueError("all vectors in one add operation must have one dimension")
        dimension = dimensions.pop()
        if self._dimension is not None and dimension != self._dimension:
            raise ValueError("vector dimension does not match the existing index")

        new_ids: list[str] = []
        for record in record_list:
            frame_id = _field(record, "frame_id")
            if not isinstance(frame_id, str) or not frame_id:
                raise ValueError("each record must expose a non-empty frame_id")
            new_ids.append(frame_id)
        has_existing_id = any(frame_id in self._frame_ids for frame_id in new_ids)
        if len(new_ids) != len(set(new_ids)) or has_existing_id:
            raise ValueError("frame_id values must be unique within a vector store")

        self._records.extend(record_list)
        self._vectors.extend(vector_list)
        self._frame_ids.update(new_ids)
        self._dimension = dimension

    def _search_with_faiss(
        self,
        query: tuple[float, ...],
        top_k: int,
    ) -> list[tuple[int, float]] | None:
        if not self._prefer_faiss or not self._vectors:
            return None
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy as np

            matrix = np.asarray(self._vectors, dtype="float32")
            index = faiss.IndexFlatIP(self._dimension or 0)
            index.add(matrix)
            scores, indices = index.search(
                np.asarray([query], dtype="float32"), min(top_k, len(self._vectors))
            )
        except Exception:  # Optional accelerator failure must not break the baseline.
            return None
        return [
            (int(index_id), float(score))
            for index_id, score in zip(indices[0], scores[0])
            if index_id >= 0
        ]

    def search(self, query: Vector, *, top_k: int = 10) -> list[MomentResult]:
        """Return the top ``top_k`` frames ranked by L2-normalized cosine score."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._vectors:
            return []
        normalized_query = _normalize(query)
        if len(normalized_query) != self._dimension:
            raise ValueError("query dimension does not match the existing index")

        scored = self._search_with_faiss(normalized_query, top_k)
        if scored is None:
            scored = [
                (index, sum(left * right for left, right in zip(normalized_query, vector)))
                for index, vector in enumerate(self._vectors)
            ]

        scored.sort(
            key=lambda item: (
                -item[1],
                str(_field(self._records[item[0]], "frame_id")),
                item[0],
            )
        )
        return [
            _moment_from_frame(self._records[index], score, rank, "visual:cosine")
            for rank, (index, score) in enumerate(scored[:top_k], start=1)
        ]
