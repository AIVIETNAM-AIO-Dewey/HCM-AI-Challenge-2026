"""Local BM25-style text retrieval with optional rank-bm25 and RapidFuzz."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import isfinite, log
import re
from typing import Any
import unicodedata

try:  # Contracts are created by the core package; keep this module importable alone.
    from hcm_ai.contracts import FrameRecord, MomentResult
except ModuleNotFoundError:  # pragma: no cover - only used during partial installs.
    FrameRecord = Any  # type: ignore[misc,assignment]
    MomentResult = Any  # type: ignore[misc,assignment]


_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TextDocument:
    """A lexical document mapped back to a renderable frame record."""

    record: FrameRecord
    text: str
    document_id: str | None = None


@dataclass(frozen=True, slots=True)
class _FallbackMomentResult:
    video_id: str
    frame_id: str
    timestamp: float
    image_path: str
    shot_id: str | None
    score: float
    rank: int
    modality_scores: dict[str, float]
    metadata: dict[str, Any]


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese/English Unicode text without a service dependency."""

    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return _TOKEN_PATTERN.findall(_normalize_text(text))


def _moment_from_document(
    document: TextDocument,
    *,
    score: float,
    rank: int,
    modality: str,
    bm25_score: float,
    fuzzy_score: float,
) -> Any:
    record = document.record
    metadata = {
        "document_id": document.document_id,
        "bm25_score": bm25_score,
        "fuzzy_score": fuzzy_score,
    }
    values = {
        "video_id": str(_field(record, "video_id")),
        "frame_id": str(_field(record, "frame_id")),
        "timestamp": float(_field(record, "timestamp")),
        "image_path": str(_field(record, "image_path")),
        "shot_id": _field(record, "shot_id"),
        "score": float(score),
        "rank": rank,
        "modality_scores": {modality: float(score)},
        "provenance": [f"{modality}:bm25"],
        "metadata": metadata,
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
            metadata=metadata,
        )
    return MomentResult(**values)


class BM25Store:
    """In-process lexical search for OCR or ASR documents.

    ``rank_bm25`` accelerates score calculation when present.  The default
    implementation is a small, deterministic Okapi BM25 scorer plus a fuzzy
    bonus, so no external service or optional package is required for tests.
    """

    def __init__(
        self,
        *,
        modality: str = "text",
        k1: float = 1.5,
        b: float = 0.75,
        fuzzy_weight: float = 0.25,
        fuzzy_threshold: float = 0.72,
        prefer_rank_bm25: bool = True,
    ) -> None:
        if not modality:
            raise ValueError("modality must be non-empty")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be within [0, 1]")
        if fuzzy_weight < 0 or not 0 <= fuzzy_threshold <= 1:
            raise ValueError("invalid fuzzy matching configuration")
        self.modality = modality
        self.k1 = k1
        self.b = b
        self.fuzzy_weight = fuzzy_weight
        self.fuzzy_threshold = fuzzy_threshold
        self._prefer_rank_bm25 = prefer_rank_bm25
        self._documents: list[TextDocument] = []
        self._tokens: list[list[str]] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._rank_bm25: Any | None = None

    def __len__(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        """Remove all text documents."""

        self._documents.clear()
        self._tokens.clear()
        self._term_frequencies.clear()
        self._document_frequencies.clear()
        self._rank_bm25 = None

    def documents(self) -> list[TextDocument]:
        """Return a copy of indexed documents for artifact persistence."""

        return list(self._documents)

    def _refresh_optional_engine(self) -> None:
        self._rank_bm25 = None
        if not self._prefer_rank_bm25 or not self._tokens:
            return
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]

            self._rank_bm25 = BM25Okapi(self._tokens, k1=self.k1, b=self.b)
        except Exception:
            self._rank_bm25 = None

    def add(self, records: Iterable[FrameRecord], texts: Iterable[str]) -> None:
        """Add aligned frame records and text documents."""

        record_list = list(records)
        text_list = list(texts)
        if len(record_list) != len(text_list):
            raise ValueError("records and texts must have the same length")
        offset = len(self._documents)
        documents: list[TextDocument] = []
        for index, (record, text) in enumerate(zip(record_list, text_list, strict=True)):
            if not isinstance(text, str):
                raise ValueError("texts must contain strings")
            frame_id = _field(record, "frame_id")
            if not isinstance(frame_id, str) or not frame_id:
                raise ValueError("each record must expose a non-empty frame_id")
            documents.append(TextDocument(record, text, f"{frame_id}:{offset + index}"))
        self.add_documents(documents)

    def add_documents(self, documents: Iterable[TextDocument]) -> None:
        """Add pre-assembled documents, retaining multiple ASR segments per frame."""

        pending = list(documents)
        for document in pending:
            if not isinstance(document.text, str):
                raise ValueError("document text must be a string")
            frame_id = _field(document.record, "frame_id")
            if not isinstance(frame_id, str) or not frame_id:
                raise ValueError("each document record must expose a non-empty frame_id")

        tokenized = [tokenize(document.text) for document in pending]
        self._documents.extend(pending)
        self._tokens.extend(tokenized)
        self._term_frequencies.extend(Counter(tokens) for tokens in tokenized)
        self._document_frequencies.update(
            token for tokens in tokenized for token in set(tokens)
        )
        self._refresh_optional_engine()

    def _manual_bm25_scores(self, query_tokens: Sequence[str]) -> list[float]:
        document_count = len(self._documents)
        if document_count == 0:
            return []
        average_length = sum(len(tokens) for tokens in self._tokens) / document_count
        if average_length == 0:
            return [0.0] * document_count

        scores: list[float] = []
        for tokens, term_frequencies in zip(self._tokens, self._term_frequencies, strict=True):
            document_length = len(tokens)
            score = 0.0
            for term in query_tokens:
                frequency = term_frequencies.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self._document_frequencies.get(term, 0)
                inverse_document_frequency = log(
                    1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / average_length
                )
                score += inverse_document_frequency * frequency * (self.k1 + 1.0) / denominator
            scores.append(score)
        return scores

    def _bm25_scores(self, query_tokens: Sequence[str]) -> list[float]:
        if self._rank_bm25 is not None:
            return [float(score) for score in self._rank_bm25.get_scores(list(query_tokens))]
        return self._manual_bm25_scores(query_tokens)

    @staticmethod
    def _ratio(left: str, right: str) -> float:
        try:
            from rapidfuzz.fuzz import ratio  # type: ignore[import-not-found]

            return float(ratio(left, right)) / 100.0
        except Exception:
            return SequenceMatcher(a=left, b=right).ratio()

    def _fuzzy_score(self, query: str, query_tokens: Sequence[str], document_index: int) -> float:
        document_text = _normalize_text(self._documents[document_index].text)
        if not document_text:
            return 0.0
        if query in document_text:
            return 1.0
        document_tokens = self._tokens[document_index]
        if not document_tokens or not query_tokens:
            return 0.0
        similarities = [
            max(self._ratio(token, document_token) for document_token in document_tokens)
            for token in query_tokens
        ]
        return sum(similarities) / len(similarities)

    def search(self, query: str, *, top_k: int = 10) -> list[MomentResult]:
        """Search text and return unique frame-level ``MomentResult`` records."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        normalized_query = _normalize_text(query)
        query_tokens = tokenize(normalized_query)
        if not normalized_query or not query_tokens or not self._documents:
            return []

        bm25_scores = self._bm25_scores(query_tokens)
        per_frame: dict[str, tuple[int, float, float, float]] = {}
        for index, bm25_score in enumerate(bm25_scores):
            fuzzy_score = self._fuzzy_score(normalized_query, query_tokens, index)
            bonus = self.fuzzy_weight * fuzzy_score if fuzzy_score >= self.fuzzy_threshold else 0.0
            total_score = float(bm25_score) + bonus
            if total_score <= 0:
                continue
            frame_id = str(_field(self._documents[index].record, "frame_id"))
            candidate = (index, total_score, float(bm25_score), fuzzy_score)
            existing = per_frame.get(frame_id)
            if existing is None or candidate[1:] > existing[1:]:
                per_frame[frame_id] = candidate

        ranked = sorted(
            per_frame.values(),
            key=lambda item: (
                -item[1],
                str(_field(self._documents[item[0]].record, "frame_id")),
                self._documents[item[0]].document_id or "",
            ),
        )
        return [
            _moment_from_document(
                self._documents[index],
                score=score,
                rank=rank,
                modality=self.modality,
                bm25_score=bm25_score,
                fuzzy_score=fuzzy_score,
            )
            for rank, (index, score, bm25_score, fuzzy_score) in enumerate(ranked[:top_k], start=1)
        ]
