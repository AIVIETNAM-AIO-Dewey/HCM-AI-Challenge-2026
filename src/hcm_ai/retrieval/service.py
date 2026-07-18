"""Cascaded visual/text retrieval, fusion, reranking, and temporal search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import isfinite, prod
from typing import Any

from hcm_ai.contracts import AnswerResult, MomentResult, QueryPlan, QueryRecord, SequenceResult, TaskType
from hcm_ai.embeddings import EmbeddingProvider
from hcm_ai.fusion import min_max_normalize, normalize_modalities, similarity_weighted_rrf
from hcm_ai.indexing import BM25Store, InMemoryVectorStore
from hcm_ai.planning import GroundedAnswerer, HeuristicQueryPlanner, QueryPlanner, parse_temporal_events
from hcm_ai.reranking import NullReranker, Reranker
from hcm_ai.temporal import temporal_beam_search


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """Runtime knobs for the service, mirroring the YAML profile defaults."""

    visual_top_k: int = 100
    text_top_k: int = 100
    rerank_top_k: int = 32
    rrf_k: int = 60
    reranker_weight: float = 0.3
    fused_weight: float = 0.7
    event_top_k: int = 20
    beam_width: int = 8
    temporal_alpha: float = 0.01

    def __post_init__(self) -> None:
        for name in ("visual_top_k", "text_top_k", "rerank_top_k", "rrf_k", "event_top_k", "beam_width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.temporal_alpha < 0:
            raise ValueError("temporal_alpha must be non-negative")
        if self.fused_weight < 0 or self.reranker_weight < 0:
            raise ValueError("reranker weights must be non-negative")

    @classmethod
    def from_settings(cls, settings: Any) -> "SearchSettings":
        """Make service settings from :class:`hcm_ai.config.Settings` or a mapping."""

        source = settings.model_dump() if hasattr(settings, "model_dump") else dict(settings)
        retrieval = source.get("retrieval", {})
        reranking = source.get("reranking", {})
        temporal = source.get("temporal", {})
        return cls(
            visual_top_k=int(retrieval.get("visual_top_k", 100)),
            text_top_k=int(retrieval.get("text_top_k", 100)),
            rerank_top_k=int(retrieval.get("rerank_top_k", reranking.get("candidate_top_k", 32))),
            rrf_k=int(retrieval.get("rrf_k", 60)),
            reranker_weight=float(reranking.get("reranker_weight", 0.3)),
            fused_weight=float(reranking.get("fused_weight", 0.7)),
            event_top_k=int(temporal.get("event_top_k", 20)),
            beam_width=int(temporal.get("beam_width", 8)),
            temporal_alpha=float(temporal.get("alpha", 0.01)),
        )


@dataclass(frozen=True, slots=True)
class SearchTrace:
    """Auditable facts from the most recent search, safe to serialize in notebooks."""

    query_id: str | None
    plan: QueryPlan
    modality_candidate_counts: dict[str, int]
    branch_errors: tuple[str, ...] = ()


def _record_from_text(query: str, *, task: TaskType) -> QueryRecord:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string or QueryRecord")
    return QueryRecord(query_id="adhoc", text=query, task=task)


def _coerce_query(query: QueryRecord | str, *, task: TaskType) -> QueryRecord:
    if isinstance(query, QueryRecord):
        return query if query.task == task else query.model_copy(update={"task": task})
    return _record_from_text(query, task=task)


def _stable_moment_order(results: Sequence[MomentResult]) -> list[MomentResult]:
    return sorted(
        results,
        key=lambda item: (-item.score, item.video_id, item.timestamp, item.frame_id),
    )


class SearchService:
    """The public no-HTTP interface for KIS, TRAKE, and grounded QA.

    All optional services are injected.  A fully functional no-GPU smoke path
    therefore consists of ``HashEmbeddingProvider``, ``InMemoryVectorStore``,
    ``BM25Store``, and the default heuristic planner.
    """

    def __init__(
        self,
        *,
        visual_stores: Mapping[str, InMemoryVectorStore] | InMemoryVectorStore | None = None,
        visual_embeddings: Mapping[str, EmbeddingProvider] | EmbeddingProvider | None = None,
        ocr_store: BM25Store | None = None,
        asr_store: BM25Store | None = None,
        planner: QueryPlanner | None = None,
        reranker: Reranker | None = None,
        answerer: GroundedAnswerer | None = None,
        settings: SearchSettings | Any | None = None,
    ) -> None:
        if isinstance(visual_stores, InMemoryVectorStore):
            visual_stores = {"siglip": visual_stores}
        if visual_embeddings is not None and not isinstance(visual_embeddings, Mapping):
            visual_embeddings = {"siglip": visual_embeddings}
        self.visual_stores = dict(visual_stores or {})
        self.visual_embeddings = dict(visual_embeddings or {})
        self.ocr_store = ocr_store
        self.asr_store = asr_store
        self.planner = planner or HeuristicQueryPlanner()
        self.reranker = reranker or NullReranker()
        self.answerer = answerer or GroundedAnswerer(max_retries=0)
        self.settings = (
            settings
            if isinstance(settings, SearchSettings)
            else SearchSettings.from_settings(settings) if settings is not None else SearchSettings()
        )
        self.last_trace: SearchTrace | None = None

    def _plan(self, record: QueryRecord) -> QueryPlan:
        plan = self.planner.plan(record)
        if not isinstance(plan, QueryPlan):
            raise TypeError("planner.plan must return QueryPlan")
        return plan

    @staticmethod
    def _keep_best_result(
        candidates: dict[str, MomentResult],
        result: MomentResult,
    ) -> None:
        current = candidates.get(result.frame_id)
        if current is None or result.score > current.score:
            candidates[result.frame_id] = result

    def _visual_candidates(
        self,
        plan: QueryPlan,
        errors: list[str],
    ) -> tuple[dict[str, float], dict[str, MomentResult], dict[str, list[str]]]:
        rankings: dict[str, list[tuple[str, float]]] = {}
        candidates: dict[str, MomentResult] = {}
        provenance: dict[str, list[str]] = {}
        queries = plan.visual_queries or [plan.translated_query or plan.original_query]
        for encoder_name in sorted(set(self.visual_stores) & set(self.visual_embeddings)):
            store = self.visual_stores[encoder_name]
            provider = self.visual_embeddings[encoder_name]
            try:
                vectors = provider.encode_texts(queries)
                if len(vectors) != len(queries):
                    raise ValueError("embedding provider returned a different number of query vectors")
            except Exception as error:
                errors.append(f"visual:{encoder_name}:encode:{type(error).__name__}: {error}")
                continue
            for variant_index, vector in enumerate(vectors):
                source = f"visual:{encoder_name}:variant{variant_index + 1}"
                try:
                    results = store.search(vector, top_k=self.settings.visual_top_k)
                except Exception as error:
                    errors.append(f"{source}:search:{type(error).__name__}: {error}")
                    continue
                rankings[source] = [(result.frame_id, result.score) for result in results]
                for result in results:
                    self._keep_best_result(candidates, result)
                    provenance.setdefault(result.frame_id, []).append(source)
        if not rankings:
            return {}, candidates, provenance
        return similarity_weighted_rrf(rankings, rrf_k=self.settings.rrf_k), candidates, provenance

    def _text_candidates(
        self,
        store: BM25Store | None,
        query: str | None,
        modality: str,
        errors: list[str],
    ) -> tuple[dict[str, float], dict[str, MomentResult], dict[str, list[str]]]:
        if store is None or not query or not query.strip():
            return {}, {}, {}
        try:
            results = store.search(query, top_k=self.settings.text_top_k)
        except Exception as error:
            errors.append(f"{modality}:search:{type(error).__name__}: {error}")
            return {}, {}, {}
        candidates: dict[str, MomentResult] = {}
        provenance: dict[str, list[str]] = {}
        scores: dict[str, float] = {}
        for result in results:
            if result.score > scores.get(result.frame_id, float("-inf")):
                scores[result.frame_id] = result.score
                candidates[result.frame_id] = result
            provenance.setdefault(result.frame_id, []).append(f"{modality}:bm25")
        return scores, candidates, provenance

    @staticmethod
    def _fuse_with_missing_scores(
        modality_scores: Mapping[str, Mapping[str, float]],
        *,
        weights: Mapping[str, float],
    ) -> dict[str, float]:
        """Min-max normalize each modality and retain 0 for missing candidates."""

        normalized = normalize_modalities(modality_scores)
        all_ids = {candidate_id for values in modality_scores.values() for candidate_id in values}
        fused: dict[str, float] = {}
        for candidate_id in all_ids:
            fused[candidate_id] = sum(
                float(weights.get(modality, 0.0)) * normalized.get(modality, {}).get(candidate_id, 0.0)
                for modality in ("visual", "ocr", "asr")
            )
        return fused

    def _fuse(
        self,
        plan: QueryPlan,
        visual_scores: Mapping[str, float],
        visual_candidates: Mapping[str, MomentResult],
        visual_provenance: Mapping[str, Sequence[str]],
        ocr_scores: Mapping[str, float],
        ocr_candidates: Mapping[str, MomentResult],
        ocr_provenance: Mapping[str, Sequence[str]],
        asr_scores: Mapping[str, float],
        asr_candidates: Mapping[str, MomentResult],
        asr_provenance: Mapping[str, Sequence[str]],
    ) -> list[MomentResult]:
        modalities = {"visual": visual_scores, "ocr": ocr_scores, "asr": asr_scores}
        fused_scores = self._fuse_with_missing_scores(
            modalities,
            weights={
                "visual": plan.visual_weight,
                "ocr": plan.ocr_weight,
                "asr": plan.asr_weight,
            },
        )
        source_records: dict[str, MomentResult] = {}
        for source in (visual_candidates, ocr_candidates, asr_candidates):
            for frame_id, item in source.items():
                current = source_records.get(frame_id)
                if current is None or item.score > current.score:
                    source_records[frame_id] = item
        records: list[MomentResult] = []
        for frame_id, score in fused_scores.items():
            base = source_records.get(frame_id)
            if base is None:
                continue
            provenance = list(
                dict.fromkeys(
                    [
                        *visual_provenance.get(frame_id, ()),
                        *ocr_provenance.get(frame_id, ()),
                        *asr_provenance.get(frame_id, ()),
                    ]
                )
            )
            modality_values = {
                "visual": float(visual_scores.get(frame_id, 0.0)),
                "ocr": float(ocr_scores.get(frame_id, 0.0)),
                "asr": float(asr_scores.get(frame_id, 0.0)),
            }
            records.append(
                MomentResult(
                    video_id=base.video_id,
                    frame_id=base.frame_id,
                    timestamp=base.timestamp,
                    image_path=base.image_path,
                    shot_id=base.shot_id,
                    score=score,
                    modality_scores=modality_values,
                    fused_score=score,
                    provenance=provenance,
                    metadata={
                        **base.metadata,
                        "query_id": plan.query_id,
                        "planner": plan.planner,
                    },
                )
            )
        return _stable_moment_order(records)

    def _rerank(self, plan: QueryPlan, candidates: Sequence[MomentResult], errors: list[str]) -> list[MomentResult]:
        if not candidates or self.settings.rerank_top_k <= 0 or self.settings.reranker_weight <= 0:
            return list(candidates)
        head = list(candidates[: self.settings.rerank_top_k])
        try:
            raw_scores = self.reranker.score(plan.translated_query or plan.original_query, [item.image_path for item in head])
            if len(raw_scores) != len(head):
                raise ValueError("reranker returned a different number of scores")
            scores = [float(value) for value in raw_scores]
            if not all(isfinite(value) for value in scores):
                raise ValueError("reranker returned a non-finite score")
        except Exception as error:
            errors.append(f"reranker:{type(error).__name__}: {error}")
            return list(candidates)

        # A constant score is a neutral model (the CPU fallback), so leave the
        # fused order intact instead of damping the candidate scores.
        if max(scores) == min(scores):
            return [
                item.model_copy(
                    update={
                        "reranker_score": score,
                        "provenance": [*item.provenance, "reranker:neutral"],
                    }
                )
                for item, score in zip(head, scores, strict=True)
            ] + list(candidates[len(head) :])

        normalized = min_max_normalize({str(index): score for index, score in enumerate(scores)})
        total_weight = self.settings.fused_weight + self.settings.reranker_weight
        fused_weight = self.settings.fused_weight / total_weight if total_weight else 1.0
        reranker_weight = self.settings.reranker_weight / total_weight if total_weight else 0.0
        reranked = [
            item.model_copy(
                update={
                    "score": fused_weight * item.score + reranker_weight * normalized[str(index)],
                    "reranker_score": score,
                    "provenance": [*item.provenance, "reranker:itm"],
                }
            )
            for index, (item, score) in enumerate(zip(head, scores, strict=True))
        ]
        return _stable_moment_order([*reranked, *candidates[len(head) :]])

    @staticmethod
    def _with_ranks(candidates: Sequence[MomentResult], *, top_k: int) -> list[MomentResult]:
        return [
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(_stable_moment_order(candidates)[:top_k], start=1)
        ]

    def _search_plan(self, plan: QueryPlan, *, top_k: int) -> tuple[list[MomentResult], SearchTrace]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        errors: list[str] = []
        # Candidate generation is modality-independent.  Keep its errors in
        # per-branch lists so the externally visible trace stays deterministic
        # despite parallel execution.
        visual_errors: list[str] = []
        ocr_errors: list[str] = []
        asr_errors: list[str] = []
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="hcm-retrieval") as executor:
            visual_future = executor.submit(self._visual_candidates, plan, visual_errors)
            ocr_future = executor.submit(
                self._text_candidates, self.ocr_store, plan.ocr_query, "ocr", ocr_errors
            )
            asr_future = executor.submit(
                self._text_candidates, self.asr_store, plan.asr_query, "asr", asr_errors
            )
            try:
                visual_scores, visual_candidates, visual_provenance = visual_future.result()
            except Exception as error:  # Defensive boundary around third-party stores.
                visual_scores, visual_candidates, visual_provenance = {}, {}, {}
                visual_errors.append(f"visual:unexpected:{type(error).__name__}: {error}")
            try:
                ocr_scores, ocr_candidates, ocr_provenance = ocr_future.result()
            except Exception as error:
                ocr_scores, ocr_candidates, ocr_provenance = {}, {}, {}
                ocr_errors.append(f"ocr:unexpected:{type(error).__name__}: {error}")
            try:
                asr_scores, asr_candidates, asr_provenance = asr_future.result()
            except Exception as error:
                asr_scores, asr_candidates, asr_provenance = {}, {}, {}
                asr_errors.append(f"asr:unexpected:{type(error).__name__}: {error}")
        errors.extend(visual_errors)
        errors.extend(ocr_errors)
        errors.extend(asr_errors)
        fused = self._fuse(
            plan,
            visual_scores,
            visual_candidates,
            visual_provenance,
            ocr_scores,
            ocr_candidates,
            ocr_provenance,
            asr_scores,
            asr_candidates,
            asr_provenance,
        )
        results = self._with_ranks(self._rerank(plan, fused, errors), top_k=top_k)
        trace = SearchTrace(
            query_id=plan.query_id,
            plan=plan,
            modality_candidate_counts={
                "visual": len(visual_scores),
                "ocr": len(ocr_scores),
                "asr": len(asr_scores),
                "fused": len(fused),
                "reranked": min(len(fused), self.settings.rerank_top_k),
            },
            branch_errors=tuple(errors),
        )
        return results, trace

    def search_moments(self, query: QueryRecord | str, *, top_k: int = 10) -> list[MomentResult]:
        """Retrieve ranked visual/OCR/ASR moments for a KIS-style query."""

        record = _coerce_query(query, task=TaskType.KIS)
        results, trace = self._search_plan(self._plan(record), top_k=top_k)
        self.last_trace = trace
        return results

    @staticmethod
    def _event_plan(parent: QueryPlan, event: str, index: int) -> QueryPlan:
        """Derive a no-extra-API child plan for one TRAKE event."""

        variants = [
            event,
            f"video scene: {event}",
            f"keyframe matching: {event}",
            f"visual evidence of: {event}",
        ]
        return QueryPlan(
            query_id=f"{parent.query_id or 'adhoc'}:E{index + 1}",
            original_query=event,
            translated_query=event,
            visual_queries=variants,
            ocr_query=event,
            asr_query=event,
            raw_weights=parent.raw_weights,
            planner=f"{parent.planner}:temporal_event",
            metadata={**parent.metadata, "parent_query_id": parent.query_id, "event_index": index},
        )

    def search_temporal(self, query: QueryRecord | str, *, top_k: int = 10) -> list[SequenceResult]:
        """Retrieve chronological same-video TRAKE sequences with time-gap decay."""

        record = _coerce_query(query, task=TaskType.TRAKE)
        plan = self._plan(record)
        events = plan.temporal_events or parse_temporal_events(record.text)
        if not events:
            self.last_trace = SearchTrace(
                query_id=plan.query_id,
                plan=plan,
                modality_candidate_counts={"events": 0},
                branch_errors=("temporal query has no E1…En events",),
            )
            return []
        per_event: list[list[MomentResult]] = []
        errors: list[str] = []
        counts: dict[str, int] = {"events": len(events)}
        for index, event in enumerate(events):
            candidates, trace = self._search_plan(
                self._event_plan(plan, event, index),
                top_k=self.settings.event_top_k,
            )
            per_event.append(candidates)
            counts[f"event_{index + 1}"] = len(candidates)
            errors.extend(trace.branch_errors)
        sequences = temporal_beam_search(
            per_event,
            event_descriptions=events,
            alpha=self.settings.temporal_alpha,
            beam_width=self.settings.beam_width,
            top_k=top_k,
        )
        reranked: list[SequenceResult] = []
        per_event_rerank_scores = [
            {candidate.frame_id: candidate.reranker_score for candidate in candidates}
            for candidates in per_event
        ]
        for sequence in sequences:
            scores = [
                per_event_rerank_scores[event.event_index].get(event.frame_id)
                for event in sequence.events
            ]
            usable = [score for score in scores if score is not None and score >= 0]
            if usable:
                sequence_reranker_score = prod(usable) ** (1.0 / len(usable))
                reranked.append(
                    sequence.model_copy(
                        update={
                            "score": sequence.score * sequence_reranker_score,
                            "reranker_score": sequence_reranker_score,
                            "metadata": {**sequence.metadata, "event_reranker_scores": scores},
                        }
                    )
                )
            else:
                reranked.append(sequence)
        ordered = sorted(
            reranked,
            key=lambda item: (-item.score, item.video_id, tuple(event.frame_id for event in item.events)),
        )
        results = [item.model_copy(update={"rank": rank}) for rank, item in enumerate(ordered[:top_k], start=1)]
        counts["sequences"] = len(results)
        self.last_trace = SearchTrace(
            query_id=plan.query_id,
            plan=plan,
            modality_candidate_counts=counts,
            branch_errors=tuple(errors),
        )
        return results

    def answer_question(
        self,
        query: QueryRecord | str,
        *,
        evidence_top_k: int = 5,
    ) -> AnswerResult:
        """Retrieve evidence first, then return a citation-validated QA result."""

        if evidence_top_k <= 0:
            raise ValueError("evidence_top_k must be positive")
        record = _coerce_query(query, task=TaskType.QA)
        evidence, trace = self._search_plan(self._plan(record), top_k=evidence_top_k)
        self.last_trace = trace
        return self.answerer.answer(query_id=record.query_id, question=record.text, evidence=evidence)


__all__ = ["SearchService", "SearchSettings", "SearchTrace"]
