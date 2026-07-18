from __future__ import annotations

from hcm_ai.contracts import FrameRecord, QueryRecord, TaskType
from hcm_ai.embeddings import HashEmbeddingProvider
from hcm_ai.indexing import BM25Store, InMemoryVectorStore
from hcm_ai.planning import GeminiQueryPlanner, HeuristicQueryPlanner, parse_temporal_events
from hcm_ai.retrieval import SearchService, SearchSettings
from hcm_ai.validation import validate_sequence


def _frame(frame_id: str, timestamp: float, video_id: str = "V1") -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        video_id=video_id,
        timestamp=timestamp,
        image_path=f"keyframes/{frame_id}.jpg",
    )


def _service() -> SearchService:
    provider = HashEmbeddingProvider(dimension=8)
    frames = [_frame("f1", 1.0), _frame("f2", 4.0), _frame("f3", 8.0)]
    visual = InMemoryVectorStore(prefer_faiss=False)
    visual.add(frames, provider.encode_images([frame.image_path for frame in frames]))
    ocr = BM25Store(modality="ocr", prefer_rank_bm25=False)
    ocr.add(frames, ["a red car", "welcome sign", "a person waves"])
    asr = BM25Store(modality="asr", prefer_rank_bm25=False)
    asr.add(frames, ["", "hello everyone", "goodbye"])
    return SearchService(
        visual_stores={"siglip": visual},
        visual_embeddings={"siglip": provider},
        ocr_store=ocr,
        asr_store=asr,
        settings=SearchSettings(visual_top_k=3, text_top_k=3, rerank_top_k=2, event_top_k=3),
    )


def test_temporal_parser_and_offline_planner_normalize_weights() -> None:
    text = "E2: a person waves\nE1: a car appears"
    assert parse_temporal_events(text) == ["a car appears", "a person waves"]
    plan = HeuristicQueryPlanner().plan(QueryRecord(query_id="t1", text=text, task=TaskType.TRAKE))
    assert plan.temporal_events == ["a car appears", "a person waves"]
    assert len(plan.visual_queries) == 4
    assert plan.visual_weight + plan.ocr_weight + plan.asr_weight == 1.0


def test_gemini_planner_falls_back_without_key() -> None:
    planner = GeminiQueryPlanner(max_retries=0)
    plan = planner.plan(QueryRecord(query_id="k1", text="xe màu đỏ", task=TaskType.KIS))
    assert plan.planner == "heuristic"
    assert "gemini_fallback_reason" in plan.metadata


def test_multimodal_search_keeps_scores_and_grounded_qa_fails_closed() -> None:
    service = _service()
    results = service.search_moments("welcome sign", top_k=2)
    assert results
    assert results[0].rank == 1
    assert set(results[0].modality_scores) == {"visual", "ocr", "asr"}
    assert service.last_trace is not None
    answer = service.answer_question(QueryRecord(query_id="qa1", text="What does the sign say?", task=TaskType.QA))
    assert answer.answer is None
    assert answer.evidence
    assert answer.error


def test_temporal_search_and_sequence_validator_accept_missing_event_image_path() -> None:
    service = _service()
    query = QueryRecord(
        query_id="trake1",
        task=TaskType.TRAKE,
        text="E1: a red car\nE2: a person waves",
    )
    results = service.search_temporal(query)
    assert results
    assert results[0].events[0].timestamp < results[0].events[1].timestamp
    canonical = results[0].model_dump(mode="json")
    canonical["events"][0]["image_path"] = None
    assert validate_sequence(canonical)["video_id"] == "V1"
