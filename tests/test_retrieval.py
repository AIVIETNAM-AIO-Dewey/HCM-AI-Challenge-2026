from __future__ import annotations

from math import exp

import pytest

from hcm_ai.contracts import FrameRecord, MomentResult
from hcm_ai.fusion import (
    fuse_modalities,
    min_max_normalize,
    similarity_weighted_rrf,
)
from hcm_ai.indexing import BM25Store, InMemoryVectorStore
from hcm_ai.temporal import temporal_beam_search


def _frame(frame_id: str, *, timestamp: float = 0.0, video_id: str = "V1") -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        video_id=video_id,
        timestamp=timestamp,
        image_path=f"keyframes/{frame_id}.jpg",
    )


def _moment(
    frame_id: str,
    *,
    timestamp: float,
    score: float,
    video_id: str = "V1",
) -> MomentResult:
    return MomentResult(
        frame_id=frame_id,
        video_id=video_id,
        timestamp=timestamp,
        image_path=f"keyframes/{frame_id}.jpg",
        score=score,
        provenance=["test"],
    )


def test_min_max_and_modality_fusion_handle_empty_and_equal_scores() -> None:
    assert min_max_normalize({}) == {}
    assert min_max_normalize({"a": 4.0, "b": 4.0}) == {"a": 0.0, "b": 0.0}

    fused = fuse_modalities(
        {
            "visual": {"a": 0.9, "b": 0.1},
            "ocr": {},
        },
        weights={"visual": 0.2, "ocr": 0.8},
    )
    assert fused["a"] == pytest.approx(1.0)
    assert fused["b"] == pytest.approx(0.0)


def test_similarity_weighted_rrf_rewards_agreement_and_is_deterministic() -> None:
    fused = similarity_weighted_rrf(
        {
            "beit3": {"frame_a": 0.8, "frame_b": 0.7},
            "siglip": {"frame_a": 0.9, "frame_c": 0.1},
        },
        rrf_k=60,
    )

    assert list(fused) == ["frame_a", "frame_b", "frame_c"]
    assert fused["frame_a"] == pytest.approx(0.8 / 61 + 0.9 / 61)
    assert fused["frame_a"] > fused["frame_b"] > fused["frame_c"]


def test_vector_store_uses_normalized_cosine_and_stable_ties() -> None:
    store = InMemoryVectorStore(prefer_faiss=False)
    store.add(
        [_frame("frame_b"), _frame("frame_a"), _frame("frame_c")],
        [[5.0, 0.0], [2.0, 0.0], [0.0, 3.0]],
    )

    results = store.search([10.0, 0.0], top_k=3)
    assert [result.frame_id for result in results] == ["frame_a", "frame_b", "frame_c"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].modality_scores["visual"] == pytest.approx(1.0)


def test_vector_store_rejects_zero_and_wrong_dimension_vectors() -> None:
    store = InMemoryVectorStore(prefer_faiss=False)
    with pytest.raises(ValueError, match="zero vectors"):
        store.add([_frame("bad")], [[0.0, 0.0]])

    store.add([_frame("good")], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dimension"):
        store.search([1.0, 0.0, 0.0])


def test_bm25_store_is_frame_level_and_has_a_fuzzy_fallback() -> None:
    first = _frame("frame_1", timestamp=1.0)
    second = _frame("frame_2", timestamp=2.0)
    store = BM25Store(modality="ocr", prefer_rank_bm25=False)
    store.add(
        [first, first, second],
        [
            "a football player enters the stadium",
            "ronaldo celebrates a goal",
            "a cooking show in the kitchen",
        ],
    )

    exact = store.search("ronaldo goal", top_k=5)
    assert [result.frame_id for result in exact] == ["frame_1"]
    assert exact[0].modality_scores["ocr"] > 0.0

    fuzzy = store.search("ronald", top_k=1)
    assert fuzzy[0].frame_id == "frame_1"
    assert fuzzy[0].metadata["fuzzy_score"] >= store.fuzzy_threshold


def test_temporal_beam_search_requires_same_video_order_and_unique_frames() -> None:
    results = temporal_beam_search(
        [
            [
                _moment("f1", timestamp=2.0, score=0.8),
                _moment("g1", timestamp=1.0, score=0.99, video_id="V2"),
            ],
            [
                _moment("f1", timestamp=3.0, score=1.0),  # Duplicate frame: invalid path.
                _moment("f2", timestamp=4.0, score=0.9),
                _moment("g2", timestamp=0.5, score=1.0, video_id="V2"),  # Reverse time.
            ],
        ],
        event_descriptions=["person appears", "person celebrates"],
        alpha=0.01,
        beam_width=8,
    )

    assert len(results) == 1
    result = results[0]
    assert result.video_id == "V1"
    assert [event.frame_id for event in result.events] == ["f1", "f2"]
    assert [event.description for event in result.events] == ["person appears", "person celebrates"]
    assert result.metadata["temporal_decays"] == pytest.approx([1.0, exp(-0.02)])
    assert result.score == pytest.approx(0.8 + 0.9 * exp(-0.02))
    assert result.duration == pytest.approx(2.0)


def test_temporal_beam_search_returns_no_partial_sequences() -> None:
    results = temporal_beam_search(
        [[_moment("f1", timestamp=4.0, score=1.0)], [_moment("f2", timestamp=4.0, score=1.0)]],
    )
    assert results == []
