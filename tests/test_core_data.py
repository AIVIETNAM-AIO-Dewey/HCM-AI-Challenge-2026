from __future__ import annotations

import pytest
from pydantic import ValidationError

from hcm_ai.artifacts import ArtifactStore, fingerprint, fingerprint_inputs
from hcm_ai.config import load_settings
from hcm_ai.contracts import (
    AnswerResult,
    MomentResult,
    QueryPlan,
    SequenceEvent,
    SequenceResult,
    TaskType,
)
from hcm_ai.ingestion.aic2025 import (
    build_keyframe_manifest,
    parse_excel_queries,
    parse_txt_query,
)


def _moment(frame_id: str = "V1_F001", timestamp: float = 1.0) -> MomentResult:
    return MomentResult(
        video_id="V1",
        frame_id=frame_id,
        timestamp=timestamp,
        image_path=f"keyframes/{frame_id}.jpg",
        modality_scores={"visual": 0.8},
        reranker_score=0.9,
    )


def test_query_plan_normalizes_raw_weights_and_json_serializes() -> None:
    plan = QueryPlan(
        original_query="Tìm cảnh có biển hiệu",
        raw_weights={"visual": 2, "ocr": 1, "asr": 1},
        visual_queries=["sign", "sign", ""],
    )

    assert plan.visual_weight == pytest.approx(0.5)
    assert plan.ocr_weight == pytest.approx(0.25)
    assert plan.asr_weight == pytest.approx(0.25)
    assert plan.visual_queries == ["sign"]
    assert plan.model_dump(mode="json")["raw_weights"] == {"visual": 2.0, "ocr": 1.0, "asr": 1.0}


def test_results_are_grounded_and_sequences_enforce_temporal_order() -> None:
    answer = AnswerResult(query_id="QA_01", answer="Một biển hiệu", evidence=[_moment()])
    assert answer.model_dump(mode="json")["evidence"][0]["reranker_score"] == 0.9

    sequence = SequenceResult(
        video_id="V1",
        events=[
            SequenceEvent(
                event_index=0,
                video_id="V1",
                frame_id="V1_F001",
                timestamp=1.0,
                description="first",
            ),
            SequenceEvent(
                event_index=1,
                video_id="V1",
                frame_id="V1_F002",
                timestamp=3.5,
                description="second",
            ),
        ],
    )
    assert sequence.duration == pytest.approx(2.5)

    with pytest.raises(ValidationError, match="strictly increasing"):
        SequenceResult(
            video_id="V1",
            events=[
                SequenceEvent(
                    event_index=0,
                    video_id="V1",
                    frame_id="V1_F001",
                    timestamp=3.0,
                    description="first",
                ),
                SequenceEvent(
                    event_index=1,
                    video_id="V1",
                    frame_id="V1_F002",
                    timestamp=2.0,
                    description="second",
                ),
            ],
        )


def test_config_profile_and_path_environment_override() -> None:
    settings = load_settings(environ={"DATA_ROOT": "/content/drive/MyDrive/aic-data"})

    assert settings.profile == "balanced_gpu"
    assert settings.models.active_visual_encoders == ["siglip"]
    assert settings.models.reranker == "blip_itm"
    assert settings.paths.data_root == "/content/drive/MyDrive/aic-data"
    assert settings.indexes.visual.backend == "faiss"


def test_artifact_store_is_content_addressed_and_resumable(tmp_path) -> None:
    payload = {"frames": ["V1_F001", "V1_F002"], "config": {"model": "siglip"}}
    artifact_fingerprint = fingerprint_inputs(payload)
    assert artifact_fingerprint == fingerprint({"inputs": [payload]})

    store = ArtifactStore(tmp_path / "artifacts")
    first = store.write_jsonl("manifests", artifact_fingerprint, [_moment(), _moment("V1_F002", 2.0)])
    second = store.write_jsonl("manifests", artifact_fingerprint, [_moment()])

    assert first.record_count == 2
    assert second.record_count == 2
    assert store.is_complete("manifests", artifact_fingerprint)
    assert [row["frame_id"] for row in store.iter_jsonl(first.path)] == ["V1_F001", "V1_F002"]


def test_txt_query_and_keyframe_manifest_use_stable_aic_ids(tmp_path) -> None:
    query_directory = tmp_path / "KIS"
    query_directory.mkdir()
    query_path = query_directory / "KIS_0001.txt"
    query_path.write_text("Description: người đứng trước biển hiệu", encoding="utf-8")

    query = parse_txt_query(query_path)
    assert query.task is TaskType.KIS
    assert query.query_id == "KIS_0001"
    assert query.text == "người đứng trước biển hiệu"

    keyframe = tmp_path / "keyframes" / "L01_V001" / "000050.jpg"
    keyframe.parent.mkdir(parents=True)
    keyframe.write_bytes(b"placeholder")
    frames = build_keyframe_manifest(keyframe.parents[1], fps_by_video={"L01_V001": 25.0})
    assert frames[0].frame_id == "L01_V001_000050"
    assert frames[0].timestamp == pytest.approx(2.0)


def test_excel_query_adapter_reads_query_name_description_and_trans(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "QA"
    sheet.append(["Query Name", "Description", "Trans"])
    sheet.append(["VQA_0001", "Ai đang nói?", "Who is speaking?"])
    path = tmp_path / "queries.xlsx"
    workbook.save(path)

    records = parse_excel_queries(path)
    assert len(records) == 1
    assert records[0].task is TaskType.QA
    assert records[0].text == "Ai đang nói?"
    assert records[0].translated_text == "Who is speaking?"
