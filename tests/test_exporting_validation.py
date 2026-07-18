from __future__ import annotations

from hcm_ai.exporting import read_jsonl, write_csv, write_jsonl
from hcm_ai.validation import ValidationError, validate_answer, validate_moment, validate_sequence


def _moment(frame_id: str = "V1_F1", timestamp: float = 1.0) -> dict[str, object]:
    return {
        "video_id": "V1",
        "frame_id": frame_id,
        "timestamp": timestamp,
        "image_path": f"keyframes/{frame_id}.jpg",
        "score": 0.8,
    }


def test_moment_and_sequence_validation() -> None:
    first = _moment("V1_F1", 1.0)
    second = _moment("V1_F2", 2.0)
    assert validate_moment(first)["frame_id"] == "V1_F1"
    assert validate_sequence({"video_id": "V1", "events": [first, second]})["video_id"] == "V1"


def test_sequence_rejects_reverse_time() -> None:
    try:
        validate_sequence({"video_id": "V1", "events": [_moment("V1_F1", 2.0), _moment("V1_F2", 1.0)]})
    except ValidationError as error:
        assert "increasing" in str(error)
    else:
        raise AssertionError("Expected reverse timestamps to fail")


def test_answer_requires_evidence() -> None:
    try:
        validate_answer({"query_id": "qa-1", "answer": "Khánh Hòa", "evidence": []})
    except ValidationError as error:
        assert "evidence" in str(error)
    else:
        raise AssertionError("Expected ungrounded answer to fail")


def test_canonical_exports(tmp_path) -> None:
    records = [_moment(), _moment("V1_F2", 2.0)]
    jsonl_path = write_jsonl(tmp_path / "moments.jsonl", records)
    csv_path = write_csv(tmp_path / "moments.csv", records)
    assert read_jsonl(jsonl_path)[0]["frame_id"] == "V1_F1"
    assert csv_path.read_text(encoding="utf-8").startswith("frame_id,")
