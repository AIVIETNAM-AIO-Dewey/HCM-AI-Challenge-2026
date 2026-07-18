from __future__ import annotations

from pathlib import Path

from hcm_ai.artifacts import ArtifactStore
from hcm_ai.asr import AsrSegment
from hcm_ai.contracts import FrameRecord, ShotRecord
from hcm_ai.embeddings import HashEmbeddingProvider
from hcm_ai.indexing import build_text_index, build_visual_index
from hcm_ai.ingestion import asr_records_from_segments, ocr_records_from_extractions
from hcm_ai.ocr import OcrExtraction
from hcm_ai.ocr import GeminiOcrRefiner, NullOcrProvider


class CountingHashProvider(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dimension=8)
        self.image_calls = 0

    def encode_images(self, images):  # type: ignore[no-untyped-def]
        self.image_calls += 1
        return super().encode_images(images)


def _frames() -> list[FrameRecord]:
    return [
        FrameRecord(frame_id="f1", video_id="V1", timestamp=1.0, image_path="keyframes/f1.jpg"),
        FrameRecord(frame_id="f2", video_id="V1", timestamp=5.0, image_path="keyframes/f2.jpg"),
    ]


def test_ocr_asr_alignment_preserves_nearest_renderable_frame() -> None:
    frames = _frames()
    ocr = ocr_records_from_extractions(frames, [OcrExtraction("one"), OcrExtraction("two")])
    assert [item.frame_id for item in ocr] == ["f1", "f2"]
    asr = asr_records_from_segments(
        video_id="V1",
        frames=frames,
        shots=[ShotRecord(shot_id="s1", video_id="V1", start=0.0, end=3.0)],
        segments=[AsrSegment(start=1.5, end=2.0, text="hello")],
    )
    assert asr[0].nearest_frame_id == "f1"
    assert asr[0].shot_id == "s1"


def test_index_artifacts_resume_without_reencoding(tmp_path: Path) -> None:
    frames = _frames()
    artifacts = ArtifactStore(tmp_path / "drive_artifacts")
    provider = CountingHashProvider()
    first_store, first = build_visual_index(frames, provider, artifacts=artifacts, batch_size=1)
    assert not first.reused
    assert provider.image_calls == 2
    second_store, second = build_visual_index(frames, provider, artifacts=artifacts, batch_size=1)
    assert second.reused
    assert provider.image_calls == 2
    assert len(first_store) == len(second_store) == 2


def test_text_index_artifact_resume(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "drive_artifacts")
    frames = _frames()
    first_store, first = build_text_index(frames, ["hello", "world"], modality="ocr", artifacts=artifacts)
    second_store, second = build_text_index(frames, ["hello", "world"], modality="ocr", artifacts=artifacts)
    assert not first.reused and second.reused
    assert first_store.search("hello")
    assert second_store.search("world")


def test_gemini_ocr_refinement_keeps_local_result_when_unavailable(tmp_path: Path) -> None:
    image = tmp_path / "text.jpg"
    image.write_bytes(b"not-a-real-image")
    refiner = GeminiOcrRefiner(NullOcrProvider(), max_retries=0)
    local = [OcrExtraction(text="local text", confidence=0.4)]
    result = refiner.refine([image], local, candidate_paths=[image])
    assert result == local
