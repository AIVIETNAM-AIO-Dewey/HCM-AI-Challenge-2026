from __future__ import annotations

from hcm_ai.preprocessing.keyframes import discover_keyframes
from hcm_ai.preprocessing.shots import FixedIntervalShotDetector, TransNetV2ShotDetector


def test_discover_keyframes_retains_stable_ids_and_timestamp(tmp_path) -> None:
    (tmp_path / "frame_t_0012.5.jpg").write_bytes(b"image")
    (tmp_path / "frame_2.jpg").write_bytes(b"image")
    frames = discover_keyframes(tmp_path, video_id="L25_V001", fps=2.0)
    assert len(frames) == 2
    assert frames[0].video_id == "L25_V001"
    assert all(frame.frame_id.startswith("L25_V001_") for frame in frames)
    assert any(frame.timestamp == 12.5 for frame in frames)


def test_interval_shot_detector_covers_video() -> None:
    shots = FixedIntervalShotDetector(interval_seconds=3.0).detect(7.0)
    assert [(shot.start_seconds, shot.end_seconds) for shot in shots] == [(0.0, 3.0), (3.0, 6.0), (6.0, 7.0)]


def test_transnet_adapter_normalizes_cuts() -> None:
    detector = TransNetV2ShotDetector(lambda _: [5.0, 2.0, 2.0, -1.0, 11.0])
    shots = detector.detect("video.mp4", 10.0)
    assert [(shot.start_seconds, shot.end_seconds) for shot in shots] == [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0)]
