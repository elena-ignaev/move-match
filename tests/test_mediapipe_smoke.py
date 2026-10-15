"""Real MediaPipe on a real file (skipped when the model has not been downloaded).

The fixture videos are test patterns, so no person is expected: this checks that the model loads,
that VIDEO-mode timestamps are accepted, and that an empty result is stored as 'not detected'.
"""

import pytest

from conftest import needs_ffmpeg
from process_video import MODEL_PATH, extract_landmarks

pytestmark = [needs_ffmpeg, pytest.mark.skipif(not MODEL_PATH.is_file(), reason="pose model not downloaded")]


def test_mediapipe_runs_on_a_video(media_dir, tmp_path):
    import landmark_store as ls
    stats = extract_landmarks(media_dir / "reference.mp4", tmp_path / "lm.npz")
    seq = ls.load_landmarks(tmp_path / "lm.npz")
    assert stats["frame_count"] == seq.num_frames >= 170
    assert seq.metadata["model_sha256"]
    assert stats["detected_ratio"] < 0.5           # a test pattern contains no dancer
