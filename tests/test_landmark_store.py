"""Landmark cache: save/load round trip, validation, and stale-cache detection."""

import json

import numpy as np
import pytest

import landmark_store as ls


def sequence(t=8):
    rng = np.random.default_rng(1)
    det = np.ones(t, dtype=bool)
    det[3] = False
    world = rng.normal(0, 0.3, (t, 33, 3))
    world[~det] = np.nan
    return ls.LandmarkSequence(
        timestamps=np.arange(t) * 1000 / 30, landmarks=world, visibility=np.where(det[:, None], np.full((t, 33), 0.9), 0.0),
        detected=det, image_landmarks=rng.uniform(0, 1, (t, 33, 3)),
        metadata=ls.build_metadata(fps=30, width=640, height=360, num_frames=t),
        brightness=np.linspace(40, 120, t))


def test_round_trip(tmp_path):
    path = ls.save_landmarks(tmp_path / "a.npz", sequence())
    loaded = ls.load_landmarks(path)
    original = sequence()
    np.testing.assert_allclose(loaded.timestamps, original.timestamps)
    np.testing.assert_allclose(loaded.landmarks, original.landmarks.astype(np.float32), equal_nan=True)
    np.testing.assert_array_equal(loaded.detected, original.detected)
    np.testing.assert_allclose(loaded.brightness, original.brightness, rtol=1e-6)
    assert loaded.metadata["schema_version"] == ls.SCHEMA_VERSION
    assert not list(tmp_path.glob("*.tmp.npz"))                 # atomic write leaves no temp file


def test_rejects_wrong_extension_and_bad_shapes(tmp_path):
    with pytest.raises(ValueError):
        ls.save_landmarks(tmp_path / "a.json", sequence())
    bad = sequence()
    bad.visibility = bad.visibility[:, :10]
    with pytest.raises(ls.LandmarkSchemaError):
        ls.save_landmarks(tmp_path / "b.npz", bad)
    unordered = sequence()
    unordered.timestamps = unordered.timestamps[::-1].copy()
    with pytest.raises(ls.LandmarkSchemaError):
        ls.save_landmarks(tmp_path / "c.npz", unordered)


def test_current_cache_is_detected(tmp_path):
    path = ls.save_landmarks(tmp_path / "a.npz", sequence())
    assert ls.is_current(path)


def test_stale_cache_old_schema(tmp_path):
    path = tmp_path / "old.npz"
    seq = sequence()
    meta = dict(seq.metadata, schema_version=1)
    np.savez_compressed(path, timestamps=seq.timestamps, landmarks=seq.landmarks, visibility=seq.visibility,
                        detected=seq.detected, image_landmarks=seq.image_landmarks,
                        metadata_json=np.array(json.dumps(meta)))           # a v1 file has no brightness
    assert not ls.is_current(path)
    with pytest.raises(ls.LandmarkSchemaError, match="regenerate"):
        ls.load_landmarks(path)


def test_stale_cache_extraction_version_and_model(tmp_path, monkeypatch):
    model = tmp_path / "model.task"
    model.write_bytes(b"model-v1")
    seq = sequence()
    seq.metadata = ls.build_metadata(fps=30, width=640, height=360, num_frames=seq.num_frames, model_path=model)
    path = ls.save_landmarks(tmp_path / "a.npz", seq)
    assert ls.is_current(path, model_path=model)
    model.write_bytes(b"model-v2")                                 # a different model file
    assert not ls.is_current(path, model_path=model)
    monkeypatch.setattr(ls, "EXTRACTION_VERSION", "999")
    assert not ls.is_current(path)


def test_missing_or_corrupt_file_is_not_current(tmp_path):
    assert not ls.is_current(tmp_path / "missing.npz")
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"not a zip")
    assert not ls.is_current(junk)


def test_files_load_without_pickle(tmp_path):
    path = ls.save_landmarks(tmp_path / "a.npz", sequence())
    with np.load(path, allow_pickle=False) as data:
        assert "metadata_json" in data.files
