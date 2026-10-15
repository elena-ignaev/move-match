"""Shared fixtures.

* Fixture videos are generated with FFmpeg test patterns, so no third-party footage is needed.
* Pose detection is replaced by FakeLandmarker, which returns synthetic choreography poses for a
  given timestamp. That lets the whole journey (upload -> process -> record -> feedback) run in
  seconds and with KNOWN ground truth. Real MediaPipe is exercised by test_mediapipe_smoke.py when
  the model file is present.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research", ROOT / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cases import make_choreography, project  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg/ffprobe not installed")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory):
    if not HAS_FFMPEG:
        pytest.skip("FFmpeg/ffprobe not installed")
    d = tmp_path_factory.mktemp("media")
    # 6 s reference with music (a sine tone), 30 fps
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=6",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "reference.mp4"))
    # Variable frame rate: irregular frame gaps (like a phone video)
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=4",
           "-vf", "select='not(mod(n\\,3))+between(n\\,30\\,60)'", "-fps_mode", "vfr",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "vfr.mkv"))
    # Browser-style recording: VP8 WebM, no audio, 8 s
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=8",
           "-c:v", "libvpx", "-b:v", "300k", "-an", str(d / "attempt.webm"))
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=2400x200:rate=10:duration=1",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "too_wide.mp4"))
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(d / "audio_only.m4a"))
    (d / "corrupt.mp4").write_bytes(np.random.default_rng(0).bytes(50_000))
    return d


# ---------------------------------------------------------------- fake pose detection
def _landmarks(world: np.ndarray, image: np.ndarray, vis: float):
    world_lms = [SimpleNamespace(x=float(x), y=float(y), z=float(z), visibility=vis) for x, y, z in world]
    image_lms = [SimpleNamespace(x=float(x), y=float(y), z=float(z), visibility=vis) for x, y, z in image]
    return image_lms, world_lms


class FakeLandmarker:
    """Stands in for MediaPipe's PoseLandmarker: pose_fn(t_seconds) -> world (33,3) or None."""

    def __init__(self, pose_fn):
        self.pose_fn = pose_fn
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def detect_for_video(self, _image, timestamp_ms):
        self.calls += 1
        world = self.pose_fn(timestamp_ms / 1000.0)
        if world is None:
            return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
        image_lms, world_lms = _landmarks(world, project(world[None])[0], 0.95)
        return SimpleNamespace(pose_landmarks=[image_lms], pose_world_landmarks=[world_lms])


class PoseScript:
    """What the fake 'camera' sees. Tests set .attempt to describe the learner's performance."""

    def __init__(self):
        self.choreo = make_choreography(0)
        self.reference = lambda t: self.choreo.pose(np.array([t]))[0]
        self.attempt = lambda t: None

    def factory(self, kind):
        return FakeLandmarker(self.reference if kind == "reference" else self.attempt)


@pytest.fixture
def poses():
    return PoseScript()


# ---------------------------------------------------------------- app
@pytest.fixture
def make_app(tmp_path, poses, monkeypatch):
    """Factory for isolated apps (own SQLite file and storage dir), migrated with Alembic."""
    monkeypatch.delenv("MOVEMATCH_PUBLIC_DEMO", raising=False)
    monkeypatch.delenv("MOVEMATCH_ENV", raising=False)
    from flask_migrate import upgrade

    import security
    from app import create_app

    created = []

    def _make(**overrides):
        n = len(created)
        config = {
            "TESTING": True,
            "JOBS_SYNC": True,
            "SECRET_KEY": "test-secret-" + "x" * 32,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / f'test{n}.sqlite3'}",
            "STORAGE_DIR": tmp_path / f"storage{n}",
            "WTF_CSRF_ENABLED": False,
            "LANDMARKER_FACTORY": poses.factory,
            "MODEL_PATH": tmp_path / "no-model.task",
        }
        config.update(overrides)
        app = create_app(config)
        with app.app_context():
            upgrade(directory=str(ROOT / "migrations"))
        security.rate_limiter.reset()
        created.append(app)
        return app

    yield _make
    from extensions import db
    for app in created:
        with app.app_context():
            db.engine.dispose()


@pytest.fixture
def app(make_app):
    return make_app()


@pytest.fixture
def client(app):
    return app.test_client()


def sync_payload(*, speed: float, start: float, end: float, lead_in_ms: float = 1000.0,
                 recorder_start_ms: float = 500.0, frame_ms: float = 1000 / 30, stall=None) -> dict:
    """What ghost_mode.js would send: reference frames presented from start to end at `speed`.

    stall=(at_ref_s, length_ms) freezes the reference (buffering) for that long.
    """
    samples, events = [], []
    session = recorder_start_ms + lead_in_ms
    ref = start
    events.append({"type": "reference_play", "session_time_ms": session, "reference_time": ref, "playback_rate": speed})
    stalled = False
    while ref <= end:
        samples.append([round(session, 2), round(ref, 4)])
        if stall and not stalled and ref >= stall[0]:
            stalled = True
            session += stall[1]                     # no frames presented while buffering
            continue
        session += frame_ms
        ref += frame_ms / 1000.0 * speed
    events.append({"type": "section_end", "session_time_ms": session, "reference_time": end, "playback_rate": speed})
    return {"recorder_start_ms": recorder_start_ms, "samples": samples, "events": events}
