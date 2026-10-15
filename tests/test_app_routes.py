"""Uploads, attempt saving, retries, JSON errors, deletion, CSRF, ownership and configuration."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import needs_ffmpeg, sync_payload

pytestmark = needs_ffmpeg


def upload(client, path: Path, name: str | None = None):
    with open(path, "rb") as f:
        return client.post("/upload", data={"video_file": (f, name or path.name)},
                           content_type="multipart/form-data")


def attempt_form(media_dir, **overrides):
    data = {
        "recording": (open(media_dir / "attempt.webm", "rb"), "attempt.webm", "video/webm"),
        "speed": "0.5", "section_start": "1.0", "section_end": "4.0", "view_mirrored": "1",
        "follow_mode": "same_side", "upload_id": "upload-1",
        "sync": json.dumps(sync_payload(speed=0.5, start=1.0, end=4.0)),
    }
    data.update(overrides)
    return data


def processed_video(app, client, media_dir):
    from models import Video
    assert upload(client, media_dir / "reference.mp4").status_code == 303
    with app.app_context():
        return Video.query.one().id


# ---------------------------------------------------------------- reference uploads
def test_upload_processes_in_background_keeps_audio_and_constant_frame_rate(app, client, media_dir):
    import media
    from landmark_store import load_landmarks
    from models import Video

    response = upload(client, media_dir / "reference.mp4")
    assert response.status_code == 303 and "/videos/" in response.headers["Location"]
    with app.app_context():
        video = Video.query.one()
        assert video.status == "completed", video.error_message
        playback = Path(app.config["STORAGE_DIR"]) / "playback" / video.playback_filename
        info = media.probe(playback)
        assert info.has_audio and video.has_audio                     # 2.1: the music is kept
        assert not info.variable_frame_rate
        seq = load_landmarks(Path(app.config["STORAGE_DIR"]) / video.landmarks_path)
        assert seq.detected.all() and seq.num_frames == video.frame_count
        # timestamps are exactly frame_index / fps of the file the browser plays
        assert np.allclose(seq.timestamps, np.arange(seq.num_frames) * 1000 / video.fps, atol=1.0)
    status = client.get(f"/api/videos/{video.id}/status").get_json()
    assert status["status"] == "completed" and status["progress"] == 1.0
    lm = client.get(f"/api/videos/{video.id}/landmarks").get_json()
    assert len(lm["frames"]) == len(lm["timestamps_ms"]) == video.frame_count
    assert client.get(f"/videos/{video.id}/media").status_code == 200


def test_variable_frame_rate_upload_is_normalized(app, client, media_dir):
    import media
    from models import Video
    upload(client, media_dir / "vfr.mkv")
    with app.app_context():
        video = Video.query.one()
        assert video.status == "completed", video.error_message
        assert media.probe(media_dir / "vfr.mkv").variable_frame_rate
        info = media.probe(Path(app.config["STORAGE_DIR"]) / "playback" / video.playback_filename)
        assert not info.variable_frame_rate


@pytest.mark.parametrize("fixture,message", [
    ("corrupt.mp4", "could not be read"),
    ("audio_only.m4a", "Unsupported file type"),
    ("too_wide.mp4", "at most 1920 pixels"),
])
def test_bad_uploads_are_rejected_and_removed(app, client, media_dir, fixture, message):
    from models import Video
    response = upload(client, media_dir / fixture)
    assert response.status_code == 302
    page = client.get("/").get_data(as_text=True)
    assert message in page
    with app.app_context():
        assert Video.query.count() == 0
    assert not any((Path(app.config["STORAGE_DIR"]) / "uploads").glob("*"))


def test_extension_is_not_trusted(app, client, media_dir):
    """A non-video renamed to .mp4 is caught by inspecting the contents."""
    from models import Video
    upload(client, media_dir / "audio_only.m4a", name="sneaky.mp4")
    assert "no video track" in client.get("/").get_data(as_text=True)
    with app.app_context():
        assert Video.query.count() == 0


def test_too_long_video_is_rejected(make_app, media_dir):
    app = make_app(MAX_REFERENCE_SECONDS=3.0)
    client = app.test_client()
    upload(client, media_dir / "reference.mp4")
    assert "MoveMatch works with short clips" in client.get("/").get_data(as_text=True)


def test_oversized_upload_returns_friendly_errors(make_app, media_dir):
    app = make_app(MAX_REFERENCE_MB=1, MAX_ATTEMPT_MB=1)
    client = app.test_client()
    big = io.BytesIO(b"0" * (2 * 1024 * 1024))
    response = client.post("/upload", data={"video_file": (big, "big.mp4")}, content_type="multipart/form-data")
    assert response.status_code == 303                       # HTML form: redirect with a message
    assert "larger than the 1 MB limit" in client.get("/").get_data(as_text=True)


def test_oversized_recording_gets_json_413(make_app, media_dir):
    app = make_app(MAX_ATTEMPT_MB=1)
    client = app.test_client()
    video_id = processed_video(app, client, media_dir)
    form = attempt_form(media_dir, recording=(io.BytesIO(b"0" * (2 * 1024 * 1024)), "a.webm", "video/webm"))
    response = client.post(f"/videos/{video_id}/attempts", data=form, content_type="multipart/form-data")
    assert response.status_code == 413
    assert "limit" in response.get_json()["error"]


def test_missing_model_fails_with_setup_instructions(make_app, media_dir, tmp_path):
    app = make_app(LANDMARKER_FACTORY=None, MODEL_PATH=tmp_path / "missing.task")
    client = app.test_client()
    from models import Video
    upload(client, media_dir / "reference.mp4")
    with app.app_context():
        video = Video.query.one()
        assert video.status == "failed"
        assert "pose model is not installed" in video.error_message
    page = client.get(f"/videos/{video.id}").get_data(as_text=True)
    assert "Try processing again" in page
    # retry from the UI works once the problem is fixed
    from conftest import PoseScript
    app.config["LANDMARKER_FACTORY"] = PoseScript().factory
    client.post(f"/videos/{video.id}/reprocess")
    with app.app_context():
        assert Video.query.one().status == "completed"


def test_internal_errors_are_not_shown_to_visitors(app, client, media_dir, monkeypatch):
    import jobs
    from models import Video

    def boom(*a, **k):
        raise RuntimeError("secret internal path /srv/private/thing")
    monkeypatch.setattr(jobs, "extract_landmarks", boom)
    upload(client, media_dir / "reference.mp4")
    with app.app_context():
        video = Video.query.one()
    page = client.get(f"/videos/{video.id}").get_data(as_text=True)
    assert "Processing failed unexpectedly" in page and "/srv/private" not in page
    # failed processing leaves no partial outputs
    assert not any((Path(app.config["STORAGE_DIR"]) / "playback").glob("*"))


# ---------------------------------------------------------------- attempts
def test_attempt_is_saved_processed_and_replayable(app, client, media_dir, poses):
    video_id = processed_video(app, client, media_dir)
    poses.attempt = lambda t: poses.choreo.pose(np.array([max(0.0, 1.0 + (t - 1.0) * 0.5)]))[0]
    response = client.post(f"/videos/{video_id}/attempts", data=attempt_form(media_dir),
                           content_type="multipart/form-data")
    assert response.status_code == 201, response.get_json()
    attempt_id = response.get_json()["attempt_id"]
    data = client.get(f"/api/attempts/{attempt_id}").get_json()
    assert data["status"] in ("ready", "withheld"), data["error"]
    assert data["recording_ready"] and data["replay_map"]
    assert data["feedback"]["algorithm_version"]
    # the raw WebM is deleted once the seekable replay exists
    assert not any((Path(app.config["STORAGE_DIR"]) / "attempts").glob("*"))
    replay = client.get(data["recording_url"])
    assert replay.status_code == 200 and replay.mimetype == "video/mp4"
    assert client.get(f"/attempts/{attempt_id}").status_code == 200


def test_retrying_an_upload_does_not_duplicate_the_attempt(app, client, media_dir):
    from models import Attempt
    video_id = processed_video(app, client, media_dir)
    first = client.post(f"/videos/{video_id}/attempts", data=attempt_form(media_dir), content_type="multipart/form-data")
    second = client.post(f"/videos/{video_id}/attempts", data=attempt_form(media_dir), content_type="multipart/form-data")
    assert first.status_code == 201 and second.status_code == 200
    assert second.get_json()["duplicate"] and second.get_json()["attempt_id"] == first.get_json()["attempt_id"]
    with app.app_context():
        assert Attempt.query.count() == 1


@pytest.mark.parametrize("field,value,error", [
    ("speed", "2", "Unsupported practice speed"),
    ("speed", "nan", "Invalid value"),
    ("follow_mode", "sideways", "dance-side"),
    ("section_end", "1.2", "Invalid practice section"),
    ("sync", "{not json", "not valid JSON"),
])
def test_attempt_validation_errors_are_json(app, client, media_dir, field, value, error):
    video_id = processed_video(app, client, media_dir)
    response = client.post(f"/videos/{video_id}/attempts", data=attempt_form(media_dir, **{field: value}),
                           content_type="multipart/form-data")
    assert response.status_code == 400
    assert error in response.get_json()["error"]


def test_corrupt_recording_fails_with_message_and_can_be_deleted(app, client, media_dir):
    video_id = processed_video(app, client, media_dir)
    form = attempt_form(media_dir, recording=(open(media_dir / "corrupt.mp4", "rb"), "a.webm", "video/webm"))
    attempt_id = client.post(f"/videos/{video_id}/attempts", data=form,
                             content_type="multipart/form-data").get_json()["attempt_id"]
    data = client.get(f"/api/attempts/{attempt_id}").get_json()
    assert data["status"] == "failed" and "could not be read" in data["error"]
    assert client.post(f"/attempts/{attempt_id}/delete").status_code == 303
    assert client.get(f"/api/attempts/{attempt_id}").status_code == 404


def test_attempt_without_sync_data_is_replayable_but_feedback_withheld(app, client, media_dir):
    video_id = processed_video(app, client, media_dir)
    attempt_id = client.post(f"/videos/{video_id}/attempts", data=attempt_form(media_dir, sync="{}"),
                             content_type="multipart/form-data").get_json()["attempt_id"]
    data = client.get(f"/api/attempts/{attempt_id}").get_json()
    assert data["status"] == "withheld" and data["recording_ready"]
    assert data["feedback"]["withheld"]


def test_recording_requires_processed_reference(app, client, media_dir):
    from extensions import db
    from models import Video
    with app.app_context():
        video = Video(original_name="x.mp4", stored_filename="x.mp4", status="processing")
        db.session.add(video)
        db.session.commit()
        vid = video.id
    response = client.post(f"/videos/{vid}/attempts", data=attempt_form(media_dir), content_type="multipart/form-data")
    assert response.status_code == 400 and "processed reference" in response.get_json()["error"]
    assert client.get(f"/videos/{vid}/ghost").status_code == 302


# ---------------------------------------------------------------- deletion, errors, security
def test_delete_video_removes_files_and_attempts(app, client, media_dir):
    video_id = processed_video(app, client, media_dir)
    client.post(f"/videos/{video_id}/attempts", data=attempt_form(media_dir), content_type="multipart/form-data")
    storage = Path(app.config["STORAGE_DIR"])
    assert any(storage.rglob("*.mp4"))
    assert client.post(f"/videos/{video_id}/delete").status_code == 303
    assert not [p for p in storage.rglob("*") if p.is_file()]
    assert client.get(f"/videos/{video_id}").status_code == 404


def test_api_errors_are_json_and_pages_are_html(client):
    response = client.get("/api/attempts/999")
    assert response.status_code == 404 and response.is_json
    page = client.get("/videos/999")
    assert page.status_code == 404 and "text/html" in page.content_type


def test_csrf_is_required_for_posts(make_app, media_dir):
    app = make_app(WTF_CSRF_ENABLED=True)
    client = app.test_client()
    assert client.post("/videos/1/delete").status_code == 400
    response = client.post("/videos/1/attempts", data={}, content_type="multipart/form-data")
    assert response.status_code == 400 and response.get_json()["code"] == "csrf"


def test_public_demo_hides_other_visitors_items(make_app, media_dir):
    app = make_app(PUBLIC_DEMO=True, ALLOW_REFERENCE_UPLOADS=True)
    owner, stranger = app.test_client(), app.test_client()
    video_id = processed_video(app, owner, media_dir)
    attempt_id = owner.post(f"/videos/{video_id}/attempts", data=attempt_form(media_dir),
                            content_type="multipart/form-data").get_json()["attempt_id"]
    assert owner.get(f"/videos/{video_id}").status_code == 200
    for url in (f"/videos/{video_id}", f"/videos/{video_id}/media", f"/api/videos/{video_id}/landmarks",
                f"/attempts/{attempt_id}", f"/attempts/{attempt_id}/recording", f"/api/attempts/{attempt_id}"):
        assert stranger.get(url).status_code == 404, url
    assert stranger.post(f"/videos/{video_id}/delete").status_code == 404
    assert stranger.post(f"/attempts/{attempt_id}/delete").status_code == 404


def test_public_demo_samples_are_readable_but_not_deletable(make_app, media_dir):
    from extensions import db
    from models import Video
    app = make_app(PUBLIC_DEMO=True, ALLOW_REFERENCE_UPLOADS=True)
    owner, visitor = app.test_client(), app.test_client()
    video_id = processed_video(app, owner, media_dir)
    with app.app_context():
        db.session.get(Video, video_id).is_sample = True
        db.session.commit()
    assert visitor.get(f"/videos/{video_id}").status_code == 200
    assert visitor.post(f"/videos/{video_id}/delete").status_code == 404
    assert visitor.post(f"/videos/{video_id}/attempts", data=attempt_form(media_dir),
                        content_type="multipart/form-data").status_code == 201


def test_public_demo_disables_reference_uploads_by_default(make_app, media_dir):
    app = make_app(PUBLIC_DEMO=True, ALLOW_REFERENCE_UPLOADS=False)
    assert upload(app.test_client(), media_dir / "reference.mp4").status_code == 403


def test_upload_rate_limit(make_app, media_dir):
    app = make_app(REFERENCE_UPLOADS_PER_HOUR=1)
    client = app.test_client()
    upload(client, media_dir / "reference.mp4")
    upload(client, media_dir / "reference.mp4")
    assert "Upload limit reached" in client.get("/").get_data(as_text=True)


def test_retention_purges_old_attempts(make_app, media_dir):
    from datetime import datetime, timedelta

    import jobs
    from extensions import db
    from models import Attempt
    app = make_app(RETENTION_HOURS=1)
    client = app.test_client()
    video_id = processed_video(app, client, media_dir)
    client.post(f"/videos/{video_id}/attempts", data=attempt_form(media_dir), content_type="multipart/form-data")
    with app.app_context():
        attempt = Attempt.query.one()
        attempt.created_at = datetime.utcnow() - timedelta(hours=2)
        db.session.commit()
        assert jobs.purge_expired() >= 1
        assert Attempt.query.count() == 0


def test_security_headers(client):
    response = client.get("/")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]


def test_production_requires_a_real_secret(monkeypatch):
    from config import ConfigError, load_config
    monkeypatch.setenv("MOVEMATCH_ENV", "production")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(ConfigError):
        load_config()
    monkeypatch.setenv("SECRET_KEY", "a" * 40)
    monkeypatch.setenv("MOVEMATCH_DEBUG", "1")
    with pytest.raises(ConfigError):
        load_config()
    monkeypatch.setenv("MOVEMATCH_DEBUG", "0")
    assert load_config()["SESSION_COOKIE_SECURE"] is True


def test_unmigrated_database_shows_setup_page(tmp_path, monkeypatch):
    monkeypatch.delenv("MOVEMATCH_ENV", raising=False)
    from app import create_app
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'empty.sqlite3'}",
                      "STORAGE_DIR": tmp_path / "s", "SECRET_KEY": "x" * 40})
    response = app.test_client().get("/")
    assert response.status_code == 503 and "flask db upgrade" in response.get_data(as_text=True)
