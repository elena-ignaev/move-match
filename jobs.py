"""Background processing, kept out of the upload request.

Uploading a reference only saves and checks the file; transcoding and pose extraction run here,
on a small thread pool, and write progress to the database for the page to poll. Attempts are
processed the same way: transcode for replay, extract the learner's landmarks, then compute
feedback. A failure leaves a message that is safe to show (details go to the server log) and the
original file, so the job can be retried from the UI.

A thread pool is the simplest thing that works for one person on a laptop or a small demo. Jobs
interrupted by a restart are picked up again on the next start (see recover_interrupted).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from flask import current_app

import feedback as fb
import media
from extensions import db
from landmark_store import LandmarkSchemaError, is_current, load_landmarks
from models import Attempt, Evaluation, Video
from process_video import ModelMissingError, extract_landmarks, require_model
from sync_timeline import SyncError, build_timeline

log = logging.getLogger("movematch.jobs")

GENERIC_FAILURE = ("Processing failed unexpectedly. Try again, or use a shorter MP4 (H.264) clip. "
                   "The details are in the server log.")


# --------------------------------------------------------------------------- storage layout
def storage(kind: str) -> Path:
    """storage/<kind>/: uploads, playback, landmarks, attempts (raw), replays."""
    path = Path(current_app.config["STORAGE_DIR"]) / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def storage_path(relative: str | None) -> Path | None:
    return None if not relative else Path(current_app.config["STORAGE_DIR"]) / relative


def relative_to_storage(path: Path) -> str:
    return Path(path).resolve().relative_to(Path(current_app.config["STORAGE_DIR"]).resolve()).as_posix()


def remove_files(*paths: Path | None) -> None:
    for p in paths:
        if p is not None:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                log.warning("Could not delete %s", p, exc_info=True)


def landmarker_factory(kind: str):
    """Test hook: config LANDMARKER_FACTORY(kind) -> factory; None means MediaPipe."""
    custom = current_app.config.get("LANDMARKER_FACTORY")
    return (lambda: custom(kind)) if custom else None


def model_path() -> Path:
    return Path(current_app.config["MODEL_PATH"])


# --------------------------------------------------------------------------- runner
class JobRunner:
    """One runner per Flask app (kept in app.extensions), so several apps never share a pool."""

    def __init__(self, app) -> None:
        self.app = app
        self.executor = None
        if not app.config.get("JOBS_SYNC"):
            self.executor = ThreadPoolExecutor(max_workers=max(1, app.config.get("JOB_WORKERS", 1)),
                                               thread_name_prefix="movematch-job")

    def submit(self, fn, *args) -> None:
        if self.executor is None:
            self._run(fn, args)
        else:
            self.executor.submit(self._run, fn, args)

    def _run(self, fn, args) -> None:
        with self.app.app_context():
            try:
                fn(*args)
            except Exception:                       # pragma: no cover - last-resort guard
                log.exception("Job %s%s crashed", fn.__name__, args)
            finally:
                db.session.remove()


class _Jobs:
    """Facade: jobs.init_app(app) once, then jobs.submit(...) inside any app or request context."""

    def init_app(self, app) -> None:
        app.extensions["movematch_jobs"] = JobRunner(app)

    def submit(self, fn, *args) -> None:
        current_app.extensions["movematch_jobs"].submit(fn, *args)


jobs = _Jobs()


class _Progress:
    """Throttled progress writer (at most every 0.4 s or 2 %)."""

    def __init__(self, obj, lo: float, hi: float) -> None:
        self.obj, self.lo, self.hi = obj, lo, hi
        self._last_t, self._last_p = 0.0, -1.0

    def __call__(self, fraction: float) -> None:
        p = self.lo + (self.hi - self.lo) * float(fraction)
        now = time.monotonic()
        if p - self._last_p >= 0.02 or now - self._last_t >= 0.4 or fraction >= 1.0:
            self.obj.progress = round(p, 3)
            db.session.commit()
            self._last_t, self._last_p = now, p


# --------------------------------------------------------------------------- reference videos
def process_reference(video_id: int) -> None:
    video = db.session.get(Video, video_id)
    if video is None:
        return
    cfg = current_app.config
    src = storage("uploads") / video.stored_filename
    stem = Path(video.stored_filename).stem
    playback = storage("playback") / f"{stem}.mp4"
    landmarks = storage("landmarks") / f"{stem}.npz"

    def step(progress: float, message: str) -> None:
        video.status, video.progress, video.progress_message = "processing", progress, message
        video.error_message = None
        db.session.commit()

    try:
        step(0.02, "Checking the video")
        if landmarker_factory("reference") is None:
            require_model(model_path())
        info = media.probe(src)
        media.validate(info, max_seconds=cfg["MAX_REFERENCE_SECONDS"], max_dimension=cfg["MAX_SOURCE_DIMENSION"])
        fps = media.target_fps(info, cfg["PLAYBACK_MAX_FPS"])

        step(0.05, "Converting for the browser (keeping the music)")
        media.transcode_reference(src, playback, info=info, fps=fps, max_height=cfg["PLAYBACK_MAX_HEIGHT"],
                                  max_seconds=cfg["MAX_REFERENCE_SECONDS"], progress=_Progress(video, 0.05, 0.3))

        step(0.3, "Finding the dancer in each frame")
        stats = extract_landmarks(playback, landmarks, landmarker_factory=landmarker_factory("reference"),
                                  progress=_Progress(video, 0.3, 0.98), model_path=model_path())
        played = media.probe(playback)

        video.playback_filename = playback.name
        video.landmarks_path = relative_to_storage(landmarks)
        video.frame_count = stats["frame_count"]
        video.fps = stats["fps"]
        video.width, video.height = stats["width"], stats["height"]
        video.duration_s = played.duration_s
        video.has_audio = played.has_audio
        video.detected_ratio = stats["detected_ratio"]
        video.status, video.progress = "completed", 1.0
        video.progress_message = None
        if stats["detected_ratio"] < 0.5:
            video.progress_message = (f"The dancer was found in only {stats['detected_ratio']:.0%} of frames. "
                                      "Feedback will be withheld where the reference cannot be tracked.")
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        _fail(video, exc, playback, landmarks)


def _user_message(exc: Exception) -> str:
    if isinstance(exc, media.MediaError):
        return str(exc)
    if isinstance(exc, ModelMissingError):
        return "The pose model is not installed on this server (see docs/setup.md, 'Download the model')."
    if isinstance(exc, media.ToolMissingError):
        return "FFmpeg is not installed on this server (see docs/setup.md)."
    if isinstance(exc, SyncError):
        return str(exc)
    return GENERIC_FAILURE


def _fail(obj, exc: Exception, *partial_files: Path) -> None:
    if not isinstance(exc, (media.MediaError, SyncError)):
        log.exception("%s %s failed", type(obj).__name__, getattr(obj, "id", "?"))
    else:
        log.info("%s %s rejected: %s", type(obj).__name__, getattr(obj, "id", "?"), exc)
    remove_files(*partial_files)
    obj.status = "failed"
    obj.error_message = _user_message(exc)
    if isinstance(obj, Video):
        obj.progress_message = None
        obj.playback_filename = None
        obj.landmarks_path = None
    db.session.commit()


# --------------------------------------------------------------------------- attempts
def process_attempt(attempt_id: int) -> None:
    attempt = db.session.get(Attempt, attempt_id)
    if attempt is None:
        return
    video = attempt.video
    cfg = current_app.config
    raw = storage("attempts") / attempt.recording_filename if attempt.recording_filename else None
    stem = Path(attempt.recording_filename or attempt.replay_filename or f"attempt_{attempt.id}").stem
    replay = storage("replays") / f"{stem}.mp4"
    landmarks = storage("landmarks") / f"attempt_{stem}.npz"
    attempt.status, attempt.progress, attempt.error_message = "processing", 0.02, None
    db.session.commit()

    try:
        if landmarker_factory("attempt") is None:
            require_model(model_path())
        details = dict(attempt.details_json or {})

        # 1. Seekable constant-frame-rate replay file (once; reevaluation reuses it).
        if raw is not None and raw.is_file():
            info = media.probe(raw)
            media.validate(info, max_seconds=cfg["MAX_ATTEMPT_SECONDS"], max_dimension=cfg["MAX_SOURCE_DIMENSION"],
                           min_seconds=0.5)
            media.transcode_recording(raw, replay, info=info, fps=30, max_height=cfg["PLAYBACK_MAX_HEIGHT"],
                                      progress=_Progress(attempt, 0.02, 0.25))
            details["recording_start_offset_s"] = info.start_time_s
            details["recording_duration_s"] = info.duration_s
            details["recording_variable_frame_rate"] = info.variable_frame_rate
            attempt.replay_filename = replay.name
            attempt.recording_filename = None
            attempt.details_json = details
            db.session.commit()                 # persist before deleting the raw file
            remove_files(raw)                   # the raw WebM is an intermediate: the replay replaces it
        elif not (attempt.replay_filename and (storage("replays") / attempt.replay_filename).is_file()):
            raise media.MediaError("The recording file is missing, so it cannot be processed.")
        replay = storage("replays") / attempt.replay_filename

        # 2. Replay map: recording time -> reference time, so replay works even if feedback fails.
        timeline = build_timeline(details, recording_offset_s=details.get("recording_start_offset_s", 0.0))
        duration = details.get("recording_duration_s") or media.probe(replay).duration_s
        details["replay_map"] = timeline.replay_map(duration)
        details["sync_source"] = timeline.source
        attempt.details_json = details
        attempt.progress = 0.3
        db.session.commit()

        # 3. Learner landmarks.
        if not is_current(landmarks):
            extract_landmarks(replay, landmarks, landmarker_factory=landmarker_factory("attempt"),
                              progress=_Progress(attempt, 0.3, 0.9), model_path=model_path())
        attempt.landmarks_path = relative_to_storage(landmarks)
        db.session.commit()

        # 4. Feedback.
        ref_path = storage_path(video.landmarks_path)
        if video.status != "completed" or ref_path is None or not ref_path.is_file():
            raise media.MediaError("The reference video has not been processed yet. Reprocess it, then re-evaluate.")
        ref = load_landmarks(ref_path)
        learner = load_landmarks(landmarks)
        mapped = timeline.reference_times(learner.timestamps / 1000.0)
        end = attempt.section_end_s if attempt.section_end_s else (video.duration_s or float(ref.timestamps[-1]) / 1000)
        result = fb.evaluate_attempt(
            ref, learner, mapped, section=(attempt.section_start_s or 0.0, end), speed=attempt.speed,
            follow_mode=attempt.follow_mode, capture_mirrored=attempt.capture_mirrored)
        save_evaluation(attempt, result)
        attempt.status = "withheld" if result["withheld"] else "ready"
        attempt.progress = 1.0
        db.session.commit()
    except SyncError as exc:
        db.session.rollback()
        attempt = db.session.get(Attempt, attempt_id)
        save_evaluation(attempt, {"algorithm_version": fb.ALGORITHM_VERSION, "withheld": True,
                                  "withheld_reasons": [str(exc)], "headline": f"Feedback withheld: {exc}"})
        attempt.status, attempt.progress = "withheld", 1.0
        db.session.commit()
    except LandmarkSchemaError as exc:
        db.session.rollback()
        log.warning("Stale landmark file for attempt %s: %s", attempt_id, exc)
        _fail(db.session.get(Attempt, attempt_id),
              media.MediaError("The reference landmarks are out of date. Reprocess the reference video, then re-evaluate."))
    except Exception as exc:
        db.session.rollback()
        _fail(db.session.get(Attempt, attempt_id), exc, replay if not attempt.replay_filename else None)


def save_evaluation(attempt: Attempt, result: dict) -> Evaluation:
    evaluation = attempt.evaluation or Evaluation(attempt_id=attempt.id, video_id=attempt.video_id)
    posture = result.get("posture") or {}
    timing = result.get("timing") or {}
    evaluation.algorithm_version = result.get("algorithm_version", fb.ALGORITHM_VERSION)
    evaluation.withheld = bool(result.get("withheld"))
    evaluation.tracking_status = (result.get("tracking") or {}).get("status")
    evaluation.timing_offset_s = timing.get("offset_s")
    evaluation.timing_confident = timing.get("confident")
    evaluation.moments_total = posture.get("moments_total")
    evaluation.moments_close = posture.get("close")
    evaluation.moments_review = posture.get("review")
    evaluation.median_angle_diff_deg = posture.get("median_abs_diff_deg")
    evaluation.details_json = _jsonable(result)
    evaluation.created_at = datetime.now(timezone.utc)
    db.session.add(evaluation)
    return evaluation


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# --------------------------------------------------------------------------- deletion and housekeeping
def delete_attempt_files(attempt: Attempt) -> None:
    remove_files(
        storage("attempts") / attempt.recording_filename if attempt.recording_filename else None,
        storage("replays") / attempt.replay_filename if attempt.replay_filename else None,
        storage_path(attempt.landmarks_path),
    )


def delete_video_files(video: Video) -> None:
    for attempt in video.attempts:
        delete_attempt_files(attempt)
    remove_files(
        storage("uploads") / video.stored_filename,
        storage("playback") / video.playback_filename if video.playback_filename else None,
        storage_path(video.landmarks_path),
    )


def recover_interrupted() -> None:
    """Re-queue work that was running when the server stopped."""
    for video in Video.query.filter(Video.status.in_(("queued", "processing"))).all():
        if (storage("uploads") / video.stored_filename).is_file():
            video.status, video.progress, video.progress_message = "queued", 0.0, "Waiting to restart processing"
            db.session.commit()
            jobs.submit(process_reference, video.id)
        else:
            video.status, video.error_message = "failed", "The uploaded file is missing. Upload it again."
            db.session.commit()
    # "uploaded" covers attempts saved by the prototype (they use the legacy event timeline) and
    # uploads whose job was lost in a restart.
    for attempt in Attempt.query.filter(Attempt.status.in_(("uploaded", "processing"))).all():
        attempt.status = "uploaded"
        db.session.commit()
        jobs.submit(process_attempt, attempt.id)


def purge_expired() -> int:
    """Public demo: delete visitors' attempts and videos older than RETENTION_HOURS."""
    hours = current_app.config.get("RETENTION_HOURS") or 0
    if hours <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_naive = cutoff.replace(tzinfo=None)                 # SQLite returns naive datetimes
    removed = 0
    for attempt in Attempt.query.filter(Attempt.created_at < cutoff_naive).all():
        delete_attempt_files(attempt)
        db.session.delete(attempt)
        removed += 1
    for video in Video.query.filter(Video.is_sample.is_(False), Video.created_at < cutoff_naive).all():
        delete_video_files(video)
        db.session.delete(video)
        removed += 1
    db.session.commit()
    if removed:
        log.info("Retention: deleted %d expired items", removed)
    return removed
