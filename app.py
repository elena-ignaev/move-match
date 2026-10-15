"""MoveMatch web app.

Journey: choose a short reference -> practise a section (slow, looped, mirrored, with a skeleton
overlay) -> record an attempt with a countdown -> replay it next to the reference -> read feedback
(timing offset, joint angles at key moments, tracking problems) -> retry and compare.

Run locally:  flask db upgrade && flask run      (see docs/setup.md)
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from pathlib import Path

from flask import (Flask, abort, flash, jsonify, redirect, render_template, request, send_from_directory,
                   url_for)
from flask_wtf.csrf import CSRFError, generate_csrf
from sqlalchemy import event
from sqlalchemy.engine import Engine
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

import feedback as fb
import jobs as jobs_module
import media
import security
from config import BASE_DIR, load_config
from extensions import csrf, db, migrate
from jobs import jobs, storage, storage_path
from landmark_store import load_landmarks
from models import Attempt, Video
from sync_timeline import SyncError, build_timeline, sanitize_payload

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "webm", "mkv", "m4v"}
ALLOWED_SPEEDS = (0.25, 0.5, 0.75, 1.0)
FOLLOW_MODES = ("same_side", "mirror")
ATTEMPT_EXTENSIONS_BY_MIME = {"video/webm": ".webm", "video/mp4": ".mp4", "video/x-matroska": ".mkv"}

log = logging.getLogger("movematch")


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record):
    """WAL lets the page poll progress while a background job writes."""
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_app(overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(load_config(overrides))
    app.config.setdefault("SQLALCHEMY_ENGINE_OPTIONS", {"connect_args": {"timeout": 15}}
                          if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite") else {})
    Path(app.config["STORAGE_DIR"]).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    migrate.init_app(app, db, directory=str(BASE_DIR / "migrations"))
    csrf.init_app(app)
    jobs.init_app(app)

    import dev_tools
    dev_tools.init_app(app)

    app.after_request(security.security_headers)
    register_cli(app)
    register_error_handlers(app)
    register_routes(app)
    register_template_helpers(app)
    app.extensions["movematch_first_request_done"] = False

    @app.before_request
    def _first_request():
        if not app.extensions["movematch_first_request_done"]:
            app.extensions["movematch_first_request_done"] = True
            problem = schema_problem()
            app.extensions["movematch_schema_problem"] = problem
            if problem is None and not app.config.get("TESTING"):
                jobs_module.recover_interrupted()
        if app.extensions.get("movematch_schema_problem") and request.endpoint != "static":
            return render_template("setup_needed.html", problem=app.extensions["movematch_schema_problem"]), 503
        if app.config.get("RETENTION_HOURS"):
            maybe_purge(app)
        return None

    return app


def schema_problem() -> str | None:
    """None if the database is at the latest migration, else what to run."""
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    script = ScriptDirectory(str(BASE_DIR / "migrations"))
    heads = set(script.get_heads())
    with db.engine.connect() as conn:
        current = set(MigrationContext.configure(conn).get_current_heads())
    if current == heads:
        return None
    if not current:
        return ("The database has not been created yet. Stop the server and run `flask db upgrade`. "
                "If you have a database from the first prototype, see docs/setup.md (Existing databases).")
    return f"The database is at revision {', '.join(sorted(current))}. Run `flask db upgrade`."


_last_purge = {"t": 0.0}


def maybe_purge(app) -> None:
    import time
    if time.monotonic() - _last_purge["t"] > 600:
        _last_purge["t"] = time.monotonic()
        jobs.submit(jobs_module.purge_expired)


# --------------------------------------------------------------------------- helpers
def wants_json() -> bool:
    if request.path.startswith("/api/") or request.path.endswith("/attempts"):
        return True
    best = request.accept_mimetypes.best_match(["application/json", "text/html"])
    return best == "application/json" and request.accept_mimetypes[best] > request.accept_mimetypes["text/html"]


def json_error(message: str, status: int, **extra):
    return jsonify({"error": message, **extra}), status


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def attempt_extension(recording_file) -> str:
    mime_type = (recording_file.mimetype or "").split(";")[0].strip().lower()
    if mime_type in ATTEMPT_EXTENSIONS_BY_MIME:
        return ATTEMPT_EXTENSIONS_BY_MIME[mime_type]
    suffix = Path(secure_filename(recording_file.filename or "")).suffix.lower()
    return suffix if suffix in {".webm", ".mp4", ".mkv"} else ".webm"


def _float_arg(form, name: str, default=None):
    raw = form.get(name)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(name)
    if not math.isfinite(value):
        raise ValueError(name)
    return value


def _bool_arg(form, name: str, default: bool) -> bool:
    raw = form.get(name)
    if raw is None:
        return default
    return str(raw).lower() in {"1", "true", "yes", "on"}


def video_summary(video: Video) -> dict:
    return {
        "id": video.id, "status": video.status, "progress": round(video.progress or 0.0, 3),
        "message": video.progress_message, "error": video.error_message,
        "detail_url": url_for("video_detail", video_id=video.id),
        "ready": video.status == "completed",
    }


def evaluation_details(attempt: Attempt) -> dict | None:
    return attempt.evaluation.details_json if attempt.evaluation else None


def previous_attempt(attempt: Attempt) -> Attempt | None:
    """The most recent earlier attempt at the same section that has feedback."""
    query = security.visible_attempts().filter(
        Attempt.video_id == attempt.video_id, Attempt.id < attempt.id,
        Attempt.status.in_(("ready", "withheld")))
    for other in query.order_by(Attempt.id.desc()).all():
        if same_section(other, attempt):
            return other
    return None


def same_section(a: Attempt, b: Attempt) -> bool:
    return (abs((a.section_start_s or 0) - (b.section_start_s or 0)) < 0.05
            and abs((a.section_end_s or 0) - (b.section_end_s or 0)) < 0.05)


def attempt_payload(attempt: Attempt) -> dict:
    details = attempt.details_json or {}
    evaluation = evaluation_details(attempt)
    prev = previous_attempt(attempt)
    comparison = fb.compare_attempts(evaluation, evaluation_details(prev)) if prev else []
    return {
        "attempt_id": attempt.id,
        "video_id": attempt.video_id,
        "status": attempt.status,
        "progress": round(attempt.progress or 0.0, 3),
        "error": attempt.error_message,
        "speed": attempt.speed,
        "section": [attempt.section_start_s, attempt.section_end_s],
        "view_mirrored": attempt.view_mirrored,
        "follow_mode": attempt.follow_mode,
        "created_at": attempt.created_at.isoformat() if attempt.created_at else None,
        "recording_url": url_for("attempt_recording", attempt_id=attempt.id),
        "recording_ready": bool(attempt.replay_filename),
        "replay_map": details.get("replay_map", []),
        "feedback": evaluation,
        "previous_attempt": ({"id": prev.id, "url": url_for("attempt_detail", attempt_id=prev.id)} if prev else None),
        "comparison": comparison,
        "page_url": url_for("attempt_detail", attempt_id=attempt.id),
        "status_url": url_for("api_attempt", attempt_id=attempt.id),
        # kept for clients of the first prototype
        "details": {k: v for k, v in details.items() if k not in ("sync", "replay_map")},
    }


# --------------------------------------------------------------------------- command line
def register_cli(app: Flask) -> None:
    import shutil

    import click

    @app.cli.command("add-sample")
    @click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
    @click.option("--name", help="Title shown in the app (defaults to the file name).")
    def add_sample(path: Path, name: str | None):
        """Add a prepared reference video that every visitor can see (processed immediately).

        Only add footage you have the right to share (e.g. you recorded it yourself).
        """
        if not allowed_file(path.name):
            raise click.ClickException("Unsupported file type.")
        unique = f"{uuid.uuid4().hex}{path.suffix.lower()}"
        shutil.copyfile(path, storage("uploads") / unique)
        video = Video(original_name=name or path.name, stored_filename=unique, status="queued", is_sample=True)
        db.session.add(video)
        db.session.commit()
        click.echo(f"Processing {path.name} …")
        jobs_module.process_reference(video.id)
        db.session.refresh(video)
        if video.status != "completed":
            raise click.ClickException(f"Processing failed: {video.error_message}")
        click.echo(f"Added sample video {video.id}: {video.original_name} "
                   f"({video.duration_s:.1f} s, dancer tracked in {video.detected_ratio:.0%} of frames)")

    @app.cli.command("purge-expired")
    def purge_expired():
        """Delete visitors' uploads and attempts older than MOVEMATCH_RETENTION_HOURS."""
        click.echo(f"Deleted {jobs_module.purge_expired()} expired item(s).")


# --------------------------------------------------------------------------- errors
def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        message = f"The file is larger than the {limit} MB limit."
        if wants_json():
            return json_error(message, 413)
        flash(message, "error")
        return redirect(url_for("index")), 303

    @app.errorhandler(CSRFError)
    def csrf_failed(error):
        message = "Your session expired or the form was not sent from this site. Reload the page and try again."
        if wants_json():
            return json_error(message, 400, code="csrf")
        return render_template("error.html", message=message, status=400), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        if wants_json():
            return json_error(error.description or error.name, error.code)
        return render_template("error.html", message=error.description, status=error.code), error.code

    @app.errorhandler(Exception)
    def unexpected(error):
        log.exception("Unhandled error on %s", request.path)
        message = "Something went wrong on the server. The details were written to the server log."
        if wants_json():
            return json_error(message, 500)
        return render_template("error.html", message=message, status=500), 500


def register_template_helpers(app: Flask) -> None:
    @app.context_processor
    def inject():
        return {
            "csrf_token": generate_csrf,
            "public_demo": app.config["PUBLIC_DEMO"],
            "uploads_allowed": app.config["ALLOW_REFERENCE_UPLOADS"],
            "limits": {
                "reference_mb": app.config["MAX_REFERENCE_MB"],
                "reference_s": app.config["MAX_REFERENCE_SECONDS"],
                "attempt_mb": app.config["MAX_ATTEMPT_MB"],
                "retention_h": app.config["RETENTION_HOURS"],
            },
        }

    @app.template_filter("clock")
    def clock(seconds):
        if seconds is None:
            return "end"
        m, s = divmod(max(0.0, float(seconds)), 60.0)
        return f"{int(m)}:{s:04.1f}"

    @app.template_filter("signed_s")
    def signed_s(value):
        if value is None:
            return "–"
        return "0.00 s" if abs(value) < 0.005 else f"{value:+.2f} s"


# --------------------------------------------------------------------------- routes
def register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        videos = security.visible_videos()
        samples = videos.filter(Video.is_sample.is_(True)).order_by(Video.created_at.desc()).all()
        recent = videos.filter(Video.is_sample.is_(False)).order_by(Video.created_at.desc()).limit(4).all()
        return render_template("home.html", samples=samples, recent_videos=recent)

    @app.route("/library")
    def library():
        videos = security.visible_videos().order_by(Video.created_at.desc()).all()
        return render_template("library.html", videos=videos)

    @app.route("/privacy")
    def privacy():
        return render_template("privacy.html")

    # ---- reference upload and processing
    @app.route("/upload", methods=["POST"])
    def upload():
        if not app.config["ALLOW_REFERENCE_UPLOADS"]:
            abort(403, "Uploading reference videos is turned off on this server. Use a sample video.")
        if not security.rate_limiter.allow("reference_upload", app.config["REFERENCE_UPLOADS_PER_HOUR"]):
            flash("Upload limit reached. Try again in an hour.", "error")
            return redirect(url_for("index"))
        uploaded = request.files.get("video_file")
        if uploaded is None or not uploaded.filename:
            flash("Please choose a video file.", "error")
            return redirect(url_for("index"))
        if not allowed_file(uploaded.filename):
            flash("Unsupported file type. Use MP4, MOV, AVI, WEBM, MKV or M4V.", "error")
            return redirect(url_for("index"))
        if request.content_length and request.content_length > app.config["MAX_REFERENCE_MB"] * 1024 * 1024:
            flash(f"The file is larger than the {app.config['MAX_REFERENCE_MB']} MB limit.", "error")
            return redirect(url_for("index"))

        original_name = secure_filename(uploaded.filename) or "video"
        unique_name = f"{uuid.uuid4().hex}{Path(original_name).suffix.lower()}"
        path = storage("uploads") / unique_name
        uploaded.save(path)
        try:
            info = media.probe(path)             # real contents, not the file name
            media.validate(info, max_seconds=app.config["MAX_REFERENCE_SECONDS"],
                           max_dimension=app.config["MAX_SOURCE_DIMENSION"])
        except media.MediaError as exc:
            jobs_module.remove_files(path)
            flash(str(exc), "error")
            return redirect(url_for("index"))
        except media.ToolMissingError:
            jobs_module.remove_files(path)
            log.exception("FFmpeg missing")
            flash("FFmpeg is not installed on this server (see docs/setup.md).", "error")
            return redirect(url_for("index"))

        video = Video(original_name=original_name, stored_filename=unique_name, status="queued",
                      progress=0.0, progress_message="Waiting to start", owner_id=security.current_owner_id(),
                      duration_s=info.duration_s, width=info.width, height=info.height, has_audio=info.has_audio)
        db.session.add(video)
        db.session.commit()
        jobs.submit(jobs_module.process_reference, video.id)
        flash("Upload received. Processing runs in the background; this page updates by itself.", "success")
        return redirect(url_for("video_detail", video_id=video.id), 303)

    @app.route("/videos/<int:video_id>")
    def video_detail(video_id):
        video = security.get_video_or_404(video_id)
        attempts = security.visible_attempts().filter(Attempt.video_id == video.id) \
            .order_by(Attempt.created_at.desc()).all()
        return render_template("video_detail.html", video=video, attempts=attempts,
                               can_modify=security.can_modify_video(video))

    @app.route("/api/videos/<int:video_id>/status")
    def api_video_status(video_id):
        return jsonify(video_summary(security.get_video_or_404(video_id)))

    @app.route("/videos/<int:video_id>/reprocess", methods=["POST"])
    def reprocess_video(video_id):
        video = security.get_video_or_404(video_id, modify=True)
        if video.status in ("queued", "processing"):
            flash("This video is already being processed.", "error")
        elif not (storage("uploads") / video.stored_filename).is_file():
            flash("The original upload is missing, so it cannot be reprocessed. Upload it again.", "error")
        else:
            video.status, video.progress, video.error_message = "queued", 0.0, None
            video.progress_message = "Waiting to start"
            db.session.commit()
            jobs.submit(jobs_module.process_reference, video.id)
            flash("Reprocessing started.", "success")
        return redirect(url_for("video_detail", video_id=video.id), 303)

    @app.route("/videos/<int:video_id>/delete", methods=["POST"])
    def delete_video(video_id):
        video = security.get_video_or_404(video_id, modify=True)
        if video.status in ("queued", "processing"):
            flash("Wait until processing finishes before deleting this video.", "error")
            return redirect(url_for("video_detail", video_id=video.id), 303)
        name = video.original_name
        jobs_module.delete_video_files(video)
        db.session.delete(video)
        db.session.commit()
        flash(f"Deleted {name} and all of its attempts.", "success")
        return redirect(url_for("library"), 303)

    @app.route("/videos/<int:video_id>/media")
    def video_media(video_id):
        video = security.get_video_or_404(video_id)
        if not video.playback_filename:
            abort(404, "This video has no playable version yet.")
        return send_from_directory(storage("playback"), video.playback_filename, mimetype="video/mp4",
                                   conditional=True, max_age=0)

    @app.route("/videos/<int:video_id>/original")
    def original_video(video_id):
        video = security.get_video_or_404(video_id, modify=True)
        return send_from_directory(storage("uploads"), video.stored_filename, conditional=True,
                                   as_attachment=True, download_name=video.original_name)

    @app.route("/api/videos/<int:video_id>/landmarks")
    def api_video_landmarks(video_id):
        """Image-space landmarks for drawing the reference skeleton on a transparent canvas."""
        video = security.get_video_or_404(video_id)
        path = storage_path(video.landmarks_path)
        if video.status != "completed" or path is None or not path.is_file():
            abort(404, "Landmarks are not available yet.")
        seq = load_landmarks(path)
        frames = []
        for detected, image, vis in zip(seq.detected, seq.image_landmarks, seq.visibility):
            if not detected:
                frames.append(None)
                continue
            frames.append([[round(float(x), 4), round(float(y), 4), round(float(v), 2)]
                           for (x, y, _z), v in zip(image, vis)])
        response = jsonify({
            "fps": seq.metadata.get("extraction_fps"),
            "width": seq.metadata.get("width"), "height": seq.metadata.get("height"),
            "timestamps_ms": [round(float(t), 2) for t in seq.timestamps],
            "frames": frames,
            "connections": __import__("process_video").POSE_CONNECTIONS,
        })
        response.headers["Cache-Control"] = "private, max-age=3600"
        return response

    # ---- practice and recording
    def require_ready(video: Video, what: str):
        if video.status != "completed" or not video.playback_filename:
            flash(f"{what} is available after the video has been processed.", "error")
            return redirect(url_for("video_detail", video_id=video.id))
        return None

    @app.route("/videos/<int:video_id>/practice")
    def video_practice(video_id):
        video = security.get_video_or_404(video_id)
        return require_ready(video, "Practice") or render_template("practice.html", video=video)

    @app.route("/dance-practice/<int:video_id>")
    def dance_practice(video_id):
        return redirect(url_for("video_practice", video_id=video_id), 301)

    @app.route("/videos/<int:video_id>/ghost")
    def ghost_mode(video_id):
        video = security.get_video_or_404(video_id)
        return require_ready(video, "Ghost mode") or render_template(
            "ghost_mode.html", video=video, speeds=ALLOWED_SPEEDS)

    @app.route("/videos/<int:video_id>/attempts", methods=["POST"])
    def create_attempt(video_id):
        video = security.get_video_or_404(video_id)
        if video.status != "completed" or not video.landmarks_path:
            return json_error("Recording needs a processed reference video.", 400)
        if request.content_length and request.content_length > app.config["MAX_ATTEMPT_MB"] * 1024 * 1024:
            return json_error(f"The recording is larger than the {app.config['MAX_ATTEMPT_MB']} MB limit.", 413)

        upload_id = (request.form.get("upload_id") or "").strip()[:64]
        if upload_id:
            existing = Attempt.query.filter_by(upload_id=upload_id).first()
            if existing is not None:              # a retried upload that already succeeded
                if not security.can_view_attempt(existing) or existing.video_id != video.id:
                    return json_error("This upload id belongs to another recording.", 409)
                return jsonify({**attempt_payload(existing), "duplicate": True}), 200

        if not security.rate_limiter.allow("attempt_upload", app.config["ATTEMPT_UPLOADS_PER_HOUR"]):
            return json_error("Recording upload limit reached. Try again later.", 429)

        recording = request.files.get("recording")
        if recording is None or not recording.filename:
            return json_error("No recording was received.", 400)
        try:
            speed = _float_arg(request.form, "speed")
            start = _float_arg(request.form, "section_start", 0.0)
            end = _float_arg(request.form, "section_end", None)
        except ValueError as exc:
            return json_error(f"Invalid value for {exc}.", 400)
        if speed not in ALLOWED_SPEEDS:
            return json_error("Unsupported practice speed.", 400)
        duration = video.duration_s or 0
        if start < 0 or (duration and start >= duration) or (end is not None and end <= start + 0.5):
            return json_error("Invalid practice section.", 400)
        if end is not None and duration:
            end = min(end, duration)
        follow_mode = request.form.get("follow_mode", "same_side")
        if follow_mode not in FOLLOW_MODES:
            return json_error("Unsupported dance-side mode.", 400)

        try:
            raw_sync = json.loads(request.form.get("sync") or "{}")
        except json.JSONDecodeError:
            return json_error("The synchronization data was not valid JSON.", 400)
        details = {
            "sync": sanitize_payload(raw_sync),
            "mime_type": (recording.mimetype or "")[:80],
            "client": {k: (request.form.get(k) or "")[:120] for k in ("user_agent", "camera_label")},
        }
        try:
            timeline = build_timeline(details)
            details["replay_map"] = timeline.replay_map(
                max(timeline.playing_seconds(), (max(st.session_ms[-1] for st in timeline.stretches)
                                                  - timeline.recorder_start_ms) / 1000.0) + 1.0)
        except SyncError as exc:
            details["sync_problem"] = str(exc)

        filename = f"{uuid.uuid4().hex}{attempt_extension(recording)}"
        path = storage("attempts") / filename
        recording.save(path)
        if path.stat().st_size == 0:
            jobs_module.remove_files(path)
            return json_error("The recording was empty.", 400)

        attempt = Attempt(
            video_id=video.id, owner_id=security.current_owner_id(), upload_id=upload_id or None,
            recording_filename=filename, speed=speed, section_start_s=start, section_end_s=end,
            view_mirrored=_bool_arg(request.form, "view_mirrored", True), follow_mode=follow_mode,
            capture_mirrored=False, status="uploaded", progress=0.0, details_json=details)
        db.session.add(attempt)
        db.session.commit()
        jobs.submit(jobs_module.process_attempt, attempt.id)
        return jsonify(attempt_payload(attempt)), 201

    # ---- replay and feedback
    @app.route("/attempts/<int:attempt_id>")
    def attempt_detail(attempt_id):
        attempt = security.get_attempt_or_404(attempt_id)
        video = attempt.video
        history = security.visible_attempts().filter(Attempt.video_id == video.id) \
            .order_by(Attempt.created_at.asc()).all()
        history = [a for a in history if same_section(a, attempt)]
        return render_template("attempt.html", attempt=attempt, video=video, history=history,
                               payload=attempt_payload(attempt))

    @app.route("/api/attempts/<int:attempt_id>")
    def api_attempt(attempt_id):
        return jsonify(attempt_payload(security.get_attempt_or_404(attempt_id)))

    @app.route("/attempts/<int:attempt_id>/status")
    def attempt_status(attempt_id):
        return api_attempt(attempt_id)

    @app.route("/attempts/<int:attempt_id>/recording")
    def attempt_recording(attempt_id):
        attempt = security.get_attempt_or_404(attempt_id)
        if attempt.replay_filename:
            return send_from_directory(storage("replays"), attempt.replay_filename, mimetype="video/mp4",
                                       conditional=True, max_age=0)
        if attempt.recording_filename:           # still processing: serve the raw upload
            return send_from_directory(storage("attempts"), attempt.recording_filename, conditional=True, max_age=0)
        abort(404, "The recording file is missing.")

    @app.route("/attempts/<int:attempt_id>/reevaluate", methods=["POST"])
    def reevaluate_attempt(attempt_id):
        attempt = security.get_attempt_or_404(attempt_id)
        if attempt.status in ("uploaded", "processing"):
            flash("This attempt is already being processed.", "error")
        else:
            attempt.status, attempt.progress, attempt.error_message = "uploaded", 0.0, None
            db.session.commit()
            jobs.submit(jobs_module.process_attempt, attempt.id)
            flash("Re-evaluating this attempt.", "success")
        return redirect(url_for("attempt_detail", attempt_id=attempt.id), 303)

    @app.route("/attempts/<int:attempt_id>/delete", methods=["POST"])
    def delete_attempt(attempt_id):
        attempt = security.get_attempt_or_404(attempt_id)
        video_id = attempt.video_id
        jobs_module.delete_attempt_files(attempt)
        db.session.delete(attempt)
        db.session.commit()
        flash("Attempt deleted.", "success")
        return redirect(url_for("video_detail", video_id=video_id), 303)


if __name__ == "__main__":
    # `flask run` finds create_app() by itself; this is for `python app.py`.
    application = create_app()
    application.run(debug=application.config["DEBUG"])
