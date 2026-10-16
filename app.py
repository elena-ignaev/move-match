from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, abort
from extensions import db, migrate
from models import Attempt, Video, Evaluation
from sqlalchemy import inspect, text
from werkzeug.utils import secure_filename
from pathlib import Path

from io import BytesIO
import json
import os
import uuid

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import mediapipe as mp

from process_video import analyze_video

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
ANNOTATED_DIR = STORAGE_DIR / "annotated"
LANDMARK_DIR = STORAGE_DIR / "landmarks"
ATTEMPT_DIR = STORAGE_DIR / "attempts"
MODEL_PATH = BASE_DIR / "pretrained_models" / "pose_landmarker_lite.task"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
ATTEMPT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

import dev_tools
dev_tools.init_app(app)

app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'videos.sqlite3').as_posix()}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

db.init_app(app)
migrate.init_app(app, db)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "webm", "mkv", "m4v"}
ALLOWED_GHOST_SPEEDS = {0.25, 0.5, 0.75}
ATTEMPT_EXTENSIONS_BY_MIME = {
    "video/webm": ".webm",
    "video/mp4": ".mp4",
    "video/ogg": ".ogv",
}

def ensure_sqlite_columns():
    """Keep the local dev database compatible without adding a migration stack."""
    if not app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite:///"):
        return

    inspector = inspect(db.engine)
    if not inspector.has_table("video"):
        video_missing_columns = []
    else:
        existing_columns = {column["name"] for column in inspector.get_columns("video")}
        video_missing_columns = []
        if "width" not in existing_columns:
            video_missing_columns.append("ALTER TABLE video ADD COLUMN width INTEGER")
        if "height" not in existing_columns:
            video_missing_columns.append("ALTER TABLE video ADD COLUMN height INTEGER")

    attempt_missing_columns = []
    if inspector.has_table("attempt"):
        attempt_columns = {column["name"] for column in inspector.get_columns("attempt")}
        if "details_json" not in attempt_columns:
            attempt_missing_columns.append("ALTER TABLE attempt ADD COLUMN details_json JSON")

    missing_columns = video_missing_columns + attempt_missing_columns
    for statement in missing_columns:
        db.session.execute(text(statement))
    if missing_columns:
        db.session.commit()

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def attempt_extension(recording_file):
    mime_type = (recording_file.mimetype or "").split(";")[0].strip().lower()
    if mime_type in ATTEMPT_EXTENSIONS_BY_MIME:
        return ATTEMPT_EXTENSIONS_BY_MIME[mime_type]

    suffix = Path(secure_filename(recording_file.filename or "")).suffix.lower()
    if suffix in {".webm", ".mp4", ".ogv", ".ogg"}:
        return suffix

    return ".webm"

def parse_sync_events(raw_events):
    if not raw_events:
        return []

    try:
        events = json.loads(raw_events)
    except json.JSONDecodeError:
        return []

    if not isinstance(events, list):
        return []

    parsed_events = []
    for event in events:
        if isinstance(event, dict):
            parsed_events.append({
                "type": event.get("type"),
                "session_time_ms": event.get("session_time_ms"),
                "reference_time": event.get("reference_time"),
                "playback_rate": event.get("playback_rate"),
                "wall_time": event.get("wall_time"),
            })
    return parsed_events

@app.errorhandler(413)
def too_large(_error):
    flash("The video exceeds the 200 MB upload limit.", "error")
    return redirect(url_for("index")), 303

@app.route('/')
def index():
    recent_videos = Video.query.order_by(Video.created_at.desc()).limit(3).all()
    return render_template("home.html", recent_videos=recent_videos)

@app.route("/library")
def library():
    videos = Video.query.order_by(Video.created_at.desc()).all()
    return render_template("library.html", videos=videos)

# Handle video uploads
@app.route('/upload', methods=['POST'])
def upload():
    if "video_file" not in request.files:
        flash("No video was provided.")
        return redirect(url_for("index"))

    uploaded_file = request.files.get("video_file")

    if not uploaded_file.filename: 
        flash("Please select a video.", "error")
        return redirect(url_for("index"))
    
    if not allowed_file(uploaded_file.filename):
        flash("Unsupported file extension. Use MP4, MOV, AVI, WEBM, MKV, OR M4V.", "error")
        return redirect(url_for("index"))
    
    original_name = secure_filename(uploaded_file.filename)
    extension = Path(original_name).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{extension}"
    input_path = UPLOAD_DIR / unique_name
    uploaded_file.save(input_path)

    video = Video(
        original_name=original_name,
        stored_filename=unique_name,
        status="processing"
    )
    db.session.add(video)
    db.session.commit()

    try:
        output_name = f"{Path(unique_name).stem}_annotated.mp4"
        landmarks_name = f"{Path(unique_name).stem}.npz"
        metadata = analyze_video(input_path, ANNOTATED_DIR / output_name, landmarks_path= LANDMARK_DIR / landmarks_name)
        browser_path = Path(metadata["output_path"])
        video.annotated_filename = browser_path.name
        video.landmarks_path = Path(metadata["landmarks_path"]).relative_to(STORAGE_DIR).as_posix()
        video.status = "completed"
        video.frame_count = metadata["frame_count"]
        video.fps = metadata["fps"]
        video.width = metadata["width"]
        video.height = metadata["height"]
        db.session.commit()
        flash("Upload and pose analysis completed", "success")
    except Exception as exc:
        app.logger.exception("Pose analysis failed for video ID %s", video.id)
        video.status = "failed"
        video.error_message = str(exc)
        db.session.commit()
        flash("The video was uploaded, but pose analysis failed. See the details for the error.", "error")
    return redirect(url_for("video_detail", video_id = video.id))

@app.route("/videos/<int:video_id>")
def video_detail(video_id):
    video = db.get_or_404(Video, video_id)
    attempts = Attempt.query.filter_by(video_id=video.id).order_by(Attempt.created_at.desc()).all()
    return render_template("video_detail.html", video=video, attempts=attempts)

@app.route("/videos/<int:video_id>/original")
def original_video(video_id):
    video = db.get_or_404(Video, video_id)
    return send_from_directory(UPLOAD_DIR, video.stored_filename, conditional=True)

@app.route("/videos/<int:video_id>/annotated")
def annotated_video(video_id):
    video = db.get_or_404(Video,video_id)

    if not video.annotated_filename:
        abort(404)
    return send_from_directory(ANNOTATED_DIR, video.annotated_filename, mimetype="video/mp4", conditional=True)

@app.route("/dance-practice/<int:video_id>")
def dance_practice(video_id):
    video = db.get_or_404(Video, video_id)
    if not video.annotated_filename:
        flash("Practice is available after the annotated video has been created.", "error")
        return redirect(url_for("video_detail", video_id=video.id))
    return render_template(
        "frame_player.html",
        video=video
    )

@app.route("/videos/<int:video_id>/practice")
def video_practice(video_id):
    return redirect(url_for("dance_practice", video_id=video_id))

@app.route("/videos/<int:video_id>/ghost")
def ghost_mode(video_id):
    video = db.get_or_404(Video, video_id)
    if not video.annotated_filename:
        flash("Ghost mode is available after the annotated video has been created.", "error")
        return redirect(url_for("video_detail", video_id=video.id))

    return render_template("ghost_mode.html", video=video, speeds=sorted(ALLOWED_GHOST_SPEEDS))

@app.route("/videos/<int:video_id>/attempts", methods=["POST"])
def create_attempt(video_id):
    video = db.get_or_404(Video, video_id)

    if not video.annotated_filename:
        return jsonify({"error": "Ghost mode requires an annotated reference video."}), 400

    recording = request.files.get("recording")
    if recording is None or not recording.filename:
        return jsonify({"error": "No learner recording was provided."}), 400

    try:
        speed = float(request.form.get("speed", ""))
    except ValueError:
        return jsonify({"error": "Invalid practice speed."}), 400

    if speed not in ALLOWED_GHOST_SPEEDS:
        return jsonify({"error": "Unsupported practice speed."}), 400

    filename = f"{uuid.uuid4().hex}{attempt_extension(recording)}"
    recording.save(ATTEMPT_DIR / filename)

    attempt = Attempt(
        video_id=video.id,
        recording_filename=filename,
        speed=speed,
        status="recorded",
        details_json={
            "mime_type": recording.mimetype,
            "reference_started_at": request.form.get("reference_started_at"),
            "reference_stopped_at": request.form.get("reference_stopped_at"),
            "reference_time": request.form.get("reference_time"),
            "sync_events": parse_sync_events(request.form.get("sync_events")),
        }
    )
    db.session.add(attempt)
    db.session.commit()

    return jsonify({
        "attempt_id": attempt.id,
        "status": attempt.status,
        "status_url": url_for("attempt_status", attempt_id=attempt.id),
    }), 201

@app.route("/attempts/<int:attempt_id>/status")
def attempt_status(attempt_id):
    attempt = db.get_or_404(Attempt, attempt_id)
    return jsonify({
        "attempt_id": attempt.id,
        "video_id": attempt.video_id,
        "speed": attempt.speed,
        "status": attempt.status,
        "details": attempt.details_json or {},
        "created_at": attempt.created_at.isoformat() if attempt.created_at else None,
<<<<<<< Updated upstream
    })
=======
        "recording_url": url_for("attempt_recording", attempt_id=attempt.id),
        "recording_ready": bool(attempt.replay_filename),
        "replay_map": details.get("replay_map", []),
        "feedback": evaluation,
        "score": fb.score_of(evaluation),
        "previous_attempt": ({"id": prev.id, "url": url_for("attempt_detail", attempt_id=prev.id),
                              "score": (fb.score_of(evaluation_details(prev)) or {}).get("value")} if prev else None),
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

    @app.template_filter("attempt_score")
    def attempt_score_filter(attempt):
        """Score value for tables (computed for evaluations stored before scoring existed)."""
        ev = attempt.evaluation
        if ev is None:
            return None
        if ev.score is not None:
            return int(round(ev.score))
        return (fb.score_of(ev.details_json) or {}).get("value")

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
>>>>>>> Stashed changes

with app.app_context():
    db.create_all()
    ensure_sqlite_columns()

if __name__ == "__main__":
    app.run(debug=True)
