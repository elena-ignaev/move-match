from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, abort
from extensions import db
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
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{(BASE_DIR / 'videos.sqlite3').as_posix()}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

db.init_app(app)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "webm", "mkv", "m4v"}
ALLOWED_GHOST_SPEEDS = {0.25, 0.5, 0.75}
ATTEMPT_EXTENSIONS_BY_MIME = {
    "video/webm": ".webm",
    "video/mp4": ".mp4",
    "video/ogg": ".ogv",
}

import os
if os.environ.get("ENABLE_DEV_TOOLS") == "1":
    from dev_tools import bp as dev_landmarks_bp
    app.register_blueprint(dev_landmarks_bp)

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
    })

with app.app_context():
    db.create_all()
    ensure_sqlite_columns()

if __name__ == "__main__":
    app.run(debug=True)
