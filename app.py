from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, abort
from extensions import db
from models import Video, Evaluation
from werkzeug.utils import secure_filename
from pathlib import Path

from io import BytesIO
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
MODEL_PATH = BASE_DIR / "pretrained_models" / "pose_landmarker_lite.task"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{(BASE_DIR / 'videos.sqlite3').as_posix()}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

db.init_app(app)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "webm", "mkv", "m4v"}

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@app.errorhandler(413)
def too_large(_error):
    flash("The video exceeds the 200 MB upload limit.", "error")
    return redirect(url_for("index")), 303

@app.route('/')
def index():
    videos = Video.query.order_by(Video.created_at.desc()).all()
    return render_template("home.html", videos=videos)

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
        metadata = analyze_video(input_path, ANNOTATED_DIR / output_name)
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
    return render_template("video_detail.html", video=video)

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
    return render_template(
        "frame_player.html",
        video=video
    )

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)