"""Database models. The schema is owned by Alembic (migrations/): change a model, then run
`flask db migrate -m "..."`, review the generated file, and `flask db upgrade`."""

from datetime import datetime, timezone

from extensions import db


def utcnow():
    return datetime.now(timezone.utc)


class Video(db.Model):
    """A reference dance clip."""

    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.String(64), nullable=True, index=True)
    is_sample = db.Column(db.Boolean, nullable=False, default=False)

    original_name = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), unique=True, nullable=False)    # original upload
    playback_filename = db.Column(db.String(255), nullable=True)                # H.264/AAC, constant frame rate
    landmarks_path = db.Column(db.String(255), nullable=True)                   # relative to storage/

    # queued -> processing -> completed | failed;  needs_reprocessing for pre-0003 rows
    status = db.Column(db.String(30), nullable=False, default="queued")
    progress = db.Column(db.Float, nullable=False, default=0.0)
    progress_message = db.Column(db.String(255), nullable=True)
    error_message = db.Column(db.Text, nullable=True)                           # safe to show visitors

    frame_count = db.Column(db.Integer, nullable=True)
    fps = db.Column(db.Float, nullable=True)
    duration_s = db.Column(db.Float, nullable=True)
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    has_audio = db.Column(db.Boolean, nullable=True)
    detected_ratio = db.Column(db.Float, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=True, default=utcnow, onupdate=utcnow)

    attempts = db.relationship("Attempt", backref="video", lazy=True, cascade="all, delete-orphan",
                               order_by="Attempt.created_at.desc()")


class Attempt(db.Model):
    """One recorded practice take of a section of a reference video."""

    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey("video.id"), nullable=False, index=True)
    owner_id = db.Column(db.String(64), nullable=True, index=True)
    upload_id = db.Column(db.String(64), nullable=True, unique=True)    # client-generated, makes retries idempotent

    recording_filename = db.Column(db.String(255), nullable=True)      # raw upload, deleted after transcoding
    replay_filename = db.Column(db.String(255), nullable=True)         # constant-frame-rate MP4 for replay
    landmarks_path = db.Column(db.String(255), nullable=True)

    speed = db.Column(db.Float, nullable=False)
    section_start_s = db.Column(db.Float, nullable=False, default=0.0)
    section_end_s = db.Column(db.Float, nullable=True)

    # Mirror convention (see pose_pipeline.py): the display preference and the dance-side policy are
    # stored separately. Recordings are never flipped (capture_mirrored=False).
    view_mirrored = db.Column(db.Boolean, nullable=False, default=True)
    follow_mode = db.Column(db.String(20), nullable=False, default="same_side")
    capture_mirrored = db.Column(db.Boolean, nullable=False, default=False)

    # uploaded -> processing -> ready | withheld | failed
    status = db.Column(db.String(30), nullable=False, default="uploaded")
    progress = db.Column(db.Float, nullable=False, default=0.0)
    error_message = db.Column(db.Text, nullable=True)
    details_json = db.Column(db.JSON)                                   # sync timeline + client info

    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    evaluation = db.relationship("Evaluation", backref="attempt", uselist=False,
                                 cascade="all, delete-orphan")


class Evaluation(db.Model):
    """Feedback for exactly one attempt: interpretable measurements plus a provisional score built from them."""

    id = db.Column(db.Integer, primary_key=True)
    attempt_id = db.Column(db.Integer, db.ForeignKey("attempt.id"), nullable=False, unique=True)
    video_id = db.Column(db.Integer, db.ForeignKey("video.id"), nullable=False, index=True)

    algorithm_version = db.Column(db.String(20), nullable=False)
    withheld = db.Column(db.Boolean, nullable=False, default=False)
    tracking_status = db.Column(db.String(20), nullable=True)           # good | partial | poor

    timing_offset_s = db.Column(db.Float, nullable=True)                # + = learner late, wall-clock seconds
    timing_confident = db.Column(db.Boolean, nullable=True)
    moments_total = db.Column(db.Integer, nullable=True)
    moments_close = db.Column(db.Integer, nullable=True)
    moments_review = db.Column(db.Integer, nullable=True)
    median_angle_diff_deg = db.Column(db.Float, nullable=True)
    score = db.Column(db.Float, nullable=True)                         # provisional 0-100, None when withheld
    score_version = db.Column(db.String(20), nullable=True)

    details_json = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
