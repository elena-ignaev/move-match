from datetime import datetime, timezone
from extensions import db

# database initialization
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    original_name = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), unique=True, nullable=False)
    annotated_filename = db.Column(db.String(255), nullable=True)
    landmarks_path = db.Column(db.String(255), nullable=True)    
    status = db.Column(db.String(30), nullable=False, default="uploaded")

    frame_count = db.Column(db.Integer, nullable=True)
    fps = db.Column(db.Float, nullable=True)
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    
    error_message = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    evaluations = db.relationship(
        "Evaluation",
        backref="video",
        lazy=True,
        cascade="all, delete-orphan"
    )

    attempts = db.relationship(
        "Attempt",
        backref="video",
        lazy=True,
        cascade="all, delete-orphan"
    )

class Attempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    video_id = db.Column(
        db.Integer,
        db.ForeignKey("video.id"),
        nullable=False
    )

    recording_filename = db.Column(db.String(255), nullable=False)
    speed = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(30), nullable=False, default="recorded")
    details_json = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

class Evaluation(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    
    video_id = db.Column(
        db.Integer,
        db.ForeignKey("video.id"),
        nullable = False
    )

<<<<<<< Updated upstream
    speed = db.Column(db.Float, nullable=False)
    score = db.Column(db.Float)
    pose_score = db.Column(db.Float)
    timing_score = db.Column(db.Float)
=======
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
>>>>>>> Stashed changes

    details_json = db.Column(db.JSON)
    completed_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
