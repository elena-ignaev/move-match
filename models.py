from datetime import datetime, timezone
from extensions import db

# database initialization
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    original_name = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), unique=True, nullable=False)
    annotated_filename = db.Column(db.String(255), nullable=True)
    
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

class Evaluation(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    
    video_id = db.Column(
        db.Integer,
        db.ForeignKey("video.id"),
        nullable = False
    )

    speed = db.Column(db.Float, nullable=False)
    score = db.Column(db.Float)
    pose_score = db.Column(db.Float)
    timing_score = db.Column(db.Float)

    details_json = db.Column(db.JSON)
    completed_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
