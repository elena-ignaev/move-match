"""attempt replay, background processing, ownership, per-attempt feedback

Revision ID: 0003_replay_feedback
Revises: b57605eb2ee6
Create Date: 2026-10-14

* video: playback file with audio (replaces the silent annotated video), processing progress,
  ownership for public-demo mode, sample flag, media facts.
* attempt: replay file, section, explicit mirror convention, idempotent upload id, progress.
* evaluation: rebuilt. It now belongs to one attempt and stores interpretable measurements
  (timing offset, key moments, angle differences) instead of unvalidated percentage scores.
  The old table held no feedback that the app ever produced, so it is dropped, not converted.
"""
from alembic import op
import sqlalchemy as sa


revision = "0003_replay_feedback"
down_revision = "b57605eb2ee6"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("video", schema=None) as batch_op:
        batch_op.add_column(sa.Column("owner_id", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("is_sample", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("playback_filename", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("progress", sa.Float(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("progress_message", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("duration_s", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("has_audio", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("detected_ratio", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("updated_at", sa.DateTime(), nullable=True))
        batch_op.drop_column("annotated_filename")
        batch_op.create_index("ix_video_owner_id", ["owner_id"])

    # Rows from the prototype have a silent annotated video and landmarks timed by nominal FPS.
    # They keep their original upload, so they can be reprocessed from the UI.
    op.execute("UPDATE video SET status = 'needs_reprocessing', progress = 0 WHERE status <> 'failed'")

    with op.batch_alter_table("attempt", schema=None) as batch_op:
        batch_op.add_column(sa.Column("owner_id", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("upload_id", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("replay_filename", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("landmarks_path", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("section_start_s", sa.Float(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("section_end_s", sa.Float(), nullable=True))
        # The prototype mirrored only the camera preview and recorded the raw camera image.
        batch_op.add_column(sa.Column("view_mirrored", sa.Boolean(), nullable=False, server_default=sa.true()))
        batch_op.add_column(sa.Column("follow_mode", sa.String(length=20), nullable=False,
                                      server_default="same_side"))
        batch_op.add_column(sa.Column("capture_mirrored", sa.Boolean(), nullable=False,
                                      server_default=sa.false()))
        batch_op.add_column(sa.Column("progress", sa.Float(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("error_message", sa.Text(), nullable=True))
        batch_op.alter_column("recording_filename", existing_type=sa.String(length=255), nullable=True)
        batch_op.create_index("ix_attempt_owner_id", ["owner_id"])
        batch_op.create_index("ix_attempt_video_id", ["video_id"])
        batch_op.create_unique_constraint("uq_attempt_upload_id", ["upload_id"])

    op.execute("UPDATE attempt SET status = 'uploaded' WHERE status = 'recorded'")

    op.drop_table("evaluation")
    op.create_table(
        "evaluation",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("attempt_id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("algorithm_version", sa.String(length=20), nullable=False),
        sa.Column("withheld", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("tracking_status", sa.String(length=20), nullable=True),
        sa.Column("timing_offset_s", sa.Float(), nullable=True),
        sa.Column("timing_confident", sa.Boolean(), nullable=True),
        sa.Column("moments_total", sa.Integer(), nullable=True),
        sa.Column("moments_close", sa.Integer(), nullable=True),
        sa.Column("moments_review", sa.Integer(), nullable=True),
        sa.Column("median_angle_diff_deg", sa.Float(), nullable=True),
        sa.Column("details_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["attempt_id"], ["attempt.id"]),
        sa.ForeignKeyConstraint(["video_id"], ["video.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("attempt_id", name="uq_evaluation_attempt_id"),
    )
    op.create_index("ix_evaluation_video_id", "evaluation", ["video_id"])


def downgrade():
    op.drop_index("ix_evaluation_video_id", table_name="evaluation")
    op.drop_table("evaluation")
    op.create_table(
        "evaluation",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("speed", sa.Float(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("pose_score", sa.Float(), nullable=True),
        sa.Column("timing_score", sa.Float(), nullable=True),
        sa.Column("details_json", sa.JSON(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["video_id"], ["video.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute("DELETE FROM attempt WHERE recording_filename IS NULL")
    with op.batch_alter_table("attempt", schema=None) as batch_op:
        batch_op.drop_constraint("uq_attempt_upload_id", type_="unique")
        batch_op.drop_index("ix_attempt_video_id")
        batch_op.drop_index("ix_attempt_owner_id")
        batch_op.alter_column("recording_filename", existing_type=sa.String(length=255), nullable=False)
        for column in ("error_message", "progress", "capture_mirrored", "follow_mode", "view_mirrored",
                       "section_end_s", "section_start_s", "landmarks_path", "replay_filename",
                       "upload_id", "owner_id"):
            batch_op.drop_column(column)
    with op.batch_alter_table("video", schema=None) as batch_op:
        batch_op.drop_index("ix_video_owner_id")
        batch_op.add_column(sa.Column("annotated_filename", sa.String(length=255), nullable=True))
        for column in ("updated_at", "detected_ratio", "has_audio", "duration_s", "progress_message",
                       "progress", "playback_filename", "is_sample", "owner_id"):
            batch_op.drop_column(column)
