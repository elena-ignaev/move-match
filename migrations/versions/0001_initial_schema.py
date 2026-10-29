"""initial schema (video, attempt, evaluation) as created by the first prototype

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-14

Earlier versions built these tables with db.create_all() at startup, so the first real migration
(b57605eb2ee6) could only ALTER an existing table and failed on a fresh database. This baseline
creates the tables explicitly. A database that was created by create_all() already has them: mark
it instead of re-creating them (see docs/setup.md, "Existing databases"):

    flask db stamp b57605eb2ee6 && flask db upgrade
"""
from alembic import op
import sqlalchemy as sa


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "video",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("original_name", sa.String(length=255), nullable=False),
        sa.Column("stored_filename", sa.String(length=255), nullable=False),
        sa.Column("annotated_filename", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("frame_count", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stored_filename"),
    )
    op.create_table(
        "attempt",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("recording_filename", sa.String(length=255), nullable=False),
        sa.Column("speed", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("details_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["video.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
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


def downgrade():
    op.drop_table("evaluation")
    op.drop_table("attempt")
    op.drop_table("video")
