"""provisional attempt score on evaluations

Revision ID: 0004_attempt_score
Revises: 0003_replay_feedback
Create Date: 2026-10-15

The score is derived from the stored measurements (feedback.attempt_score), so evaluations made
before this revision get a score when they are displayed or re-evaluated; nothing is backfilled here.
"""
from alembic import op
import sqlalchemy as sa


revision = "0004_attempt_score"
down_revision = "0003_replay_feedback"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("evaluation", schema=None) as batch_op:
        batch_op.add_column(sa.Column("score", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("score_version", sa.String(length=20), nullable=True))


def downgrade():
    with op.batch_alter_table("evaluation", schema=None) as batch_op:
        batch_op.drop_column("score_version")
        batch_op.drop_column("score")
