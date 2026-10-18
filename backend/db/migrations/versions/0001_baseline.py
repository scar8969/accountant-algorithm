"""baseline schema (all tables as of feature schema 1.0.0)

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-18

The baseline is generated from the SQLAlchemy metadata rather than hand-written
DDL, so a fresh database and the ORM can never disagree. Later revisions should
be produced with `alembic revision --autogenerate -m "..."` against a database
already stamped at this revision.
"""

from __future__ import annotations

from alembic import op

from backend.db.models import Base

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
