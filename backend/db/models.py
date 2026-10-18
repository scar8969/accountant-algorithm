"""SQLAlchemy models mirroring the blueprint schema (section 5).

Deviations, documented on purpose:
  * JSON instead of JSONB so the same schema runs on SQLite for local dev and on
    PostgreSQL in deployment (SQLAlchemy JSON maps to JSONB there when using
    postgresql.JSONB; see db/session.py comment).
  * `participants.source` distinguishes synthetic demo rows from real pilot
    participants, so no synthetic data can be mistaken for human evidence.
  * `participants.token_hash` stores a hash of the participant's access token,
    never the token itself.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, CheckConstraint, DateTime, Float,
                        ForeignKey, Integer, String, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    public_code: Mapped[str] = mapped_column(String(36), unique=True, default=_uuid)
    status: Mapped[str] = mapped_column(String(16), default="active")
    source: Mapped[str] = mapped_column(String(16), default="pilot")
    token_hash: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        CheckConstraint("status IN ('active', 'withdrawn', 'deleted')",
                        name="ck_participant_status"),
    )


class ConsentRecord(Base):
    __tablename__ = "consent_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    participant_id: Mapped[str] = mapped_column(ForeignKey("participants.id"), index=True)
    study_version: Mapped[str] = mapped_column(String(32))
    purpose: Mapped[str] = mapped_column(String(64))
    consented: Mapped[bool] = mapped_column(Boolean)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True),
                                                          nullable=True)


class Profile(Base):
    __tablename__ = "profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    participant_id: Mapped[str] = mapped_column(ForeignKey("participants.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    feature_schema_version: Mapped[str] = mapped_column(String(16))
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("participant_id", "version", name="uq_profile_version"),)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120))
    hypothesis: Mapped[str] = mapped_column(String(2000))
    protocol: Mapped[dict] = mapped_column(JSON, default=dict)
    random_seed: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(32))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    feature_schema_version: Mapped[str] = mapped_column(String(16))
    code_revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_version"),)


class PairOutcome(Base):
    """Directional feedback. Mutual interest is known only when both sides answered."""

    __tablename__ = "pair_outcomes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"),
                                                      nullable=True)
    profile_a: Mapped[str] = mapped_column(ForeignKey("profiles.id"), index=True)
    profile_b: Mapped[str] = mapped_column(ForeignKey("profiles.id"), index=True)
    exposure_recorded: Mapped[bool] = mapped_column(Boolean, default=False)
    interest_a: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    interest_b: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    conversation_quality_a: Mapped[int | None] = mapped_column(Integer, nullable=True)
    conversation_quality_b: Mapped[int | None] = mapped_column(Integer, nullable=True)
    continued_interest_a: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    continued_interest_b: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        CheckConstraint("profile_a <> profile_b", name="ck_pair_distinct"),
        CheckConstraint("conversation_quality_a BETWEEN 1 AND 5 OR "
                        "conversation_quality_a IS NULL", name="ck_quality_a"),
        CheckConstraint("conversation_quality_b BETWEEN 1 AND 5 OR "
                        "conversation_quality_b IS NULL", name="ck_quality_b"),
    )


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"))
    model_version_id: Mapped[str | None] = mapped_column(ForeignKey("model_versions.id"),
                                                         nullable=True)
    dataset_version: Mapped[str] = mapped_column(String(120))
    split_config: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True),
                                                        nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True),
                                                          nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("experiment_runs.id"),
                                                nullable=True)
    profile_a: Mapped[str] = mapped_column(ForeignKey("profiles.id"), index=True)
    profile_b: Mapped[str] = mapped_column(ForeignKey("profiles.id"), index=True)
    task: Mapped[str] = mapped_column(String(24))
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    explanation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        CheckConstraint("probability IS NULL OR (probability >= 0 AND probability <= 1)",
                        name="ck_probability_range"),
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    participant_id: Mapped[str] = mapped_column(ForeignKey("participants.id"), index=True)
    prediction_id: Mapped[str | None] = mapped_column(ForeignKey("predictions.id"),
                                                       nullable=True)
    feedback_type: Mapped[str] = mapped_column(String(32))
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("participant_id", "prediction_id", "feedback_type",
                         name="uq_feedback_dedupe"),
    )
