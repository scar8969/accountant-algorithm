"""Consent enforcement and the withdrawal / deletion workflow.

Consent is checked in the application on every write that creates research data,
not merely stored. Withdrawal scrubs profile answers, deletes this participant's
pair outcomes, predictions and feedback, and marks the participant deleted.
Aggregate run metrics (no personal data) are retained so past experiments stay
reproducible; that retention is stated in the consent text.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from ..db.models import (ConsentRecord, Feedback, PairOutcome, Participant,
                         Prediction, Profile)

STUDY_VERSION = "2026-10-pilot"
PURPOSE = "research_v1"


def record_consent(session: Session, participant_id: str, consented: bool,
                   study_version: str = STUDY_VERSION,
                   purpose: str = PURPOSE) -> ConsentRecord:
    record = ConsentRecord(participant_id=participant_id, study_version=study_version,
                           purpose=purpose, consented=consented,
                           withdrawn_at=None if consented else datetime.now(timezone.utc))
    session.add(record)
    session.flush()
    return record


def has_active_consent(session: Session, participant_id: str,
                       study_version: str = STUDY_VERSION,
                       purpose: str = PURPOSE) -> bool:
    record = session.execute(
        select(ConsentRecord)
        .where(ConsentRecord.participant_id == participant_id,
               ConsentRecord.study_version == study_version,
               ConsentRecord.purpose == purpose,
               ConsentRecord.consented.is_(True),
               ConsentRecord.withdrawn_at.is_(None))
        .order_by(ConsentRecord.recorded_at.desc())
    ).scalars().first()
    return record is not None


def require_consent(session: Session, participant_id: str,
                    purpose: str = PURPOSE) -> None:
    if purpose == PURPOSE and not has_active_consent(session, participant_id):
        raise HTTPException(status_code=403,
                            detail="active research consent required for this operation")


def profile_ids_for(session: Session, participant_id: str) -> list[str]:
    return list(session.execute(
        select(Profile.id).where(Profile.participant_id == participant_id)
    ).scalars())


def withdraw(session: Session, participant_id: str) -> dict:
    """Withdraw consent and delete this participant's research data."""
    participant = session.get(Participant, participant_id)
    if participant is None:
        raise HTTPException(status_code=404, detail="participant not found")

    now = datetime.now(timezone.utc)
    for record in session.execute(
            select(ConsentRecord).where(ConsentRecord.participant_id == participant_id)
    ).scalars():
        if record.withdrawn_at is None:
            record.withdrawn_at = now

    profile_ids = profile_ids_for(session, participant_id)

    deleted = {"profiles": 0, "predictions": 0, "pair_outcomes": 0, "feedback": 0}
    if profile_ids:
        deleted["predictions"] = session.execute(
            delete(Prediction).where(Prediction.profile_a.in_(profile_ids)
                                     | Prediction.profile_b.in_(profile_ids))
        ).rowcount or 0
        deleted["pair_outcomes"] = session.execute(
            delete(PairOutcome).where(PairOutcome.profile_a.in_(profile_ids)
                                      | PairOutcome.profile_b.in_(profile_ids))
        ).rowcount or 0
        deleted["profiles"] = session.execute(
            delete(Profile).where(Profile.id.in_(profile_ids))
        ).rowcount or 0

    deleted["feedback"] = session.execute(
        delete(Feedback).where(Feedback.participant_id == participant_id)
    ).rowcount or 0

    participant.status = "deleted"
    participant.token_hash = None  # token is useless after withdrawal
    session.flush()

    return {"participant_id": participant_id, "status": participant.status,
            "withdrawn_at": now.isoformat(), "deleted": deleted,
            "note": ("Profile answers and pair-level records were deleted. Aggregate "
                     "run metrics contain no personal data and are retained.")}


def export(session: Session, participant_id: str) -> dict:
    participant = session.get(Participant, participant_id)
    if participant is None:
        raise HTTPException(status_code=404, detail="participant not found")

    profiles = list(session.execute(
        select(Profile).where(Profile.participant_id == participant_id)
        .order_by(Profile.version)
    ).scalars())
    consent = list(session.execute(
        select(ConsentRecord).where(ConsentRecord.participant_id == participant_id)
    ).scalars())
    feedback = list(session.execute(
        select(Feedback).where(Feedback.participant_id == participant_id)
    ).scalars())

    profile_ids = [p.id for p in profiles]
    outcomes = []
    if profile_ids:
        rows = session.execute(
            select(PairOutcome).where(or_(PairOutcome.profile_a.in_(profile_ids),
                                          PairOutcome.profile_b.in_(profile_ids)))
        ).scalars().all()
        own = set(profile_ids)
        for row in rows:
            side_is_a = row.profile_a in own
            outcomes.append({
                "other_profile_id": row.profile_b if side_is_a else row.profile_a,
                "interest": row.interest_a if side_is_a else row.interest_b,
                "conversation_quality": (row.conversation_quality_a if side_is_a
                                         else row.conversation_quality_b),
                "continued_interest": (row.continued_interest_a if side_is_a
                                       else row.continued_interest_b),
                "recorded_at": row.recorded_at.isoformat(),
            })

    return {
        "participant": {"id": participant.id, "public_code": participant.public_code,
                        "status": participant.status, "source": participant.source,
                        "created_at": participant.created_at.isoformat()},
        "consent_records": [
            {"study_version": c.study_version, "purpose": c.purpose,
             "consented": c.consented, "recorded_at": c.recorded_at.isoformat(),
             "withdrawn_at": c.withdrawn_at.isoformat() if c.withdrawn_at else None}
            for c in consent
        ],
        "profiles": [
            {"profile_id": p.id, "version": p.version,
             "feature_schema_version": p.feature_schema_version,
             "features": p.features, "created_at": p.created_at.isoformat()}
            for p in profiles
        ],
        "feedback": [
            {"feedback_type": f.feedback_type, "rating": f.rating, "payload": f.payload,
             "created_at": f.created_at.isoformat()} for f in feedback
        ],
        "pair_outcomes": outcomes,
        "note": ("Export contains only this participant's own data. Pair outcomes show "
                 "this participant's side only, never the other person's answers."),
    }
