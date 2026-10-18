"""Authentication and authorisation.

Two credentials:
  * research key  (X-API-Key)  — access to benchmarking, scoring, run inspection.
  * participant token (X-Participant-Token) — access to one participant's own data.

Neither is a password. Tokens are random, stored hashed, and never logged.
In production ACCOUNTANT_API_KEY must be set (fail closed); locally a dev key is
used so the app runs out of the box.
"""

from __future__ import annotations

import hashlib
import os
import secrets

from fastapi import Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Participant

DEV_RESEARCH_KEY = "dev-research-key"


def research_api_key() -> str:
    key = os.environ.get("ACCOUNTANT_API_KEY")
    if key:
        return key
    if os.environ.get("ACCOUNTANT_ENV", "development").lower() == "production":
        raise RuntimeError("ACCOUNTANT_API_KEY must be set when ACCOUNTANT_ENV=production")
    return DEV_RESEARCH_KEY


def require_research_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    expected = research_api_key()
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing research API key")
    return x_api_key


def new_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def participant_from_token(session: Session, token: str | None) -> Participant:
    if not token:
        raise HTTPException(status_code=401, detail="missing participant token")
    participant = session.execute(
        select(Participant).where(Participant.token_hash == hash_token(token))
    ).scalar_one_or_none()
    if participant is None:
        raise HTTPException(status_code=401, detail="invalid participant token")
    return participant


def authorise_participant(session: Session, participant_id: str,
                          token: str | None) -> Participant:
    participant = participant_from_token(session, token)
    if participant.id != participant_id:
        raise HTTPException(status_code=403, detail="token does not match participant")
    return participant

