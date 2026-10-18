"""Database session helpers.

DATABASE_URL controls the backend:
    sqlite:///data/accountant.db      (default, local dev)
    postgresql+psycopg://user:pass@host/db   (deployment)

On PostgreSQL, swap `JSON` for `JSONB` in db/models.py if you want indexed
document queries; the application code does not care.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

ROOT = Path(__file__).resolve().parents[2]


def database_url() -> str:
    url = os.environ.get("ACCOUNTANT_DB_URL")
    if url:
        return url
    data_dir = Path(os.environ.get("ACCOUNTANT_DATA_DIR", ROOT / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir / 'accountant.db'}"


_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine():
    global _engine
    if _engine is None:
        url = database_url()
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(url, **kwargs)
    return _engine


def get_sessionmaker() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())


def reset_engine() -> None:
    """Drop the cached engine/session factory (used by tests with a temp DB)."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None


def get_db():
    """FastAPI dependency."""
    session: Session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
