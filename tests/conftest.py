"""Shared fixtures. Environment is configured before the app is imported."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="accountant-tests-"))
os.environ.setdefault("ACCOUNTANT_DB_URL", f"sqlite:///{(_TMP / 'test.db').as_posix()}")
os.environ.setdefault("ACCOUNTANT_ENV", "test")
os.environ.setdefault("ACCOUNTANT_API_KEY", "test-research-key")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402

RESEARCH_KEY = os.environ["ACCOUNTANT_API_KEY"]


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def research_headers() -> dict:
    return {"X-API-Key": RESEARCH_KEY}


def make_profile(**overrides) -> dict:
    base = {
        "traits": {"ambition": 4, "family": 5, "independence": 3, "directness": 4,
                   "conflict_approach": 3, "social_energy": 2,
                   "routine_structure": 4, "frequency": 3},
        "preferences": {"ambition": {"range": [3, 5], "weight": 4},
                        "frequency": {"range": [2, 4], "weight": 5}},
        "goal": "long_term",
        "flags": {"smoker": False, "drinks_regularly": False,
                  "wants_children": True, "has_pets": False},
        "dealbreakers": ["smoker"],
        "interests": ["hiking", "cooking", "reading"],
        "text": "I like hiking, cooking and quiet evenings with a book.",
    }
    base.update(overrides)
    return base


@pytest.fixture
def profile_factory():
    return make_profile


@pytest.fixture
def participant(client):
    """Create a consented participant; returns (id, token, headers)."""

    def _make(profile: dict | None = None):
        created = client.post("/api/v1/participants", json={}).json()
        token = created["participant_token"]
        pid = created["participant_id"]
        headers = {"X-Participant-Token": token}
        client.post(f"/api/v1/participants/{pid}/consent",
                    json={"study_version": "2026-10-pilot", "consented": True},
                    headers=headers)
        profile_id = None
        if profile is not None:
            response = client.post("/api/v1/profiles", json=profile, headers=headers)
            assert response.status_code == 201, response.text
            profile_id = response.json()["profile_id"]
        return {"id": pid, "token": token, "headers": headers,
                "profile_id": profile_id}

    return _make
