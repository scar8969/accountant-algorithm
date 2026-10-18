"""Integration tests for the versioned API: consent, scoring, feedback, deletion."""

from __future__ import annotations

import pytest

from backend.core.ratelimit import LIMITER
from backend.db.models import Profile
from backend.db.session import get_sessionmaker


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["feature_schema_version"]


def test_research_endpoints_require_key(client):
    assert client.get("/api/v1/models").status_code == 401
    assert client.post("/api/v1/experiments", json={"name": "x", "hypothesis": "y"}
                       ).status_code == 401


def test_models_endpoint_lists_registry(client, research_headers):
    body = client.get("/api/v1/models", headers=research_headers).json()
    names = {m["name"] for m in body["models"]}
    assert {"random", "structured", "directional_logistic"} <= names


def test_profile_requires_consent(client, profile_factory):
    created = client.post("/api/v1/participants", json={}).json()
    token = created["participant_token"]
    response = client.post("/api/v1/profiles", json=profile_factory(),
                           headers={"X-Participant-Token": token})
    assert response.status_code == 403
    assert "consent" in response.json()["detail"].lower()

    client.post(f"/api/v1/participants/{created['participant_id']}/consent",
                json={"study_version": "2026-10-pilot", "consented": True},
                headers={"X-Participant-Token": token})
    response = client.post("/api/v1/profiles", json=profile_factory(),
                           headers={"X-Participant-Token": token})
    assert response.status_code == 201
    assert response.json()["version"] == 1


def test_invalid_features_are_rejected(client, participant, profile_factory):
    party = participant()
    bad = profile_factory(traits={"ambition": 12})
    response = client.post("/api/v1/profiles", json=bad, headers=party["headers"])
    assert response.status_code == 422
    assert "errors" in response.json()["detail"]

    wrong_schema = profile_factory()
    wrong_schema["feature_schema_version"] = "0.0.1"
    response = client.post("/api/v1/profiles", json=wrong_schema,
                           headers=party["headers"])
    assert response.status_code == 400


def test_participant_cannot_read_another_profile(client, participant, profile_factory):
    a = participant(profile_factory())
    b = participant(profile_factory(goal="casual"))
    assert client.get(f"/api/v1/profiles/{b['profile_id']}",
                      headers=a["headers"]).status_code == 403
    assert client.get(f"/api/v1/profiles/{a['profile_id']}",
                      headers=a["headers"]).status_code == 200


def test_score_returns_explanation_without_private_answers(client, participant,
                                                           profile_factory,
                                                           research_headers):
    a = participant(profile_factory())
    b = participant(profile_factory(goal="casual",
                                    traits={"ambition": 2, "family": 1,
                                            "independence": 5, "directness": 2,
                                            "conflict_approach": 2,
                                            "social_energy": 5,
                                            "routine_structure": 1,
                                            "frequency": 5}))
    body = client.post("/api/v1/compatibility/score",
                       json={"profile_a_id": a["profile_id"],
                             "profile_b_id": b["profile_id"],
                             "model_name": "structured_baseline"},
                       headers=research_headers).json()

    assert body["task"] == "compatibility_score"
    assert 0 <= body["score"] <= 100
    assert body["scale"] == "0-100"
    assert body["probability"] is None
    assert body["dimensions"]
    assert body["explanation"]["contributions"]
    assert body["limitations"]
    serialised = str(body)
    assert "preferences" not in serialised and "interests" not in serialised
    assert a["profile_id"] not in serialised and b["profile_id"] not in serialised


def test_rank_excludes_self_and_returns_ordered_items(client, participant,
                                                      profile_factory):
    query = participant(profile_factory())
    for _ in range(3):
        participant(profile_factory())
    body = client.post("/api/v1/compatibility/rank",
                       json={"profile_id": query["profile_id"], "limit": 5},
                       headers=query["headers"]).json()
    assert body["items"]
    scores = [item["score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all(item["profile_id"] != query["profile_id"] for item in body["items"])


def test_feedback_is_idempotent_and_detects_mutual_interest(client, participant,
                                                            profile_factory):
    a = participant(profile_factory())
    b = participant(profile_factory())

    payload = {"feedback_type": "interest", "other_profile_id": b["profile_id"],
               "value": True}
    first = client.post("/api/v1/feedback", json=payload, headers=a["headers"]).json()
    second = client.post("/api/v1/feedback", json=payload, headers=a["headers"]).json()
    assert first["feedback_id"] == second["feedback_id"]
    assert second["duplicate"] is True
    assert first["mutual_interest_recorded"] is False

    payload_b = {"feedback_type": "interest", "other_profile_id": a["profile_id"],
                 "value": True}
    response = client.post("/api/v1/feedback", json=payload_b, headers=b["headers"]).json()
    assert response["mutual_interest_recorded"] is True


def test_feedback_validates_values(client, participant, profile_factory):
    a = participant(profile_factory())
    b = participant(profile_factory())
    bad = client.post("/api/v1/feedback",
                      json={"feedback_type": "conversation_quality",
                            "other_profile_id": b["profile_id"], "value": 9},
                      headers=a["headers"])
    assert bad.status_code == 422


def test_withdraw_deletes_profile_and_pair_data(client, participant, profile_factory):
    a = participant(profile_factory())
    b = participant(profile_factory())
    client.post("/api/v1/feedback",
                json={"feedback_type": "interest",
                      "other_profile_id": b["profile_id"], "value": True},
                headers=a["headers"])

    export = client.post(f"/api/v1/participants/{a['id']}/export",
                         headers=a["headers"]).json()
    assert export["profiles"] and export["feedback"]

    result = client.post(f"/api/v1/participants/{a['id']}/withdraw",
                         headers=a["headers"]).json()
    assert result["status"] == "deleted"
    assert result["deleted"]["profiles"] >= 1
    assert result["deleted"]["pair_outcomes"] >= 1
    assert result["deleted"]["feedback"] >= 1

    # profile is gone and the token no longer works
    assert client.get(f"/api/v1/profiles/{a['profile_id']}",
                      headers=a["headers"]).status_code in (401, 404)
    assert client.post(f"/api/v1/participants/{a['id']}/export",
                       headers=a["headers"]).status_code == 401


def test_consent_withdrawal_prevents_further_writes(client, participant,
                                                    profile_factory):
    a = participant(profile_factory())
    b = participant(profile_factory())
    client.post(f"/api/v1/participants/{a['id']}/consent",
                json={"study_version": "2026-10-pilot", "consented": False},
                headers=a["headers"])
    response = client.post("/api/v1/feedback",
                           json={"feedback_type": "interest",
                                 "other_profile_id": b["profile_id"], "value": True},
                           headers=a["headers"])
    assert response.status_code in (401, 403)


def test_consent_rejects_unknown_study_version(client, participant):
    party = participant()
    response = client.post(f"/api/v1/participants/{party['id']}/consent",
                           json={"study_version": "1999-ancient", "consented": True},
                           headers=party["headers"])
    assert response.status_code == 409
    assert response.json()["detail"]["supported"]


def test_consent_rejects_unknown_purpose(client, participant):
    party = participant()
    response = client.post(f"/api/v1/participants/{party['id']}/consent",
                           json={"study_version": "2026-10-pilot",
                                 "purpose": "advertising", "consented": True},
                           headers=party["headers"])
    assert response.status_code == 400


def test_oversized_payload_is_rejected(client, participant, profile_factory):
    party = participant()
    payload = profile_factory(interests=[f"topic-{i}" for i in range(60)])
    response = client.post("/api/v1/profiles", json=payload, headers=party["headers"])
    assert response.status_code == 422


def test_profile_with_old_schema_cannot_be_scored(client, participant, profile_factory,
                                                  research_headers):
    a = participant(profile_factory())
    b = participant(profile_factory())
    session = get_sessionmaker()()
    session.query(Profile).filter(Profile.id == b["profile_id"]).update(
        {"feature_schema_version": "0.9.0"})
    session.commit()
    session.close()

    response = client.post("/api/v1/compatibility/score",
                           json={"profile_a_id": a["profile_id"],
                                 "profile_b_id": b["profile_id"]},
                           headers=research_headers)
    assert response.status_code == 409
    assert response.json()["detail"]["supported"]


def test_export_contains_own_pair_outcomes_only(client, participant, profile_factory):
    a = participant(profile_factory())
    b = participant(profile_factory())
    client.post("/api/v1/feedback",
                json={"feedback_type": "interest", "other_profile_id": b["profile_id"],
                      "value": True}, headers=a["headers"])
    client.post("/api/v1/feedback",
                json={"feedback_type": "interest", "other_profile_id": a["profile_id"],
                      "value": False}, headers=b["headers"])

    body = client.post(f"/api/v1/participants/{a['id']}/export",
                       headers=a["headers"]).json()
    assert len(body["pair_outcomes"]) == 1
    outcome = body["pair_outcomes"][0]
    assert outcome["interest"] is True           # A's own answer
    assert outcome["other_profile_id"] == b["profile_id"]

    other = client.post(f"/api/v1/participants/{b['id']}/export",
                        headers=b["headers"]).json()
    assert other["pair_outcomes"][0]["interest"] is False
    assert other["pair_outcomes"][0]["other_profile_id"] == a["profile_id"]


def test_rate_limit_is_enforced(client, monkeypatch):
    monkeypatch.setenv("ACCOUNTANT_RL_PARTICIPANTS", "2/60")
    LIMITER.reset()
    try:
        assert client.post("/api/v1/participants", json={}).status_code == 201
        assert client.post("/api/v1/participants", json={}).status_code == 201
        blocked = client.post("/api/v1/participants", json={})
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers
    finally:
        monkeypatch.delenv("ACCOUNTANT_RL_PARTICIPANTS", raising=False)
        LIMITER.reset()


def test_experiment_seed_is_respected(client, research_headers):
    experiment = client.post("/api/v1/experiments", json={
        "name": "seed-check",
        "hypothesis": "the experiment seed is used",
        "protocol": {"synthetic": {"n_profiles": 40, "exposure_k": 8},
                     "protocol": {"bootstrap_iters": 30}},
        "random_seed": 4242,
    }, headers=research_headers).json()

    run = client.post(f"/api/v1/experiments/{experiment['experiment_id']}/runs",
                      json={}, headers=research_headers).json()
    status = client.get(f"/api/v1/runs/{run['run_id']}", headers=research_headers).json()
    assert status["seed"] == 4242
    assert status["comparison"]["mutual"]["ci_width"] is not None


def test_failed_run_is_persisted(client, research_headers):
    experiment = client.post("/api/v1/experiments", json={
        "name": "invalid-run",
        "hypothesis": "config validation fails before modelling",
        "protocol": {"synthetic": {"n_profiles": 10, "exposure_k": 5}},
        "random_seed": 1,
    }, headers=research_headers).json()

    response = client.post(f"/api/v1/experiments/{experiment['experiment_id']}/runs",
                           json={}, headers=research_headers)
    assert response.status_code in (400, 500)

    runs = client.get(f"/api/v1/experiments/{experiment['experiment_id']}/runs",
                      headers=research_headers).json()["items"]
    assert runs
    assert runs[0]["status"] in {"failed", "running", "queued"}


def test_health_reports_database(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["participants"] >= 0


def test_experiment_run_endpoint(client, research_headers):
    experiment = client.post("/api/v1/experiments", json={
        "name": "integration-run",
        "hypothesis": "pipeline works",
        "protocol": {"synthetic": {"n_profiles": 40, "exposure_k": 8},
                     "protocol": {"bootstrap_iters": 30}},
        "random_seed": 5,
    }, headers=research_headers).json()

    run = client.post(f"/api/v1/experiments/{experiment['experiment_id']}/runs",
                      json={}, headers=research_headers)
    assert run.status_code == 201, run.text
    body = run.json()
    assert body["status"] == "completed"
    assert body["comparison"]["mutual"]["best_model"]

    status = client.get(f"/api/v1/runs/{body['run_id']}", headers=research_headers).json()
    assert status["status"] == "completed"

    results = client.get(f"/api/v1/runs/{body['run_id']}/results",
                         headers=research_headers).json()
    assert results["protocol"]["leakage"]["disjoint_participants"] is True
