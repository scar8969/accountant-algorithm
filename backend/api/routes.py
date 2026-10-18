"""API v1 routes.

Rules enforced here (blueprint section 6):
  * feature ranges validated before storage;
  * participant-specific operations require a matching participant token;
  * research operations require the research key;
  * match responses never include the other person's private answers;
  * consent is checked before any research write;
  * feedback is idempotent;
  * withdrawal deletes data.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..algorithms.registry import build_model, registered_models
from ..core import consent as consent_mod
from ..core.ratelimit import limit as rate_limit
from ..core.security import (authorise_participant, hash_token, new_token,
                             participant_from_token, require_research_key)
from ..db.models import (Experiment, ExperimentRun, Feedback, PairOutcome,
                         Participant, Prediction, Profile)
from ..db.session import get_db
from ..experiments.runner import DEFAULT_CONFIG, deep_merge, run_experiment
from ..features.schema import FEATURE_SCHEMA_VERSION
from ..features.structured import explanation, score_pair
from .schemas import (ConsentRequest, ExperimentIn, FeedbackIn, ParticipantCreate,
                      ParticipantCreated, ProfileCreated, ProfileIn, RankRequest,
                      RunIn, ScoreRequest)

router = APIRouter()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _profile_payload(profile: Profile) -> dict:
    return {
        "profile_id": profile.id,
        "version": profile.version,
        "feature_schema_version": profile.feature_schema_version,
        "features": profile.features,
        "created_at": profile.created_at.isoformat(),
    }


def _latest_profile(session: Session, participant_id: str) -> Profile | None:
    return session.execute(
        select(Profile).where(Profile.participant_id == participant_id)
        .order_by(Profile.version.desc())
    ).scalars().first()


def _validate_features(payload: ProfileIn, *, expected_schema: str | None) -> dict:
    from ..features.structured import validate_profile
    features = payload.model_dump(exclude={"feature_schema_version"})
    errors = validate_profile(features)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    if expected_schema and payload.feature_schema_version != expected_schema:
        raise HTTPException(status_code=400,
                            detail=f"unsupported feature_schema_version "
                                   f"'{payload.feature_schema_version}', "
                                   f"expected '{expected_schema}'")
    return features


# --------------------------------------------------------------------------- #
# participants and consent
# --------------------------------------------------------------------------- #
@router.post("/participants", response_model=ParticipantCreated, status_code=201,
             dependencies=[Depends(rate_limit("participants",
                                              "ACCOUNTANT_RL_PARTICIPANTS",
                                              "200/60"))])
def create_participant(payload: ParticipantCreate, session: Session = Depends(get_db)):
    token = new_token()
    participant = Participant(source=payload.source, token_hash=hash_token(token))
    session.add(participant)
    session.flush()
    return ParticipantCreated(participant_id=participant.id,
                              public_code=participant.public_code,
                              participant_token=token,
                              study_version=consent_mod.STUDY_VERSION)


@router.post("/participants/{participant_id}/consent")
def give_consent(participant_id: str, payload: ConsentRequest,
                 session: Session = Depends(get_db),
                 x_participant_token: str | None = Header(default=None,
                                                          alias="X-Participant-Token")):
    authorise_participant(session, participant_id, x_participant_token)
    if payload.study_version != consent_mod.STUDY_VERSION:
        raise HTTPException(
            status_code=409,
            detail={"error": "unsupported study version",
                    "supported": consent_mod.STUDY_VERSION})
    if payload.purpose != consent_mod.PURPOSE:
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported consent purpose",
                    "supported": consent_mod.PURPOSE})
    record = consent_mod.record_consent(session, participant_id, payload.consented,
                                        study_version=payload.study_version,
                                        purpose=payload.purpose)
    result = {"consent_id": record.id, "consented": record.consented,
              "study_version": record.study_version, "purpose": record.purpose}
    if not payload.consented:
        result["deleted"] = consent_mod.withdraw(session, participant_id)["deleted"]
    return result


@router.post("/participants/{participant_id}/export")
def export_participant(participant_id: str, session: Session = Depends(get_db),
                       x_participant_token: str | None = Header(default=None,
                                                                alias="X-Participant-Token")):
    authorise_participant(session, participant_id, x_participant_token)
    return consent_mod.export(session, participant_id)


@router.post("/participants/{participant_id}/withdraw")
def withdraw_participant(participant_id: str, session: Session = Depends(get_db),
                         x_participant_token: str | None = Header(default=None,
                                                                  alias="X-Participant-Token")):
    authorise_participant(session, participant_id, x_participant_token)
    return consent_mod.withdraw(session, participant_id)


# --------------------------------------------------------------------------- #
# profiles
# --------------------------------------------------------------------------- #
@router.post("/profiles", response_model=ProfileCreated, status_code=201)
def create_profile(payload: ProfileIn, session: Session = Depends(get_db),
                   x_participant_token: str | None = Header(default=None,
                                                            alias="X-Participant-Token")):
    participant = participant_from_token(session, x_participant_token)
    consent_mod.require_consent(session, participant.id)
    features = _validate_features(payload, expected_schema=FEATURE_SCHEMA_VERSION)
    existing = _latest_profile(session, participant.id)
    version = 1 if existing is None else existing.version + 1
    profile = Profile(participant_id=participant.id, version=version,
                      feature_schema_version=FEATURE_SCHEMA_VERSION, features=features)
    session.add(profile)
    session.flush()
    return ProfileCreated(profile_id=profile.id, version=version,
                          feature_schema_version=FEATURE_SCHEMA_VERSION)


@router.post("/profiles/{profile_id}/versions", response_model=ProfileCreated,
             status_code=201)
def add_profile_version(profile_id: str, payload: ProfileIn,
                        session: Session = Depends(get_db),
                        x_participant_token: str | None = Header(default=None,
                                                                 alias="X-Participant-Token")):
    base = session.get(Profile, profile_id)
    if base is None:
        raise HTTPException(status_code=404, detail="profile not found")
    authorise_participant(session, base.participant_id, x_participant_token)
    consent_mod.require_consent(session, base.participant_id)
    features = _validate_features(payload, expected_schema=FEATURE_SCHEMA_VERSION)
    latest = _latest_profile(session, base.participant_id)
    version = (latest.version if latest else 0) + 1
    profile = Profile(participant_id=base.participant_id, version=version,
                      feature_schema_version=FEATURE_SCHEMA_VERSION, features=features)
    session.add(profile)
    session.flush()
    return ProfileCreated(profile_id=profile.id, version=version,
                          feature_schema_version=FEATURE_SCHEMA_VERSION)


@router.get("/profiles/{profile_id}")
def get_profile(profile_id: str, session: Session = Depends(get_db),
                x_participant_token: str | None = Header(default=None,
                                                         alias="X-Participant-Token"),
                x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    profile = session.get(Profile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="profile not found")

    if x_api_key:
        require_research_key(x_api_key)
    else:
        participant = participant_from_token(session, x_participant_token)
        if participant.id != profile.participant_id:
            raise HTTPException(status_code=403, detail="not your profile")
    return _profile_payload(profile)


# --------------------------------------------------------------------------- #
# compatibility
# --------------------------------------------------------------------------- #
def _load_pair(session: Session, a_id: str, b_id: str) -> tuple[Profile, Profile]:
    a, b = session.get(Profile, a_id), session.get(Profile, b_id)
    if a is None or b is None:
        raise HTTPException(status_code=404, detail="profile not found")
    if a.participant_id == b.participant_id:
        raise HTTPException(status_code=400, detail="cannot score a profile against itself")
    return a, b


def _cfg_hash(cfg: dict) -> str:
    import hashlib
    import json as _json
    payload = _json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _model_for(name: str, seed: int = 0):
    try:
        return build_model(name, seed=seed)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models", dependencies=[Depends(require_research_key)])
def list_models():
    out = []
    for name in registered_models():
        model = build_model(name)
        out.append(model.get_config())
    return {"feature_schema_version": FEATURE_SCHEMA_VERSION, "models": out}


@router.post("/compatibility/score",
             dependencies=[Depends(rate_limit("score", "ACCOUNTANT_RL_SCORE", "240/60"))])
def compatibility_score(payload: ScoreRequest, session: Session = Depends(get_db),
                        x_participant_token: str | None = Header(default=None,
                                                                 alias="X-Participant-Token"),
                        x_api_key: str | None = Header(default=None,
                                                       alias="X-API-Key")):
    a, b = _load_pair(session, payload.profile_a_id, payload.profile_b_id)
    for profile in (a, b):
        if profile.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise HTTPException(
                status_code=409,
                detail={"error": "profile uses an unsupported feature schema",
                        "found": profile.feature_schema_version,
                        "supported": FEATURE_SCHEMA_VERSION,
                        "action": "re-submit the profile under the current schema"})

    if x_api_key:
        require_research_key(x_api_key)
    else:
        participant = participant_from_token(session, x_participant_token)
        if participant.id not in (a.participant_id, b.participant_id):
            raise HTTPException(status_code=403, detail="not authorised for this pair")

    model = _model_for(payload.model_name)
    if payload.model_version and payload.model_version != model.version:
        raise HTTPException(status_code=400,
                            detail=f"model_version {payload.model_version} not available "
                                   f"(have {model.version})")

    ps = score_pair(a.features, b.features)
    exp = explanation(ps)

    prediction = Prediction(profile_a=a.id, profile_b=b.id, task="compatibility_score",
                            score=ps.score, probability=None, explanation=exp)
    session.add(prediction)
    session.flush()

    return {
        "task": "compatibility_score",
        "prediction_id": prediction.id,
        "model": {"name": model.name, "version": model.version,
                  "feature_schema_version": FEATURE_SCHEMA_VERSION},
        "score": ps.score,
        "scale": "0-100",
        "probability": None,
        "sufficient_data": not ps.insufficient,
        "coverage": ps.coverage,
        "evidence": ps.evidence,
        "uncertainty": ps.uncertainty,
        "dimensions": ps.dimensions,
        "explanation": exp,
        "gates": {
            "a_disqualifies_b": ps.gate_a,
            "b_disqualifies_a": ps.gate_b,
        },
        "limitations": ps.limitations,
    }


@router.post("/compatibility/rank")
def compatibility_rank(payload: RankRequest, session: Session = Depends(get_db),
                       x_participant_token: str | None = Header(default=None,
                                                                alias="X-Participant-Token")):
    query = session.get(Profile, payload.profile_id)
    if query is None:
        raise HTTPException(status_code=404, detail="profile not found")
    authorise_participant(session, query.participant_id, x_participant_token)
    consent_mod.require_consent(session, query.participant_id)

    rows = session.execute(
        select(Profile, Participant)
        .join(Participant, Participant.id == Profile.participant_id)
        .where(Participant.status == "active", Participant.id != query.participant_id)
    ).all()

    # Only latest version per participant, and only participants with consent.
    latest: dict[str, Profile] = {}
    for profile, participant in rows:
        if not consent_mod.has_active_consent(session, participant.id):
            continue
        current = latest.get(participant.id)
        if current is None or profile.version > current.version:
            latest[participant.id] = profile

    model = _model_for(payload.model_name)
    scored = []
    skipped_schema = 0
    for candidate in latest.values():
        if candidate.feature_schema_version != query.feature_schema_version:
            skipped_schema += 1
            continue
        ps = score_pair(query.features, candidate.features)
        if ps.score is None or ps.gate_a:
            continue
        scored.append((ps, candidate))
    scored.sort(key=lambda item: item[0].score, reverse=True)

    results = []
    for rank, (ps, candidate) in enumerate(scored[: payload.limit], start=1):
        exp = explanation(ps, top=2)
        session.add(Prediction(profile_a=query.id, profile_b=candidate.id,
                               task="rank_candidates", score=ps.score,
                               rank_position=rank, explanation=exp))
        results.append({
            "rank": rank,
            "profile_id": candidate.id,
            "score": ps.score,
            "scale": "0-100",
            "dimensions": ps.dimensions,
            "positive_signals": exp["contributions"],
            "conflicts": exp["conflicts"],
            "missing": exp["missing"],
            "note": ("Aggregate comparison only. The candidate's answers and identity "
                     "are not disclosed."),
        })
    if not results:
        return {"items": [], "status": "insufficient_data",
                "message": "No eligible candidates with sufficient answers."}

    session.flush()
    return {
        "items": results,
        "model": {"name": model.name, "version": model.version},
        "candidates_considered": len(scored),
        "excluded_insufficient_or_gated": (len(latest) - len(scored) - skipped_schema),
        "excluded_schema_mismatch": skipped_schema,
        "limitations": ["Score is an experimental ranking signal, not a probability.",
                        "Ranking excludes deal-breaker conflicts in this direction."],
    }


# --------------------------------------------------------------------------- #
# feedback
# --------------------------------------------------------------------------- #
@router.post("/feedback", status_code=201,
             dependencies=[Depends(rate_limit("feedback", "ACCOUNTANT_RL_FEEDBACK",
                                              "120/60"))])
def submit_feedback(payload: FeedbackIn, session: Session = Depends(get_db),
                    x_participant_token: str | None = Header(default=None,
                                                             alias="X-Participant-Token")):
    participant = participant_from_token(session, x_participant_token)
    consent_mod.require_consent(session, participant.id)

    own = _latest_profile(session, participant.id)
    if own is None:
        raise HTTPException(status_code=400, detail="create a profile before feedback")
    other = session.get(Profile, payload.other_profile_id)
    if other is None:
        raise HTTPException(status_code=404, detail="other profile not found")
    if other.participant_id == participant.id:
        raise HTTPException(status_code=400, detail="cannot give feedback on your own profile")

    dedupe_key = payload.prediction_id or f"{own.id}:{other.id}:{payload.feedback_type}"
    existing = session.execute(
        select(Feedback).where(Feedback.participant_id == participant.id,
                               Feedback.dedupe_key == dedupe_key)
    ).scalars().first()
    if existing is not None:
        return {"feedback_id": existing.id, "duplicate": True,
                "message": "identical feedback already recorded"}

    if payload.feedback_type == "interest" and not isinstance(payload.value, bool):
        raise HTTPException(status_code=422, detail="interest feedback requires a boolean")
    if payload.feedback_type == "conversation_quality":
        if not isinstance(payload.value, int) or not 1 <= payload.value <= 5:
            raise HTTPException(status_code=422,
                                detail="conversation_quality requires an integer 1-5")
    if payload.feedback_type == "continued_interest" and not isinstance(payload.value, bool):
        raise HTTPException(status_code=422,
                            detail="continued_interest requires a boolean")

    a_id, b_id = sorted([own.id, other.id])
    outcome = session.execute(
        select(PairOutcome).where(PairOutcome.profile_a == a_id,
                                  PairOutcome.profile_b == b_id)
    ).scalars().first()
    if outcome is None:
        outcome = PairOutcome(profile_a=a_id, profile_b=b_id,
                              exposure_recorded=payload.prediction_id is not None)
        session.add(outcome)

    own_side_is_a = own.id == a_id
    if payload.feedback_type == "interest":
        if own_side_is_a:
            outcome.interest_a = bool(payload.value)
        else:
            outcome.interest_b = bool(payload.value)
    elif payload.feedback_type == "conversation_quality":
        if own_side_is_a:
            outcome.conversation_quality_a = int(payload.value)
        else:
            outcome.conversation_quality_b = int(payload.value)
    else:
        if own_side_is_a:
            outcome.continued_interest_a = bool(payload.value)
        else:
            outcome.continued_interest_b = bool(payload.value)

    record = Feedback(participant_id=participant.id,
                      prediction_id=payload.prediction_id,
                      feedback_type=payload.feedback_type,
                      dedupe_key=dedupe_key,
                      rating=int(payload.value) if isinstance(payload.value, int)
                      and not isinstance(payload.value, bool) else None,
                      payload={**payload.payload, "dedupe_key": dedupe_key,
                               "other_profile_id": other.id})
    session.add(record)
    try:
        session.flush()
    except IntegrityError:
        # Two concurrent identical submissions: the unique index wins, we return
        # the row that landed first instead of raising.
        session.rollback()
        row = session.execute(
            select(Feedback).where(Feedback.participant_id == participant.id,
                                   Feedback.dedupe_key == dedupe_key)
        ).scalars().first()
        return {"feedback_id": row.id if row else None, "duplicate": True,
                "message": "identical feedback already recorded"}
    return {"feedback_id": record.id, "duplicate": False,
            "mutual_interest_recorded": (outcome.interest_a is not None
                                         and outcome.interest_b is not None)}


# --------------------------------------------------------------------------- #
# experiments and runs
# --------------------------------------------------------------------------- #
@router.post("/experiments", status_code=201,
             dependencies=[Depends(require_research_key)])
def create_experiment(payload: ExperimentIn, session: Session = Depends(get_db)):
    experiment = Experiment(name=payload.name, hypothesis=payload.hypothesis,
                            protocol=payload.protocol, random_seed=payload.random_seed)
    session.add(experiment)
    session.flush()
    return {"experiment_id": experiment.id, "status": experiment.status,
            "random_seed": experiment.random_seed}


@router.post("/experiments/{experiment_id}/runs", status_code=201,
             dependencies=[Depends(require_research_key)])
def start_run(experiment_id: str, payload: RunIn, session: Session = Depends(get_db)):
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="experiment not found")

    protocol_cfg = experiment.protocol or {}
    cfg = deep_merge(DEFAULT_CONFIG, protocol_cfg)
    cfg = deep_merge(cfg, payload.config or {})
    # The experiment's own seed must win unless the protocol or the request sets one;
    # DEFAULT_CONFIG always carries a seed, so a naive merge would silently ignore it.
    if "seed" not in protocol_cfg and "seed" not in (payload.config or {}):
        cfg["seed"] = int(experiment.random_seed)
    cfg["seed"] = int(cfg["seed"])

    n_profiles = int((cfg.get("synthetic") or {}).get("n_profiles", 0))
    if n_profiles > 400:
        raise HTTPException(
            status_code=400,
            detail=("runs started through the API are capped at 400 synthetic "
                    "profiles; run larger experiments from the CLI"))

    run = ExperimentRun(experiment_id=experiment.id, model_version_id=None,
                        dataset_version="pending",
                        split_config=cfg.get("protocol", {}), status="running",
                        started_at=datetime.now(timezone.utc))
    session.add(run)
    session.flush()

    try:
        results = run_experiment(cfg, write_artifacts=False)
    except Exception as exc:
        # Commit the failure before raising: get_db rolls back on exceptions, and a
        # silently vanishing failed run is worse than the error itself.
        run.status = "failed"
        run.completed_at = datetime.now(timezone.utc)
        run.metrics = {"error": str(exc), "config_hash": _cfg_hash(cfg)}
        session.commit()
        raise HTTPException(status_code=500, detail=f"run failed: {exc}") from exc

    run.status = "completed"
    run.completed_at = datetime.now(timezone.utc)
    run.dataset_version = results["dataset"]["version"]
    run.metrics = {
        "comparison": results["comparison"],
        "dataset": results["dataset"],
        "protocol": results["protocol"],
        "results": results["results"],
        "limitations": results["limitations"],
        "power_note": results.get("power_note"),
        "config_hash": results["experiment"]["config_hash"],
        "seed": results["experiment"]["seed"],
        "full_results": results,
    }
    session.flush()
    return {"run_id": run.id, "status": run.status,
            "primary_metric": results["experiment"]["primary_metric"],
            "comparison": results["comparison"]}


@router.get("/runs/{run_id}", dependencies=[Depends(require_research_key)])
def get_run(run_id: str, session: Session = Depends(get_db)):
    run = session.get(ExperimentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    metrics = run.metrics or {}
    return {
        "run_id": run.id, "experiment_id": run.experiment_id, "status": run.status,
        "dataset_version": run.dataset_version, "started_at": run.started_at,
        "completed_at": run.completed_at,
        "seed": metrics.get("seed"),
        "config_hash": metrics.get("config_hash"),
        "power_note": metrics.get("power_note"),
        "comparison": metrics.get("comparison"),
        "dataset": metrics.get("dataset"),
    }


@router.get("/experiments/{experiment_id}/runs",
            dependencies=[Depends(require_research_key)])
def list_runs(experiment_id: str, session: Session = Depends(get_db)):
    runs = session.execute(
        select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id)
        .order_by(ExperimentRun.started_at.desc())
    ).scalars().all()
    return {"items": [{"run_id": r.id, "status": r.status,
                       "started_at": r.started_at.isoformat() if r.started_at else None,
                       "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                       "dataset_version": r.dataset_version} for r in runs]}


@router.get("/runs/{run_id}/results",
            dependencies=[Depends(require_research_key)])
def get_run_results(run_id: str, session: Session = Depends(get_db)):
    run = session.get(ExperimentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    metrics = run.metrics or {}
    if "full_results" in metrics:
        return metrics["full_results"]
    return {"run_id": run.id, "status": run.status, "metrics": metrics}


@router.get("/health")
def health(session: Session = Depends(get_db)):
    """Liveness plus a real database round-trip, so a broken DB is not 'ok'."""
    try:
        session.execute(text("SELECT 1"))
        participants = session.execute(
            select(func.count()).select_from(Participant)).scalar_one()
    except Exception as exc:  # pragma: no cover - depends on broken infrastructure
        raise HTTPException(status_code=503,
                            detail=f"database unavailable: {type(exc).__name__}") from exc
    return {"status": "ok", "database": "ok", "participants": int(participants),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "time": datetime.now(timezone.utc).isoformat()}
