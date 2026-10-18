"""Pydantic request/response schemas (API v1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..features.schema import FEATURE_SCHEMA_VERSION

TaskName = Literal["compatibility_score", "directional_interest", "mutual_interest",
                   "interaction_quality"]


class ParticipantCreate(BaseModel):
    source: Literal["pilot", "synthetic"] = "pilot"


class ParticipantCreated(BaseModel):
    participant_id: str
    public_code: str
    participant_token: str = Field(description="Shown once. Store it client-side.")
    study_version: str


class ConsentRequest(BaseModel):
    study_version: str
    purpose: str = "research_v1"
    consented: bool


class ProfileFeatures(BaseModel):
    traits: dict[str, float] = Field(default_factory=dict)
    preferences: dict[str, dict] = Field(default_factory=dict)
    goal: str | None = None
    flags: dict[str, bool] = Field(default_factory=dict)
    dealbreakers: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    text: str | None = None

    @field_validator("text")
    @classmethod
    def cap_text(cls, value: str | None) -> str | None:
        if value is not None and len(value) > 4000:
            raise ValueError("text must be at most 4000 characters")
        return value


class ProfileIn(ProfileFeatures):
    feature_schema_version: str = FEATURE_SCHEMA_VERSION


class ProfileCreated(BaseModel):
    profile_id: str
    version: int
    feature_schema_version: str


class ScoreRequest(BaseModel):
    profile_a_id: str
    profile_b_id: str
    model_name: str = "structured_baseline"
    model_version: str | None = None


class RankRequest(BaseModel):
    profile_id: str
    model_name: str = "structured_baseline"
    limit: int = Field(default=10, ge=1, le=50)


class FeedbackIn(BaseModel):
    feedback_type: Literal["interest", "conversation_quality", "continued_interest"]
    other_profile_id: str
    value: bool | int
    prediction_id: str | None = None
    payload: dict = Field(default_factory=dict)


class ExperimentIn(BaseModel):
    name: str
    hypothesis: str
    protocol: dict = Field(default_factory=dict)
    random_seed: int = 20261018


class RunIn(BaseModel):
    config: dict = Field(default_factory=dict)
