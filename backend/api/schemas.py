"""Pydantic request/response schemas (API v1)."""

from __future__ import annotations

import json
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

    @field_validator("traits", "preferences")
    @classmethod
    def cap_maps(cls, value: dict) -> dict:
        if len(value) > 20:
            raise ValueError("at most 20 entries allowed")
        if any(len(str(k)) > 40 for k in value):
            raise ValueError("key names must be at most 40 characters")
        return value

    @field_validator("interests")
    @classmethod
    def cap_interests(cls, value: list[str]) -> list[str]:
        if len(value) > 50:
            raise ValueError("at most 50 interests")
        if any(len(item) > 60 for item in value):
            raise ValueError("each interest must be at most 60 characters")
        return value

    @field_validator("flags", "dealbreakers")
    @classmethod
    def cap_flags(cls, value):
        if len(value) > 20:
            raise ValueError("at most 20 entries allowed")
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
    other_profile_id: str = Field(min_length=8, max_length=64)
    value: bool | int
    prediction_id: str | None = Field(default=None, max_length=64)
    payload: dict = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def cap_payload(cls, value: dict) -> dict:
        if len(json.dumps(value, default=str)) > 4096:
            raise ValueError("payload must be at most 4096 bytes when serialised")
        return value


class ExperimentIn(BaseModel):
    name: str = Field(min_length=3, max_length=120)
    hypothesis: str = Field(min_length=3, max_length=2000)
    protocol: dict = Field(default_factory=dict)
    random_seed: int = 20261018

    @field_validator("protocol")
    @classmethod
    def cap_protocol(cls, value: dict) -> dict:
        if len(json.dumps(value, default=str)) > 65536:
            raise ValueError("protocol config must be at most 64 KB")
        return value


class RunIn(BaseModel):
    config: dict = Field(default_factory=dict)

    @field_validator("config")
    @classmethod
    def cap_config(cls, value: dict) -> dict:
        if len(json.dumps(value, default=str)) > 65536:
            raise ValueError("run config must be at most 64 KB")
        return value
