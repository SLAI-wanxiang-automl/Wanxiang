"""Contracts for constrained AutoML updates of an existing station."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator

from energy_tsf.contracts.artifact import ActiveRef, CandidateRef
from energy_tsf.contracts.common import ContractModel
from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.contracts.modeling import TrialSpec, frequency_seconds


class UpdateRequest(ContractModel):
    request_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    station_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    data_uri: str = Field(min_length=1)
    target: str = Field(min_length=1)
    timestamp_column: str = Field(min_length=1)
    frequency: str
    timezone: str = Field(min_length=1)
    horizon: Literal["1d"] = "1d"
    stride: Literal["1d"] = "1d"
    validation_days: Literal[7] = 7
    prediction_mode: Literal["univariate", "multivariate"] = "univariate"
    covariates: CovariateSpec = Field(default_factory=CovariateSpec)
    top_k: int = Field(default=5, gt=0)
    max_parallel_trials: int = Field(default=2, gt=0)
    publish: bool = True
    seed: int = Field(default=42, ge=0)
    task_type: Literal["update"] = "update"

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, value: str) -> str:
        value = value.strip().lower()
        frequency_seconds(value)
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value

    @model_validator(mode="after")
    def validate_covariates(self) -> "UpdateRequest":
        reserved = {self.target, self.timestamp_column}
        overlap = sorted(reserved & set(self.covariates.columns))
        if overlap:
            raise ValueError(f"target/timestamp cannot be covariates: {', '.join(overlap)}")
        if self.prediction_mode == "multivariate" and not self.covariates.multivariate:
            raise ValueError("multivariate update requires covariates")
        if self.prediction_mode == "univariate" and self.covariates.multivariate:
            raise ValueError("univariate update cannot declare covariates")
        return self


class UpdateInputProfile(ContractModel):
    rows: int = Field(gt=0)
    train_rows: int = Field(gt=0)
    validation_rows: int = Field(gt=0)
    validation_days: Literal[7] = 7
    window_count: Literal[7] = 7
    horizon_steps: int = Field(gt=0)
    start_time: datetime
    end_time: datetime
    original_missing: int = Field(ge=0)
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class UpdatePlan(ContractModel):
    plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str
    station_id: str
    parent_solution_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_pool_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_end: int = Field(gt=0)
    validation_origins: tuple[int, ...]
    trials: tuple[TrialSpec, ...]
    unavailable: dict[str, str] = Field(default_factory=dict)
    created_at: datetime

    @model_validator(mode="after")
    def validate_plan(self) -> "UpdatePlan":
        if len(self.validation_origins) != 7:
            raise ValueError("update plan requires exactly seven validation origins")
        if not self.trials:
            raise ValueError("update plan requires at least one executable pool entry")
        if len({trial.trial_id for trial in self.trials}) != len(self.trials):
            raise ValueError("update trial ids must be unique")
        return self


class UpdateResult(ContractModel):
    request_id: str
    status: Literal["evaluated", "released", "release_failed", "insufficient_candidates"]
    plan_id: str
    parent_solution_id: str
    source_pool_id: str
    next_pool_id: str | None = None
    new_solution_id: str | None = None
    candidate: CandidateRef | None = None
    active: ActiveRef | None = None
    published: bool = False
    report_path: str
    leaderboard_path: str
    window_leaderboard_path: str
    result_path: str
    release_reasons: tuple[str, ...] = ()
