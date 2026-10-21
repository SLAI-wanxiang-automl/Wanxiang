"""Contracts for deterministic rolling backtests and ensemble updates."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator

from energy_tsf.contracts.artifact import ActiveRef, CandidateRef
from energy_tsf.contracts.common import ContractModel
from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.contracts.modeling import frequency_seconds


class BacktestRequest(ContractModel):
    request_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    station_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    data_uri: str = Field(min_length=1)
    target: str = Field(min_length=1)
    timestamp_column: str = Field(min_length=1)
    frequency: str
    timezone: str = Field(min_length=1)
    horizon: Literal["1d"] = "1d"
    stride: Literal["1d"] = "1d"
    prediction_mode: Literal["univariate", "multivariate"] = "univariate"
    covariates: CovariateSpec = Field(default_factory=CovariateSpec)
    publish: bool = True
    epsilon: float = Field(default=1e-3, gt=0)
    temperature: float = Field(default=0.10, gt=0)
    prior_strength: float | None = Field(default=None, gt=0)
    task_type: Literal["backtest"] = "backtest"

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
    def validate_covariates(self) -> "BacktestRequest":
        reserved = {self.target, self.timestamp_column}
        overlap = sorted(reserved & set(self.covariates.columns))
        if overlap:
            raise ValueError(f"target/timestamp cannot be covariates: {', '.join(overlap)}")
        if self.prediction_mode == "multivariate" and not self.covariates.multivariate:
            raise ValueError("multivariate backtest requires covariates")
        if self.prediction_mode == "univariate" and self.covariates.multivariate:
            raise ValueError("univariate backtest cannot declare covariates")
        return self


class BacktestInputProfile(ContractModel):
    rows: int = Field(gt=0)
    start_time: datetime
    end_time: datetime
    window_count: int = Field(gt=0)
    horizon_steps: int = Field(gt=0)
    original_missing: int = Field(ge=0)
    irregular_intervals: int = Field(ge=0)
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class BacktestWindow(ContractModel):
    window_id: str = Field(min_length=1)
    origin: datetime
    end_time: datetime
    horizon_steps: int = Field(gt=0)
    valid_observations: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "BacktestWindow":
        if self.end_time <= self.origin:
            raise ValueError("backtest window end_time must be after origin")
        return self


class BacktestPlan(ContractModel):
    plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str
    station_id: str
    parent_solution_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_snapshot_id: str
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    member_trial_ids: tuple[str, ...]
    windows: tuple[BacktestWindow, ...]
    created_at: datetime

    @model_validator(mode="after")
    def validate_contents(self) -> "BacktestPlan":
        if not self.member_trial_ids:
            raise ValueError("backtest plan requires at least one member")
        if len(set(self.member_trial_ids)) != len(self.member_trial_ids):
            raise ValueError("backtest plan member ids must be unique")
        if len(self.windows) < 1:
            raise ValueError("backtest plan requires at least one window")
        return self


class WindowEvidence(ContractModel):
    window_id: str
    normalized_rmse: dict[str, float]
    responsibilities: dict[str, float]


class BayesianWeightUpdate(ContractModel):
    member_trial_ids: tuple[str, ...]
    previous_weights: tuple[float, ...]
    prior_weights: tuple[float, ...]
    prior_alpha: tuple[float, ...]
    window_evidence: tuple[WindowEvidence, ...]
    posterior_alpha: tuple[float, ...]
    new_weights: tuple[float, ...]
    epsilon: float = Field(gt=0)
    temperature: float = Field(gt=0)
    prior_strength: float = Field(gt=0)
    scale: float = Field(gt=0)
    scale_source: Literal["capacity", "truth_q95"]

    @model_validator(mode="after")
    def validate_vectors(self) -> "BayesianWeightUpdate":
        size = len(self.member_trial_ids)
        vectors = (
            self.previous_weights,
            self.prior_weights,
            self.prior_alpha,
            self.posterior_alpha,
            self.new_weights,
        )
        if not size or any(len(values) != size for values in vectors):
            raise ValueError("Bayesian weight vectors must match member_trial_ids")
        if any(value < 0 for values in vectors for value in values):
            raise ValueError("Bayesian weight vectors must be non-negative")
        for name, values in (("previous_weights", self.previous_weights), ("prior_weights", self.prior_weights), ("new_weights", self.new_weights)):
            if abs(sum(values) - 1.0) > 1e-8:
                raise ValueError(f"{name} must sum to one")
        return self


class BacktestResult(ContractModel):
    request_id: str
    status: Literal["evaluated", "released", "release_failed"]
    plan_id: str
    parent_solution_id: str
    new_solution_id: str | None = None
    candidate: CandidateRef | None = None
    active: ActiveRef | None = None
    published: bool = False
    predictions_path: str
    metrics_path: str
    weights_path: str
    report_path: str
    release_reasons: tuple[str, ...] = ()
