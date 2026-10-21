"""Contracts used at the deterministic modeling-task boundary."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from energy_tsf.contracts.common import ContractModel
from energy_tsf.contracts.features import (
    CovariateSpec,
    FeatureSelectionEvidence,
    FeatureSpec,
    StatisticalSelectionSpec,
    WrapperSelectionSpec,
)

DEFAULT_PROVIDERS = (
    "arima",
    "sarima",
    "ridge",
    "lightgbm",
    "xgboost",
    "timesfm-2.5",
    "moirai-2",
    "chronos-2",
)
_FREQUENCY = re.compile(r"^(?P<count>[1-9]\d*)(?P<unit>s|min|h|d)$")
_UNIT_SECONDS = {"s": 1, "min": 60, "h": 3600, "d": 86400}


def frequency_seconds(value: str) -> int:
    """Return seconds for the canonical fixed-frequency syntax."""

    match = _FREQUENCY.fullmatch(value.strip().lower())
    if match is None:
        raise ValueError("frequency must look like 10min, 1h, or 1d")
    seconds = int(match.group("count")) * _UNIT_SECONDS[match.group("unit")]
    if seconds > 86400 or 86400 % seconds:
        raise ValueError("frequency must divide one day into equal steps")
    return seconds


class SplitConfig(ContractModel):
    """Fixed chronological split for the first modeling release."""

    train: float = 0.70
    validation: float = 0.20
    test: float = 0.10
    strategy: Literal["chronological"] = "chronological"

    @model_validator(mode="after")
    def validate_fixed_split(self) -> "SplitConfig":
        values = (self.train, self.validation, self.test)
        expected = (0.70, 0.20, 0.10)
        if any(abs(value - target) > 1e-9 for value, target in zip(values, expected)):
            raise ValueError("split must be the fixed chronological 0.70/0.20/0.10")
        return self


class SearchBudget(ContractModel):
    """Bounded local AutoML budget supplied by the Agent."""

    max_trials: int = Field(default=64, gt=0)
    max_parallel_trials: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_parallelism(self) -> "SearchBudget":
        if self.max_parallel_trials > self.max_trials:
            raise ValueError("max_parallel_trials cannot exceed max_trials")
        return self


class ModelingRequest(ContractModel):
    """Authoritative structured input produced by the external Agent."""

    request_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    station_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    data_uri: str = Field(min_length=1)
    target: str = Field(min_length=1)
    timestamp_column: str | None = None
    frequency: str
    horizon: Literal["1d"] = "1d"
    split: SplitConfig = Field(default_factory=SplitConfig)
    prediction_mode: Literal["univariate", "multivariate"] = "univariate"
    covariates: CovariateSpec = Field(default_factory=CovariateSpec)
    top_k: int = Field(default=5, gt=0)
    providers: tuple[str, ...] = DEFAULT_PROVIDERS
    foundation_checkpoints: dict[str, str] = Field(default_factory=dict)
    budget: SearchBudget = Field(default_factory=SearchBudget)
    publish: bool = True
    seed: int = Field(default=42, ge=0)
    task_type: Literal["modeling"] = "modeling"

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, value: str) -> str:
        value = value.strip().lower()
        frequency_seconds(value)
        return value

    @field_validator("timestamp_column")
    @classmethod
    def normalize_optional_column(cls, value: str | None) -> str | None:
        return value or None

    @field_validator("providers")
    @classmethod
    def normalize_providers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value.strip().lower() for value in values))
        if not normalized:
            raise ValueError("at least one provider is required")
        unknown = sorted(set(normalized) - set(DEFAULT_PROVIDERS))
        if unknown:
            raise ValueError(f"unsupported providers: {', '.join(unknown)}")
        return normalized

    @field_validator("foundation_checkpoints")
    @classmethod
    def validate_foundation_checkpoints(cls, values: dict[str, str]) -> dict[str, str]:
        allowed = {"timesfm-2.5", "moirai-2", "chronos-2"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"checkpoints are supported only for foundation providers: {', '.join(unknown)}")
        if any(not path.strip() for path in values.values()):
            raise ValueError("foundation checkpoint paths cannot be empty")
        return values

    @model_validator(mode="after")
    def validate_covariates(self) -> "ModelingRequest":
        reserved = {self.target}
        if self.timestamp_column is not None:
            reserved.add(self.timestamp_column)
        overlap = sorted(reserved & set(self.covariates.columns))
        if overlap:
            raise ValueError(f"target/timestamp cannot be covariates: {', '.join(overlap)}")
        if self.prediction_mode == "multivariate" and not self.covariates.multivariate:
            raise ValueError("multivariate modeling requires at least one covariate")
        if self.prediction_mode == "univariate" and self.covariates.multivariate:
            raise ValueError("univariate modeling cannot declare covariates")
        return self


class PreflightIssue(ContractModel):
    """Stable machine-readable preflight finding."""

    code: str
    message: str


class PreflightReport(ContractModel):
    """Result of cheap checks performed before a task directory is created."""

    valid: bool
    errors: tuple[PreflightIssue, ...] = ()
    warnings: tuple[PreflightIssue, ...] = ()


class TrialSpec(ContractModel):
    trial_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    provider: str
    params: dict[str, Any]
    input_mode: Literal["univariate", "multivariate"] = "univariate"
    covariate_spec: CovariateSpec = Field(default_factory=CovariateSpec)
    feature_spec: FeatureSpec | None = None
    feature_selection: FeatureSelectionEvidence = Field(default_factory=FeatureSelectionEvidence)
    statistical_selection: StatisticalSelectionSpec = Field(default_factory=StatisticalSelectionSpec)
    wrapper_selection: WrapperSelectionSpec = Field(default_factory=WrapperSelectionSpec)
    context_length: int | None = Field(default=None, gt=0)
    seed: int = Field(ge=0)
    data_snapshot_id: str
    split_id: str


class ExperimentPlan(ContractModel):
    plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    trials: tuple[TrialSpec, ...]
    unavailable: dict[str, str] = Field(default_factory=dict)
    max_trials: int = Field(gt=0)
    seed: int = Field(ge=0)


TrialStatus = Literal["planned", "running", "succeeded", "failed", "timed_out", "cancelled"]


class TrialRecord(ContractModel):
    spec: TrialSpec
    status: TrialStatus
    validation_metrics: dict[str, float | None] = Field(default_factory=dict)
    prediction_ref: str | None = None
    artifact_ref: str | None = None
    truth_fingerprint: str | None = None
    valid_observations: int = 0
    fit_duration_seconds: float = 0.0
    predict_duration_seconds: float = 0.0
    runtime: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    test_metrics: dict[str, float | None] = Field(default_factory=dict)


class TopKSelection(ContractModel):
    status: Literal["complete", "insufficient"]
    requested: int = Field(gt=0)
    records: tuple[TrialRecord, ...]
    truth_fingerprint: str | None = None
