"""Serializable specifications for covariates and leakage-safe features."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from energy_tsf.contracts.common import ContractModel


class CovariateSpec(ContractModel):
    """Explicit availability roles for exogenous variables.

    ``past_only`` values may only be read strictly before a forecast origin.
    ``known_future`` values must also be supplied for the forecast horizon.
    """

    past_only: tuple[str, ...] = ()
    known_future: tuple[str, ...] = ()

    @field_validator("past_only", "known_future")
    @classmethod
    def normalize_columns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
        return normalized

    @model_validator(mode="after")
    def validate_disjoint(self) -> "CovariateSpec":
        overlap = sorted(set(self.past_only) & set(self.known_future))
        if overlap:
            raise ValueError(f"covariate roles overlap: {', '.join(overlap)}")
        return self

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.past_only, *self.known_future)

    @property
    def multivariate(self) -> bool:
        return bool(self.columns)


class StatisticalSelectionSpec(ContractModel):
    variance_threshold: float = Field(default=1e-10, ge=0)
    correlation_threshold: float = Field(default=0.98, gt=0, le=1)
    mutual_information_top_n: int = Field(default=64, gt=0)
    max_selected_features: int = Field(default=96, gt=0)


class WrapperSelectionSpec(ContractModel):
    max_selected_covariates: int = Field(default=4, gt=0)
    max_evaluations_per_provider: int = Field(default=16, gt=0)
    selection_windows: int = Field(default=3, gt=0)
    minimum_improvement: float = Field(default=0.0, ge=0)


class FeatureSelectionEvidence(ContractModel):
    """Frozen output of a train/validation-only feature selector."""

    method: Literal["none", "statistical", "wrapper"] = "none"
    selected: tuple[str, ...] = ()
    scores: dict[str, float] = Field(default_factory=dict)
    evaluated_subsets: tuple[tuple[str, ...], ...] = ()


class FeatureSpec(ContractModel):
    season_length: int = Field(gt=0)
    daily_lags: tuple[int, ...] = (1, 2, 3, 7)
    recent_lags: int = Field(default=12, gt=0)
    rolling_windows: tuple[int, ...] = (6, 12, 144)
    calendar: bool = True
    target_differences: tuple[int, ...] = ()
    covariate_lags: tuple[int, ...] = (1,)
    covariate_differences: tuple[int, ...] = (1,)
    covariate_rolling_windows: tuple[int, ...] = ()
    covariates: CovariateSpec = Field(default_factory=CovariateSpec)
    selected_features: tuple[str, ...] = ()

    @field_validator(
        "daily_lags",
        "rolling_windows",
        "target_differences",
        "covariate_lags",
        "covariate_differences",
        "covariate_rolling_windows",
    )
    @classmethod
    def validate_positive_unique(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value <= 0 for value in values):
            raise ValueError("feature lags and windows must be positive")
        return tuple(dict.fromkeys(values))

    @property
    def required_history(self) -> int:
        requirements = [self.recent_lags, 1]
        requirements.extend(day * self.season_length for day in self.daily_lags)
        requirements.extend(self.rolling_windows)
        requirements.extend(value + 1 for value in self.target_differences)
        requirements.extend(self.covariate_lags)
        requirements.extend(value + 1 for value in self.covariate_differences)
        requirements.extend(self.covariate_rolling_windows)
        return max(requirements)
