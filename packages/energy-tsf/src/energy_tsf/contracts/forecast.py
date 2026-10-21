"""Contracts at the deterministic forecast-task boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from energy_tsf.contracts.common import ContractModel
from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.contracts.station import StationSpec


def _require_aware(value: datetime | None) -> datetime | None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError("timestamp must include a UTC offset")
    return value


class ForecastRequest(ContractModel):
    """Authoritative structured request produced by the external host Agent."""

    request_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    station: StationSpec
    data_uri: str = Field(min_length=1)
    future_covariates_uri: str | None = None
    horizon: Literal["1d"] = "1d"
    prediction_mode: Literal["univariate", "multivariate"] = "univariate"
    covariates: CovariateSpec = Field(default_factory=CovariateSpec)
    issue_time: datetime | None = None
    allow_similar: bool = True
    allow_similar_on_direct_failure: bool = False
    max_source_solutions: int = Field(default=3, ge=1, le=3)
    deadline_seconds: int | None = Field(default=None, gt=0)
    task_type: Literal["forecast"] = "forecast"

    @field_validator("issue_time")
    @classmethod
    def validate_issue_time(cls, value: datetime | None) -> datetime | None:
        return _require_aware(value)

    @model_validator(mode="after")
    def validate_multivariate_input(self) -> "ForecastRequest":
        reserved = {self.station.target}
        if self.station.timestamp_column is not None:
            reserved.add(self.station.timestamp_column)
        overlap = sorted(reserved & set(self.covariates.columns))
        if overlap:
            raise ValueError(f"target/timestamp cannot be covariates: {', '.join(overlap)}")
        if self.prediction_mode == "multivariate":
            if not self.covariates.multivariate:
                raise ValueError("multivariate forecast requires covariates")
            if self.covariates.known_future and not self.future_covariates_uri:
                raise ValueError("known-future covariates require future_covariates_uri")
            if self.station.timestamp_column is None:
                raise ValueError("multivariate forecast requires a timestamp column")
        elif self.covariates.multivariate or self.future_covariates_uri is not None:
            raise ValueError("univariate forecast cannot declare covariate inputs")
        return self


class SimilarityScore(ContractModel):
    """Transparent score components applied only after hard compatibility."""

    technology: float = Field(default=0.0, ge=0, le=1)
    location: float = Field(default=0.0, ge=0, le=1)
    capacity: float = Field(default=0.0, ge=0, le=1)
    season: float = Field(default=0.0, ge=0, le=1)
    quality: float = Field(default=0.0, ge=0, le=1)
    recency: float = Field(default=0.0, ge=0, le=1)

    @property
    def weighted_total(self) -> float:
        return (
            0.25 * self.technology
            + 0.25 * self.location
            + 0.15 * self.capacity
            + 0.15 * self.season
            + 0.15 * self.quality
            + 0.05 * self.recency
        )


class SimilarSolutionCandidate(ContractModel):
    """One Active solution considered by deterministic retrieval."""

    station_id: str
    solution_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatible: bool
    required_history: int = Field(default=0, ge=0)
    available_members: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    score: SimilarityScore = Field(default_factory=SimilarityScore)
    total_score: float = Field(default=0.0, ge=0, le=1)
    recommended_weight: float = Field(default=0.0, ge=0, le=1)
    scale_method: Literal["none", "capacity_ratio"] = "none"
    distance_km: float | None = Field(default=None, ge=0)
    capacity_ratio: float | None = Field(default=None, gt=0)
    timezone_match: bool = False
    technology_match: bool | None = None
    quality_metric: float | None = Field(default=None, ge=0)
    source_created_at: str | None = None

    @model_validator(mode="after")
    def validate_compatibility_evidence(self) -> "SimilarSolutionCandidate":
        if self.compatible and self.rejection_reasons:
            raise ValueError("compatible candidate cannot have rejection reasons")
        if not self.compatible and not self.rejection_reasons:
            raise ValueError("incompatible candidate must explain why it was rejected")
        return self


class ForecastAssetInspection(ContractModel):
    """Read-only retrieval result presented to the external Agent."""

    mode: Literal["direct", "similar", "none"]
    direct: SimilarSolutionCandidate | None = None
    candidates: tuple[SimilarSolutionCandidate, ...] = ()
    rejected: tuple[SimilarSolutionCandidate, ...] = ()

    @model_validator(mode="after")
    def validate_mode(self) -> "ForecastAssetInspection":
        if self.mode == "direct" and self.direct is None:
            raise ValueError("direct inspection requires a direct candidate")
        if self.mode != "direct" and self.direct is not None:
            raise ValueError("only direct inspection may contain a direct candidate")
        if self.mode == "similar" and not self.candidates:
            raise ValueError("similar inspection requires compatible candidates")
        return self


class ForecastSourcePlan(ContractModel):
    """One frozen source Solution and its outer-ensemble weight."""

    station_id: str = Field(min_length=1)
    solution_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    weight: float = Field(ge=0, le=1)
    scale_method: Literal["none", "capacity_ratio"] = "none"
    selection_reasons: tuple[str, ...] = Field(min_length=1)


class ForecastSelection(ContractModel):
    """Small public choice written by the external Agent after inspection."""

    sources: tuple[ForecastSourcePlan, ...] = Field(min_length=1, max_length=3)
    renormalize_on_source_failure: bool = True


class ForecastPlan(ContractModel):
    """Immutable execution choice frozen after retrieval and Agent review."""

    plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str = Field(min_length=1)
    target_station_id: str = Field(min_length=1)
    mode: Literal["direct", "similar"]
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    sources: tuple[ForecastSourcePlan, ...] = Field(min_length=1, max_length=3)
    renormalize_on_source_failure: bool = True
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _require_aware(value)  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_sources(self) -> "ForecastPlan":
        keys = [(source.station_id, source.solution_id) for source in self.sources]
        if len(set(keys)) != len(keys):
            raise ValueError("forecast plan cannot repeat a source solution")
        if abs(sum(source.weight for source in self.sources) - 1.0) > 1e-8:
            raise ValueError("forecast source weights must sum to one")
        if self.mode == "direct":
            if len(self.sources) != 1 or self.sources[0].station_id != self.target_station_id:
                raise ValueError("direct plan must contain one source from the target station")
            if self.sources[0].scale_method != "none":
                raise ValueError("direct plan must not apply cross-station scaling")
        elif any(source.station_id == self.target_station_id for source in self.sources):
            raise ValueError("similar plan sources must come from other stations")
        return self


ForecastStatus = Literal[
    "prepared",
    "delivered",
    "rejected",
    "no_compatible_solution",
    "deadline_exceeded",
    "failed",
]


class ForecastResult(ContractModel):
    """Small machine-readable summary returned by the forecast workflow."""

    request_id: str
    status: ForecastStatus
    plan_id: str | None = None
    input_fingerprint: str | None = None
    forecast_path: str | None = None
    report_path: str | None = None
    source_solution_ids: tuple[str, ...] = ()
    succeeded_members: tuple[str, ...] = ()
    failed_members: dict[str, str] = Field(default_factory=dict)
    degraded: bool = False
    warnings: tuple[str, ...] = ()


class ConstraintSummary(ContractModel):
    minimum: float | None = None
    maximum: float | None = None
    modified_points: int = Field(ge=0)
    max_abs_adjustment: float = Field(ge=0)
