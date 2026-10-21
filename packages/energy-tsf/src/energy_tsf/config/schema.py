"""Strict schema for generated and resolved modeling TOML."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator

from energy_tsf.contracts.common import ContractModel
from energy_tsf.contracts.features import CovariateSpec, StatisticalSelectionSpec, WrapperSelectionSpec
from energy_tsf.contracts.modeling import SplitConfig, frequency_seconds
from energy_tsf.contracts.station import StationSpec


class RequestConfig(ContractModel):
    task_type: Literal["modeling"] = "modeling"
    request_id: str
    station_id: str
    target: str
    horizon: Literal["1d"] = "1d"
    prediction_mode: Literal["univariate", "multivariate"] = "univariate"
    top_k: int = Field(default=5, gt=0)
    publish: bool = True


class DataConfig(ContractModel):
    path: str
    timestamp_column: str | None = None
    frequency: str
    covariates: CovariateSpec = Field(default_factory=CovariateSpec)

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, value: str) -> str:
        value = value.strip().lower()
        frequency_seconds(value)
        return value


class CleaningConfig(ContractModel):
    missing: Literal["linear_interpolate"] = "linear_interpolate"
    edge_fill: Literal["nearest"] = "nearest"
    record_imputation_mask: bool = True


class FeatureConfig(ContractModel):
    daily_lags: tuple[int, ...] = (1, 2, 3, 7)
    recent_lags: int = Field(default=12, gt=0)
    rolling_windows: tuple[int, ...] = (6, 12, 144)
    calendar: bool = True
    target_differences: tuple[int, ...] = (1,)
    covariate_lags: tuple[int, ...] = (1, 4)
    covariate_differences: tuple[int, ...] = (1,)
    covariate_rolling_windows: tuple[int, ...] = (4, 16)
    statistical_selection: StatisticalSelectionSpec = Field(default_factory=StatisticalSelectionSpec)
    tsfm_wrapper: WrapperSelectionSpec = Field(default_factory=WrapperSelectionSpec)


class SearchProvidersConfig(ContractModel):
    enabled: tuple[str, ...]
    checkpoints: dict[str, str] = Field(default_factory=dict)


class SearchConfig(ContractModel):
    seed: int = Field(default=42, ge=0)
    primary_metric: Literal["rmse"] = "rmse"
    secondary_metric: Literal["mae"] = "mae"
    max_trials: int = Field(default=64, gt=0)
    max_parallel_trials: int = Field(default=2, gt=0)
    providers: SearchProvidersConfig


class EnsembleConfig(ContractModel):
    method: Literal["nnls"] = "nnls"
    fallback: Literal["inverse_rmse"] = "inverse_rmse"
    non_negative: bool = True
    normalize: bool = True


class StorageConfig(ContractModel):
    memory_root: str = "asset_store"
    work_root: str = "work_dirs/train"


class ModelingConfig(ContractModel):
    """Fully resolved task configuration consumed by deterministic code."""

    request: RequestConfig
    data: DataConfig
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    search: SearchConfig
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "ModelingConfig":
        _validate_covariates(
            self.request.prediction_mode,
            self.data.covariates,
            self.request.target,
            self.data.timestamp_column,
            "modeling",
        )
        if self.search.max_parallel_trials > self.search.max_trials:
            raise ValueError("max_parallel_trials cannot exceed max_trials")
        return self


class ForecastRequestConfig(ContractModel):
    task_type: Literal["forecast"] = "forecast"
    request_id: str
    horizon: Literal["1d"] = "1d"
    prediction_mode: Literal["univariate", "multivariate"] = "univariate"
    issue_time: datetime | None = None


class ForecastDataConfig(ContractModel):
    path: str
    future_covariates_path: str | None = None
    covariates: CovariateSpec = Field(default_factory=CovariateSpec)


class ForecastRetrievalConfig(ContractModel):
    allow_similar: bool = True
    allow_similar_on_direct_failure: bool = False
    max_source_solutions: int = Field(default=3, ge=1, le=3)
    deadline_seconds: int | None = Field(default=None, gt=0)


class ForecastStorageConfig(ContractModel):
    memory_root: str = "asset_store"
    work_root: str = "work_dirs/forecast"


class ForecastConfig(ContractModel):
    """Resolved forecast configuration consumed by deterministic code."""

    request: ForecastRequestConfig
    station: StationSpec
    data: ForecastDataConfig
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    retrieval: ForecastRetrievalConfig = Field(default_factory=ForecastRetrievalConfig)
    storage: ForecastStorageConfig = Field(default_factory=ForecastStorageConfig)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "ForecastConfig":
        _validate_covariates(
            self.request.prediction_mode,
            self.data.covariates,
            self.station.target,
            self.station.timestamp_column,
            "forecast",
        )
        if self.request.prediction_mode == "multivariate":
            if self.data.covariates.known_future and not self.data.future_covariates_path:
                raise ValueError("known-future covariates require future_covariates_path")
            if self.station.timestamp_column is None:
                raise ValueError("multivariate forecast requires a timestamp column")
        elif self.data.future_covariates_path is not None:
            raise ValueError("univariate forecast cannot declare future covariates")
        return self


class BacktestRequestConfig(ContractModel):
    task_type: Literal["backtest"] = "backtest"
    request_id: str
    station_id: str
    target: str
    horizon: Literal["1d"] = "1d"
    stride: Literal["1d"] = "1d"
    prediction_mode: Literal["univariate", "multivariate"] = "univariate"
    timezone: str
    publish: bool = True

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value


class BacktestDataConfig(ContractModel):
    path: str
    timestamp_column: str = Field(min_length=1)
    frequency: str
    covariates: CovariateSpec = Field(default_factory=CovariateSpec)

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, value: str) -> str:
        value = value.strip().lower()
        frequency_seconds(value)
        return value


class BayesianUpdateConfig(ContractModel):
    epsilon: float = Field(default=1e-3, gt=0)
    temperature: float = Field(default=0.10, gt=0)
    prior_strength: float | None = Field(default=None, gt=0)


class BacktestStorageConfig(ContractModel):
    memory_root: str = "asset_store"
    work_root: str = "work_dirs/backtest"


class BacktestConfig(ContractModel):
    request: BacktestRequestConfig
    data: BacktestDataConfig
    bayesian: BayesianUpdateConfig = Field(default_factory=BayesianUpdateConfig)
    storage: BacktestStorageConfig = Field(default_factory=BacktestStorageConfig)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "BacktestConfig":
        _validate_covariates(
            self.request.prediction_mode,
            self.data.covariates,
            self.request.target,
            self.data.timestamp_column,
            "backtest",
        )
        return self


class UpdateRequestConfig(ContractModel):
    task_type: Literal["update"] = "update"
    request_id: str
    station_id: str
    target: str
    horizon: Literal["1d"] = "1d"
    stride: Literal["1d"] = "1d"
    timezone: str
    validation_days: Literal[7] = 7
    prediction_mode: Literal["univariate", "multivariate"] = "univariate"
    top_k: int = Field(default=5, gt=0)
    publish: bool = True
    seed: int = Field(default=42, ge=0)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value


class UpdateDataConfig(ContractModel):
    path: str
    timestamp_column: str = Field(min_length=1)
    frequency: str
    covariates: CovariateSpec = Field(default_factory=CovariateSpec)

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, value: str) -> str:
        value = value.strip().lower()
        frequency_seconds(value)
        return value


class UpdateSearchConfig(ContractModel):
    max_parallel_trials: int = Field(default=2, gt=0)


class UpdateStorageConfig(ContractModel):
    memory_root: str = "asset_store"
    work_root: str = "work_dirs/update"


class UpdateConfig(ContractModel):
    request: UpdateRequestConfig
    data: UpdateDataConfig
    search: UpdateSearchConfig = Field(default_factory=UpdateSearchConfig)
    storage: UpdateStorageConfig = Field(default_factory=UpdateStorageConfig)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "UpdateConfig":
        _validate_covariates(
            self.request.prediction_mode,
            self.data.covariates,
            self.request.target,
            self.data.timestamp_column,
            "update",
        )
        return self


def _validate_covariates(
    prediction_mode: str,
    covariates: CovariateSpec,
    target: str,
    timestamp_column: str | None,
    task: str,
) -> None:
    reserved = {target}
    if timestamp_column is not None:
        reserved.add(timestamp_column)
    overlap = sorted(reserved & set(covariates.columns))
    if overlap:
        raise ValueError(f"target/timestamp cannot be covariates: {', '.join(overlap)}")
    if prediction_mode == "multivariate" and not covariates.multivariate:
        raise ValueError(f"multivariate {task} requires covariates")
    if prediction_mode == "univariate" and covariates.multivariate:
        raise ValueError(f"univariate {task} cannot declare covariates")
