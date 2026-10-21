"""In-memory series containers used by the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from energy_tsf.contracts.data import DataQualityReport, ImputationReport, TemporalSplit
from energy_tsf.contracts.backtest import BacktestInputProfile, BacktestWindow
from energy_tsf.contracts.data import ForecastContextProfile
from energy_tsf.contracts.features import CovariateSpec


@dataclass(frozen=True)
class RawSeries:
    frame: pd.DataFrame
    target: str
    timestamp_column: str | None
    frequency: str
    covariates: CovariateSpec = CovariateSpec()


@dataclass(frozen=True)
class PreparedSeries:
    frame: pd.DataFrame
    target: str
    timestamp_column: str | None
    frequency: str
    imputation_mask: pd.Series
    split: TemporalSplit
    quality: DataQualityReport
    imputation: ImputationReport
    covariates: CovariateSpec = CovariateSpec()


@dataclass(frozen=True)
class PreparedHistory:
    """Clean observed history without modeling-only split semantics."""

    frame: pd.DataFrame
    target: str
    timestamp_column: str | None
    frequency: str
    imputation_mask: pd.Series
    quality: DataQualityReport
    covariates: CovariateSpec = CovariateSpec()


@dataclass(frozen=True)
class ForecastContext:
    history: PreparedHistory
    future_index: pd.Index
    profile: ForecastContextProfile
    future_covariates: pd.DataFrame | None = None

    @property
    def origin(self) -> int:
        return len(self.history.frame)

    @property
    def horizon_steps(self) -> int:
        return len(self.future_index)


@dataclass(frozen=True)
class BacktestInput:
    """Timestamped truth kept separate from the historical model context."""

    frame: pd.DataFrame
    target: str
    timestamp_column: str
    frequency: str
    observed_mask: pd.Series
    windows: tuple[BacktestWindow, ...]
    profile: BacktestInputProfile
    covariates: CovariateSpec = CovariateSpec()


@dataclass(frozen=True)
class BacktestWindowData:
    spec: BacktestWindow
    context: ForecastContext
    truth: pd.Series
    observed_mask: pd.Series


PreparedData = PreparedSeries | PreparedHistory
