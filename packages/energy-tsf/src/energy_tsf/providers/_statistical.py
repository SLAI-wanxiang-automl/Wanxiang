"""Shared behavior for fitted statsmodels univariate providers."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy_tsf.contracts.result import ForecastTable
from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.data.schemas import PreparedData
from energy_tsf.providers._common import forecast_table, load_joblib, save_joblib
from energy_tsf.providers.base import TrainableProvider
from energy_tsf.tools.features import FeatureSpec


@dataclass(frozen=True)
class FittedStatisticalModel:
    result: Any
    train_end: int
    params: dict[str, Any]


class StatisticalProvider(TrainableProvider):
    def available(self) -> tuple[bool, str | None]:
        available = importlib.util.find_spec("statsmodels") is not None
        return available, None if available else "missing optional dependency: statsmodels"

    def fit_result(self, values: np.ndarray, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def fit(
        self,
        series: PreparedData,
        origins: tuple[int, ...],
        *,
        train_end: int,
        params: dict[str, Any],
        feature_spec: FeatureSpec | None = None,
        future_index: pd.Index | None = None,
        covariate_spec: CovariateSpec | None = None,
        future_covariates: pd.DataFrame | None = None,
    ) -> FittedStatisticalModel:
        validated = self.validate_params(params)
        values = series.frame[series.target].to_numpy(dtype=float)[:train_end]
        context_length = validated.get("context_length")
        if context_length is not None:
            values = values[-context_length:]
        fit_params = {key: value for key, value in validated.items() if key != "context_length"}
        return FittedStatisticalModel(self.fit_result(values, fit_params), train_end, validated)

    def predict(
        self,
        model: FittedStatisticalModel,
        series: PreparedData,
        origins: tuple[int, ...],
        horizon_steps: int,
        *,
        params: dict[str, Any],
        feature_spec: FeatureSpec | None = None,
        future_index: pd.Index | None = None,
        covariate_spec: CovariateSpec | None = None,
        future_covariates: pd.DataFrame | None = None,
    ) -> ForecastTable:
        values = series.frame[series.target].to_numpy(dtype=float)
        predictions: list[np.ndarray] = []
        for origin in origins:
            start = 0
            context_length = model.params.get("context_length")
            if context_length is not None:
                start = max(0, origin - context_length)
            updated = model.result.apply(values[start:origin], refit=False)
            predictions.append(np.asarray(updated.forecast(horizon_steps), dtype=float))
        return forecast_table(self.name, series, origins, predictions, future_index=future_index)

    def save(self, model: FittedStatisticalModel, path: str | Path) -> Path:
        return save_joblib(model, path)

    def load(self, path: str | Path) -> FittedStatisticalModel:
        model = load_joblib(path)
        if not isinstance(model, FittedStatisticalModel):
            raise TypeError("artifact is not a fitted statistical model")
        return model
