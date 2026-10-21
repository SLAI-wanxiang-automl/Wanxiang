"""Shared direct-horizon implementation for supervised regressors."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy_tsf.contracts.result import ForecastTable
from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.data.schemas import PreparedData, PreparedSeries
from energy_tsf.providers._common import forecast_table, load_joblib, save_joblib
from energy_tsf.providers.base import TrainableProvider
from energy_tsf.tools.features import FeatureSpec, build_supervised_dataset, feature_names, features_for_origin


@dataclass(frozen=True)
class FittedFeatureModel:
    estimator: Any
    feature_spec: FeatureSpec
    feature_names: tuple[str, ...]
    params: dict[str, Any]


class TabularProvider(TrainableProvider):
    dependency: str

    def available(self) -> tuple[bool, str | None]:
        available = importlib.util.find_spec(self.dependency) is not None
        return available, None if available else f"missing optional dependency: {self.dependency}"

    def build_estimator(self, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def fit(
        self,
        series: PreparedSeries,
        origins: tuple[int, ...],
        *,
        train_end: int,
        params: dict[str, Any],
        feature_spec: FeatureSpec | None = None,
        future_index: pd.Index | None = None,
        covariate_spec: CovariateSpec | None = None,
        future_covariates: pd.DataFrame | None = None,
    ) -> FittedFeatureModel:
        if feature_spec is None:
            raise ValueError(f"{self.name} requires a FeatureSpec")
        validated = self.validate_params(params)
        usable = tuple(origin for origin in origins if origin < train_end)
        dataset = build_supervised_dataset(series, usable, feature_spec)
        keep = dataset.target_indices < train_end
        if not keep.any():
            raise ValueError("no supervised training rows fall before train_end")
        estimator = self.build_estimator(validated)
        estimator.fit(dataset.x[keep], dataset.y[keep])
        return FittedFeatureModel(estimator, feature_spec, dataset.feature_names, validated)

    def predict(
        self,
        model: FittedFeatureModel,
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
        spec = feature_spec or model.feature_spec
        if feature_names(spec) != model.feature_names:
            raise ValueError("prediction FeatureSpec does not match the fitted feature order")
        predictions = [
            np.asarray(
                model.estimator.predict(
                    features_for_origin(
                        series,
                        origin,
                        spec,
                        horizon_steps,
                        future_index=future_index,
                        future_covariates=future_covariates,
                    )
                ),
                dtype=float,
            )
            for origin in origins
        ]
        return forecast_table(self.name, series, origins, predictions, future_index=future_index)

    def save(self, model: FittedFeatureModel, path: str | Path) -> Path:
        return save_joblib(model, path)

    def load(self, path: str | Path) -> FittedFeatureModel:
        model = load_joblib(path)
        if not isinstance(model, FittedFeatureModel):
            raise TypeError("artifact is not a fitted feature model")
        return model
