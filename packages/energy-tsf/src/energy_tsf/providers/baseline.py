"""Seasonal-naive reference provider used for pipeline verification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from energy_tsf.contracts.result import ForecastTable
from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.data.schemas import PreparedData
from energy_tsf.providers._common import forecast_table
from energy_tsf.providers.base import ForecastProvider, ProviderCapabilities, ProviderParams
from energy_tsf.tools.features import FeatureSpec


class SeasonalNaiveParams(ProviderParams):
    season_length: int = Field(gt=0)


class SeasonalNaiveProvider(ForecastProvider):
    name = "seasonal-naive"
    params_model = SeasonalNaiveParams
    capabilities = ProviderCapabilities(
        name=name,
        family="baseline",
        trainable=False,
        requires_features=False,
        artifact_kind="none",
    )

    def predict(
        self,
        model: Any,
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
        validated = self.params_model.model_validate(params)
        predictions: list[np.ndarray] = []
        for origin in origins:
            source = origin + np.arange(horizon_steps) - validated.season_length
            if source.min() < 0 or source.max() >= origin:
                raise ValueError("seasonal-naive source must be fully observable at the origin")
            predictions.append(values[source])
        return forecast_table(
            self.name,
            series,
            origins,
            predictions,
            future_index=future_index,
        )
