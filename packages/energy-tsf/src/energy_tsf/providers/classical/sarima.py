"""statsmodels SARIMA provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import Field

from energy_tsf.providers._statistical import StatisticalProvider
from energy_tsf.providers.base import ProviderCapabilities, ProviderParams


@dataclass(frozen=True)
class SeasonallyDifferencedResult:
    """Statsmodels result plus the history needed to restore level forecasts."""

    result: Any
    season_length: int
    history: np.ndarray

    def apply(self, values: np.ndarray, *, refit: bool = False) -> "SeasonallyDifferencedResult":
        values = np.asarray(values, dtype=float)
        if len(values) <= self.season_length:
            raise ValueError("SARIMA context must exceed one seasonal period")
        differenced = values[self.season_length :] - values[: -self.season_length]
        return SeasonallyDifferencedResult(
            result=self.result.apply(differenced, refit=refit),
            season_length=self.season_length,
            history=values[-self.season_length :].copy(),
        )

    def forecast(self, steps: int) -> np.ndarray:
        differences = np.asarray(self.result.forecast(steps), dtype=float)
        restored = np.empty(steps, dtype=float)
        for index, difference in enumerate(differences):
            base = self.history[index] if index < self.season_length else restored[index - self.season_length]
            restored[index] = base + difference
        return restored


class SARIMAParams(ProviderParams):
    p: int = Field(default=1, ge=0, le=3)
    d: int = Field(default=0, ge=0, le=2)
    q: int = Field(default=0, ge=0, le=3)
    seasonal_p: int = Field(default=0, ge=0, le=2)
    seasonal_d: int = Field(default=1, ge=0, le=1)
    seasonal_q: int = Field(default=0, ge=0, le=2)
    season_length: int = Field(gt=1)
    maxiter: int = Field(default=50, gt=0)
    context_length: int | None = Field(default=None, gt=0)


class SARIMAProvider(StatisticalProvider):
    name = "sarima"
    params_model = SARIMAParams
    capabilities = ProviderCapabilities(
        name=name,
        family="classical",
        trainable=True,
        requires_features=False,
        artifact_kind="joblib",
    )

    def fit_result(self, values: np.ndarray, params: dict[str, Any]):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        order = (params["p"], params["d"], params["q"])
        season_length = params["season_length"]
        seasonal_d = params["seasonal_d"]
        model_values = np.asarray(values, dtype=float)
        if seasonal_d:
            if len(model_values) <= season_length:
                raise ValueError("SARIMA context must exceed one seasonal period")
            history = model_values[-season_length:].copy()
            model_values = model_values[season_length:] - model_values[:-season_length]
        seasonal = (
            params["seasonal_p"],
            0,
            params["seasonal_q"],
            season_length,
        )
        result = SARIMAX(
            model_values,
            order=order,
            seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=params["maxiter"])
        if seasonal_d:
            return SeasonallyDifferencedResult(result, season_length, history)
        return result
