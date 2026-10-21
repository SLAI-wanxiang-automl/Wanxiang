"""statsmodels ARIMA provider."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from energy_tsf.providers._statistical import StatisticalProvider
from energy_tsf.providers.base import ProviderCapabilities, ProviderParams


class ARIMAParams(ProviderParams):
    p: int = Field(default=1, ge=0, le=5)
    d: int = Field(default=1, ge=0, le=2)
    q: int = Field(default=0, ge=0, le=5)
    maxiter: int = Field(default=50, gt=0)
    context_length: int | None = Field(default=None, gt=0)


class ARIMAProvider(StatisticalProvider):
    name = "arima"
    params_model = ARIMAParams
    capabilities = ProviderCapabilities(
        name=name,
        family="classical",
        trainable=True,
        requires_features=False,
        artifact_kind="joblib",
    )

    def fit_result(self, values: np.ndarray, params: dict[str, Any]):
        from statsmodels.tsa.arima.model import ARIMA

        order = (params["p"], params["d"], params["q"])
        return ARIMA(values, order=order).fit(method_kwargs={"maxiter": params["maxiter"]})
