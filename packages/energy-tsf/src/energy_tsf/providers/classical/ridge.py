"""Ridge direct-horizon provider with persisted scaling and feature order."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from energy_tsf.providers._tabular import TabularProvider
from energy_tsf.providers.base import ProviderCapabilities, ProviderParams


class RidgeParams(ProviderParams):
    alpha: float = Field(default=1.0, gt=0)


class RidgeProvider(TabularProvider):
    name = "ridge"
    dependency = "sklearn"
    params_model = RidgeParams
    capabilities = ProviderCapabilities(
        name=name,
        family="ml",
        trainable=True,
        requires_features=True,
        artifact_kind="joblib",
    )

    def build_estimator(self, params: dict[str, Any]):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), Ridge(**params))
