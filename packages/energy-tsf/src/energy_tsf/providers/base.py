"""Provider contracts shared by classical, ML, and foundation models."""

from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from energy_tsf.contracts.common import ContractModel
from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.contracts.result import ForecastTable
from energy_tsf.data.schemas import PreparedData, PreparedSeries
from energy_tsf.tools.features import FeatureSpec


class ProviderParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderCapabilities(ContractModel):
    name: str
    family: Literal["baseline", "classical", "ml", "foundation"]
    trainable: bool
    requires_features: bool
    artifact_kind: str
    runtime: str = "local"


class ForecastProvider(ABC):
    """Common inference contract implemented by every model adapter.

    Providers expose capabilities, availability, parameter validation, runtime
    fingerprints, and forecasts in one normalized table format. Workflows own
    model selection and publication; Providers do not publish assets.
    """

    name: str
    params_model: type[ProviderParams]
    capabilities: ProviderCapabilities

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate provider-specific parameters and return JSON-safe values."""
        return self.params_model.model_validate(params).model_dump(mode="json")

    def available(self) -> tuple[bool, str | None]:
        """Report whether this provider can run in the current environment."""
        return True, None

    def runtime_fingerprint(self) -> dict[str, str]:
        """Describe runtime versions that can affect prediction reproducibility."""
        return {
            "provider": self.name,
            "python": platform.python_version(),
            "numpy": np.__version__,
        }

    @abstractmethod
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
        """Produce normalized finite point forecasts."""


class TrainableProvider(ForecastProvider):
    """Provider contract for models that fit and persist local artifacts."""
    @abstractmethod
    def fit(
        self,
        series: PreparedSeries,
        origins: tuple[int, ...],
        *,
        train_end: int,
        params: dict[str, Any],
        feature_spec: FeatureSpec | None = None,
        covariate_spec: CovariateSpec | None = None,
    ) -> Any:
        """Fit and return an opaque model handle."""

    @abstractmethod
    def save(self, model: Any, path: str | Path) -> Path:
        """Persist a fitted model artifact."""

    @abstractmethod
    def load(self, path: str | Path) -> Any:
        """Reload a fitted model artifact."""
