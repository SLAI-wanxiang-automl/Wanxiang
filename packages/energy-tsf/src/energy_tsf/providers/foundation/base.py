"""Offline-only inference base for official time-series foundation models."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
from pydantic import Field

from energy_tsf.contracts.result import ForecastTable
from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.data.schemas import PreparedData
from energy_tsf.providers._common import forecast_table
from energy_tsf.providers.base import ForecastProvider, ProviderCapabilities, ProviderParams
from energy_tsf.tools.features import FeatureSpec

RuntimeFactory = Callable[..., Any]


class FoundationParams(ProviderParams):
    checkpoint_path: str
    context_length: int = Field(default=4032, gt=0)
    device: str = "cpu"


class FoundationProvider(ForecastProvider):
    dependency: str
    runtime_factory: RuntimeFactory | None

    def __init__(self, runtime_factory: RuntimeFactory | None = None) -> None:
        self.runtime_factory = runtime_factory

    def available(self) -> tuple[bool, str | None]:
        if self.runtime_factory is not None:
            return True, None
        available = importlib.util.find_spec(self.dependency) is not None
        return available, None if available else f"missing isolated runtime dependency: {self.dependency}"

    def checkpoint_available(self, value: str | None) -> tuple[bool, str | None]:
        """Cheap plan-time check for an explicitly local, materialized checkpoint."""

        if not value:
            return False, "local checkpoint path was not provided"
        checkpoint = Path(value).expanduser()
        if not checkpoint.exists():
            return False, f"local checkpoint does not exist: {checkpoint}"
        if checkpoint.is_file():
            return True, None
        weight_patterns = ("*.safetensors", "*.bin", "*.pt", "*.ckpt")
        if not any(next(checkpoint.glob(pattern), None) is not None for pattern in weight_patterns):
            return False, f"local checkpoint has no model weight files: {checkpoint}"
        return True, None

    def build_runtime(
        self,
        params: FoundationParams,
        horizon_steps: int,
        covariate_spec: CovariateSpec,
    ) -> Any:
        factory = self.runtime_factory or self.default_runtime_factory
        checkpoint = Path(params.checkpoint_path).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"foundation checkpoint does not exist: {checkpoint}")
        values = params.model_dump(mode="json")
        values["checkpoint_path"] = str(checkpoint)
        return factory(
            **values,
            horizon_steps=horizon_steps,
            known_future_covariates=len(covariate_spec.known_future),
            past_only_covariates=len(covariate_spec.past_only),
        )

    def default_runtime_factory(self, **kwargs: Any) -> Any:
        raise NotImplementedError

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
        validated = cast(FoundationParams, self.params_model.model_validate(params))
        covariates = covariate_spec or CovariateSpec()
        values = series.frame[series.target].to_numpy(dtype=float)
        contexts: list[np.ndarray] = []
        past_batches: list[dict[str, np.ndarray]] = []
        future_batches: list[dict[str, np.ndarray]] = []
        for origin in origins:
            if origin < validated.context_length:
                raise ValueError(f"origin {origin} has less than {validated.context_length} context rows")
            contexts.append(values[origin - validated.context_length : origin].astype(np.float32, copy=False))
            past_batches.append(
                {
                    column: series.frame[column]
                    .iloc[origin - validated.context_length : origin]
                    .to_numpy(dtype=np.float32)
                    for column in covariates.columns
                }
            )
            known: dict[str, np.ndarray] = {}
            for column in covariates.known_future:
                if origin + horizon_steps <= len(series.frame):
                    raw = series.frame[column].iloc[origin : origin + horizon_steps]
                elif origin == len(series.frame) and future_covariates is not None:
                    raw = future_covariates[column]
                else:
                    raise ValueError(f"future values are required for known covariate: {column}")
                array = np.asarray(raw, dtype=np.float32)
                if len(array) != horizon_steps or not np.isfinite(array).all():
                    raise ValueError(f"future covariate {column} must contain finite horizon values")
                known[column] = array
            future_batches.append(known)
        runtime = model or self.build_runtime(validated, horizon_steps, covariates)
        if covariates.multivariate:
            predictions = np.asarray(
                runtime.predict(
                    contexts,
                    horizon_steps,
                    past_covariates=past_batches,
                    future_covariates=future_batches,
                ),
                dtype=float,
            )
        else:
            predictions = np.asarray(runtime.predict(contexts, horizon_steps), dtype=float)
        expected = (len(origins), horizon_steps)
        predictions = np.squeeze(predictions)
        if len(origins) == 1 and predictions.shape == (horizon_steps,):
            predictions = predictions[np.newaxis, :]
        if predictions.shape != expected:
            raise ValueError(f"unexpected foundation forecast shape {predictions.shape}, expected {expected}")
        if not np.isfinite(predictions).all():
            raise ValueError("foundation forecast contains non-finite values")
        return forecast_table(
            self.name,
            series,
            origins,
            [row for row in predictions],
            future_index=future_index,
        )


def foundation_capabilities(name: str) -> ProviderCapabilities:
    return ProviderCapabilities(
        name=name,
        family="foundation",
        trainable=False,
        requires_features=False,
        artifact_kind="runtime_reference",
        runtime="isolated",
    )
