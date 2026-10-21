"""Window adapter for MLForecast's native AutoML parameter search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import KEY_COLUMNS, ForecastWindow
from .window_models import ModelCapabilities


@dataclass(frozen=True)
class NativeTuningDecision:
    origin: pd.Timestamp
    best_config: dict[str, Any]
    inner_score: float
    inner_n_windows: int
    inner_step_size: int | None
    inner_refit: bool | int
    num_samples: int


class AutoMLForecastWindowAdapter:
    fit_on_full_history = True
    """Use AutoMLForecast's own search, then its refit model for prediction.

    This adapter deliberately passes only ``unique_id``, ``ds``, and ``y``.
    Models that consume known-future covariates need a different adapter.
    """

    def __init__(
        self,
        auto_model: Any,
        *,
        alias: str,
        season_length: int,
        inner_n_windows: int = 3,
        inner_step_size: int | None = None,
        inner_refit: bool | int = True,
        num_samples: int = 10,
        context_length: int | None = None,
        loss: Callable | None = None,
    ):
        if season_length < 1 or inner_n_windows < 1 or num_samples < 1:
            raise ValueError(
                "season_length, inner_n_windows, and num_samples must be positive"
            )
        if inner_step_size is not None and inner_step_size < 1:
            raise ValueError("inner_step_size must be positive")
        self.auto_model = auto_model
        self.alias = alias
        self.season_length = season_length
        self.inner_n_windows = inner_n_windows
        self.inner_step_size = inner_step_size
        self.inner_refit = inner_refit
        self.num_samples = num_samples
        self.loss = loss
        self.metric = "mae"
        self.seasonality = season_length
        self.capabilities = ModelCapabilities(
            context_length=context_length, tuning_mode="native"
        )
        self.selection_log: list[NativeTuningDecision] = []

    def minimum_training_history(self, h):
        return max(
            self.capabilities.context_length or 0, 2 * self.season_length + h + 1
        )

    def _common_loss(self, df, train_df):
        values = []
        for uid, group in df.groupby("unique_id", sort=False):
            errors = group.y.to_numpy(dtype=float) - group["model"].to_numpy(
                dtype=float
            )
            value = (
                float(np.sqrt(np.square(errors).mean()))
                if self.metric == "rmse"
                else float(np.abs(errors).mean())
            )
            if self.metric == "mase":
                y = train_df.loc[train_df.unique_id == uid, "y"].to_numpy(dtype=float)
                scale = np.abs(y[self.seasonality :] - y[: -self.seasonality]).mean()
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("Cannot tune MASE with an invalid training scale")
                value /= scale
            values.append(value)
        return float(np.mean(values))

    def frozen_candidate(
        self, origin: str | pd.Timestamp | None = None
    ) -> FixedMLForecastWindowAdapter:
        """Reuse a selected config without running AutoML search again."""
        decisions = self.selection_log
        if origin is not None:
            decisions = [d for d in decisions if d.origin == pd.Timestamp(origin)]
        if not decisions:
            raise ValueError("No native AutoML decision is available for this origin")
        return FixedMLForecastWindowAdapter(
            self.auto_model,
            best_config=decisions[-1].best_config,
            alias=self.alias,
            context_length=self.capabilities.context_length,
        )

    def tune(self, window: ForecastWindow, freq: str) -> FixedMLForecastWindowAdapter:
        self._search(window, freq)
        return self.frozen_candidate()

    def forecast(self, window: ForecastWindow, freq: str) -> pd.DataFrame:
        search, h = self._search(window, freq)
        predicted = search.predict(h=h)
        prediction_cols = [col for col in predicted if col not in KEY_COLUMNS]
        if len(prediction_cols) != 1:
            raise ValueError(
                "AutoMLForecast returned an unexpected number of model columns"
            )
        return predicted.rename(columns={prediction_cols[0]: self.alias}).loc[
            :, [*KEY_COLUMNS, self.alias]
        ]

    def _search(self, window: ForecastWindow, freq: str):
        from mlforecast.auto import AutoMLForecast

        history = window.history.loc[:, [*KEY_COLUMNS, "y"]]
        h = len(window.future_covariates) // history.unique_id.nunique()
        search = AutoMLForecast(
            models=[self.auto_model],
            freq=freq,
            season_length=self.season_length,
        )
        search.fit(
            df=history,
            n_windows=self.inner_n_windows,
            step_size=self.inner_step_size,
            refit=self.inner_refit,
            h=h,
            num_samples=self.num_samples,
            loss=self.loss or self._common_loss,
        )
        if len(search.results_) != 1:
            raise ValueError("AutoMLForecast adapter requires exactly one model result")
        study = next(iter(search.results_.values()))
        trial = study.best_trial
        config = trial.user_attrs.get("config")
        if config is None:
            raise ValueError("AutoMLForecast did not expose its best configuration")
        self.selection_log.append(
            NativeTuningDecision(
                origin=window.origin,
                best_config=config,
                inner_score=float(trial.value),
                inner_n_windows=self.inner_n_windows,
                inner_step_size=self.inner_step_size,
                inner_refit=self.inner_refit,
                num_samples=self.num_samples,
            )
        )
        return search, h


class FixedMLForecastWindowAdapter:
    fit_on_full_history = True
    """Refit a previously selected AutoML configuration without searching."""

    def __init__(
        self,
        auto_model: Any,
        *,
        best_config: dict[str, Any],
        alias: str,
        context_length: int | None,
    ):
        self.auto_model = auto_model
        self.best_config = best_config
        self.alias = alias
        self.capabilities = ModelCapabilities(
            context_length=context_length, tuning_mode="fixed"
        )

    def fit_frozen(
        self, window: ForecastWindow, freq: str
    ) -> FittedMLForecastWindowAdapter:
        from mlforecast import MLForecast
        from sklearn.base import clone

        estimator = clone(self.auto_model.model)
        estimator.set_params(**self.best_config["model_params"])
        forecast = MLForecast(
            models={self.alias: estimator},
            freq=freq,
            **self.best_config["mlf_init_params"],
        )
        history = window.history.loc[:, [*KEY_COLUMNS, "y"]]
        forecast.fit(history, **self.best_config["mlf_fit_params"])
        return FittedMLForecastWindowAdapter(
            forecast, self.alias, self.capabilities, history
        )

    def get_params(self):
        from copy import deepcopy

        return deepcopy(self.best_config)

    def forecast(self, window: ForecastWindow, freq: str) -> pd.DataFrame:
        return self.fit_frozen(window, freq).forecast(window, freq)


class FittedMLForecastWindowAdapter:
    """Reuse fitted MLForecast weights with updated observed lag context."""

    def __init__(
        self, fitted: Any, alias: str, capabilities: ModelCapabilities, history=None
    ):
        from .rolling_models.common import ModelAudit

        self.fitted = fitted
        self.alias = alias
        self.capabilities = capabilities
        self.audit = ModelAudit()
        if history is not None:
            self.audit.fitted(history)

    def forecast(self, window: ForecastWindow, freq: str) -> pd.DataFrame:
        del freq
        history = window.history.loc[:, [*KEY_COLUMNS, "y"]]
        h = len(window.future_covariates) // history.unique_id.nunique()
        predicted = self.fitted.predict(h=h, new_df=history)
        self.audit.predictions.append(dict(origin=window.origin, h=h))
        return predicted.loc[:, [*KEY_COLUMNS, self.alias]]
