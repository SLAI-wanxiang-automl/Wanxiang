"""Model adapters that forecast from a point-in-time data window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .data import KEY_COLUMNS, ForecastData, ForecastWindow


@dataclass(frozen=True)
class ModelCapabilities:
    """Data requirements used to filter candidate models before evaluation."""

    uses_historical_covariates: bool = False
    uses_future_covariates: bool = False
    requires_future_covariates_in_history: bool = False
    required_future_covariate_cols: tuple[str, ...] = ()
    min_history: int = 2
    context_length: int | None = None
    tuning_mode: Literal["fixed", "native", "framework"] = "fixed"
    update_mode: Literal["frozen", "state_update"] = "frozen"
    supported_update_modes: tuple[str, ...] = ("frozen",)

    def validate(self, data: ForecastData, h: int) -> None:
        if self.update_mode not in self.supported_update_modes:
            raise ValueError("Model does not support the requested update_mode")
        if self.update_mode == "state_update" and self.context_length is not None:
            raise ValueError(
                "State updates require complete history, not truncated context"
            )
        if h < 1:
            raise ValueError("h must be positive")
        if self.uses_historical_covariates and not data.historical_covariate_cols:
            raise ValueError("Model requires historical covariates")
        if self.uses_future_covariates and not data.future_covariate_cols:
            raise ValueError("Model requires future covariates")
        if self.requires_future_covariates_in_history:
            missing = set(
                self.required_future_covariate_cols or data.future_covariate_cols
            ) - set(data.historical_covariate_cols)
            if missing:
                raise ValueError(
                    "Model requires past values of future covariates: "
                    f"{sorted(missing)}"
                )
        if self.context_length is not None and self.context_length < self.min_history:
            raise ValueError("context_length is shorter than min_history")


class UnivariateWindowAdapter:
    """Run an existing EneygyExpert forecaster on a safe univariate window."""

    def __init__(
        self,
        model: Any,
        context_length: int | None = None,
        tuning_mode: Literal["fixed", "native"] = "fixed",
    ):
        self.model = model
        self.alias = model.alias
        self.capabilities = ModelCapabilities(
            context_length=context_length, tuning_mode=tuning_mode
        )

    def forecast(self, window: ForecastWindow, freq: str) -> pd.DataFrame:
        history = window.history.loc[:, [*KEY_COLUMNS, "y"]]
        h = len(window.future_covariates) // history.unique_id.nunique()
        result = self.model.forecast(df=history, h=h, freq=freq)
        return result.loc[:, [*KEY_COLUMNS, self.alias]]


class DirectRidgeWindowModel:
    """Direct, per-horizon linear forecasts using past state and known future data.

    Each series and lead time gets its own ridge fit. Historical covariates are
    read only at a training origin; future covariates are read at the target
    timestamp. Training requires archived values of future covariates in the
    history frame, with the same column names.
    """

    def __init__(
        self,
        *,
        historical_covariate_cols: tuple[str, ...],
        future_covariate_cols: tuple[str, ...],
        lags: int = 4,
        context_length: int | None = None,
        alpha: float = 1.0,
        alias: str = "DirectRidge",
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        if lags < 1 or alpha <= 0:
            raise ValueError("lags and alpha must be positive")
        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise ValueError("lower_bound must not exceed upper_bound")
        self.alias = alias
        self.historical_covariate_cols = tuple(historical_covariate_cols)
        self.future_covariate_cols = tuple(future_covariate_cols)
        self.lags = lags
        self.alpha = alpha
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.capabilities = ModelCapabilities(
            uses_historical_covariates=bool(historical_covariate_cols),
            uses_future_covariates=bool(future_covariate_cols),
            requires_future_covariates_in_history=bool(future_covariate_cols),
            required_future_covariate_cols=self.future_covariate_cols,
            min_history=lags + 2,
            context_length=context_length,
        )

    def get_params(self) -> dict[str, Any]:
        """Configuration needed to reconstruct this model after selection."""
        return {
            "historical_covariate_cols": self.historical_covariate_cols,
            "future_covariate_cols": self.future_covariate_cols,
            "lags": self.lags,
            "context_length": self.capabilities.context_length,
            "alpha": self.alpha,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    @staticmethod
    def _fit_ridge(
        x: np.ndarray, y: np.ndarray, alpha: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        z = (x - mean) / scale
        design = np.column_stack((np.ones(len(z)), z))
        penalty = np.eye(design.shape[1]) * alpha
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return coef, mean, scale

    def fit_frozen(
        self, window: ForecastWindow, freq: str
    ) -> FittedDirectRidgeWindowModel:
        """Fit once before testing; later origins reuse these coefficients."""
        del freq
        future = window.future_covariates
        required_history = {
            *KEY_COLUMNS,
            "y",
            *self.historical_covariate_cols,
            *self.future_covariate_cols,
        }
        missing_history = required_history - set(window.history)
        missing_future = set(self.future_covariate_cols) - set(future)
        if missing_history or missing_future:
            raise ValueError(
                f"Missing model inputs: history={sorted(missing_history)}, "
                f"future={sorted(missing_future)}"
            )

        fitted = {}
        for uid, future_uid in future.groupby("unique_id", sort=False):
            past = window.history.loc[window.history.unique_id == uid].sort_values("ds")
            future_uid = future_uid.sort_values("ds")
            h = len(future_uid)
            n = len(past)
            if n < self.lags + h + 1:
                raise ValueError(
                    f"Series {uid} needs at least {self.lags + h + 1} history rows "
                    f"to fit all {h} horizons; got {n}"
                )
            y = past["y"].to_numpy(dtype=float)
            hist_cov = past[list(self.historical_covariate_cols)].to_numpy(dtype=float)
            known_past = past[list(self.future_covariate_cols)].to_numpy(dtype=float)
            if not all(np.isfinite(a).all() for a in (y, hist_cov, known_past)):
                raise ValueError(f"Series {uid} has missing or non-finite model inputs")

            lead_models = []
            for lead in range(1, h + 1):
                origins = np.arange(self.lags - 1, n - lead)
                lag_matrix = np.column_stack(
                    [y[origins - lag] for lag in range(self.lags)]
                )
                x = np.column_stack(
                    (lag_matrix, hist_cov[origins], known_past[origins + lead])
                )
                target = y[origins + lead]
                lead_models.append(self._fit_ridge(x, target, self.alpha))
            fitted[uid] = lead_models
        result = FittedDirectRidgeWindowModel(self, fitted)
        result.audit.fitted(window.history)
        return result

    def forecast(self, window: ForecastWindow, freq: str) -> pd.DataFrame:
        return self.fit_frozen(window, freq).forecast(window, freq)


class FittedDirectRidgeWindowModel:
    """DirectRidge coefficients fitted once, with fresh inputs at each origin."""

    def __init__(
        self,
        source: DirectRidgeWindowModel,
        fitted: dict[Any, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    ):
        from .rolling_models.common import ModelAudit

        self.source = source
        self.fitted = fitted
        self.alias = source.alias
        self.capabilities = source.capabilities
        self.audit = ModelAudit()

    def forecast(self, window: ForecastWindow, freq: str) -> pd.DataFrame:
        del freq
        outputs = []
        for uid, future_uid in window.future_covariates.groupby(
            "unique_id", sort=False
        ):
            past = window.history.loc[window.history.unique_id == uid].sort_values("ds")
            future_uid = future_uid.sort_values("ds")
            models = self.fitted.get(uid)
            if models is None or len(future_uid) != len(models):
                raise ValueError(
                    f"No frozen coefficients for series {uid} and this horizon"
                )
            y = past["y"].to_numpy(dtype=float)
            historical = past[list(self.source.historical_covariate_cols)].to_numpy(
                dtype=float
            )
            known = future_uid[list(self.source.future_covariate_cols)].to_numpy(
                dtype=float
            )
            if len(y) < self.source.lags or not all(
                np.isfinite(a).all() for a in (y, historical, known)
            ):
                raise ValueError(f"Series {uid} has insufficient or non-finite inputs")
            current_state = np.r_[y[-self.source.lags :][::-1], historical[-1]]
            predictions = []
            for (coef, mean, scale), future_row in zip(models, known, strict=True):
                row = np.r_[current_state, future_row]
                value = float(coef[0] + ((row - mean) / scale) @ coef[1:])
                if self.source.lower_bound is not None:
                    value = max(value, self.source.lower_bound)
                if self.source.upper_bound is not None:
                    value = min(value, self.source.upper_bound)
                predictions.append(value)
            result = future_uid.loc[:, list(KEY_COLUMNS)].copy()
            result[self.alias] = predictions
            outputs.append(result)
        self.audit.predictions.append(dict(origin=window.origin))
        return pd.concat(outputs, ignore_index=True)


def forecast_from_data(
    model: UnivariateWindowAdapter | DirectRidgeWindowModel,
    data: ForecastData,
    origin: str | pd.Timestamp,
    h: int,
) -> pd.DataFrame:
    """Validate model requirements, create its context window, and forecast."""
    model.capabilities.validate(data, h)
    source = model
    while hasattr(source, "selected"):
        source = source.selected
    if hasattr(source, "source"):
        source = source.source
    if hasattr(source, "historical_covariate_cols") and not set(
        source.historical_covariate_cols
    ) <= set(data.historical_covariate_cols):
        raise ValueError(
            "Model historical covariates are absent from the data contract"
        )
    if hasattr(source, "future_covariate_cols") and not set(
        source.future_covariate_cols
    ) <= set(data.future_covariate_cols):
        raise ValueError("Model future covariates are absent from the data contract")
    context = (
        None
        if getattr(model, "fit_on_full_history", False)
        else model.capabilities.context_length
    )
    window = data.window(origin, h, context_length=context)
    counts = window.history.groupby("unique_id").size()
    if (counts < model.capabilities.min_history).any():
        raise ValueError("Some series do not meet the model's minimum history length")
    result = model.forecast(window.model_input(), data.freq)
    expected = window.future_covariates.loc[:, list(KEY_COLUMNS)]
    if not result.loc[:, list(KEY_COLUMNS)].equals(expected.reset_index(drop=True)):
        raise ValueError(
            "Model returned timestamps that differ from the forecast window"
        )
    return result
