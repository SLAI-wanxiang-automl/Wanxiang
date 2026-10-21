"""StatsForecast structures frozen after selection; parameters fit once per phase.

Stateful forecasts replay observed history with fixed parameters, avoiding
optimization. This is deterministic state reconstruction, not constant-time
incremental filtering. Theta's fitted seasonal factors are also kept fixed.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from ..data import KEY_COLUMNS
from ..window_models import ModelCapabilities
from .common import ModelAudit, horizon, model_history

STATS_MODELS = (
    "AutoARIMA",
    "AutoETS",
    "AutoCES",
    "Theta",
    "DynamicOptimizedTheta",
    "ADIDA",
    "IMAPA",
    "CrostonClassic",
    "HistoricAverage",
    "SeasonalNaive",
    "ZeroModel",
)
STATE_MODELS = ("AutoARIMA", "AutoETS", "AutoCES", "Theta", "DynamicOptimizedTheta")


def _backend(name, params):
    from statsforecast import models

    return getattr(models, name)(**deepcopy(params))


def _structure(name, fitted):
    model = fitted.model_
    if name == "AutoARIMA":
        p, q, P, Q, m, d, D = model["arma"]
        return dict(
            order=(p, d, q),
            seasonal_order=(P, D, Q),
            season_length=m,
            include_mean="intercept" in model["coef"],
            include_drift="drift" in model["coef"],
        )
    if name == "AutoETS":
        return dict(
            model="".join(model["components"][:3]),
            damped=model["components"][3] != "N",
            season_length=model["m"],
        )
    if name == "AutoCES":
        return dict(model=model["seasontype"], season_length=model["m"])
    return {}


class StatsForecastAdapter:
    """Fixed constructor settings; use NativeStatsAdapter for automatic structures."""

    def __init__(self, name, *, params=None, alias=None, per_series_params=None):
        if name not in STATS_MODELS:
            raise ValueError(f"Unknown statistical model {name}")
        self.name, self.params = name, deepcopy(params or {})
        self.alias = alias or name
        self.per_series_params = deepcopy(per_series_params or {})
        mode = "state_update" if name in STATE_MODELS else "frozen"
        self.capabilities = ModelCapabilities(
            min_history=max(4, 2 * self.params.get("season_length", 1)),
            update_mode=mode,
            supported_update_modes=(mode,),
        )

    def get_params(self):
        return dict(
            name=self.name,
            params=deepcopy(self.params),
            per_series_params=deepcopy(self.per_series_params),
        )

    def fit_frozen(self, window, freq):
        history = model_history(window)
        fitted = {}
        for uid, group in history.groupby("unique_id", sort=False):
            params = {**self.params, **self.per_series_params.get(uid, {})}
            # Fixed ARIMA orders avoid another automatic search in final fitting.
            backend = (
                "ARIMA" if self.name == "AutoARIMA" and "order" in params else self.name
            )
            params["alias"] = self.alias
            model = _backend(backend, params)
            model.fit(group.y.to_numpy(dtype=float))
            fitted[uid] = model
        return FittedStatsForecast(self, fitted, history, freq)

    def forecast(self, window, freq):
        return self.fit_frozen(window, freq).forecast(window, freq)


class NativeStatsAdapter(StatsForecastAdapter):
    """Native IC search proposes structures; shared inner CV chooses among them.

    At most num_samples historical prefixes supply native proposals. Every
    proposed structure is then scored on all common inner folds. No structure
    search is repeated during validation or final testing.
    """

    def __init__(self, name, *, num_samples=3, **kwargs):
        if name not in ("AutoARIMA", "AutoETS", "AutoCES"):
            raise ValueError("Native structure search is for ARIMA, ETS, CES")
        super().__init__(name, **kwargs)
        from dataclasses import replace

        self.capabilities = replace(self.capabilities, tuning_mode="native")
        self.num_samples = num_samples
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.inner_n_windows = 1
        self.inner_step_size = None
        self.metric = "mae"
        self.seasonality = 1
        self.selection_log = []

    def tune(self, window, freq):
        from ..backtesting import rolling_cross_validate
        from ..data import ForecastData

        history = model_history(window)
        dates = pd.DatetimeIndex(sorted(history.ds.unique()))
        h, step = horizon(window), self.inner_step_size or horizon(window)
        first = len(dates) - h - (self.inner_n_windows - 1) * step
        if first < self.capabilities.min_history:
            raise ValueError("Insufficient history for native statistical tuning")
        # The final train prefix also contributes its native IC winner.
        prefixes = np.unique(
            np.linspace(first, len(dates), self.num_samples, dtype=int)
        )
        proposals = []
        for count in prefixes:
            settings = {}
            for uid, group in history.loc[history.ds <= dates[count - 1]].groupby(
                "unique_id"
            ):
                native = _backend(self.name, {**self.params, "alias": self.alias})
                native.fit(group.y.to_numpy(dtype=float))
                settings[uid] = _structure(self.name, native)
            if settings not in proposals:
                proposals.append(settings)
        candidates = [
            StatsForecastAdapter(
                self.name,
                params=self._fixed_base(),
                per_series_params=config,
                alias=f"{self.alias}_structure_{i}",
            )
            for i, config in enumerate(proposals)
        ]
        result = rolling_cross_validate(
            ForecastData(history, freq),
            candidates,
            validation_start=dates[0],
            validation_end=dates[-1],
            h=h,
            n_windows=self.inner_n_windows,
            step_size=step,
            seasonality=self.seasonality,
        )
        winner = result.best_model(self.metric)
        selected = next(c for c in candidates if c.alias == winner)
        selected.alias = self.alias
        self.selection_log = [
            dict(
                origin=window.origin,
                params=selected.get_params(),
                inner_n_windows=self.inner_n_windows,
                score=float(
                    result.summary.set_index("model").loc[winner, f"mean_{self.metric}"]
                ),
            )
        ]
        return selected

    def _fixed_base(self):
        if self.name == "AutoARIMA":
            # Auto-only search controls are replaced by the selected orders.
            return {
                k: v
                for k, v in self.params.items()
                if k in ("method", "blambda", "biasadj")
            }
        return {k: v for k, v in self.params.items() if k not in ("model", "damped")}

    def forecast(self, window, freq):
        return self.tune(window, freq).forecast(window, freq)


class FittedStatsForecast:
    def __init__(self, source, models, history, freq):
        self.alias, self.capabilities = source.alias, source.capabilities
        self.name, self.models = source.name, models
        self.history, self.freq = history.copy(), freq
        self.audit = ModelAudit()
        self.audit.fitted(history)
        self.initial_models = {
            uid: deepcopy(getattr(m, "model_", {})) for uid, m in models.items()
        }

    def update(self, history, freq):
        if freq != self.freq:
            raise ValueError("Frequency cannot change while updating state")
        rows = []
        for uid, old in self.history.groupby("unique_id", sort=False):
            current = history.loc[history.unique_id == uid, [*KEY_COLUMNS, "y"]]
            if current.empty or current.ds.max() < old.ds.max():
                raise ValueError("State origins must move forward")
            overlap = old.merge(
                current, on=list(KEY_COLUMNS), suffixes=("_old", "_new")
            )
            if len(overlap) != len(old) or not np.array_equal(
                overlap.y_old, overlap.y_new
            ):
                raise ValueError(
                    "State history must include unchanged prior observations"
                )
            new = current.loc[current.ds > old.ds.max()]
            if len(new):
                expected = pd.date_range(old.ds.max(), periods=len(new) + 1, freq=freq)[
                    1:
                ]
                if not pd.DatetimeIndex(new.ds).equals(expected):
                    raise ValueError("New state observations must be contiguous")
                rows.append(new)
        if set(history.unique_id) != set(self.models):
            raise ValueError("Cannot add or remove series during state updates")
        if rows:
            additions = pd.concat(rows, ignore_index=True)
            self.history = pd.concat(
                [self.history, additions], ignore_index=True
            ).sort_values(list(KEY_COLUMNS))
            self.audit.updates.append(dict(end=additions.ds.max(), rows=len(additions)))

    def _theta(self, uid, y, h):
        from statsforecast.theta import forecast_theta, thetamodel

        initial = self.initial_models[uid]
        season = None
        if initial.get("decompose", False):
            # Stored seasonal forecast starts at the first point after fitting.
            # Rotate it to the initial series origin and preserve that pattern.
            m = initial["m"]
            pattern = np.roll(initial["seas_forecast"]["mean"], initial["n"] % m)
            season = np.resize(pattern, len(y) + h)
            y = (
                y - season[: len(y)]
                if initial["decomposition_type"] == "additive"
                else y / season[: len(y)]
            )
        params = initial["par"]
        updated = thetamodel(
            y=y,
            m=initial["m"],
            modeltype=initial["modeltype"],
            initial_smoothed=params["initial_smoothed"],
            alpha=params["alpha"],
            theta=params["theta"],
            nmse=3,
        )
        values = forecast_theta(updated, h)["mean"]
        if season is not None:
            values = (
                values + season[len(y) :]
                if initial["decomposition_type"] == "additive"
                else values * season[len(y) :]
            )
        return values

    def _state_forecast(self, uid, y, h):
        """Reconstruct state and explicitly check all learned coefficients."""
        initial = deepcopy(self.initial_models[uid])
        if self.name == "AutoARIMA":
            from statsforecast.arima import forecast_arima, forward_arima

            updated = forward_arima(initial, y, method=self.models[uid].method)
            for key, value in initial["coef"].items():
                if not np.allclose(value, updated["coef"][key], equal_nan=True):
                    raise ValueError("ARIMA state update changed coefficients")
            return forecast_arima(updated, h)["mean"]
        if self.name == "AutoETS":
            # ets_f's constant-series shortcut optimizes a new model. A
            # still-constant observed prefix has unchanged forecasts instead.
            if np.ptp(y) == 0:
                return self.models[uid].predict(h=h)["mean"]
            from statsforecast.ets import forecast_ets, forward_ets

            updated = forward_ets(initial, y)
            if not np.allclose(initial["par"], updated["par"], equal_nan=True):
                raise ValueError("ETS state update changed parameters")
            return forecast_ets(updated, h)["mean"]
        if self.name == "AutoCES":
            from statsforecast.ces import forecast_ces, forward_ces

            updated = forward_ces(initial, y)
            for key, value in initial["par"].items():
                if not np.allclose(value, updated["par"][key], equal_nan=True):
                    raise ValueError("CES state update changed parameters")
            return forecast_ces(updated, h)["mean"]
        return self._theta(uid, y, h)

    def forecast(self, window, freq):
        if self.capabilities.update_mode == "state_update":
            self.update(window.history, freq)
        h = horizon(window)
        rows = []
        for uid, future in window.future_covariates.groupby("unique_id", sort=False):
            if uid not in self.models:
                raise ValueError(f"Unknown fitted series {uid}")
            model = self.models[uid]
            if self.name == "SeasonalNaive":
                y = window.history.loc[window.history.unique_id == uid, "y"].to_numpy(
                    dtype=float
                )
                values = np.resize(y[-model.season_length :], h)
            elif self.capabilities.update_mode == "state_update":
                y = self.history.loc[self.history.unique_id == uid, "y"].to_numpy(
                    dtype=float
                )
                values = self._state_forecast(uid, y, h)
            else:
                values = model.predict(h=h)["mean"]
            frame = future.loc[:, list(KEY_COLUMNS)].copy()
            frame[self.alias] = values
            rows.append(frame)
        self.audit.predictions.append(dict(origin=window.origin, h=h))
        return pd.concat(rows, ignore_index=True)
