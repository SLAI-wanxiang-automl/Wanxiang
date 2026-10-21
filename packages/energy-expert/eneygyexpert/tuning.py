"""Framework-owned hyperparameter search on nested time-series windows."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from .backtesting import WindowModel, rolling_cross_validate
from .data import KEY_COLUMNS, ForecastData, ForecastWindow
from .window_models import DirectRidgeWindowModel, ModelCapabilities


@dataclass(frozen=True)
class TuningDecision:
    outer_origin: pd.Timestamp
    inner_last_target: pd.Timestamp
    selected_candidate: str
    selected_params: dict
    metric: str
    score: float
    inner_h: int
    inner_n_windows: int
    inner_step_size: int | None


class FixedAliasWindowModel:
    """Keep the selected model's public alias when freezing its parameters."""

    def __init__(self, selected: WindowModel, alias: str):
        self.selected = selected
        self.alias = alias
        self.capabilities = selected.capabilities
        self.fit_on_full_history = getattr(selected, "fit_on_full_history", False)

    def get_params(self) -> dict:
        return (
            self.selected.get_params() if hasattr(self.selected, "get_params") else {}
        )

    def fit_frozen(self, window: ForecastWindow, freq: str) -> FixedAliasWindowModel:
        if not hasattr(self.selected, "fit_frozen"):
            raise ValueError("Selected model does not support frozen-weight prediction")
        return FixedAliasWindowModel(self.selected.fit_frozen(window, freq), self.alias)

    def forecast(self, window: ForecastWindow, freq: str) -> pd.DataFrame:
        return self.selected.forecast(window, freq).rename(
            columns={self.selected.alias: self.alias}
        )

    def update(self, history: pd.DataFrame, freq: str) -> None:
        self.selected.update(history, freq)


class GridTunedWindowModel:
    """Select a fixed-parameter candidate using only each origin's past data.

    The wrapper is one outer candidate. Its parameter search is repeated at
    every outer origin, including rolling test origins. The model family and
    parameter grid are fixed before validation and testing.
    """

    def __init__(
        self,
        candidates: list[WindowModel],
        *,
        alias: str,
        inner_h: int,
        inner_n_windows: int | None = None,
        inner_step_size: int | None = None,
        metric: str = "mae",
        seasonality: int = 1,
    ):
        if not candidates or len({c.alias for c in candidates}) != len(candidates):
            raise ValueError("Candidates must have unique aliases and cannot be empty")
        if any(c.capabilities.tuning_mode != "fixed" for c in candidates):
            raise ValueError("Framework search requires fixed-parameter candidates")
        if metric not in {"mae", "rmse", "mase"}:
            raise ValueError("metric must be mae, rmse, or mase")
        if (
            inner_h < 1
            or (inner_n_windows is not None and inner_n_windows < 1)
            or seasonality < 1
        ):
            raise ValueError(
                "inner_h, inner_n_windows, and seasonality must be positive"
            )
        if inner_step_size is not None and inner_step_size < 1:
            raise ValueError("inner_step_size must be positive")
        self.alias = alias
        self.candidates = {candidate.alias: candidate for candidate in candidates}
        self.inner_h = inner_h
        self.inner_n_windows = inner_n_windows
        self.inner_step_size = inner_step_size
        self.metric = metric
        self.seasonality = seasonality
        self.selection_log: list[TuningDecision] = []
        self.capabilities = ModelCapabilities(
            uses_historical_covariates=any(
                c.capabilities.uses_historical_covariates for c in candidates
            ),
            uses_future_covariates=any(
                c.capabilities.uses_future_covariates for c in candidates
            ),
            requires_future_covariates_in_history=any(
                c.capabilities.requires_future_covariates_in_history for c in candidates
            ),
            required_future_covariate_cols=tuple(
                dict.fromkeys(
                    col
                    for c in candidates
                    for col in c.capabilities.required_future_covariate_cols
                )
            ),
            min_history=min(c.capabilities.min_history for c in candidates),
            context_length=None,
            tuning_mode="framework",
        )

    def frozen_candidate(self, origin: str | pd.Timestamp | None = None) -> WindowModel:
        """Return fixed parameters chosen at a completed tuning origin.

        Use this fixed candidate in rolling tests when parameters must stay
        unchanged; its model coefficients are still refitted at each origin.
        """
        decisions = self.selection_log
        if origin is not None:
            decisions = [d for d in decisions if d.outer_origin == pd.Timestamp(origin)]
        if not decisions:
            raise ValueError("No tuning decision is available for the requested origin")
        return FixedAliasWindowModel(
            self.candidates[decisions[-1].selected_candidate], self.alias
        )

    def tune(self, window: ForecastWindow, freq: str) -> WindowModel:
        """Search within the supplied history and return fixed parameters."""
        historical = window.history.copy()
        historical_cols = tuple(c for c in historical if c not in {*KEY_COLUMNS, "y"})
        future_cols = tuple(
            dict.fromkeys(
                col
                for candidate in self.candidates.values()
                for col in (
                    getattr(candidate, "future_covariate_cols", ())
                    or candidate.capabilities.required_future_covariate_cols
                )
            )
        )
        missing = set(future_cols) - set(historical_cols)
        if missing:
            raise ValueError(
                f"Tuning needs archived future covariates: {sorted(missing)}"
            )
        inner_data = ForecastData(
            history=historical,
            freq=freq,
            historical_covariate_cols=historical_cols,
            future_covariate_cols=future_cols,
            future_covariates=(
                historical.loc[:, [*KEY_COLUMNS, *future_cols]] if future_cols else None
            ),
        )
        inner_dates = pd.DatetimeIndex(sorted(historical.ds.unique()))
        count = self.inner_n_windows
        if count is None:
            from .pipeline import automatic_window_count

            minimum = max(
                max(
                    c.capabilities.min_history,
                    c.capabilities.context_length or 0,
                    getattr(c, "lags", 0) + self.inner_h + 1,
                )
                for c in self.candidates.values()
            )
            count = automatic_window_count(
                len(inner_dates),
                self.inner_h,
                self.inner_step_size or self.inner_h,
                minimum,
            )
        result = rolling_cross_validate(
            inner_data,
            list(self.candidates.values()),
            validation_start=inner_dates[0],
            validation_end=inner_dates[-1],
            h=self.inner_h,
            n_windows=count,
            step_size=self.inner_step_size,
            seasonality=self.seasonality,
        )
        selected_alias = result.best_model(self.metric)
        selected = self.candidates[selected_alias]
        selected_row = result.summary.set_index("model").loc[selected_alias]
        params = selected.get_params() if hasattr(selected, "get_params") else {}
        self.selection_log.append(
            TuningDecision(
                outer_origin=window.origin,
                inner_last_target=result.forecasts.ds.max(),
                selected_candidate=selected_alias,
                selected_params=params,
                metric=self.metric,
                score=float(selected_row[f"mean_{self.metric}"]),
                inner_h=self.inner_h,
                inner_n_windows=count,
                inner_step_size=self.inner_step_size,
            )
        )
        return self.frozen_candidate(window.origin)

    def forecast(self, window: ForecastWindow, freq: str) -> pd.DataFrame:
        selected = self.tune(window, freq)
        context = selected.capabilities.context_length
        selected_window = ForecastWindow(
            origin=window.origin,
            history=(
                window.history.groupby("unique_id", sort=False).tail(context)
                if context is not None and not selected.fit_on_full_history
                else window.history
            ),
            future_covariates=window.future_covariates,
            actuals=window.actuals,
        )
        return selected.forecast(selected_window, freq)


def direct_ridge_grid(
    *,
    historical_covariate_cols: tuple[str, ...],
    future_covariate_cols: tuple[str, ...],
    lags: tuple[int, ...],
    context_lengths: tuple[int | None, ...],
    alphas: tuple[float, ...],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> list[DirectRidgeWindowModel]:
    """Construct distinct fixed candidates for nested DirectRidge tuning."""
    if not lags or not context_lengths or not alphas:
        raise ValueError("DirectRidge parameter grid axes cannot be empty")
    options = list(dict.fromkeys(product(lags, context_lengths, alphas)))
    return [
        DirectRidgeWindowModel(
            historical_covariate_cols=historical_covariate_cols,
            future_covariate_cols=future_covariate_cols,
            lags=lag,
            context_length=context,
            alpha=alpha,
            alias=f"DirectRidge_option_{i}",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        for i, (lag, context, alpha) in enumerate(options)
    ]
