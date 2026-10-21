"""Multi-window validation and deterministic model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .data import ForecastData, ForecastWindow, KEY_COLUMNS
from .window_models import ModelCapabilities, forecast_from_data


class WindowModel(Protocol):
    alias: str
    capabilities: ModelCapabilities

    def forecast(self, window: ForecastWindow, freq: str) -> pd.DataFrame: ...


class SeasonalNaiveWindowModel:
    """Repeat the most recent observed seasonal cycle."""

    def __init__(self, seasonality: int, alias: str = "SeasonalNaive"):
        if seasonality < 1:
            raise ValueError("seasonality must be positive")
        self.alias = alias
        self.seasonality = seasonality
        self.capabilities = ModelCapabilities(min_history=seasonality, tuning_mode="fixed")

    def fit_frozen(self, window: ForecastWindow, freq: str) -> SeasonalNaiveWindowModel:
        """No learned weights; the seasonal rule itself stays fixed."""
        return self

    def forecast(self, window: ForecastWindow, freq: str) -> pd.DataFrame:
        del freq
        output = []
        for uid, future in window.future_covariates.groupby("unique_id", sort=False):
            past = window.history.loc[window.history.unique_id == uid].sort_values("ds")
            cycle = past.y.to_numpy(dtype=float)[-self.seasonality:]
            if len(cycle) < self.seasonality:
                raise ValueError(f"Series {uid} has insufficient seasonal history")
            result = future.loc[:, list(KEY_COLUMNS)].copy()
            result[self.alias] = np.resize(cycle, len(result))
            output.append(result)
        return pd.concat(output, ignore_index=True)


@dataclass
class BacktestResult:
    """Window-level predictions, scores, eligible summaries, and model failures."""

    forecasts: pd.DataFrame
    scores: pd.DataFrame
    summary: pd.DataFrame
    failures: pd.DataFrame

    def best_model(self, metric: str = "mae") -> str:
        if metric not in {"mae", "rmse", "mase"}:
            raise ValueError("metric must be mae, rmse, or mase")
        valid = self.summary.loc[self.summary[f"mean_{metric}"].notna()]
        if valid.empty:
            raise ValueError(f"No model has a valid {metric} score across all windows")
        return str(valid.sort_values([f"mean_{metric}", "model"]).iloc[0].model)


def rolling_cross_validate(
    data: ForecastData,
    models: list[WindowModel],
    *,
    validation_start: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    h: int,
    n_windows: int,
    step_size: int | None = None,
    seasonality: int = 1,
) -> BacktestResult:
    """Evaluate historical forecast origins without using data after each origin.

    The last window ends at ``validation_end``. Earlier origins move backward
    by ``step_size`` observations. All series must have the same regular grid
    over the validation interval. Only models that succeed on every requested
    window and series are eligible for ``best_model``.
    """
    if h < 1 or n_windows < 1 or seasonality < 1:
        raise ValueError("h, n_windows, and seasonality must be positive")
    step = h if step_size is None else step_size
    if step < 1:
        raise ValueError("step_size must be positive")
    start = pd.Timestamp(validation_start)
    end = pd.Timestamp(validation_end)
    if start > end:
        raise ValueError("validation_start must not exceed validation_end")
    if not models or len({model.alias for model in models}) != len(models):
        raise ValueError("models must have unique aliases and cannot be empty")

    all_dates = pd.DatetimeIndex(sorted(data.history.ds.unique()))
    if end not in all_dates:
        raise ValueError("validation_end must be an observed timestamp")
    end_pos = all_dates.get_loc(end)
    origins = []
    for i in range(n_windows - 1, -1, -1):
        origin_pos = end_pos - h - i * step
        if origin_pos < 0:
            raise ValueError("Not enough history for the requested windows")
        origin = all_dates[origin_pos]
        first_target = origin + pd.tseries.frequencies.to_offset(data.freq)
        if first_target < start:
            raise ValueError("Requested windows extend before validation_start")
        origins.append(origin)

    prediction_rows = []
    score_rows = []
    failure_rows = []
    expected_scores = n_windows * data.history.unique_id.nunique()
    for origin in origins:
        window = data.window(origin, h)
        if len(window.actuals) != len(window.future_covariates):
            raise ValueError(f"Missing held-out target values at origin {origin}")
        for model in models:
            try:
                predicted = forecast_from_data(model, data, origin, h)
                if predicted[model.alias].isna().any() or not np.isfinite(
                    predicted[model.alias].to_numpy(dtype=float)
                ).all():
                    raise ValueError("Model produced non-finite predictions")
                joined = window.actuals.merge(
                    predicted, on=list(KEY_COLUMNS), how="inner", validate="one_to_one"
                )
                if len(joined) != len(window.actuals):
                    raise ValueError("Prediction keys do not match held-out targets")
                joined["origin"] = origin
                joined["model"] = model.alias
                joined["lead"] = joined.groupby("unique_id").cumcount() + 1
                joined = joined.rename(columns={model.alias: "y_pred"})
                prediction_rows.append(joined)
                for uid, group in joined.groupby("unique_id", sort=False):
                    train_y = window.history.loc[
                        window.history.unique_id == uid, "y"
                    ].to_numpy(dtype=float)
                    scale = (
                        np.abs(train_y[seasonality:] - train_y[:-seasonality]).mean()
                        if len(train_y) > seasonality
                        else np.nan
                    )
                    errors = group.y.to_numpy(dtype=float) - group.y_pred.to_numpy(
                        dtype=float
                    )
                    mae = float(np.abs(errors).mean())
                    score_rows.append(
                        {
                            "model": model.alias,
                            "unique_id": uid,
                            "origin": origin,
                            "mae": mae,
                            "rmse": float(np.sqrt(np.square(errors).mean())),
                            "mase": mae / scale if np.isfinite(scale) and scale > 0 else np.nan,
                        }
                    )
            except Exception as exc:
                failure_rows.append(
                    {"model": model.alias, "origin": origin, "error": str(exc)}
                )

    forecasts = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    scores = pd.DataFrame(score_rows)
    failures = pd.DataFrame(failure_rows, columns=["model", "origin", "error"])
    if scores.empty:
        summary = pd.DataFrame(
            columns=["model", "mean_mae", "std_mae", "mean_rmse", "std_rmse",
                     "mean_mase", "std_mase"]
        )
    else:
        eligible = scores.groupby("model").size()
        eligible = eligible[eligible == expected_scores].index
        summary = (
            scores.loc[scores.model.isin(eligible)]
            .groupby("model", as_index=False)
            .agg(
                mean_mae=("mae", "mean"),
                std_mae=("mae", "std"),
                mean_rmse=("rmse", "mean"),
                std_rmse=("rmse", "std"),
                mean_mase=("mase", "mean"),
                std_mase=("mase", "std"),
            )
        )
        invalid_mase = scores.groupby("model").mase.apply(lambda values: values.isna().any())
        summary.loc[summary.model.map(invalid_mase), ["mean_mase", "std_mase"]] = np.nan
    return BacktestResult(forecasts, scores, summary, failures)
