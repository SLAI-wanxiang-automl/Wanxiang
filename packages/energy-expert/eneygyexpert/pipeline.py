"""Chronological train/tune, frozen validation, and final frozen test."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

import pandas as pd

from .data import ForecastData
from .test_rolling import RollingTestResult, rolling_test_forecast


def automatic_window_count(length: int, h: int, step: int, min_history: int = 0) -> int:
    """Count complete horizons, reserving an initial training prefix."""
    if min(length, h, step) < 1 or min_history < 0:
        raise ValueError("length, h, step must be positive; min_history nonnegative")
    count = (length - min_history - h) // step + 1
    if count < 1:
        raise ValueError("Interval is too short for one complete forecasting window")
    return count


@dataclass
class PipelineResult:
    boundaries: dict
    inner_n_windows: int
    fixed_models: dict
    validation: dict[str, RollingTestResult]
    leaderboard: pd.DataFrame
    selected_model: str
    test: RollingTestResult
    tuning_logs: dict = field(default_factory=dict)
    test_results: dict[str, RollingTestResult] = field(default_factory=dict)
    selected_models: list[str] = field(default_factory=list)
    ensemble_weights: dict[str, float] = field(default_factory=dict)


def train_validation_test(
    data: ForecastData,
    models: list,
    *,
    h: int,
    step_size: int | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    min_train_size: int | None = None,
    seasonality: int = 1,
    metric: str = "mae",
    top_k: int = 1,
    bounds: tuple[float | None, float | None] | None = None,
) -> PipelineResult:
    """Tune once on train; compare frozen models on validation; test the winner.

    All tuners receive identical inner h/count/step. Native tuners must expose
    inner_n_windows and inner_step_size. min_train_size controls the initial
    inner fitting prefix, independently of rolling prediction context. There
    is no tuning on validation or test. Only the validation top_k are tested.
    Ensemble weights use validation MAE only. Selection uses validation.
    Observed targets can enter later
    rolling inputs while weights remain fixed. Partial horizons are omitted.
    """
    if not (
        0 < train_fraction < 1
        and 0 < validation_fraction < 1
        and train_fraction + validation_fraction < 1
    ):
        raise ValueError("Fractions must leave nonempty train, validation, and test")
    if metric not in {"mae", "rmse", "mase"}:
        raise ValueError("metric must be mae, rmse, or mase")
    if not models or len({m.alias for m in models}) != len(models):
        raise ValueError("Models must be nonempty with unique aliases")
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= len(models)
    ):
        raise ValueError("top_k must be an integer between 1 and the candidate count")
    if any(m.alias in {"MedianEnsemble", "WeightedEnsemble"} for m in models):
        raise ValueError("MedianEnsemble and WeightedEnsemble are reserved aliases")
    dates = pd.DatetimeIndex(sorted(data.history.ds.unique()))
    expected = pd.date_range(dates[0], dates[-1], freq=data.freq)
    if not dates.equals(expected) or any(
        not pd.DatetimeIndex(group.ds).equals(dates)
        for _, group in data.history.groupby("unique_id", sort=False)
    ):
        raise ValueError("All series must share a complete regular timestamp grid")
    train_n = int(len(dates) * train_fraction)
    val_end_n = train_n + int(len(dates) * validation_fraction)
    step = h if step_size is None else step_size
    options = [
        candidate
        for m in models
        for candidate in (
            list(m.candidates.values()) if hasattr(m, "candidates") else [m]
        )
    ]
    required = max(
        max(
            c.minimum_training_history(h)
            if hasattr(c, "minimum_training_history")
            else c.capabilities.min_history,
            getattr(c, "lags", 0) + h + 1,
            c.capabilities.context_length or 0,
        )
        for c in options
    )
    minimum = required if min_train_size is None else min_train_size
    if minimum < required:
        raise ValueError(
            f"min_train_size must be at least {required} for these candidates"
        )
    inner_count = automatic_window_count(train_n, h, step, minimum)
    automatic_window_count(val_end_n - train_n, h, step)
    automatic_window_count(len(dates) - val_end_n, h, step)
    train_window = data.window(dates[train_n - 1], h).model_input()
    fixed, validation, rows, tuning_logs = {}, {}, [], {}
    for original in models:
        model = copy(original)
        mode = model.capabilities.tuning_mode
        if mode != "fixed":
            model.inner_n_windows = inner_count
            model.inner_step_size = step
            if hasattr(model, "inner_refit"):
                model.inner_refit = True
            model.metric = metric
            model.seasonality = seasonality
            if mode == "framework":
                model.inner_h = h
                selected = model.tune(train_window, data.freq)
            elif mode == "native":
                # Native AutoML also refits its winner; its best config is then
                # reused by the frozen adapter without another parameter search.
                selected = model.tune(train_window, data.freq)
            else:
                raise ValueError(f"Unsupported tuning mode: {mode}")
        else:
            selected = model
        fixed[selected.alias] = selected
        tuning_logs[selected.alias] = list(getattr(model, "selection_log", []))
        result = rolling_test_forecast(
            data,
            selected,
            test_start=dates[train_n],
            test_end=dates[val_end_n - 1],
            h=h,
            step_size=step,
            seasonality=seasonality,
            full_fit_history=True,
            bounds=bounds,
        )
        validation[selected.alias] = result
        rows.append(result.summary.iloc[0].to_dict())
    leaderboard = pd.DataFrame(rows)
    column = "mean_mase" if metric == "mase" else metric
    eligible = leaderboard.loc[leaderboard[column].notna()].sort_values(
        [column, "model"]
    )
    if eligible.empty:
        raise ValueError(f"No valid validation {metric} score")
    if len(eligible) < top_k:
        raise ValueError("Fewer valid validation models than top_k")
    selected_models = eligible.head(top_k).model.astype(str).tolist()
    winner = selected_models[0]
    test_results = {}
    for alias in selected_models:
        test_results[alias] = rolling_test_forecast(
            data,
            fixed[alias],
            test_start=dates[val_end_n],
            test_end=dates[-1],
            h=h,
            step_size=step,
            seasonality=seasonality,
            full_fit_history=True,
            bounds=bounds,
        )
    test = test_results[winner]
    from .rolling_models.prediction_ensemble import ensemble_test_results

    ensembles, weights = ensemble_test_results(
        data, test_results, validation, seasonality=seasonality
    )
    test_results.update(ensembles)
    return PipelineResult(
        dict(
            train_end=dates[train_n - 1],
            validation_start=dates[train_n],
            validation_end=dates[val_end_n - 1],
            test_start=dates[val_end_n],
            test_end=dates[-1],
        ),
        inner_count,
        fixed,
        validation,
        leaderboard,
        winner,
        test,
        tuning_logs,
        test_results,
        selected_models,
        weights,
    )
