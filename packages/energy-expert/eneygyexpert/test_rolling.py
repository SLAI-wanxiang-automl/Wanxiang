"""Held-out rolling evaluation after model selection is complete."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtesting import WindowModel
from .data import KEY_COLUMNS, ForecastData
from .window_models import forecast_from_data


@dataclass
class RollingTestResult:
    """Test predictions and diagnostics; no model-selection operation is exposed."""

    forecasts: pd.DataFrame
    scores: pd.DataFrame
    by_lead: pd.DataFrame
    summary: pd.DataFrame
    fitted_model: object | None = None


def rolling_test_forecast(
    data: ForecastData,
    model: WindowModel,
    *,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    h: int,
    step_size: int | None = None,
    seasonality: int = 1,
    weight_policy: str | None = None,
    full_fit_history: bool = True,
    bounds: tuple[float | None, float | None] | None = None,
) -> RollingTestResult:
    """Score a selected model at rolling test origins.

    The first origin is immediately before ``test_start``. Later origins move
    forward by ``step_size`` observations. Previously observed test targets may
    enter later origins' input histories. By default the model's declared
    policy is used. With ``weight_policy='frozen'``, weights are fitted once
    before the test and reused at every
    origin. ``state_update`` keeps parameters fixed while ingesting new data.
    Legacy ``'refit'`` retrains at every origin. Prediction context may be
    truncated; fitting uses full history by default. The last incomplete
    horizon is omitted. ``bounds`` is an optional ``(lower, upper)`` clip
    applied to every model's predictions before scoring, e.g. ``(0.0, None)``
    for non-negative power.
    """
    if weight_policy is None:
        weight_policy = model.capabilities.update_mode
    if weight_policy not in {"frozen", "state_update", "refit"}:
        raise ValueError("weight_policy must be frozen or state_update (legacy: refit)")
    if (
        weight_policy != "refit"
        and weight_policy not in model.capabilities.supported_update_modes
    ):
        raise ValueError("Model does not support the requested update policy")
    model.capabilities.validate(data, h)
    if h < 1 or seasonality < 1:
        raise ValueError("h and seasonality must be positive")
    step = h if step_size is None else step_size
    if step < 1:
        raise ValueError("step_size must be positive")
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    if start > end:
        raise ValueError("test_start must not exceed test_end")
    if bounds is not None:
        lower, upper = bounds
        if (lower is not None and upper is not None and lower > upper) or (
            lower is None and upper is None
        ):
            raise ValueError("bounds must be a nonempty (lower, upper) interval")

    dates = pd.DatetimeIndex(sorted(data.history.ds.unique()))
    if start not in dates or end not in dates:
        raise ValueError("test_start and test_end must be observed timestamps")
    start_pos = dates.get_loc(start)
    end_pos = dates.get_loc(end)
    if start_pos == 0 or start_pos + h - 1 > end_pos:
        raise ValueError("Test interval needs a prior origin and one complete horizon")
    first_origin = dates[start_pos - 1]
    if first_origin + pd.tseries.frequencies.to_offset(data.freq) != start:
        raise ValueError("test_start must immediately follow the first origin")

    prediction_model = model
    if weight_policy in {"frozen", "state_update"}:
        if not hasattr(model, "fit_frozen"):
            raise ValueError(
                "This model cannot freeze weights; choose fixed parameters and "
                "a frozen-weight adapter, or set weight_policy='refit'"
            )
        training_window = data.window(
            first_origin,
            h,
            context_length=(
                None if full_fit_history else model.capabilities.context_length
            ),
        ).model_input()
        prediction_model = model.fit_frozen(training_window, data.freq)
        if weight_policy == "state_update" and not hasattr(prediction_model, "update"):
            raise ValueError("state_update requires a fitted model update() interface")

    predictions = []
    scores = []
    for origin_pos in range(start_pos - 1, end_pos - h + 1, step):
        origin = dates[origin_pos]
        window = data.window(origin, h)
        if weight_policy == "state_update":
            prediction_model.update(window.history, data.freq)
        if len(window.actuals) != len(window.future_covariates):
            raise ValueError(f"Missing held-out target values at origin {origin}")
        forecast = forecast_from_data(prediction_model, data, origin, h)
        values = forecast[prediction_model.alias].to_numpy(dtype=float)
        if bounds is not None:
            lower, upper = bounds
            if lower is not None:
                values = np.maximum(values, lower)
            if upper is not None:
                values = np.minimum(values, upper)
            forecast[prediction_model.alias] = values
        if not np.isfinite(values).all():
            raise ValueError(
                f"Model produced non-finite predictions at origin {origin}"
            )
        joined = window.actuals.merge(
            forecast, on=list(KEY_COLUMNS), how="inner", validate="one_to_one"
        )
        if len(joined) != len(window.actuals):
            raise ValueError(
                f"Prediction keys differ from test targets at origin {origin}"
            )
        joined = joined.rename(columns={prediction_model.alias: "y_pred"})
        joined["origin"] = origin
        joined["lead"] = joined.groupby("unique_id").cumcount() + 1
        joined["model"] = model.alias
        joined["absolute_error"] = (joined.y - joined.y_pred).abs()
        joined["squared_error"] = (joined.y - joined.y_pred) ** 2
        predictions.append(joined)

        for uid, group in joined.groupby("unique_id", sort=False):
            train_y = window.history.loc[window.history.unique_id == uid, "y"].to_numpy(
                dtype=float
            )
            scale = (
                np.abs(train_y[seasonality:] - train_y[:-seasonality]).mean()
                if len(train_y) > seasonality
                else np.nan
            )
            mae = float(group.absolute_error.mean())
            scores.append(
                {
                    "model": model.alias,
                    "unique_id": uid,
                    "origin": origin,
                    "mae": mae,
                    "rmse": float(np.sqrt(group.squared_error.mean())),
                    "mase": mae / scale if np.isfinite(scale) and scale > 0 else np.nan,
                }
            )

    forecasts = pd.concat(predictions, ignore_index=True)
    scores_df = pd.DataFrame(scores)
    by_lead = forecasts.groupby("lead", as_index=False).agg(
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        n=("absolute_error", "size"),
    )
    by_lead["rmse"] = np.sqrt(by_lead.pop("mse"))
    summary = pd.DataFrame(
        [
            {
                "model": model.alias,
                "weight_policy": weight_policy,
                "n_origins": forecasts.origin.nunique(),
                "n_predictions": len(forecasts),
                "mae": float(forecasts.absolute_error.mean()),
                "rmse": float(np.sqrt(forecasts.squared_error.mean())),
                "mean_mase": (
                    float(scores_df.mase.mean())
                    if scores_df.mase.notna().all()
                    else np.nan
                ),
            }
        ]
    )
    return RollingTestResult(forecasts, scores_df, by_lead, summary, prediction_model)
