"""Combine aligned rolling predictions using weights fixed on validation."""

import numpy as np
import pandas as pd

from ..test_rolling import RollingTestResult


def ensemble_test_results(data, results, validation, *, seasonality=1):
    """Inverse-MAE weights; zero-MAE winners share all weight equally."""
    aliases = list(results)
    errors = np.array([validation[a].summary.iloc[0].mae for a in aliases], dtype=float)
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Ensemble requires finite nonnegative validation MAE")
    if (errors == 0).any():
        weights = (errors == 0).astype(float)
    else:
        # Equivalent to 1/MAE, without overflow for very small positive errors.
        weights = errors.min() / errors
    weights /= weights.sum()
    keys = ["unique_id", "origin", "ds", "lead"]
    base = results[aliases[0]].forecasts.sort_values(keys).reset_index(drop=True)
    values = []
    for alias in aliases:
        frame = results[alias].forecasts.sort_values(keys).reset_index(drop=True)
        if not frame[keys + ["y"]].equals(base[keys + ["y"]]):
            raise ValueError("Ensemble members must have identical keys and targets")
        values.append(frame.y_pred.to_numpy(dtype=float))
    matrix = np.stack(values)
    ensembles = {}
    for alias, predictions in (
        ("MedianEnsemble", np.median(matrix, axis=0)),
        ("WeightedEnsemble", weights @ matrix),
    ):
        forecasts = base.copy()
        forecasts["model"] = alias
        forecasts["y_pred"] = predictions
        forecasts["absolute_error"] = abs(forecasts.y - predictions)
        forecasts["squared_error"] = (forecasts.y - predictions) ** 2
        rows = []
        for (uid, origin), group in forecasts.groupby(["unique_id", "origin"]):
            history = data.history.loc[
                (data.history.unique_id == uid) & (data.history.ds <= origin), "y"
            ].to_numpy(dtype=float)
            scale = (
                np.abs(history[seasonality:] - history[:-seasonality]).mean()
                if len(history) > seasonality
                else np.nan
            )
            mae = float(group.absolute_error.mean())
            rows.append(
                dict(
                    model=alias,
                    unique_id=uid,
                    origin=origin,
                    mae=mae,
                    rmse=float(np.sqrt(group.squared_error.mean())),
                    mase=mae / scale if np.isfinite(scale) and scale > 0 else np.nan,
                )
            )
        scores = pd.DataFrame(rows)
        by_lead = forecasts.groupby("lead", as_index=False).agg(
            mae=("absolute_error", "mean"),
            mse=("squared_error", "mean"),
            n=("absolute_error", "size"),
        )
        by_lead["rmse"] = np.sqrt(by_lead.pop("mse"))
        summary = pd.DataFrame(
            [
                dict(
                    model=alias,
                    weight_policy="prediction_ensemble",
                    n_origins=forecasts.origin.nunique(),
                    n_predictions=len(forecasts),
                    mae=float(forecasts.absolute_error.mean()),
                    rmse=float(np.sqrt(forecasts.squared_error.mean())),
                    mean_mase=float(scores.mase.mean())
                    if scores.mase.notna().all()
                    else np.nan,
                )
            ]
        )
        ensembles[alias] = RollingTestResult(forecasts, scores, by_lead, summary)
    return ensembles, dict(zip(aliases, weights.tolist(), strict=True))
