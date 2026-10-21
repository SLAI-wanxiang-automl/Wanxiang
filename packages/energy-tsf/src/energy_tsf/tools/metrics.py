"""Deterministic point-forecast metrics suited to renewable energy."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


def point_metrics(
    y_true,
    y_pred,
    *,
    include_mask=None,
    training_values=None,
    season_length: int | None = None,
) -> Mapping[str, float]:
    """Compute finite point metrics, optionally excluding masked observations."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction must have the same shape")
    if include_mask is not None:
        mask = np.asarray(include_mask, dtype=bool).reshape(-1)
        if mask.shape != truth.shape:
            raise ValueError("metric mask must match truth shape")
        truth, prediction = truth[mask], prediction[mask]
    if truth.size == 0:
        raise ValueError("no observations remain for metric calculation")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("metrics require finite truth and predictions")

    error = prediction - truth
    absolute = np.abs(error)
    denominator = np.abs(truth) + np.abs(prediction)
    smape_terms = np.divide(2.0 * absolute, denominator, out=np.zeros_like(absolute), where=denominator > 0)
    truth_magnitude = np.abs(truth).sum()
    result: dict[str, float] = {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(absolute)),
        "smape": float(np.mean(smape_terms)),
        "wape": float(absolute.sum() / truth_magnitude) if truth_magnitude > 0 else math.nan,
    }
    if training_values is not None and season_length is not None:
        train = np.asarray(training_values, dtype=float).reshape(-1)
        if season_length <= 0 or len(train) <= season_length:
            raise ValueError("training history must exceed the positive season length")
        scale = float(np.mean(np.abs(train[season_length:] - train[:-season_length])))
        result["mase"] = float(np.mean(absolute) / scale) if scale > 0 else math.nan
    return result
