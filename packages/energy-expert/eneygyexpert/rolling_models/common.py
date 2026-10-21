"""Shared safe inputs and audit records for rolling model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data import KEY_COLUMNS


@dataclass
class ModelAudit:
    """Counters distinguish fitting, state ingestion, and prediction."""

    fits: list[dict] = field(default_factory=list)
    updates: list[dict] = field(default_factory=list)
    predictions: list[dict] = field(default_factory=list)

    def fitted(self, history):
        self.fits.append(dict(end=history.ds.max(), rows=len(history)))


def horizon(window):
    sizes = window.future_covariates.groupby("unique_id").size()
    if sizes.empty or sizes.nunique() != 1:
        raise ValueError("Each series must have the same nonempty forecast horizon")
    return int(sizes.iloc[0])


def model_history(window, historical=(), future=()):
    cols = list(dict.fromkeys([*KEY_COLUMNS, "y", *historical, *future]))
    missing = set(cols) - set(window.history)
    if missing:
        raise ValueError(f"Missing archived model inputs: {sorted(missing)}")
    frame = window.history.loc[:, cols].copy()
    if not np.isfinite(
        frame.drop(columns=list(KEY_COLUMNS)).to_numpy(dtype=float)
    ).all():
        raise ValueError("Model inputs must be finite")
    return frame


def model_future(window, future=()):
    cols = [*KEY_COLUMNS, *future]
    return window.future_covariates.loc[:, cols].copy()


def point_output(predicted, alias):
    """Normalize NeuralForecast median output without dropping timestamps."""
    if alias not in predicted:
        median = f"{alias}-median"
        if median not in predicted:
            raise ValueError(f"Model did not return point/median column {alias}")
        predicted = predicted.rename(columns={median: alias})
    return predicted.loc[:, [*KEY_COLUMNS, alias]].reset_index(drop=True)
