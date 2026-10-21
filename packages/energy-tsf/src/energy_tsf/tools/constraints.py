"""Deterministic physical constraints applied to normalized forecasts."""

from __future__ import annotations

import numpy as np

from energy_tsf.contracts.forecast import ConstraintSummary
from energy_tsf.contracts.result import ForecastTable


def constrain_forecast(
    table: ForecastTable,
    *,
    minimum: float | None = 0.0,
    maximum: float | None = None,
) -> ForecastTable:
    """Return a constrained copy while preserving forecast alignment columns."""

    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum prediction constraint cannot exceed maximum")
    frame = table.frame.copy(deep=True)
    values = frame["y_pred"].to_numpy(dtype=float)
    if minimum is not None:
        values = np.maximum(values, minimum)
    if maximum is not None:
        values = np.minimum(values, maximum)
    frame["y_pred"] = values
    return ForecastTable(frame)


def constrain_forecast_with_summary(
    table: ForecastTable,
    *,
    minimum: float | None = 0.0,
    maximum: float | None = None,
) -> tuple[ForecastTable, ConstraintSummary]:
    """Apply output bounds and report how many predictions were adjusted."""
    constrained = constrain_forecast(table, minimum=minimum, maximum=maximum)
    before = table.predictions
    after = constrained.predictions
    difference = np.abs(after - before)
    return constrained, ConstraintSummary(
        minimum=minimum,
        maximum=maximum,
        modified_points=int(np.count_nonzero(difference)),
        max_abs_adjustment=float(difference.max()) if difference.size else 0.0,
    )
