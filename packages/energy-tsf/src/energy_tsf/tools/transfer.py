"""Explicit capacity-based scale adaptation for cross-station forecasts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from energy_tsf.contracts.station import (
    StationSpec,
    capacity_in_target_unit,
    is_per_unit,
)
from energy_tsf.data.schemas import PreparedHistory


def _to_per_unit(values: np.ndarray, station: StationSpec) -> np.ndarray:
    if is_per_unit(station.target_unit):
        return values.astype(float, copy=True)
    capacity = capacity_in_target_unit(station)
    if capacity is None:
        raise ValueError(f"station {station.station_id} has no convertible capacity")
    return values.astype(float, copy=True) / capacity


def _from_per_unit(values: np.ndarray, station: StationSpec) -> np.ndarray:
    if is_per_unit(station.target_unit):
        return values.astype(float, copy=True)
    capacity = capacity_in_target_unit(station)
    if capacity is None:
        raise ValueError(f"station {station.station_id} has no convertible capacity")
    return values.astype(float, copy=True) * capacity


def adapt_history_to_source(
    history: PreparedHistory,
    target: StationSpec,
    source: StationSpec,
) -> PreparedHistory:
    frame = history.frame.copy(deep=True)
    values = frame[history.target].to_numpy(dtype=float)
    frame[history.target] = _from_per_unit(_to_per_unit(values, target), source)
    return replace(history, frame=frame)


def adapt_predictions_to_target(
    predictions,
    source: StationSpec,
    target: StationSpec,
) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    return _from_per_unit(_to_per_unit(values, source), target)
