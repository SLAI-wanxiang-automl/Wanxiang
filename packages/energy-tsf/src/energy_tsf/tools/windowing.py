"""Timestamp-based input extraction used by an external Agent before workflow calls."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.data.splits import horizon_steps


@dataclass(frozen=True)
class ExtractedForecastInput:
    origin: pd.Timestamp
    history: pd.DataFrame
    future_covariates: pd.DataFrame | None


def extract_forecast_input(
    frame: pd.DataFrame,
    *,
    origin: str | pd.Timestamp,
    timestamp_column: str,
    target: str,
    frequency: str,
    covariates: CovariateSpec,
    history_days: int = 28,
) -> ExtractedForecastInput:
    """Return causal history and a target-free one-day known-future table."""

    ordered = frame.copy()
    ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column], errors="raise")
    ordered = ordered.sort_values(timestamp_column, kind="stable")
    point = pd.Timestamp(origin)
    steps = horizon_steps(frequency)
    offset = pd.to_timedelta(86400 // steps, unit="s")
    start = point - pd.Timedelta(days=history_days)
    end = point + pd.Timedelta(days=1)
    history_columns = [timestamp_column, target, *covariates.columns]
    history = ordered.loc[
        (ordered[timestamp_column] >= start) & (ordered[timestamp_column] < point),
        history_columns,
    ].reset_index(drop=True)
    if len(history) != history_days * steps:
        raise ValueError("forecast history does not contain the requested complete window")
    expected_future = pd.date_range(point, periods=steps, freq=offset)
    future = None
    if covariates.known_future:
        future = ordered.loc[
            (ordered[timestamp_column] >= point) & (ordered[timestamp_column] < end),
            [timestamp_column, *covariates.known_future],
        ].reset_index(drop=True)
        if len(future) != steps or not pd.DatetimeIndex(future[timestamp_column]).equals(expected_future):
            raise ValueError("known-future covariates do not cover the one-day horizon")
        if target in future.columns:
            raise RuntimeError("future target leaked into forecast input")
    return ExtractedForecastInput(origin=point, history=history, future_covariates=future)


def extract_interval(
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timestamp_column: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Extract a half-open timestamp interval with explicit columns."""

    ordered = frame.copy()
    ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column], errors="raise")
    selected = ordered.loc[
        (ordered[timestamp_column] >= pd.Timestamp(start))
        & (ordered[timestamp_column] < pd.Timestamp(end)),
        [timestamp_column, *columns],
    ]
    return selected.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
