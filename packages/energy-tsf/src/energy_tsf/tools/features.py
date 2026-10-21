"""Serializable, leakage-safe direct-horizon feature construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energy_tsf.contracts.features import FeatureSpec
from energy_tsf.data.schemas import PreparedData, PreparedSeries
from energy_tsf.data.splits import horizon_steps


@dataclass(frozen=True)
class SupervisedDataset:
    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    origins: np.ndarray
    target_indices: np.ndarray


def default_feature_spec(series: PreparedSeries) -> FeatureSpec:
    return FeatureSpec(season_length=horizon_steps(series.frequency))


def _all_feature_names(spec: FeatureSpec) -> tuple[str, ...]:
    names = [f"daily_lag_{day}" for day in spec.daily_lags]
    names.extend(f"recent_lag_{lag}" for lag in range(1, spec.recent_lags + 1))
    for window in spec.rolling_windows:
        names.extend(
            (
                f"rolling_mean_{window}",
                f"rolling_std_{window}",
                f"rolling_min_{window}",
                f"rolling_max_{window}",
            )
        )
    names.extend(f"target_diff_{lag}" for lag in spec.target_differences)
    names.extend(("horizon_sin", "horizon_cos", "horizon_index"))
    if spec.calendar:
        names.extend(("hour_sin", "hour_cos", "weekday_sin", "weekday_cos"))
    for column in spec.covariates.past_only:
        names.extend(f"{column}__lag_{lag}" for lag in spec.covariate_lags)
        names.extend(f"{column}__diff_{lag}" for lag in spec.covariate_differences)
        for window in spec.covariate_rolling_windows:
            names.extend(
                (
                    f"{column}__rolling_mean_{window}",
                    f"{column}__rolling_std_{window}",
                    f"{column}__rolling_min_{window}",
                    f"{column}__rolling_max_{window}",
                )
            )
    for column in spec.covariates.known_future:
        names.append(f"{column}__future")
        names.extend(f"{column}__lag_{lag}" for lag in spec.covariate_lags)
        names.extend(f"{column}__diff_{lag}" for lag in spec.covariate_differences)
        for window in spec.covariate_rolling_windows:
            names.extend(
                (
                    f"{column}__rolling_mean_{window}",
                    f"{column}__rolling_std_{window}",
                    f"{column}__rolling_min_{window}",
                    f"{column}__rolling_max_{window}",
                )
            )
    return tuple(names)


def feature_names(spec: FeatureSpec) -> tuple[str, ...]:
    names = _all_feature_names(spec)
    if not spec.selected_features:
        return names
    missing = sorted(set(spec.selected_features) - set(names))
    if missing:
        raise ValueError(f"selected features are not constructible: {', '.join(missing)}")
    return spec.selected_features


def _known_future_values(
    series: PreparedData,
    column: str,
    origin: int,
    steps: int,
    future_covariates: pd.DataFrame | None,
) -> np.ndarray:
    if origin + steps <= len(series.frame):
        return series.frame[column].iloc[origin : origin + steps].to_numpy(dtype=float)
    if origin != len(series.frame) or future_covariates is None:
        raise ValueError(f"future values are required for known covariate: {column}")
    if column not in future_covariates:
        raise ValueError(f"future covariate column is missing: {column}")
    values = future_covariates[column].to_numpy(dtype=float)
    if len(values) != steps:
        raise ValueError(f"future covariate {column} must contain {steps} rows")
    return values


def features_for_origin(
    series: PreparedData,
    origin: int,
    spec: FeatureSpec,
    steps: int | None = None,
    *,
    future_index: pd.Index | None = None,
    future_covariates: pd.DataFrame | None = None,
) -> np.ndarray:
    """Build direct-horizon rows using only target values before origin."""

    steps = steps or horizon_steps(series.frequency)
    if origin < spec.required_history:
        raise ValueError(f"origin {origin} needs at least {spec.required_history} history rows")
    if future_index is None and origin + steps > len(series.frame):
        raise ValueError("target horizon extends beyond the prepared series")
    if future_index is not None and len(future_index) != steps:
        raise ValueError("future_index length must match forecast steps")

    values = series.frame[series.target].to_numpy(dtype=float)
    offsets = np.arange(steps, dtype=int)
    columns: list[np.ndarray] = []
    for day in spec.daily_lags:
        columns.append(values[origin + offsets - day * spec.season_length])
    for lag in range(1, spec.recent_lags + 1):
        columns.append(np.full(steps, values[origin - lag]))
    for window in spec.rolling_windows:
        history = values[origin - window : origin]
        columns.extend(
            (
                np.full(steps, history.mean()),
                np.full(steps, history.std(ddof=0)),
                np.full(steps, history.min()),
                np.full(steps, history.max()),
            )
        )
    for lag in spec.target_differences:
        difference = values[origin - 1] - values[origin - 1 - lag]
        columns.append(np.full(steps, difference))

    angle = 2.0 * np.pi * offsets / steps
    columns.extend((np.sin(angle), np.cos(angle), offsets / max(steps - 1, 1)))
    if spec.calendar:
        if future_index is not None and isinstance(future_index, pd.DatetimeIndex):
            timestamps = pd.Series(future_index)
            hour = timestamps.dt.hour.to_numpy() + timestamps.dt.minute.to_numpy() / 60.0
            weekday = timestamps.dt.dayofweek.to_numpy()
            columns.extend(
                (
                    np.sin(2.0 * np.pi * hour / 24.0),
                    np.cos(2.0 * np.pi * hour / 24.0),
                    np.sin(2.0 * np.pi * weekday / 7.0),
                    np.cos(2.0 * np.pi * weekday / 7.0),
                )
            )
        elif series.timestamp_column is None:
            columns.extend((np.zeros(steps), np.ones(steps), np.zeros(steps), np.ones(steps)))
        else:
            timestamps = series.frame[series.timestamp_column].iloc[origin : origin + steps].dt
            hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
            weekday = timestamps.dayofweek.to_numpy()
            columns.extend(
                (
                    np.sin(2.0 * np.pi * hour / 24.0),
                    np.cos(2.0 * np.pi * hour / 24.0),
                    np.sin(2.0 * np.pi * weekday / 7.0),
                    np.cos(2.0 * np.pi * weekday / 7.0),
                )
            )
    for column in spec.covariates.past_only:
        covariate = series.frame[column].to_numpy(dtype=float)
        for lag in spec.covariate_lags:
            columns.append(np.full(steps, covariate[origin - lag]))
        for lag in spec.covariate_differences:
            columns.append(np.full(steps, covariate[origin - 1] - covariate[origin - 1 - lag]))
        for window in spec.covariate_rolling_windows:
            history = covariate[origin - window : origin]
            columns.extend(
                (
                    np.full(steps, history.mean()),
                    np.full(steps, history.std(ddof=0)),
                    np.full(steps, history.min()),
                    np.full(steps, history.max()),
                )
            )

    for column in spec.covariates.known_future:
        history = series.frame[column].iloc[:origin].to_numpy(dtype=float)
        future = _known_future_values(series, column, origin, steps, future_covariates)
        combined = np.concatenate((history, future))
        columns.append(future)
        for lag in spec.covariate_lags:
            indices = origin + offsets - lag
            columns.append(combined[indices])
        for lag in spec.covariate_differences:
            current = combined[origin + offsets]
            previous = combined[origin + offsets - lag]
            columns.append(current - previous)
        for window in spec.covariate_rolling_windows:
            past = history[-window:]
            columns.extend(
                (
                    np.full(steps, past.mean()),
                    np.full(steps, past.std(ddof=0)),
                    np.full(steps, past.min()),
                    np.full(steps, past.max()),
                )
            )

    matrix = np.column_stack(columns).astype(float, copy=False)
    all_names = _all_feature_names(spec)
    if matrix.shape[1] != len(all_names):
        raise RuntimeError("constructed feature matrix and names are inconsistent")
    if spec.selected_features:
        selected_indices = [all_names.index(name) for name in spec.selected_features]
        matrix = matrix[:, selected_indices]
    return matrix


def build_supervised_dataset(
    series: PreparedSeries,
    origins: tuple[int, ...] | list[int] | np.ndarray,
    spec: FeatureSpec,
    steps: int | None = None,
) -> SupervisedDataset:
    """Build aligned direct-horizon rows using only data available at each origin."""
    steps = steps or horizon_steps(series.frequency)
    origin_values = np.asarray(origins, dtype=int)
    if origin_values.size == 0:
        raise ValueError("at least one forecast origin is required")
    matrices = [features_for_origin(series, int(origin), spec, steps) for origin in origin_values]
    values = series.frame[series.target].to_numpy(dtype=float)
    targets = [values[origin : origin + steps] for origin in origin_values]
    target_indices = np.concatenate([np.arange(origin, origin + steps) for origin in origin_values])
    repeated_origins = np.repeat(origin_values, steps)
    return SupervisedDataset(
        x=np.concatenate(matrices),
        y=np.concatenate(targets),
        feature_names=feature_names(spec),
        origins=repeated_origins,
        target_indices=target_indices,
    )


def eligible_origins(origins: tuple[int, ...], spec: FeatureSpec) -> tuple[int, ...]:
    """Keep origins with enough preceding observations for the feature recipe."""
    return tuple(origin for origin in origins if origin >= spec.required_history)
