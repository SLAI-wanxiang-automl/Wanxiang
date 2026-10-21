"""Split-local missing-value interpolation with an auditable mask."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from energy_tsf.contracts.data import (
    DataQualityReport,
    ForecastContextProfile,
    ImputationReport,
    TemporalSplit,
)
from energy_tsf.contracts.backtest import BacktestInputProfile, BacktestWindow
from energy_tsf.contracts.modeling import frequency_seconds
from energy_tsf.contracts.update import UpdateInputProfile
from energy_tsf.data.schemas import BacktestInput, ForecastContext, PreparedHistory, PreparedSeries, RawSeries
from energy_tsf.data.splits import horizon_steps
from energy_tsf.tools.point_in_time import future_time_index, relative_horizon_index


def _irregular_intervals(raw: RawSeries) -> int:
    if raw.timestamp_column is None or len(raw.frame) < 2:
        return 0
    differences = raw.frame[raw.timestamp_column].diff().iloc[1:].dt.total_seconds()
    return int((differences != frequency_seconds(raw.frequency)).sum())


def _clean_partition(values: pd.Series, *, name: str) -> pd.Series:
    if values.notna().sum() == 0:
        raise ValueError(f"target is entirely missing in {name}")
    return values.interpolate(method="linear", limit_direction="both")


def _clean_causal_partition(values: pd.Series, *, name: str, seed: float) -> pd.Series:
    """Fill an evaluation partition using information available at each row."""

    if values.notna().sum() == 0:
        raise ValueError(f"target is entirely missing in {name}")
    seeded = pd.concat(
        [pd.Series([float(seed)]), values.reset_index(drop=True)],
        ignore_index=True,
    )
    return seeded.ffill().iloc[1:]


def _clean_covariate_columns(frame: pd.DataFrame, raw: RawSeries, split: TemporalSplit) -> None:
    """Clean exogenous columns without reading later evaluation values."""

    train_start, train_end = split.bounds("train")
    for column in raw.covariates.columns:
        values = frame[column]
        finite = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise ValueError(f"covariate contains infinite values: {column}")
        train = _clean_partition(values.iloc[train_start:train_end], name=f"{column} train split")
        frame.loc[frame.index[train_start:train_end], column] = train.to_numpy()
        previous = float(train.iloc[-1])
        for name in ("validation", "test"):
            start, end = split.bounds(name)
            if start == end:
                continue
            part = frame.iloc[start:end][column]
            cleaned = _clean_causal_partition(part, name=f"{column} {name} split", seed=previous)
            frame.loc[frame.index[start:end], column] = cleaned.to_numpy()
            previous = float(cleaned.iloc[-1])


def clean_series(raw: RawSeries, split: TemporalSplit) -> PreparedSeries:
    """Clean train freely and evaluation partitions causally, preserving missingness."""

    if len(raw.frame) != split.n_rows:
        raise ValueError("split row count does not match the input series")
    frame = raw.frame.copy(deep=True)
    if raw.timestamp_column is None:
        frame = frame.reset_index(drop=True)
        frame.index = frame.index.rename("relative_time")
    values = frame[raw.target]
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError("target contains infinite values")
    mask = values.isna().copy()
    counts: dict[str, int] = {}
    _clean_covariate_columns(frame, raw, split)

    train_start, train_end = split.bounds("train")
    train = _clean_partition(
        frame.iloc[train_start:train_end][raw.target],
        name="train split",
    )
    frame.loc[frame.index[train_start:train_end], raw.target] = train.to_numpy()
    counts["train"] = int(mask.iloc[train_start:train_end].sum())

    previous = float(train.iloc[-1])
    for name in ("validation", "test"):
        start, end = split.bounds(name)
        part = frame.iloc[start:end][raw.target]
        cleaned = _clean_causal_partition(part, name=f"{name} split", seed=previous)
        frame.loc[frame.index[start:end], raw.target] = cleaned.to_numpy()
        counts[name] = int(mask.iloc[start:end].sum())
        previous = float(cleaned.iloc[-1])

    non_finite = int((~np.isfinite(frame[raw.target].to_numpy(dtype=float))).sum())
    if non_finite:
        raise ValueError("target contains non-finite values after cleaning")

    quality = DataQualityReport(
        rows=len(frame),
        original_missing=int(mask.sum()),
        imputed=int(mask.sum()),
        non_finite_after_cleaning=non_finite,
        timestamp_sorted=True,
        irregular_intervals=_irregular_intervals(raw),
    )
    imputation = ImputationReport(
        total_imputed=int(mask.sum()),
        train_imputed=counts["train"],
        validation_imputed=counts["validation"],
        test_imputed=counts["test"],
    )
    return PreparedSeries(
        frame=frame,
        target=raw.target,
        timestamp_column=raw.timestamp_column,
        frequency=raw.frequency,
        imputation_mask=mask.rename(raw.target),
        split=split,
        quality=quality,
        imputation=imputation,
        covariates=raw.covariates,
    )


def prepare_series(raw: RawSeries) -> PreparedSeries:
    """Apply the standard chronological split and leakage-safe cleaning policy."""
    from energy_tsf.data.splits import chronological_split

    return clean_series(raw, chronological_split(len(raw.frame)))


def clean_update_series(
    raw: RawSeries,
    *,
    timezone: str,
    validation_days: int = 7,
    horizon: str = "1d",
) -> tuple[PreparedSeries, UpdateInputProfile, tuple[int, ...]]:
    """Create a train + causal seven-day validation series for model updates."""

    if raw.timestamp_column is None:
        raise ValueError("update input requires a timestamp column")
    frame = raw.frame.sort_values(raw.timestamp_column, kind="stable").reset_index(drop=True).copy()
    timestamps = frame[raw.timestamp_column]
    if timestamps.duplicated().any():
        raise ValueError("timestamp column contains duplicates")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    else:
        timestamps = timestamps.dt.tz_convert(timezone)
    frame[raw.timestamp_column] = timestamps
    normalized = RawSeries(frame, raw.target, raw.timestamp_column, raw.frequency, raw.covariates)
    irregular = _irregular_intervals(normalized)
    if irregular:
        raise ValueError(f"timestamp column contains {irregular} irregular intervals")

    steps = horizon_steps(raw.frequency, horizon)
    validation_rows = validation_days * steps
    if validation_days != 7:
        raise ValueError("the first update release requires seven validation days")
    if len(frame) % steps:
        raise ValueError("update input must contain only complete one-day windows")
    if len(frame) < 8 * steps:
        raise ValueError("update input requires at least eight complete days")
    train_end = len(frame) - validation_rows
    split = TemporalSplit(
        n_rows=len(frame),
        train_end=train_end,
        validation_end=len(frame),
        partition_kind="update",
    )
    values = frame[raw.target]
    finite = values.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError("update target contains infinite values")
    mask = values.isna().copy()
    for day in range(validation_days):
        start = train_end + day * steps
        if values.iloc[start : start + steps].notna().sum() == 0:
            raise ValueError(f"update validation window {day + 1} has no observed truth")
    train = _clean_partition(values.iloc[:train_end], name="update train")
    frame.loc[frame.index[:train_end], raw.target] = train.to_numpy()
    validation = values.iloc[train_end:].copy()
    causal = pd.concat(
        [pd.Series([float(train.iloc[-1])]), validation.reset_index(drop=True)], ignore_index=True
    ).ffill().iloc[1:]
    frame.loc[frame.index[train_end:], raw.target] = causal.to_numpy()
    _clean_covariate_columns(frame, raw, split)
    if not np.isfinite(frame[raw.target].to_numpy(dtype=float)).all():
        raise ValueError("update target contains non-finite values after cleaning")

    quality = DataQualityReport(
        rows=len(frame),
        original_missing=int(mask.sum()),
        imputed=int(mask.sum()),
        non_finite_after_cleaning=0,
        timestamp_sorted=True,
        irregular_intervals=0,
    )
    imputation = ImputationReport(
        total_imputed=int(mask.sum()),
        train_imputed=int(mask.iloc[:train_end].sum()),
        validation_imputed=int(mask.iloc[train_end:].sum()),
        test_imputed=0,
    )
    prepared = PreparedSeries(
        frame=frame,
        target=raw.target,
        timestamp_column=raw.timestamp_column,
        frequency=raw.frequency,
        imputation_mask=mask.rename(raw.target),
        split=split,
        quality=quality,
        imputation=imputation,
        covariates=raw.covariates,
    )
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    digest.update(mask.to_numpy(dtype=bool).tobytes())
    origins = tuple(train_end + day * steps for day in range(validation_days))
    profile = UpdateInputProfile(
        rows=len(frame),
        train_rows=train_end,
        validation_rows=validation_rows,
        horizon_steps=steps,
        start_time=timestamps.iloc[0].to_pydatetime(),
        end_time=timestamps.iloc[-1].to_pydatetime(),
        original_missing=int(mask.sum()),
        input_fingerprint=digest.hexdigest(),
    )
    return prepared, profile, origins


def _history_fingerprint(
    history: PreparedHistory,
    future_covariates: pd.DataFrame | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(history.target.encode("utf-8"))
    digest.update(history.frequency.encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(history.frame, index=True).to_numpy().tobytes())
    digest.update(history.imputation_mask.to_numpy(dtype=bool).tobytes())
    if future_covariates is not None:
        digest.update(pd.util.hash_pandas_object(future_covariates, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def clean_forecast_context(
    raw: RawSeries,
    *,
    timezone: str,
    horizon: str = "1d",
    future_covariates: pd.DataFrame | None = None,
) -> ForecastContext:
    """Clean one observed history and create an explicit future index."""

    frame = raw.frame.copy(deep=True)
    timestamp_column = raw.timestamp_column
    if timestamp_column is None:
        frame = frame.reset_index(drop=True)
        frame.index = frame.index.rename("relative_time")
    else:
        frame = frame.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
        if frame[timestamp_column].duplicated().any():
            raise ValueError("timestamp column contains duplicates")
        timestamps = frame[timestamp_column]
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        else:
            timestamps = timestamps.dt.tz_convert(timezone)
        frame[timestamp_column] = timestamps

    values = frame[raw.target]
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError("target contains infinite values")
    mask = values.isna().copy()
    frame[raw.target] = _clean_partition(values, name="forecast history").to_numpy()
    for column in raw.covariates.columns:
        values = frame[column]
        finite = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise ValueError(f"covariate contains infinite values: {column}")
        frame[column] = _clean_partition(values, name=f"forecast history {column}").to_numpy()

    normalized = RawSeries(
        frame=frame,
        target=raw.target,
        timestamp_column=timestamp_column,
        frequency=raw.frequency,
        covariates=raw.covariates,
    )
    irregular = _irregular_intervals(normalized)
    if irregular:
        raise ValueError(f"timestamp column contains {irregular} irregular intervals")
    non_finite = int((~np.isfinite(frame[raw.target].to_numpy(dtype=float))).sum())
    if non_finite:
        raise ValueError("target contains non-finite values after cleaning")

    quality = DataQualityReport(
        rows=len(frame),
        original_missing=int(mask.sum()),
        imputed=int(mask.sum()),
        non_finite_after_cleaning=non_finite,
        timestamp_sorted=True,
        irregular_intervals=irregular,
    )
    history = PreparedHistory(
        frame=frame,
        target=raw.target,
        timestamp_column=timestamp_column,
        frequency=raw.frequency,
        imputation_mask=mask.rename(raw.target),
        quality=quality,
        covariates=raw.covariates,
    )
    steps = horizon_steps(raw.frequency, horizon)
    if timestamp_column is None:
        future_index: pd.Index = relative_horizon_index(steps)
        start_time = end_time = future_start = future_end = None
    else:
        timestamps = frame[timestamp_column]
        future_index = future_time_index(timestamps.iloc[-1], raw.frequency, steps)
        start_time = timestamps.iloc[0].isoformat()
        end_time = timestamps.iloc[-1].isoformat()
        future_start = future_index[0].isoformat()
        future_end = future_index[-1].isoformat()
    normalized_future = None
    if raw.covariates.known_future:
        if timestamp_column is None or future_covariates is None:
            raise ValueError("known-future covariates require a timestamped future table")
        normalized_future = future_covariates.copy(deep=True)
        timestamps = pd.to_datetime(normalized_future[timestamp_column], errors="raise")
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        else:
            timestamps = timestamps.dt.tz_convert(timezone)
        normalized_future[timestamp_column] = timestamps
        normalized_future = normalized_future.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
        if len(normalized_future) != steps:
            raise ValueError(f"future covariates must contain exactly {steps} rows")
        if not pd.DatetimeIndex(normalized_future[timestamp_column]).equals(pd.DatetimeIndex(future_index)):
            raise ValueError("future covariate timestamps do not match the forecast horizon")
        for column in raw.covariates.known_future:
            values = normalized_future[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"future covariate contains missing or non-finite values: {column}")
    profile = ForecastContextProfile(
        rows=len(frame),
        timestamped=timestamp_column is not None,
        start_time=start_time,
        end_time=end_time,
        future_start=future_start,
        future_end=future_end,
        horizon_steps=steps,
        original_missing=int(mask.sum()),
        imputed=int(mask.sum()),
        irregular_intervals=irregular,
        input_fingerprint=_history_fingerprint(history, normalized_future),
    )
    return ForecastContext(
        history=history,
        future_index=future_index,
        profile=profile,
        future_covariates=normalized_future,
    )


def clean_backtest_input(raw: RawSeries, *, timezone: str, horizon: str = "1d") -> BacktestInput:
    """Normalize timestamped truth without imputing labels used for evaluation."""

    if raw.timestamp_column is None:
        raise ValueError("backtest input requires a timestamp column")
    frame = raw.frame.sort_values(raw.timestamp_column, kind="stable").reset_index(drop=True).copy()
    timestamps = frame[raw.timestamp_column]
    if timestamps.duplicated().any():
        raise ValueError("timestamp column contains duplicates")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    else:
        timestamps = timestamps.dt.tz_convert(timezone)
    frame[raw.timestamp_column] = timestamps
    normalized = RawSeries(
        frame=frame,
        target=raw.target,
        timestamp_column=raw.timestamp_column,
        frequency=raw.frequency,
        covariates=raw.covariates,
    )
    irregular = _irregular_intervals(normalized)
    if irregular:
        raise ValueError(f"timestamp column contains {irregular} irregular intervals")
    values = frame[raw.target]
    finite = values.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError("backtest truth contains infinite values")

    steps = horizon_steps(raw.frequency, horizon)
    if len(frame) < 2 * steps:
        raise ValueError("backtest input requires at least two complete one-day windows")
    if len(frame) % steps:
        raise ValueError("backtest input must contain only complete one-day windows")
    observed = values.notna().rename("observed")
    offset = pd.to_timedelta(frequency_seconds(raw.frequency), unit="s")
    windows: list[BacktestWindow] = []
    for index, start in enumerate(range(0, len(frame), steps), 1):
        part = frame.iloc[start : start + steps]
        valid = int(observed.iloc[start : start + steps].sum())
        if valid == 0:
            raise ValueError(f"backtest window {index} has no observed truth")
        origin = part[raw.timestamp_column].iloc[0]
        windows.append(
            BacktestWindow(
                window_id=f"w{index:03d}",
                origin=origin.to_pydatetime(),
                end_time=(part[raw.timestamp_column].iloc[-1] + offset).to_pydatetime(),
                horizon_steps=steps,
                valid_observations=valid,
            )
        )

    digest = hashlib.sha256()
    digest.update(raw.target.encode("utf-8"))
    digest.update(raw.frequency.encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    digest.update(observed.to_numpy(dtype=bool).tobytes())
    profile = BacktestInputProfile(
        rows=len(frame),
        start_time=timestamps.iloc[0].to_pydatetime(),
        end_time=timestamps.iloc[-1].to_pydatetime(),
        window_count=len(windows),
        horizon_steps=steps,
        original_missing=int((~observed).sum()),
        irregular_intervals=irregular,
        input_fingerprint=digest.hexdigest(),
    )
    return BacktestInput(
        frame=frame,
        target=raw.target,
        timestamp_column=raw.timestamp_column,
        frequency=raw.frequency,
        observed_mask=observed,
        windows=tuple(windows),
        profile=profile,
        covariates=raw.covariates,
    )
