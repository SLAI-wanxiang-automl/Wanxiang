"""Generic CSV/Parquet ingestion for one target series."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.data.schemas import RawSeries


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported data format: {suffix}")


def inspect_target_schema(
    path: str | Path,
    target: str,
    timestamp_column: str | None,
    covariates: CovariateSpec | None = None,
) -> None:
    """Read a small projection and reject missing or incompatible columns."""

    source = Path(path).expanduser()
    covariates = covariates or CovariateSpec()
    requested = [target, *covariates.columns] + ([timestamp_column] if timestamp_column is not None else [])
    requested = list(dict.fromkeys(requested))
    try:
        if source.suffix.lower() == ".csv":
            sample = pd.read_csv(source, usecols=requested, nrows=1000)
        elif source.suffix.lower() in {".parquet", ".pq"}:
            sample = pd.read_parquet(source, columns=requested).head(1000)
        else:
            raise ValueError(f"unsupported data format: {source.suffix.lower()}")
    except (KeyError, ValueError) as exc:
        raise ValueError("target or timestamp column was not found") from exc
    try:
        pd.to_numeric(sample[target], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target column must be numeric: {target}") from exc
    for column in covariates.columns:
        try:
            pd.to_numeric(sample[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"covariate column must be numeric: {column}") from exc
    if timestamp_column is not None:
        try:
            pd.to_datetime(sample[timestamp_column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid timestamps in column: {timestamp_column}") from exc


def load_target_series(
    path: str | Path,
    target: str,
    timestamp_column: str | None,
    frequency: str,
    covariates: CovariateSpec | None = None,
) -> RawSeries:
    """Read one target and its explicitly declared covariates."""

    source = Path(path).expanduser()
    frame = _read_table(source)
    covariates = covariates or CovariateSpec()
    if target not in frame.columns:
        raise ValueError(f"target column not found: {target}")
    missing_covariates = [column for column in covariates.columns if column not in frame.columns]
    if missing_covariates:
        raise ValueError(f"covariate columns not found: {', '.join(missing_covariates)}")
    columns = [target, *covariates.columns]
    if timestamp_column is not None:
        if timestamp_column not in frame.columns:
            raise ValueError(f"timestamp column not found: {timestamp_column}")
        columns.insert(0, timestamp_column)
    selected = frame.loc[:, columns].copy()
    try:
        selected[target] = pd.to_numeric(selected[target], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target column must be numeric: {target}") from exc
    for column in covariates.columns:
        try:
            selected[column] = pd.to_numeric(selected[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"covariate column must be numeric: {column}") from exc

    if timestamp_column is not None:
        try:
            selected[timestamp_column] = pd.to_datetime(selected[timestamp_column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid timestamps in column: {timestamp_column}") from exc
        selected = selected.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    else:
        selected = selected.reset_index(drop=True)
        selected.index = pd.RangeIndex(len(selected), name="relative_time")

    return RawSeries(
        frame=selected,
        target=target,
        timestamp_column=timestamp_column,
        frequency=frequency,
        covariates=covariates,
    )


def load_future_covariates(
    path: str | Path,
    timestamp_column: str,
    covariates: CovariateSpec,
) -> pd.DataFrame:
    """Load an explicit future-known covariate table without any target values."""

    source = Path(path).expanduser()
    frame = _read_table(source)
    columns = [timestamp_column, *covariates.known_future]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"future covariate columns not found: {', '.join(missing)}")
    selected = frame.loc[:, columns].copy()
    selected[timestamp_column] = pd.to_datetime(selected[timestamp_column], errors="raise")
    for column in covariates.known_future:
        selected[column] = pd.to_numeric(selected[column], errors="raise").astype(float)
    return selected.sort_values(timestamp_column, kind="stable").reset_index(drop=True)


def load_modeling_data(config) -> RawSeries:
    """Load from a validated ModelingConfig without coupling its sections."""

    return load_target_series(
        config.data.path,
        config.request.target,
        config.data.timestamp_column,
        config.data.frequency,
        config.data.covariates,
    )


def load_forecast_data(config) -> RawSeries:
    """Load observed history from a validated ForecastConfig."""

    return load_target_series(
        config.data.path,
        config.station.target,
        config.station.timestamp_column,
        config.station.frequency,
        config.data.covariates,
    )


def load_backtest_data(config) -> RawSeries:
    """Load timestamped truth from a validated BacktestConfig."""

    return load_target_series(
        config.data.path,
        config.request.target,
        config.data.timestamp_column,
        config.data.frequency,
        config.data.covariates,
    )


def load_update_data(config) -> RawSeries:
    """Load timestamped recent observations from an UpdateConfig."""

    return load_target_series(
        config.data.path,
        config.request.target,
        config.data.timestamp_column,
        config.data.frequency,
        config.data.covariates,
    )
