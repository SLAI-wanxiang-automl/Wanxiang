"""Shared persistence and forecast-table helpers for official providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy_tsf.contracts.result import ForecastTable
from energy_tsf.data.schemas import PreparedData


def forecast_table(
    provider: str,
    series: PreparedData,
    origins: tuple[int, ...],
    predictions: list[np.ndarray],
    *,
    future_index: pd.Index | None = None,
) -> ForecastTable:
    if future_index is not None:
        if len(origins) != 1:
            raise ValueError("future_index supports exactly one forecast origin")
        if len(future_index) != len(predictions[0]):
            raise ValueError("future_index length must match forecast length")
    rows: list[dict[str, Any]] = []
    for origin, forecast in zip(origins, predictions):
        for offset, value in enumerate(np.asarray(forecast, dtype=float)):
            target_index = origin + offset
            row: dict[str, Any] = {
                "provider": provider,
                "origin": origin,
                "horizon_step": offset + 1,
                "target_index": target_index,
                "y_pred": float(value),
            }
            if series.timestamp_column is not None:
                if future_index is not None:
                    row["timestamp"] = future_index[offset]
                else:
                    row["timestamp"] = series.frame[series.timestamp_column].iloc[target_index]
            rows.append(row)
    return ForecastTable(pd.DataFrame(rows))


def save_joblib(model: Any, path: str | Path) -> Path:
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return target


def load_joblib(path: str | Path) -> Any:
    import joblib

    return joblib.load(path)
