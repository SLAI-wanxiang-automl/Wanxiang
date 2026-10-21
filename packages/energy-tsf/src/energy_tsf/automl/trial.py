"""Single-trial evaluation helpers and stable validation alignment."""

from __future__ import annotations

import hashlib

import numpy as np

from energy_tsf.contracts.result import ForecastTable
from energy_tsf.data.schemas import PreparedSeries
from energy_tsf.tools.metrics import point_metrics


def validation_truth(
    series: PreparedSeries,
    table: ForecastTable,
    train_end: int,
    season_length: int,
) -> tuple[dict[str, float | None], int, str]:
    """Calculate validation metrics and a fingerprint for aligned truth rows."""
    indices = table.frame["target_index"].to_numpy(dtype=int)
    expected = table.frame["origin"].to_numpy(dtype=int) + table.frame["horizon_step"].to_numpy(dtype=int) - 1
    if not np.array_equal(indices, expected):
        raise ValueError("provider target indices are not aligned to origin and horizon_step")
    values = series.frame[series.target].to_numpy(dtype=float)
    if indices.size == 0 or indices.min() < 0 or indices.max() >= len(values):
        raise ValueError("provider target indices fall outside the prepared series")
    truth = values[indices]
    include = ~series.imputation_mask.iloc[indices].to_numpy(dtype=bool)
    metrics = point_metrics(
        truth,
        table.predictions,
        include_mask=include,
        training_values=values[:train_end],
        season_length=season_length,
    )
    normalized = {name: value if np.isfinite(value) else None for name, value in metrics.items()}
    digest = hashlib.sha256()
    digest.update(indices.astype("<i8", copy=False).tobytes())
    digest.update(truth.astype("<f8", copy=False).tobytes())
    return normalized, int(include.sum()), digest.hexdigest()
