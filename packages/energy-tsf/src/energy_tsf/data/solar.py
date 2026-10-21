"""Solar convenience adapter over generic ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_tsf.data.ingestion import load_target_series
from energy_tsf.data.schemas import RawSeries


def load_solar(
    path: str | Path,
    target: str = "power",
    timestamp_column: str | None = "timestamp",
    frequency: str = "10min",
) -> RawSeries:
    """Load a selected channel from the legacy headerless Solar dataset."""
    return load_target_series(path, target, timestamp_column, frequency)


def load_legacy_modern_tsf_solar(
    path: str | Path,
    *,
    column: int = 0,
    target: str = "power",
    frequency: str = "10min",
) -> RawSeries:
    """Adapt ModernTSF's headerless 137-column Solar text file."""

    selected = pd.read_csv(path, header=None, usecols=[column], dtype=float)
    selected.columns = [target]
    selected.index = pd.RangeIndex(len(selected), name="relative_time")
    return RawSeries(selected, target, None, frequency)


def export_legacy_modern_tsf_solar(
    source: str | Path,
    destination: str | Path,
    *,
    column: int = 0,
    target: str = "power",
    tail_fraction: float = 1.0,
) -> Path:
    """Convert one legacy Solar channel into the generic CSV contract.

    ``tail_fraction`` is applied after selecting the target channel. This keeps
    chronological order and makes inputs such as "the final quarter" explicit.
    """

    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in the interval (0, 1]")

    series = load_legacy_modern_tsf_solar(source, column=column, target=target)
    rows = len(series.frame)
    selected_rows = max(1, int(rows * tail_fraction))
    selected = series.frame.iloc[rows - selected_rows :].reset_index(drop=True)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    return output
