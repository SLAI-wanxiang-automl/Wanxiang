"""Point-in-time helpers for a history-only forecast input."""

from __future__ import annotations

import pandas as pd


def future_time_index(last_observation, frequency: str, steps: int) -> pd.DatetimeIndex:
    """Build valid timestamps after the final observed timestamp."""
    if steps <= 0:
        raise ValueError("forecast steps must be positive")
    offset = pd.tseries.frequencies.to_offset(frequency)
    index = pd.date_range(pd.Timestamp(last_observation) + offset, periods=steps, freq=offset)
    index.name = "valid_time"
    return index


def relative_horizon_index(steps: int) -> pd.RangeIndex:
    """Build one-based horizon steps for timestamp-free series."""
    if steps <= 0:
        raise ValueError("forecast steps must be positive")
    return pd.RangeIndex(1, steps + 1, name="horizon_step")
