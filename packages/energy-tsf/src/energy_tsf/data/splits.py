"""Chronological split and leakage-safe forecast-origin generation."""

from __future__ import annotations

import math

from energy_tsf.contracts.data import OriginSet, TemporalSplit
from energy_tsf.contracts.modeling import frequency_seconds


def chronological_split(n_rows: int) -> TemporalSplit:
    """Return floor-based 70/20/10 boundaries."""

    return TemporalSplit(
        n_rows=n_rows,
        train_end=math.floor(0.70 * n_rows),
        validation_end=math.floor(0.90 * n_rows),
    )


def horizon_steps(frequency: str, horizon: str = "1d") -> int:
    """Convert the supported one-day horizon into sample steps."""
    if horizon != "1d":
        raise ValueError("the first release supports only a one-day horizon")
    return 86400 // frequency_seconds(frequency)


def _origins(start: int, end: int, horizon: int, stride: int) -> tuple[int, ...]:
    first = max(start, 1)
    stop = end - horizon + 1
    return tuple(range(first, stop, stride)) if stop > first else ()


def build_origin_sets(
    split: TemporalSplit,
    frequency: str,
    *,
    horizon: str = "1d",
    stride: int | None = None,
) -> OriginSet:
    """Build complete, non-overlapping origins inside each chronological split."""
    steps = horizon_steps(frequency, horizon)
    stride = steps if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive")
    return OriginSet(
        train=_origins(*split.bounds("train"), steps, stride),
        validation=_origins(*split.bounds("validation"), steps, stride),
        test=_origins(*split.bounds("test"), steps, stride),
        horizon_steps=steps,
        stride=stride,
    )
