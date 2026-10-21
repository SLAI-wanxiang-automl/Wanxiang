"""Small explicit permission policy for runtime tracks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastTrackPolicy:
    read_active: bool = True
    read_candidate: bool = False
    write_active: bool = False
    write_candidate: bool = False
    allow_fit: bool = False
    allow_model_save: bool = False
    allow_checkpoint_download: bool = False
    allow_promotion: bool = False


FORECAST_POLICY = ForecastTrackPolicy()


@dataclass(frozen=True)
class BacktestTrackPolicy:
    read_active: bool = True
    read_candidate: bool = False
    write_active: bool = True
    write_candidate: bool = True
    allow_fit: bool = False
    allow_model_save: bool = False
    allow_checkpoint_download: bool = False
    allow_promotion: bool = True


BACKTEST_POLICY = BacktestTrackPolicy()


@dataclass(frozen=True)
class UpdateTrackPolicy:
    read_active: bool = True
    read_candidate: bool = False
    write_active: bool = True
    write_candidate: bool = True
    allow_fit: bool = True
    allow_model_save: bool = True
    allow_checkpoint_download: bool = False
    allow_promotion: bool = True


UPDATE_POLICY = UpdateTrackPolicy()
