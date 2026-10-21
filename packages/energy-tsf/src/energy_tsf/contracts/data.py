"""Contracts for prepared data, temporal splits, and snapshots."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from energy_tsf.contracts.common import ContractModel


class TemporalSplit(ContractModel):
    n_rows: int = Field(gt=0)
    train_end: int = Field(gt=0)
    validation_end: int = Field(gt=0)
    partition_kind: Literal["modeling", "update"] = "modeling"

    @model_validator(mode="after")
    def validate_boundaries(self) -> "TemporalSplit":
        valid = (
            0 < self.train_end < self.validation_end < self.n_rows
            if self.partition_kind == "modeling"
            else 0 < self.train_end < self.validation_end == self.n_rows
        )
        if not valid:
            raise ValueError("split boundaries do not match the partition kind")
        return self

    def bounds(self, name: str) -> tuple[int, int]:
        if name == "train":
            return 0, self.train_end
        if name == "validation":
            return self.train_end, self.validation_end
        if name == "test":
            return self.validation_end, self.n_rows
        raise KeyError(name)


class OriginSet(ContractModel):
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    horizon_steps: int = Field(gt=0)
    stride: int = Field(gt=0)


class DataQualityReport(ContractModel):
    rows: int
    original_missing: int
    imputed: int
    non_finite_after_cleaning: int
    timestamp_sorted: bool
    irregular_intervals: int


class ImputationReport(ContractModel):
    strategy: str = "linear_interpolate"
    edge_fill: str = "nearest"
    total_imputed: int
    train_imputed: int
    validation_imputed: int
    test_imputed: int


class DataSnapshotRef(ContractModel):
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str
    rows: int = Field(gt=0)
    target: str
    frequency: str


class ForecastContextProfile(ContractModel):
    """Auditable facts about the cleaned history and requested future range."""

    rows: int = Field(gt=0)
    timestamped: bool
    start_time: str | None = None
    end_time: str | None = None
    future_start: str | None = None
    future_end: str | None = None
    horizon_steps: int = Field(gt=0)
    original_missing: int = Field(ge=0)
    imputed: int = Field(ge=0)
    irregular_intervals: int = Field(ge=0)
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
