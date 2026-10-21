"""Immutable candidate-search-pool contracts."""

from __future__ import annotations

from pydantic import Field, model_validator

from energy_tsf.contracts.common import ContractModel
from energy_tsf.contracts.modeling import TrialSpec


class WindowTrialScore(ContractModel):
    window_id: str
    origin: int = Field(ge=0)
    trial_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    provider: str
    rmse: float = Field(ge=0)
    mae: float = Field(ge=0)
    valid_observations: int = Field(gt=0)
    rank: int = Field(gt=0)
    selected: bool


class SearchPoolEntry(ContractModel):
    spec: TrialSpec
    validation_metrics: dict[str, float | None]
    selected_windows: tuple[str, ...]
    best_window_rank: int = Field(gt=0)
    runtime: dict[str, str] = Field(default_factory=dict)
    source_artifact_ref: str | None = None

    @model_validator(mode="after")
    def validate_windows(self) -> "SearchPoolEntry":
        if not self.selected_windows:
            raise ValueError("search-pool entry requires at least one selected window")
        if len(set(self.selected_windows)) != len(self.selected_windows):
            raise ValueError("selected windows must be unique")
        return self


class SearchPoolArtifact(ContractModel):
    pool_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str
    station_id: str
    data_snapshot_id: str
    split_id: str
    horizon: str
    top_k_per_window: int = Field(gt=0)
    window_count: int = Field(gt=0)
    entries: tuple[SearchPoolEntry, ...]

    @model_validator(mode="after")
    def validate_entries(self) -> "SearchPoolArtifact":
        ids = [entry.spec.trial_id for entry in self.entries]
        if not ids:
            raise ValueError("search pool cannot be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("search-pool trial ids must be unique")
        if any(entry.spec.data_snapshot_id != self.data_snapshot_id for entry in self.entries):
            raise ValueError("search-pool entries must use the pool data snapshot")
        if any(entry.spec.split_id != self.split_id for entry in self.entries):
            raise ValueError("search-pool entries must use the pool split")
        return self
