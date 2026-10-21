"""Normalized result tables shared by all model providers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import Field

from energy_tsf.contracts.artifact import ActiveRef, CandidateRef
from energy_tsf.contracts.common import ContractModel
from energy_tsf.contracts.modeling import TopKSelection


@dataclass(frozen=True)
class ForecastTable:
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        required = {"provider", "origin", "horizon_step", "target_index", "y_pred"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"forecast table is missing columns: {sorted(missing)}")
        if not np.isfinite(self.frame["y_pred"].to_numpy(dtype=float)).all():
            raise ValueError("forecast values must be finite")
        if self.frame.duplicated(["origin", "horizon_step"]).any():
            raise ValueError("forecast rows must be unique by origin and horizon_step")
        for _, group in self.frame.groupby("origin", sort=False):
            expected = np.arange(1, len(group) + 1)
            if not np.array_equal(group["horizon_step"].to_numpy(), expected):
                raise ValueError("horizon_step must be contiguous and one-based per origin")

    @property
    def predictions(self) -> np.ndarray:
        return self.frame["y_pred"].to_numpy(dtype=float)

    def require_coverage(self, origins: tuple[int, ...], horizon_steps: int) -> None:
        """Require exactly one complete horizon for every requested origin."""

        if horizon_steps <= 0:
            raise ValueError("forecast horizon must be positive")
        expected_origins = tuple(origins)
        actual_origins = tuple(self.frame["origin"].drop_duplicates().astype(int))
        if actual_origins != expected_origins:
            raise ValueError(
                f"forecast origins {actual_origins} do not match requested origins {expected_origins}"
            )
        expected_steps = np.arange(1, horizon_steps + 1)
        for origin in expected_origins:
            group = self.frame.loc[self.frame["origin"] == origin]
            if not np.array_equal(group["horizon_step"].to_numpy(dtype=int), expected_steps):
                raise ValueError(f"forecast origin {origin} does not contain a complete horizon")


class ModelingWorkflowResult(ContractModel):
    request_id: str
    status: str
    plan_id: str
    planned_trials: int
    succeeded_trials: int
    failed_trials: int
    unavailable_providers: dict[str, str] = Field(default_factory=dict)
    top_k: TopKSelection
    search_pool_id: str | None = None
    solution_id: str | None = None
    candidate: CandidateRef | None = None
    active: ActiveRef | None = None
    report_path: str
    release_reasons: tuple[str, ...] = ()
