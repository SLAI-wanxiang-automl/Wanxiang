"""Bounded execution track for SearchPool retraining and final fitting."""

from __future__ import annotations

from pathlib import Path

from energy_tsf.automl.runner import TrialRunner
from energy_tsf.automl.update_finalize import FinalizedUpdateMember, finalize_update_members
from energy_tsf.contracts.modeling import ExperimentPlan, TopKSelection, TrialRecord
from energy_tsf.data.schemas import PreparedSeries
from energy_tsf.providers.registry import ProviderRegistry
from energy_tsf.runtime.policy import UPDATE_POLICY


class UpdateTrack:
    policy = UPDATE_POLICY

    def __init__(self, registry: ProviderRegistry) -> None:
        self.registry = registry

    def run_candidates(
        self,
        plan: ExperimentPlan,
        series: PreparedSeries,
        *,
        train_origins: tuple[int, ...],
        validation_origins: tuple[int, ...],
        train_end: int,
        max_parallel_trials: int,
        work_dir: str | Path,
    ) -> tuple[TrialRecord, ...]:
        """Execute only the candidate trials contained in the frozen update plan."""
        return TrialRunner(
            self.registry,
            series,
            train_origins=train_origins,
            validation_origins=validation_origins,
            train_end=train_end,
            work_root=work_dir,
        ).run_plan(plan, max_parallel_trials=max_parallel_trials)

    def finalize(
        self,
        selection: TopKSelection,
        series: PreparedSeries,
        *,
        retrain_origins: tuple[int, ...],
        work_dir: str | Path,
    ) -> tuple[FinalizedUpdateMember, ...]:
        """Refit selected update members on the final training origins."""
        return finalize_update_members(
            selection,
            series,
            self.registry,
            retrain_origins=retrain_origins,
            work_dir=work_dir,
        )
