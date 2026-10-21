"""Deterministic rolling backtest, Bayesian weight update, and publication."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from energy_tsf.assets.promotion import promote_candidate
from energy_tsf.config.loader import backtest_config_fingerprint
from energy_tsf.config.schema import BacktestConfig
from energy_tsf.contracts.artifact import EnsembleSpec, ReleaseDecision
from energy_tsf.contracts.backtest import BacktestPlan, BacktestRequest, BacktestResult
from energy_tsf.contracts.station import StationSpec, capacity_in_target_unit
from energy_tsf.data import clean_backtest_input, load_backtest_data, load_snapshot_by_id
from energy_tsf.gates.backtest_release import evaluate_backtest_release
from energy_tsf.ledger import LedgerEvent, LedgerWriter
from energy_tsf.memory import ExecutionCache, MemoryRepository, StationRepository, build_solution
from energy_tsf.reporting import write_backtest_reports
from energy_tsf.runtime.backtest_track import BacktestTrack
from energy_tsf.tools.backtest import (
    backtest_metrics,
    backtest_plan_fingerprint,
    build_backtest_plan,
    build_window_data,
    recombine_member_predictions,
)
from energy_tsf.tools.bayesian import update_ensemble_weights
from energy_tsf.tools.fingerprints import json_fingerprint, source_fingerprint


class BacktestWorkflow:
    def __init__(self, workspace_root: str | Path, *, registry=None) -> None:
        self.workspace_root = Path(workspace_root)
        self.registry = registry

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.workspace_root / path

    def _request(self, config: BacktestConfig) -> BacktestRequest:
        return BacktestRequest(
            request_id=config.request.request_id,
            station_id=config.request.station_id,
            data_uri=config.data.path,
            target=config.request.target,
            timestamp_column=config.data.timestamp_column,
            frequency=config.data.frequency,
            timezone=config.request.timezone,
            horizon=config.request.horizon,
            stride=config.request.stride,
            prediction_mode=config.request.prediction_mode,
            covariates=config.data.covariates,
            publish=config.request.publish,
            epsilon=config.bayesian.epsilon,
            temperature=config.bayesian.temperature,
            prior_strength=config.bayesian.prior_strength,
        )

    @staticmethod
    def _write_json(path: Path, value) -> None:
        payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _event(self, memory_root: Path, work_dir: Path, request_id: str, event_type: str, payload: dict) -> None:
        event = LedgerEvent(
            event_id=json_fingerprint(
                {"request_id": request_id, "event_type": event_type, "payload": payload}
            ),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_id=request_id,
            payload=payload,
        )
        LedgerWriter(memory_root).append(event)
        with (work_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")

    def _load_input(self, config: BacktestConfig):
        request = self._request(config)
        backtest = clean_backtest_input(
            load_backtest_data(config),
            timezone=request.timezone,
            horizon=request.horizon,
        )
        work_dir = self._path(config.storage.work_root) / request.request_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return request, backtest, work_dir

    @staticmethod
    def _validate_solution(request: BacktestRequest, solution) -> None:
        compatibility = solution.compatibility
        if solution.station_id != request.station_id:
            raise ValueError("Active Solution belongs to another station")
        if compatibility.target != request.target:
            raise ValueError("Active Solution target does not match the backtest")
        if compatibility.frequency != request.frequency:
            raise ValueError("Active Solution frequency does not match the backtest")
        if compatibility.horizon != request.horizon:
            raise ValueError("Active Solution horizon does not match the backtest")
        if compatibility.prediction_mode != request.prediction_mode:
            raise ValueError("Active Solution prediction mode does not match the backtest")
        if compatibility.covariates != request.covariates:
            raise ValueError("Active Solution covariates do not match the backtest")
        if compatibility.timestamp_column is None:
            raise ValueError("backtest requires a timestamped Active Solution")

    def _station(self, memory_root: Path, request: BacktestRequest) -> StationSpec:
        try:
            station = StationRepository(memory_root).load(request.station_id)
            if station.target != request.target or station.frequency != request.frequency:
                raise ValueError("registered station profile is incompatible with the backtest")
            return station
        except FileNotFoundError:
            return StationSpec(
                station_id=request.station_id,
                energy_type="unknown",
                timezone=request.timezone,
                target=request.target,
                target_unit="unknown",
                frequency=request.frequency,
                timestamp_column=request.timestamp_column,
            )

    def inspect(self, config: BacktestConfig) -> dict:
        """Pin the Active parent and freeze leakage-safe rolling windows."""
        request, backtest, work_dir = self._load_input(config)
        memory_root = self._path(config.storage.memory_root)
        repository = MemoryRepository(memory_root)
        cache = ExecutionCache(memory_root)
        self._event(
            memory_root,
            work_dir,
            request.request_id,
            "backtest.input_ready",
            {
                "input_fingerprint": backtest.profile.input_fingerprint,
                "windows": backtest.profile.window_count,
            },
        )
        solution = cache.active_solution(
            repository, request.station_id, request.target, request.horizon
        )
        self._validate_solution(request, solution)
        for member in solution.members:
            repository.resolve_reference(member.artifact_ref)
            if member.provider in {"timesfm-2.5", "moirai-2", "chronos-2"}:
                checkpoint = Path(str(member.params.get("checkpoint_path", ""))).expanduser()
                if not checkpoint.exists():
                    raise FileNotFoundError(
                        f"foundation checkpoint not found for {member.trial_id}: {checkpoint}"
                    )
        snapshot = load_snapshot_by_id(memory_root, solution.data_snapshot_id)
        build_window_data(
            backtest,
            snapshot,
            solution,
            timezone_name=request.timezone,
        )
        plan = build_backtest_plan(request.request_id, request.station_id, solution, backtest)
        self._write_json(work_dir / "input_profile.json", backtest.profile)
        self._write_json(work_dir / "plan.json", plan)
        self._event(
            memory_root,
            work_dir,
            request.request_id,
            "backtest.asset_pinned",
            {
                "plan_id": plan.plan_id,
                "parent_solution_id": solution.solution_id,
                "windows": len(plan.windows),
            },
        )
        return {
            "request_id": request.request_id,
            "config_hash": backtest_config_fingerprint(config),
            "input": backtest.profile.model_dump(mode="json"),
            "parent_solution_id": solution.solution_id,
            "members": [member.trial_id for member in solution.members],
            "plan_path": str(work_dir / "plan.json"),
        }

    def run(self, config: BacktestConfig, plan_path: str | Path | None = None) -> BacktestResult:
        """Replay the pinned model, update weights, and apply the release gate."""
        request, backtest, work_dir = self._load_input(config)
        memory_root = self._path(config.storage.memory_root)
        repository = MemoryRepository(memory_root)
        cache = ExecutionCache(memory_root)
        selected_plan = Path(plan_path) if plan_path is not None else work_dir / "plan.json"
        if plan_path is not None and not selected_plan.is_file():
            raise FileNotFoundError(f"backtest plan not found: {selected_plan}")
        if not selected_plan.is_file():
            self.inspect(config)
        plan = BacktestPlan.model_validate_json(selected_plan.read_text(encoding="utf-8"))
        if backtest_plan_fingerprint(plan) != plan.plan_id:
            raise ValueError("backtest plan fingerprint does not match its contents")
        if plan.request_id != request.request_id or plan.station_id != request.station_id:
            raise ValueError("backtest plan does not belong to this request")
        if plan.input_fingerprint != backtest.profile.input_fingerprint:
            raise ValueError("backtest input changed after the plan was frozen")

        complete = work_dir / ".complete"
        result_path = work_dir / "result.json"
        if complete.is_file() and result_path.is_file():
            previous = BacktestResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            if previous.plan_id == plan.plan_id:
                return previous
            raise ValueError("request directory already contains a different completed backtest")

        solution = cache.get(request.station_id, request.target, request.horizon)
        if solution is None or solution.solution_id != plan.parent_solution_id:
            solution = repository.load_solution(plan.parent_solution_id)
        self._validate_solution(request, solution)
        if tuple(member.trial_id for member in solution.members) != plan.member_trial_ids:
            raise ValueError("backtest plan member set does not match its parent Solution")
        snapshot = load_snapshot_by_id(memory_root, plan.data_snapshot_id)
        windows = build_window_data(
            backtest,
            snapshot,
            solution,
            timezone_name=request.timezone,
        )
        station = self._station(memory_root, request)
        track = BacktestTrack(memory_root, registry=self.registry).run(solution, station, windows)
        member_ids = solution.ensemble.member_trial_ids
        metrics_before = backtest_metrics(track.predictions)
        window_metrics = metrics_before.loc[
            (metrics_before["scope"] == "window") & (metrics_before["kind"] == "member")
        ]
        raw_window_rmse = {
            window_id: dict(zip(group["trial_id"], group["rmse"]))
            for window_id, group in window_metrics.groupby("window_id", sort=False)
        }
        window_rmse = {
            window_id: values
            for window_id, values in raw_window_rmse.items()
            if set(values) == set(member_ids)
        }
        if not window_rmse:
            raise ValueError("no window has predictions from every Active ensemble member")
        self._event(
            memory_root,
            work_dir,
            request.request_id,
            "backtest.evaluated",
            {
                "plan_id": plan.plan_id,
                "completed_windows": len(window_rmse),
                "expected_windows": len(plan.windows),
                "failures": len(track.failures),
            },
        )
        maximum = capacity_in_target_unit(station)
        if maximum is not None:
            scale = maximum
            scale_source: Literal["capacity", "truth_q95"] = "capacity"
        else:
            truth = backtest.frame.loc[backtest.observed_mask, backtest.target].abs().to_numpy(dtype=float)
            scale = max(float(np.quantile(truth, 0.95)), 1e-12)
            scale_source = "truth_q95"
        update = update_ensemble_weights(
            member_ids,
            solution.ensemble.weights,
            window_rmse,
            scale=scale,
            scale_source=scale_source,
            epsilon=config.bayesian.epsilon,
            temperature=config.bayesian.temperature,
            prior_strength=config.bayesian.prior_strength,
        )
        after = recombine_member_predictions(track.predictions, member_ids, update.new_weights)
        predictions = pd.concat([track.predictions, after], ignore_index=True)
        metrics = backtest_metrics(predictions)
        overall = metrics.loc[metrics["scope"] == "overall"].set_index("trial_id")
        before_rmse = float(overall.loc["ensemble_before", "rmse"])
        after_rmse = float(overall.loc["ensemble_after", "rmse"])
        self._event(
            memory_root,
            work_dir,
            request.request_id,
            "backtest.weights_updated",
            {"plan_id": plan.plan_id, "before_rmse": before_rmse, "after_rmse": after_rmse},
        )

        new_solution_id = None
        candidate = active = None
        decision: ReleaseDecision | None = None
        if request.publish:
            evidence_payload = {
                "request_id": request.request_id,
                "plan_id": plan.plan_id,
                "input": backtest.profile.model_dump(mode="json"),
                "weight_update": update.model_dump(mode="json"),
                "before_rmse": before_rmse,
                "after_rmse": after_rmse,
            }
            evidence = repository.put_bytes(
                (json.dumps(evidence_payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            )
            old_order = {member: index for index, member in enumerate(solution.ensemble.fallback_order)}
            weights_by_id = dict(zip(member_ids, update.new_weights))
            fallback = tuple(sorted(member_ids, key=lambda member: (-weights_by_id[member], old_order[member])))
            ensemble = EnsembleSpec(
                method="bayesian_update",
                member_trial_ids=member_ids,
                weights=update.new_weights,
                fallback_order=fallback,
                renormalize_on_failure=solution.ensemble.renormalize_on_failure,
                min_valid_members=solution.ensemble.min_valid_members,
            )
            updated = build_solution(
                request_id=request.request_id,
                station_id=solution.station_id,
                data_snapshot_id=solution.data_snapshot_id,
                compatibility=solution.compatibility,
                members=solution.members,
                ensemble=ensemble,
                validation_report_ref=solution.validation_report_ref,
                test_report_ref=solution.test_report_ref,
                validation_predictions_ref=solution.validation_predictions_ref,
                test_predictions_ref=solution.test_predictions_ref,
                constraints=solution.constraints,
                code_fingerprint=source_fingerprint(Path(__file__).parents[1]),
                created_at=datetime.now(timezone.utc).isoformat(),
                search_pool_id=solution.search_pool_id,
                parent_solution_id=solution.solution_id,
                update_evidence_ref=evidence.uri,
            )
            repository.save_solution(updated)
            candidate = repository.create_candidate(updated)
            new_solution_id = updated.solution_id
            self._event(
                memory_root,
                work_dir,
                request.request_id,
                "backtest.candidate_created",
                {"parent_solution_id": solution.solution_id, "solution_id": updated.solution_id},
            )
            decision = evaluate_backtest_release(
                repository,
                candidate,
                parent_solution_id=solution.solution_id,
                completed_windows=len(window_rmse),
                expected_windows=len(plan.windows),
                before_rmse=before_rmse,
                after_rmse=after_rmse,
            )
            if decision.approved:
                active = promote_candidate(
                    repository,
                    candidate,
                    decision,
                    expected_active_solution_id=solution.solution_id,
                )
                cache.put(updated)

        result = write_backtest_reports(
            work_dir,
            request=request,
            plan=plan,
            profile=backtest.profile,
            predictions=predictions,
            metrics=metrics,
            update=update,
            config_hash=backtest_config_fingerprint(config),
            new_solution_id=new_solution_id,
            candidate=candidate,
            active=active,
            decision=decision,
            failures=track.failures,
        )
        self._event(
            memory_root,
            work_dir,
            request.request_id,
            (
                "backtest.released"
                if result.published
                else ("backtest.release_failed" if candidate is not None else "backtest.completed")
            ),
            {
                "plan_id": plan.plan_id,
                "parent_solution_id": solution.solution_id,
                "new_solution_id": new_solution_id,
                "published": result.published,
            },
        )
        complete.write_text(plan.plan_id + "\n", encoding="utf-8")
        return result
