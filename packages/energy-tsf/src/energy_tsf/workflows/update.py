"""Constrained AutoML retraining and atomic publication for an existing station."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from energy_tsf.assets.active import read_active
from energy_tsf.assets.promotion import promote_candidate
from energy_tsf.automl.finalize import build_validation_ensemble
from energy_tsf.automl.ranking import rank_trials
from energy_tsf.automl.update_planner import build_update_plan, update_plan_fingerprint
from energy_tsf.automl.window_ranking import build_search_pool
from energy_tsf.config.loader import update_config_fingerprint
from energy_tsf.config.schema import UpdateConfig
from energy_tsf.contracts.artifact import CompatibilitySpec, ReleaseDecision, SolutionMember
from energy_tsf.contracts.data import ForecastContextProfile
from energy_tsf.contracts.forecast import ForecastSourcePlan
from energy_tsf.contracts.modeling import ExperimentPlan
from energy_tsf.contracts.search_pool import SearchPoolArtifact, WindowTrialScore
from energy_tsf.contracts.station import StationSpec
from energy_tsf.contracts.update import UpdatePlan, UpdateRequest, UpdateResult
from energy_tsf.data import clean_update_series, create_snapshot, horizon_steps, load_snapshot_by_id, load_update_data
from energy_tsf.data.schemas import ForecastContext, PreparedHistory, PreparedSeries
from energy_tsf.data.snapshots import snapshot_fingerprint
from energy_tsf.gates.update_release import evaluate_update_release
from energy_tsf.ledger import LedgerEvent, LedgerWriter
from energy_tsf.memory import ExecutionCache, MemoryRepository, StationRepository, build_solution
from energy_tsf.providers import modeling_registry
from energy_tsf.reporting import write_update_reports
from energy_tsf.runtime.forecast_track import ForecastTrack
from energy_tsf.runtime.update_track import UpdateTrack
from energy_tsf.tools.fingerprints import json_fingerprint, source_fingerprint


class UpdateWorkflow:
    def __init__(self, workspace_root: str | Path, *, registry=None) -> None:
        self.workspace_root = Path(workspace_root)
        self.registry = registry or modeling_registry()

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.workspace_root / path

    @staticmethod
    def _write_json(path: Path, value) -> None:
        payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _event(self, memory_root: Path, work_dir: Path, request_id: str, event_type: str, payload: dict) -> None:
        event = LedgerEvent(
            event_id=json_fingerprint({"request_id": request_id, "event_type": event_type, "payload": payload}),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_id=request_id,
            payload=payload,
        )
        LedgerWriter(memory_root).append(event)
        with (work_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")

    def _request(self, config: UpdateConfig) -> UpdateRequest:
        return UpdateRequest(
            request_id=config.request.request_id,
            station_id=config.request.station_id,
            data_uri=config.data.path,
            target=config.request.target,
            timestamp_column=config.data.timestamp_column,
            frequency=config.data.frequency,
            timezone=config.request.timezone,
            horizon=config.request.horizon,
            stride=config.request.stride,
            validation_days=config.request.validation_days,
            prediction_mode=config.request.prediction_mode,
            covariates=config.data.covariates,
            top_k=config.request.top_k,
            max_parallel_trials=config.search.max_parallel_trials,
            publish=config.request.publish,
            seed=config.request.seed,
        )

    def _load_input(self, config: UpdateConfig):
        request = self._request(config)
        prepared, profile, validation_origins = clean_update_series(
            load_update_data(config),
            timezone=request.timezone,
            validation_days=request.validation_days,
            horizon=request.horizon,
        )
        work_dir = self._path(config.storage.work_root) / request.request_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return request, prepared, profile, validation_origins, work_dir

    @staticmethod
    def _validate_solution(request: UpdateRequest, solution) -> None:
        compatibility = solution.compatibility
        if solution.station_id != request.station_id:
            raise ValueError("Active Solution belongs to another station")
        if compatibility.target != request.target:
            raise ValueError("Active Solution target does not match the update")
        if compatibility.frequency != request.frequency:
            raise ValueError("Active Solution frequency does not match the update")
        if compatibility.horizon != request.horizon:
            raise ValueError("Active Solution horizon does not match the update")
        if compatibility.timestamp_column != request.timestamp_column:
            raise ValueError("Active Solution timestamp column does not match the update")
        if compatibility.prediction_mode != request.prediction_mode:
            raise ValueError("Active Solution prediction mode does not match the update")
        if compatibility.covariates != request.covariates:
            raise ValueError("Active Solution covariates do not match the update")
        if not solution.search_pool_id:
            raise ValueError("Active Solution has no search pool; rerun Modeling before Update")

    def _station(self, memory_root: Path, request: UpdateRequest) -> StationSpec:
        try:
            station = StationRepository(memory_root).load(request.station_id)
            if station.target != request.target or station.frequency != request.frequency:
                raise ValueError("registered station profile is incompatible with the update")
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

    def inspect(self, config: UpdateConfig) -> dict:
        """Pin Active and its SearchPool, then freeze the update trial plan."""
        request, prepared, profile, origins, work_dir = self._load_input(config)
        memory_root = self._path(config.storage.memory_root)
        repository = MemoryRepository(memory_root)
        active = read_active(memory_root, request.station_id, request.target, request.horizon)
        solution = repository.load_solution(active.solution_id)
        self._validate_solution(request, solution)
        pool_id = solution.search_pool_id
        if pool_id is None:
            raise ValueError("Active Solution has no search pool; rerun Modeling before Update")
        pool = repository.load_search_pool(pool_id)
        data_ref = create_snapshot(
            prepared,
            memory_root,
            lineage={
                "task_type": "update",
                "request_id": request.request_id,
                "config_hash": update_config_fingerprint(config),
                "validation_days": request.validation_days,
            },
        )
        repository.register_data_snapshot(data_ref)
        split_id = json_fingerprint(prepared.split.model_dump(mode="json"))
        plan = build_update_plan(
            request_id=request.request_id,
            station_id=request.station_id,
            parent_solution_id=solution.solution_id,
            pool=pool,
            data_snapshot_id=data_ref.snapshot_id,
            split_id=split_id,
            input_fingerprint=profile.input_fingerprint,
            train_end=prepared.split.train_end,
            validation_origins=origins,
            seed=request.seed,
            registry=self.registry,
            required_candidates=request.top_k,
        )
        self._write_json(work_dir / "input_profile.json", profile)
        self._write_json(work_dir / "pool_before.json", pool)
        self._write_json(work_dir / "plan.json", plan)
        self._event(
            memory_root,
            work_dir,
            request.request_id,
            "update.plan_frozen",
            {
                "plan_id": plan.plan_id,
                "parent_solution_id": solution.solution_id,
                "source_pool_id": pool.pool_id,
                "trials": len(plan.trials),
            },
        )
        return {
            "request_id": request.request_id,
            "config_hash": update_config_fingerprint(config),
            "input": profile.model_dump(mode="json"),
            "parent_solution_id": solution.solution_id,
            "source_pool_id": pool.pool_id,
            "candidate_trials": len(plan.trials),
            "unavailable": plan.unavailable,
            "plan_path": str(work_dir / "plan.json"),
        }

    @staticmethod
    def _context(series: PreparedSeries, origin: int, steps: int) -> ForecastContext:
        frame = series.frame.iloc[:origin].copy()
        mask = series.imputation_mask.iloc[:origin].copy()
        history = PreparedHistory(
            frame=frame,
            target=series.target,
            timestamp_column=series.timestamp_column,
            frequency=series.frequency,
            imputation_mask=mask,
            quality=series.quality,
            covariates=series.covariates,
        )
        if series.timestamp_column is None:
            future_index = pd.RangeIndex(1, steps + 1, name="horizon_step")
            start = end = future_start = future_end = None
        else:
            timestamps = series.frame[series.timestamp_column]
            future_index = pd.DatetimeIndex(timestamps.iloc[origin : origin + steps])
            if len(future_index) < steps:
                offset = pd.to_timedelta(86400 // steps, unit="s")
                future_index = pd.date_range(timestamps.iloc[-1] + offset, periods=steps, freq=offset)
            start = timestamps.iloc[0].isoformat()
            end = timestamps.iloc[origin - 1].isoformat()
            future_start = future_index[0].isoformat()
            future_end = future_index[-1].isoformat()
        future_covariates = None
        if series.covariates.known_future:
            future_covariates = series.frame.iloc[origin : origin + steps][
                [series.timestamp_column, *series.covariates.known_future]
            ].copy()
        profile = ForecastContextProfile(
            rows=len(frame),
            timestamped=series.timestamp_column is not None,
            start_time=start,
            end_time=end,
            future_start=future_start,
            future_end=future_end,
            horizon_steps=steps,
            original_missing=int(mask.sum()),
            imputed=int(mask.sum()),
            irregular_intervals=0,
            input_fingerprint=json_fingerprint(
                {
                    "origin": origin,
                    "target": series.target,
                    "history": frame.astype(str).to_dict(orient="list"),
                    "future_covariates": None
                    if future_covariates is None
                    else future_covariates.astype(str).to_dict(orient="list"),
                }
            ),
        )
        return ForecastContext(
            history=history,
            future_index=future_index,
            profile=profile,
            future_covariates=future_covariates,
        )

    def _parent_baseline(
        self,
        repository: MemoryRepository,
        solution,
        station: StationSpec,
        prepared: PreparedSeries,
        origins: tuple[int, ...],
        work_dir: Path,
    ) -> float:
        track = ForecastTrack(repository.root, registry=self.registry)
        source = ForecastSourcePlan(
            station_id=solution.station_id,
            solution_id=solution.solution_id,
            weight=1.0,
            scale_method="none",
            selection_reasons=("frozen parent update baseline",),
        )
        steps = horizon_steps(prepared.frequency)
        rows = []
        errors = []
        values = prepared.frame[prepared.target].to_numpy(dtype=float)
        observed = ~prepared.imputation_mask.to_numpy(dtype=bool)
        for origin in origins:
            result = track.execute_source(source, self._context(prepared, origin, steps), station)
            indices = np.arange(origin, origin + steps)
            include = observed[indices]
            if include.any():
                errors.extend((result.predictions[include] - values[indices][include]).tolist())
            for offset, prediction in enumerate(result.predictions):
                index = origin + offset
                rows.append(
                    {
                        "origin": origin,
                        "horizon_step": offset + 1,
                        "target_index": index,
                        "y_true": values[index] if observed[index] else np.nan,
                        "y_pred": prediction,
                        "observed": bool(observed[index]),
                    }
                )
        if not errors:
            raise ValueError("parent baseline has no observed validation truth")
        pd.DataFrame(rows).to_csv(work_dir / "parent_predictions.csv", index=False)
        return float(np.sqrt(np.mean(np.square(errors))))

    def _smoke(self, repository: MemoryRepository, solution, station: StationSpec, prepared: PreparedSeries) -> bool:
        try:
            source = ForecastSourcePlan(
                station_id=solution.station_id,
                solution_id=solution.solution_id,
                weight=1.0,
                scale_method="none",
                selection_reasons=("update release smoke",),
            )
            result = ForecastTrack(repository.root, registry=self.registry).execute_source(
                source,
                self._context(prepared, len(prepared.frame), horizon_steps(prepared.frequency)),
                station,
            )
            return len(result.predictions) == horizon_steps(prepared.frequency) and np.isfinite(result.predictions).all()
        except Exception:
            return False

    def run(self, config: UpdateConfig, plan_path: str | Path | None = None) -> UpdateResult:
        """Retrain pinned candidates and publish only when the update gate passes."""
        request, prepared, profile, validation_origins, work_dir = self._load_input(config)
        memory_root = self._path(config.storage.memory_root)
        repository = MemoryRepository(memory_root)
        selected_plan = Path(plan_path) if plan_path is not None else work_dir / "plan.json"
        if plan_path is not None and not selected_plan.is_file():
            raise FileNotFoundError(f"update plan not found: {selected_plan}")
        if not selected_plan.is_file():
            self.inspect(config)
        plan = UpdatePlan.model_validate_json(selected_plan.read_text(encoding="utf-8"))
        if update_plan_fingerprint(plan) != plan.plan_id:
            raise ValueError("update plan fingerprint does not match its contents")
        if plan.request_id != request.request_id or plan.station_id != request.station_id:
            raise ValueError("update plan does not belong to this request")
        if plan.input_fingerprint != profile.input_fingerprint:
            raise ValueError("update input changed after the plan was frozen")
        result_path = work_dir / "result.json"
        if (work_dir / ".complete").is_file() and result_path.is_file():
            previous = UpdateResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            if previous.plan_id == plan.plan_id:
                return previous
            raise ValueError("request directory already contains a different completed update")

        parent = repository.load_solution(plan.parent_solution_id)
        self._validate_solution(request, parent)
        if parent.search_pool_id != plan.source_pool_id:
            raise ValueError("update plan search pool does not match its frozen parent Solution")
        repository.load_search_pool(plan.source_pool_id)
        load_snapshot_by_id(memory_root, plan.data_snapshot_id)
        expected_snapshot_id = snapshot_fingerprint(
            prepared,
            {
                "task_type": "update",
                "request_id": request.request_id,
                "config_hash": update_config_fingerprint(config),
                "validation_days": request.validation_days,
            },
        )
        if expected_snapshot_id != plan.data_snapshot_id:
            raise ValueError("update snapshot does not match the frozen input")
        experiment = ExperimentPlan(
            plan_id=plan.plan_id,
            trials=plan.trials,
            unavailable=plan.unavailable,
            max_trials=len(plan.trials),
            seed=request.seed,
        )
        steps = horizon_steps(prepared.frequency)
        train_origins = tuple(range(1, max(1, plan.train_end - steps + 1), steps))
        update_track = UpdateTrack(self.registry)
        records = update_track.run_candidates(
            experiment,
            prepared,
            train_origins=train_origins,
            validation_origins=plan.validation_origins,
            train_end=plan.train_end,
            max_parallel_trials=config.search.max_parallel_trials,
            work_dir=work_dir,
        )
        for record in records:
            repository.save_trial(request.request_id, record)
        selection = rank_trials(records, top_k=request.top_k)
        next_pool: SearchPoolArtifact | None = None
        scores: tuple[WindowTrialScore, ...] = ()
        if selection.records:
            next_pool, scores = build_search_pool(
                records,
                prepared,
                request_id=request.request_id,
                station_id=request.station_id,
                horizon=request.horizon,
                top_k_per_window=request.top_k,
            )
            repository.save_search_pool(next_pool, scores)
            self._write_json(work_dir / "pool_after.json", next_pool)

        parent_rmse = updated_rmse = None
        new_solution_id = None
        candidate = active = None
        decision: ReleaseDecision | None = None
        if selection.status == "complete":
            ensemble, ensemble_metrics = build_validation_ensemble(selection, prepared)
            rmse_value = ensemble_metrics.get("rmse")
            if rmse_value is None:
                raise ValueError("updated ensemble has no finite validation RMSE")
            updated_rmse = float(rmse_value)
            if next_pool is None:
                raise RuntimeError("complete Top-K selection did not produce a search pool")
            station = self._station(memory_root, request)
            parent_rmse = self._parent_baseline(
                repository, parent, station, prepared, plan.validation_origins, work_dir
            )
            retrain_origins = tuple(sorted(set((*train_origins, *validation_origins))))
            finalized = update_track.finalize(
                selection,
                prepared,
                retrain_origins=retrain_origins,
                work_dir=work_dir / "final" / plan.plan_id,
            )
            selected_by_id = {record.spec.trial_id: record for record in selection.records}
            members = []
            for member in finalized:
                if Path(member.artifact_ref).is_file():
                    artifact_ref = repository.put_file(member.artifact_ref).uri
                else:
                    runtime_payload = {
                        "provider": member.spec.provider,
                        "repo_id": member.spec.params.get("repo_id"),
                        "revision": member.spec.params.get("revision"),
                        "checkpoint_path": member.spec.params.get("checkpoint_path"),
                        "context_length": member.spec.params.get("context_length"),
                    }
                    ref = repository.put_bytes(
                        (json.dumps(runtime_payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                    )
                    artifact_ref = f"runtime-ref://{ref.sha256}"
                record = selected_by_id[member.spec.trial_id]
                members.append(
                    SolutionMember(
                        trial_id=member.spec.trial_id,
                        provider=member.spec.provider,
                        artifact_ref=artifact_ref,
                        params=member.spec.params,
                        input_mode=member.spec.input_mode,
                        covariate_spec=member.spec.covariate_spec,
                        feature_spec=member.spec.feature_spec,
                        feature_selection=member.spec.feature_selection,
                        runtime=member.runtime,
                        validation_metrics=record.validation_metrics,
                    )
                )
            frames = []
            for record in selection.records:
                frame = pd.read_csv(record.prediction_ref)
                frame.insert(0, "trial_id", record.spec.trial_id)
                frames.append(frame)
            validation_predictions = work_dir / "validation_predictions.csv"
            pd.concat(frames, ignore_index=True).to_csv(validation_predictions, index=False)
            validation_report = work_dir / "validation_report.json"
            self._write_json(
                validation_report,
                {
                    "plan_id": plan.plan_id,
                    "top_k": [record.spec.trial_id for record in selection.records],
                    "ensemble": ensemble.model_dump(mode="json"),
                    "ensemble_metrics": ensemble_metrics,
                },
            )
            evidence_payload = {
                "request_id": request.request_id,
                "plan": plan.model_dump(mode="json"),
                "input": profile.model_dump(mode="json"),
                "parent_rmse": parent_rmse,
                "updated_rmse": updated_rmse,
                "source_pool_id": plan.source_pool_id,
                "next_pool_id": next_pool.pool_id,
            }
            evidence = repository.put_bytes(
                (json.dumps(evidence_payload, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode("utf-8")
            )
            solution = build_solution(
                request_id=request.request_id,
                station_id=request.station_id,
                data_snapshot_id=plan.data_snapshot_id,
                compatibility=CompatibilitySpec(
                    target=request.target,
                    frequency=request.frequency,
                    horizon=request.horizon,
                    timestamp_column=request.timestamp_column,
                    prediction_mode=request.prediction_mode,
                    covariates=request.covariates,
                ),
                members=tuple(members),
                ensemble=ensemble,
                validation_report_ref=repository.put_file(validation_report).uri,
                validation_predictions_ref=repository.put_file(validation_predictions).uri,
                constraints=parent.constraints,
                code_fingerprint=source_fingerprint(Path(__file__).parents[1]),
                created_at=datetime.now(timezone.utc).isoformat(),
                search_pool_id=next_pool.pool_id,
                parent_solution_id=parent.solution_id,
                update_evidence_ref=evidence.uri,
            )
            repository.save_solution(solution)
            new_solution_id = solution.solution_id
            smoke_ok = self._smoke(repository, solution, station, prepared)
            if request.publish:
                candidate = repository.create_candidate(solution)
                decision = evaluate_update_release(
                    repository,
                    candidate,
                    parent_solution_id=parent.solution_id,
                    parent_rmse=parent_rmse,
                    updated_rmse=updated_rmse,
                    smoke_ok=smoke_ok,
                )
                if decision.approved:
                    active = promote_candidate(
                        repository,
                        candidate,
                        decision,
                        expected_active_solution_id=parent.solution_id,
                    )
                    ExecutionCache(memory_root).invalidate(
                        request.station_id, request.target, request.horizon
                    )

        result = write_update_reports(
            work_dir,
            profile=profile,
            plan=plan,
            records=records,
            selection=selection,
            next_pool=next_pool,
            window_scores=scores,
            parent_rmse=parent_rmse,
            updated_rmse=updated_rmse,
            new_solution_id=new_solution_id,
            candidate=candidate,
            active=active,
            decision=decision,
        )
        self._event(
            memory_root,
            work_dir,
            request.request_id,
            f"update.{result.status}",
            {
                "plan_id": plan.plan_id,
                "parent_solution_id": parent.solution_id,
                "new_solution_id": new_solution_id,
                "published": result.published,
            },
        )
        (work_dir / ".complete").write_text(plan.plan_id + "\n", encoding="utf-8")
        return result
