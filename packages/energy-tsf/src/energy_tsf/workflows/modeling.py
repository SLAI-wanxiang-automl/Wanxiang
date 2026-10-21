"""End-to-end deterministic modeling workflow from config to Candidate/Active."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from energy_tsf.automl.finalize import build_validation_ensemble, final_evaluate
from energy_tsf.automl.planner import PlanningInput, build_experiment_plan
from energy_tsf.automl.ranking import rank_trials
from energy_tsf.automl.runner import TrialRunner
from energy_tsf.automl.window_ranking import build_search_pool
from energy_tsf.automl.search_space import load_default_search_spaces
from energy_tsf.assets.promotion import promote_candidate
from energy_tsf.config.loader import config_fingerprint
from energy_tsf.config.schema import ModelingConfig
from energy_tsf.contracts.artifact import CompatibilitySpec, SolutionMember
from energy_tsf.contracts.result import ModelingWorkflowResult
from energy_tsf.contracts.search_pool import SearchPoolArtifact, WindowTrialScore
from energy_tsf.data import build_origin_sets, create_snapshot, load_modeling_data, prepare_series
from energy_tsf.gates.release import evaluate_release
from energy_tsf.ledger import LedgerEvent, LedgerWriter
from energy_tsf.memory import ExecutionCache, MemoryRepository, build_solution
from energy_tsf.providers import modeling_registry
from energy_tsf.providers.base import TrainableProvider
from energy_tsf.reporting import write_final_modeling_report, write_search_reports
from energy_tsf.tools.fingerprints import json_fingerprint, source_fingerprint
from energy_tsf.tools.constraints import constrain_forecast
from energy_tsf.workflows.state import ModelingState, transition


class ModelingWorkflow:
    def __init__(self, workspace_root: str | Path, *, registry=None, spaces=None) -> None:
        self.workspace_root = Path(workspace_root)
        self.registry = registry or modeling_registry()
        self.spaces = spaces or load_default_search_spaces()

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.workspace_root / path

    def run(self, config: ModelingConfig) -> ModelingWorkflowResult:
        """Run data preparation, bounded search, final evaluation, and release."""
        request_id = config.request.request_id
        work_dir = self._path(config.storage.work_root) / request_id
        memory = MemoryRepository(self._path(config.storage.memory_root))
        ledger = LedgerWriter(memory.root)
        state = ModelingState.CREATED

        def record_state(new_state: ModelingState, **payload) -> None:
            nonlocal state
            state = transition(state, new_state)
            timestamp = datetime.now(timezone.utc).isoformat()
            event_id = json_fingerprint({"task": request_id, "state": state, **payload})
            ledger.append(
                LedgerEvent(
                    event_id=event_id,
                    event_type="modeling.state",
                    timestamp=timestamp,
                    task_id=request_id,
                    payload={"status": state, **payload},
                )
            )

        raw = load_modeling_data(config)
        prepared = prepare_series(raw)
        origins = build_origin_sets(prepared.split, prepared.frequency)
        if not origins.train or not origins.validation or not origins.test:
            raise ValueError("each split must contain at least one complete one-day forecast origin")
        config_hash = config_fingerprint(config)
        data_ref = create_snapshot(prepared, memory.root, lineage={"config_hash": config_hash})
        memory.register_data_snapshot(data_ref)
        record_state(ModelingState.DATA_PREPARED, snapshot_id=data_ref.snapshot_id)

        split_id = json_fingerprint(prepared.split.model_dump(mode="json"))
        planning = PlanningInput(
            data_snapshot_id=data_ref.snapshot_id,
            split_id=split_id,
            train_end=prepared.split.train_end,
            validation_end=prepared.split.validation_end,
            frequency=prepared.frequency,
            providers=config.search.providers.enabled,
            max_trials=config.search.max_trials,
            seed=config.search.seed,
            foundation_checkpoints=config.search.providers.checkpoints,
            prediction_mode=config.request.prediction_mode,
            covariates=config.data.covariates,
            feature_options=config.features.model_dump(
                exclude={"statistical_selection", "tsfm_wrapper"}
            ),
            statistical_selection=config.features.statistical_selection,
            wrapper_selection=config.features.tsfm_wrapper,
        )
        plan = build_experiment_plan(planning, self.spaces, self.registry)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "plan.json").write_text(
            json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        record_state(ModelingState.PLANNED, plan_id=plan.plan_id, trials=len(plan.trials))

        runner = TrialRunner(
            self.registry,
            prepared,
            train_origins=origins.train,
            validation_origins=origins.validation,
            train_end=prepared.split.train_end,
            work_root=work_dir,
        )
        records = runner.run_plan(plan, max_parallel_trials=config.search.max_parallel_trials)
        for trial_record in records:
            memory.save_trial(request_id, trial_record)
        record_state(
            ModelingState.TRIALS_COMPLETED,
            succeeded=sum(item.status == "succeeded" for item in records),
            failed=sum(item.status != "succeeded" for item in records),
        )

        selection = rank_trials(records, top_k=config.request.top_k)
        search_pool: SearchPoolArtifact | None = None
        window_scores: tuple[WindowTrialScore, ...] = ()
        if selection.records:
            search_pool, window_scores = build_search_pool(
                records,
                prepared,
                request_id=request_id,
                station_id=config.request.station_id,
                horizon=config.request.horizon,
                top_k_per_window=config.request.top_k,
            )
            memory.save_search_pool(search_pool, window_scores)
        reports = write_search_reports(
            plan,
            records,
            selection,
            work_dir / "reports",
            search_pool=search_pool,
            window_scores=window_scores,
        )
        if selection.status != "complete":
            record_state(ModelingState.INSUFFICIENT_CANDIDATES)
            return ModelingWorkflowResult(
                request_id=request_id,
                status=state,
                plan_id=plan.plan_id,
                planned_trials=len(plan.trials),
                succeeded_trials=sum(item.status == "succeeded" for item in records),
                failed_trials=sum(item.status != "succeeded" for item in records),
                unavailable_providers=plan.unavailable,
                top_k=selection,
                search_pool_id=None if search_pool is None else search_pool.pool_id,
                report_path=str(reports["report_md"]),
            )
        if search_pool is None:
            raise RuntimeError("complete Top-K selection did not produce a search pool")
        record_state(ModelingState.TOPK_FROZEN, top_k=[item.spec.trial_id for item in selection.records])

        ensemble, ensemble_validation_metrics = build_validation_ensemble(selection, prepared)
        final = final_evaluate(
            selection,
            ensemble,
            prepared,
            self.registry,
            retrain_origins=tuple(sorted(set((*origins.train, *origins.validation)))),
            test_origins=origins.test,
            train_end=prepared.split.validation_end,
            work_dir=work_dir / "final" / plan.plan_id,
        )
        record_state(ModelingState.FINAL_EVALUATED)

        validation_report = json.loads(reports["validation_report"].read_text(encoding="utf-8"))
        validation_report["ensemble"] = {
            "spec": ensemble.model_dump(mode="json"),
            "metrics": ensemble_validation_metrics,
        }
        reports["validation_report"].write_text(
            json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        test_frames = []
        for member in final.members:
            frame = pd.read_csv(member.prediction_ref)
            frame.insert(0, "trial_id", member.trial_id)
            test_frames.append(frame)
        ensemble_frame = pd.read_csv(final.ensemble_prediction_ref)
        ensemble_frame.insert(0, "trial_id", "ensemble")
        test_frames.append(ensemble_frame)
        all_test_path = work_dir / "reports" / "test_predictions.csv"
        pd.concat(test_frames, ignore_index=True).to_csv(all_test_path, index=False)
        write_final_modeling_report(
            reports["report_md"],
            prepared=prepared,
            origins=origins,
            plan=plan,
            records=records,
            selection=selection,
            ensemble=ensemble,
            ensemble_validation_metrics=ensemble_validation_metrics,
            final=final,
            search_pool=search_pool,
        )

        validation_report_ref = memory.put_file(reports["validation_report"]).uri
        test_report_ref = memory.put_file(final.report_ref).uri
        validation_predictions_ref = memory.put_file(reports["validation_predictions"]).uri
        test_predictions_ref = memory.put_file(all_test_path).uri
        selected_by_id = {record.spec.trial_id: record for record in selection.records}
        members = []
        for result in final.members:
            selected = selected_by_id[result.trial_id]
            if Path(result.artifact_ref).is_file():
                artifact_ref = memory.put_file(result.artifact_ref).uri
            else:
                runtime_payload = {
                    "provider": result.provider,
                    "repo_id": selected.spec.params.get("repo_id"),
                    "revision": selected.spec.params.get("revision"),
                    "checkpoint_path": selected.spec.params.get("checkpoint_path"),
                    "context_length": selected.spec.params.get("context_length"),
                }
                runtime_object = memory.put_bytes(
                    (json.dumps(runtime_payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                )
                artifact_ref = f"runtime-ref://{runtime_object.sha256}"
            member_prediction_ref = memory.put_file(result.prediction_ref).uri
            members.append(
                SolutionMember(
                    trial_id=result.trial_id,
                    provider=result.provider,
                    artifact_ref=artifact_ref,
                    params=selected.spec.params,
                    input_mode=selected.spec.input_mode,
                    covariate_spec=selected.spec.covariate_spec,
                    feature_spec=selected.spec.feature_spec,
                    feature_selection=selected.spec.feature_selection,
                    runtime=result.runtime,
                    validation_metrics=selected.validation_metrics,
                    test_metrics=result.test_metrics,
                    test_prediction_ref=member_prediction_ref,
                )
            )

        code_hash = source_fingerprint(Path(__file__).parents[1])
        solution = build_solution(
            request_id=request_id,
            station_id=config.request.station_id,
            data_snapshot_id=data_ref.snapshot_id,
            compatibility=CompatibilitySpec(
                target=config.request.target,
                frequency=config.data.frequency,
                horizon=config.request.horizon,
                timestamp_column=config.data.timestamp_column,
                prediction_mode=config.request.prediction_mode,
                covariates=config.data.covariates,
            ),
            members=tuple(members),
            ensemble=ensemble,
            validation_report_ref=validation_report_ref,
            test_report_ref=test_report_ref,
            validation_predictions_ref=validation_predictions_ref,
            test_predictions_ref=test_predictions_ref,
            constraints={"non_negative": True, "minimum": 0.0},
            code_fingerprint=code_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            search_pool_id=search_pool.pool_id,
        )
        memory.save_solution(
            solution,
            {
                "leaderboard.csv": reports["leaderboard"],
                "top_k.json": reports["top_k"],
                "report.md": reports["report_md"],
                "window_leaderboard.csv": reports["window_leaderboard"],
                "search_pool.json": reports["search_pool"],
            },
        )
        candidate = memory.create_candidate(solution)
        record_state(ModelingState.CANDIDATE, solution_id=solution.solution_id)

        def smoke(member: SolutionMember) -> np.ndarray:
            selected = selected_by_id[member.trial_id]
            provider = self.registry.create(member.provider)
            model = None
            if isinstance(provider, TrainableProvider):
                model = provider.load(memory.resolve_reference(member.artifact_ref))
            table = constrain_forecast(provider.predict(
                model,
                prepared,
                (origins.test[0],),
                origins.horizon_steps,
                params=selected.spec.params,
                feature_spec=selected.spec.feature_spec,
                **(
                    {"covariate_spec": selected.spec.covariate_spec}
                    if selected.spec.covariate_spec.multivariate
                    else {}
                ),
            ))
            return table.predictions

        decision = evaluate_release(memory, candidate, smoke)
        active = None
        if config.request.publish:
            if decision.approved:
                active = promote_candidate(memory, candidate, decision)
                ExecutionCache(memory.root).put(solution)
                record_state(ModelingState.RELEASED, solution_id=solution.solution_id)
            else:
                record_state(ModelingState.RELEASE_FAILED, reasons=list(decision.reasons))

        return ModelingWorkflowResult(
            request_id=request_id,
            status=state,
            plan_id=plan.plan_id,
            planned_trials=len(plan.trials),
            succeeded_trials=sum(item.status == "succeeded" for item in records),
            failed_trials=sum(item.status != "succeeded" for item in records),
            unavailable_providers=plan.unavailable,
            top_k=selection,
            search_pool_id=search_pool.pool_id,
            solution_id=solution.solution_id,
            candidate=candidate,
            active=active,
            report_path=str(reports["report_md"]),
            release_reasons=decision.reasons,
        )
