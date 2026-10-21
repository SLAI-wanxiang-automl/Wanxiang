"""Deterministic forecast workflow from recent history to delivered report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from energy_tsf.config.loader import forecast_config_fingerprint
from energy_tsf.config.schema import ForecastConfig
from energy_tsf.contracts.forecast import (
    ForecastAssetInspection,
    ForecastPlan,
    ForecastRequest,
    ForecastResult,
    ForecastSelection,
    ForecastSourcePlan,
)
from energy_tsf.contracts.station import capacity_in_target_unit
from energy_tsf.data import clean_forecast_context, load_forecast_data
from energy_tsf.gates.forecast_precheck import forecast_precheck
from energy_tsf.ledger import LedgerEvent, LedgerWriter
from energy_tsf.memory import ExecutionCache, ForecastAssetRetriever, build_forecast_plan, validate_plan_candidates
from energy_tsf.reporting import write_forecast_reports
from energy_tsf.runtime.forecast_track import ForecastTrack
from energy_tsf.tools.constraints import constrain_forecast_with_summary
from energy_tsf.tools.fingerprints import json_fingerprint


class ForecastWorkflow:
    def __init__(self, workspace_root: str | Path, *, registry=None) -> None:
        self.workspace_root = Path(workspace_root)
        self.registry = registry

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.workspace_root / path

    def _request(self, config: ForecastConfig) -> ForecastRequest:
        return ForecastRequest(
            request_id=config.request.request_id,
            station=config.station,
            data_uri=config.data.path,
            future_covariates_uri=config.data.future_covariates_path,
            horizon=config.request.horizon,
            prediction_mode=config.request.prediction_mode,
            covariates=config.data.covariates,
            issue_time=config.request.issue_time,
            allow_similar=config.retrieval.allow_similar,
            allow_similar_on_direct_failure=config.retrieval.allow_similar_on_direct_failure,
            max_source_solutions=config.retrieval.max_source_solutions,
            deadline_seconds=config.retrieval.deadline_seconds,
        )

    def _prepare(self, config: ForecastConfig):
        request = self._request(config)
        raw = load_forecast_data(config)
        future_covariates = None
        if config.data.future_covariates_path is not None:
            from energy_tsf.data import load_future_covariates

            timestamp_column = request.station.timestamp_column
            if timestamp_column is None:
                raise ValueError("future covariates require a timestamp column")
            future_covariates = load_future_covariates(
                config.data.future_covariates_path,
                timestamp_column,
                request.covariates,
            )
        context = clean_forecast_context(
            raw,
            timezone=request.station.timezone,
            horizon=request.horizon,
            future_covariates=future_covariates,
        )
        work_dir = self._path(config.storage.work_root) / request.request_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return request, context, work_dir

    @staticmethod
    def _write_json(path: Path, value) -> None:
        payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _event(self, memory_root: Path, work_dir: Path, request_id: str, event_type: str, payload: dict) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        event = LedgerEvent(
            event_id=json_fingerprint(
                {"request_id": request_id, "event_type": event_type, "payload": payload}
            ),
            event_type=event_type,
            timestamp=timestamp,
            task_id=request_id,
            payload=payload,
        )
        LedgerWriter(memory_root).append(event)
        with (work_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")

    def inspect(self, config: ForecastConfig) -> dict:
        """Prepare input and inspect compatible assets without running inference."""
        request, context, work_dir = self._prepare(config)
        memory_root = self._path(config.storage.memory_root)
        reference_time = request.issue_time
        if reference_time is None and context.profile.end_time is not None:
            reference_time = datetime.fromisoformat(context.profile.end_time)
        inspection = ForecastAssetRetriever(memory_root).inspect(
            request,
            context.profile.rows,
            reference_time=reference_time,
        )
        self._write_json(work_dir / "context_profile.json", context.profile)
        self._write_json(work_dir / "candidates.json", inspection)
        plan = None
        if inspection.mode == "direct" and inspection.direct is not None:
            source = ForecastSourcePlan(
                station_id=inspection.direct.station_id,
                solution_id=inspection.direct.solution_id,
                weight=1.0,
                scale_method="none",
                selection_reasons=("exact station Active",),
            )
            plan = build_forecast_plan(
                request=request,
                input_fingerprint=context.profile.input_fingerprint,
                mode="direct",
                sources=(source,),
            )
            self._write_json(work_dir / "plan.json", plan)
        self._event(
            memory_root,
            work_dir,
            request.request_id,
            "forecast.inspected",
            {
                "mode": inspection.mode,
                "input_fingerprint": context.profile.input_fingerprint,
                "compatible_candidates": len(inspection.candidates),
                "rejected_candidates": len(inspection.rejected),
                "plan_id": None if plan is None else plan.plan_id,
            },
        )
        return {
            "request_id": request.request_id,
            "config_hash": forecast_config_fingerprint(config),
            "context": context.profile.model_dump(mode="json"),
            "inspection": inspection.model_dump(mode="json"),
            "plan_path": None if plan is None else str(work_dir / "plan.json"),
            "candidates_path": str(work_dir / "candidates.json"),
        }

    def freeze_plan(self, config: ForecastConfig, selection_path: str | Path) -> ForecastPlan:
        """Validate an Agent source selection and persist the immutable plan."""
        request, context, work_dir = self._prepare(config)
        memory_root = self._path(config.storage.memory_root)
        candidates_path = work_dir / "candidates.json"
        if not candidates_path.is_file():
            self.inspect(config)
        inspection = ForecastAssetInspection.model_validate_json(
            candidates_path.read_text(encoding="utf-8")
        )
        if inspection.mode != "similar":
            raise ValueError("an explicit selection is only needed for similar-station inspection")
        selection = ForecastSelection.model_validate_json(Path(selection_path).read_text(encoding="utf-8"))
        if len(selection.sources) > request.max_source_solutions:
            raise ValueError("selection exceeds max_source_solutions")
        plan = build_forecast_plan(
            request=request,
            input_fingerprint=context.profile.input_fingerprint,
            mode="similar",
            sources=selection.sources,
            renormalize_on_source_failure=selection.renormalize_on_source_failure,
        )
        validate_plan_candidates(
            plan,
            request,
            context.profile.input_fingerprint,
            inspection.candidates,
        )
        self._write_json(work_dir / "selection.json", selection)
        self._write_json(work_dir / "plan.json", plan)
        self._event(
            memory_root,
            work_dir,
            request.request_id,
            "forecast.plan_frozen",
            {"plan_id": plan.plan_id, "sources": [item.solution_id for item in plan.sources]},
        )
        return plan

    def run(self, config: ForecastConfig, plan_path: str | Path | None = None) -> ForecastResult:
        """Run only the frozen plan, persist predictions, and write the report."""
        request, context, work_dir = self._prepare(config)
        memory_root = self._path(config.storage.memory_root)
        candidates_path = work_dir / "candidates.json"
        if not candidates_path.is_file():
            self.inspect(config)
        selected_plan = Path(plan_path) if plan_path is not None else work_dir / "plan.json"
        if not selected_plan.is_file():
            raise ValueError("forecast plan is missing; review similar candidates and freeze a plan")
        inspection = ForecastAssetInspection.model_validate_json(
            candidates_path.read_text(encoding="utf-8")
        )
        plan = ForecastPlan.model_validate_json(selected_plan.read_text(encoding="utf-8"))
        candidates = (
            (inspection.direct,)
            if inspection.direct is not None
            else inspection.candidates
        )
        preflight = forecast_precheck(request, plan, context, candidates)
        if not preflight.valid:
            raise ValueError(
                "forecast precheck failed: "
                + "; ".join(issue.message for issue in preflight.errors)
            )

        complete = work_dir / ".complete"
        result_path = work_dir / "result.json"
        if complete.is_file() and result_path.is_file():
            previous = ForecastResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            if previous.plan_id == plan.plan_id and previous.input_fingerprint == context.profile.input_fingerprint:
                return previous
            raise ValueError("request directory already contains a different completed forecast")

        track = ForecastTrack(memory_root, registry=self.registry).run(plan, context, request.station)
        maximum = capacity_in_target_unit(request.station)
        constrained, summary = constrain_forecast_with_summary(
            track.table,
            minimum=0.0,
            maximum=maximum,
        )
        result = write_forecast_reports(
            work_dir,
            request=request,
            plan=plan,
            context=context,
            track=track,
            constrained_table=constrained,
            constraints=summary,
            config_hash=forecast_config_fingerprint(config),
        )
        self._event(
            memory_root,
            work_dir,
            request.request_id,
            "forecast.delivered",
            {
                "plan_id": plan.plan_id,
                "forecast_path": result.forecast_path,
                "degraded": result.degraded,
            },
        )
        ExecutionCache(memory_root).touch(
            request.station.station_id,
            request.station.target,
            request.horizon,
            issue_time=(
                request.issue_time.isoformat()
                if request.issue_time is not None
                else context.profile.end_time
            ),
            input_fingerprint=context.profile.input_fingerprint,
        )
        complete.write_text(plan.plan_id + "\n", encoding="utf-8")
        return result
