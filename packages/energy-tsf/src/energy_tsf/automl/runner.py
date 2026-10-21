"""Local bounded trial runner with atomic results and failure isolation."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from energy_tsf.automl.trial import validation_truth
from energy_tsf.contracts.modeling import ExperimentPlan, TrialRecord, TrialSpec
from energy_tsf.data.schemas import PreparedSeries
from energy_tsf.data.splits import horizon_steps
from energy_tsf.providers.base import TrainableProvider
from energy_tsf.providers.registry import ProviderRegistry
from energy_tsf.tools.constraints import constrain_forecast
from energy_tsf.tools.feature_selection import select_statistical_features, select_wrapper_covariates

_EXPECTED_TRIAL_ERRORS = (ValueError, FileNotFoundError, ImportError, RuntimeError, OSError, MemoryError)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_record(path: Path, record: TrialRecord) -> None:
    _atomic_text(path, json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


class TrialRunner:
    def __init__(
        self,
        registry: ProviderRegistry,
        series: PreparedSeries,
        *,
        train_origins: tuple[int, ...],
        validation_origins: tuple[int, ...],
        train_end: int,
        work_root: str | os.PathLike[str],
        timeout_seconds: float | None = None,
    ) -> None:
        self.registry = registry
        self.series = series
        self.train_origins = train_origins
        self.validation_origins = validation_origins
        self.train_end = train_end
        self.work_root = Path(work_root)
        self.timeout_seconds = timeout_seconds
        self._selection_lock = threading.Lock()
        self._statistical_cache: dict[str, tuple] = {}
        self._wrapper_cache: dict[str, tuple] = {}

    def _effective_spec(self, spec: TrialSpec, provider, steps: int) -> TrialSpec:
        if spec.feature_selection.method == "statistical":
            key = json.dumps(
                {
                    "feature": spec.feature_spec.model_dump(mode="json") if spec.feature_spec else None,
                    "selection": spec.statistical_selection.model_dump(mode="json"),
                },
                sort_keys=True,
            )
            with self._selection_lock:
                cached = self._statistical_cache.get(key)
                if cached is None:
                    if spec.feature_spec is None:
                        raise ValueError("statistical selection requires a FeatureSpec")
                    origins = tuple(
                        origin
                        for origin in self.train_origins
                        if origin >= spec.feature_spec.required_history
                    )
                    cached = select_statistical_features(
                        self.series,
                        origins,
                        train_end=self.train_end,
                        feature_spec=spec.feature_spec,
                        selection=spec.statistical_selection,
                        seed=spec.seed,
                    )
                    self._statistical_cache[key] = cached
            feature_spec, evidence = cached
            return spec.model_copy(
                update={"feature_spec": feature_spec, "feature_selection": evidence}
            )

        if spec.feature_selection.method == "wrapper":
            key = spec.provider
            with self._selection_lock:
                cached = self._wrapper_cache.get(key)
                if cached is None:
                    origins = self.validation_origins[-spec.wrapper_selection.selection_windows :]

                    def evaluate(covariates):
                        table = constrain_forecast(
                            provider.predict(
                                None,
                                self.series,
                                origins,
                                steps,
                                params=spec.params,
                                covariate_spec=covariates,
                            )
                        )
                        metrics, _, _ = validation_truth(
                            self.series, table, self.train_end, steps
                        )
                        return float(metrics["rmse"])

                    allowed = (
                        spec.covariate_spec.known_future
                        if spec.provider == "timesfm-2.5"
                        else spec.covariate_spec.columns
                    )
                    cached = select_wrapper_covariates(
                        spec.covariate_spec,
                        evaluate,
                        spec.wrapper_selection,
                        allowed=allowed,
                    )
                    self._wrapper_cache[key] = cached
            covariates, evidence = cached
            return spec.model_copy(
                update={
                    "input_mode": "multivariate" if covariates.multivariate else "univariate",
                    "covariate_spec": covariates,
                    "feature_selection": evidence,
                }
            )
        return spec

    def _cancelled(self, spec: TrialSpec, trial_dir: Path) -> TrialRecord:
        record = TrialRecord(spec=spec, status="cancelled", error="cancelled before execution")
        _write_record(trial_dir / "result.json", record)
        return record

    def run_trial(self, spec: TrialSpec, cancellation: threading.Event | None = None) -> TrialRecord:
        """Fit one planned candidate and persist its validation outputs."""
        trial_dir = self.work_root / "trials" / spec.trial_id
        result_path = trial_dir / "result.json"
        if result_path.is_file():
            return TrialRecord.model_validate_json(result_path.read_text(encoding="utf-8"))
        trial_dir.mkdir(parents=True, exist_ok=True)
        if cancellation is not None and cancellation.is_set():
            return self._cancelled(spec, trial_dir)
        _atomic_text(trial_dir / "state.json", '{"status":"running"}\n')

        provider = self.registry.create(spec.provider)
        steps = horizon_steps(self.series.frequency)
        fit_duration = 0.0
        predict_duration = 0.0
        started = time.perf_counter()
        try:
            spec = self._effective_spec(spec, provider, steps)
            model = None
            artifact_ref: str
            if isinstance(provider, TrainableProvider):
                train_origins = self.train_origins
                if spec.feature_spec is not None:
                    train_origins = tuple(
                        origin for origin in train_origins if origin >= spec.feature_spec.required_history
                    )
                fit_started = time.perf_counter()
                model = provider.fit(
                    self.series,
                    train_origins,
                    train_end=self.train_end,
                    params=spec.params,
                    feature_spec=spec.feature_spec,
                )
                fit_duration = time.perf_counter() - fit_started
                artifact = provider.save(model, trial_dir / "artifact.joblib")
                provider.load(artifact)
                artifact_ref = str(artifact)
            else:
                artifact_ref = str(spec.params.get("checkpoint_path", "runtime-reference"))

            predict_started = time.perf_counter()
            table = constrain_forecast(provider.predict(
                model,
                self.series,
                self.validation_origins,
                steps,
                params=spec.params,
                feature_spec=spec.feature_spec,
                **(
                    {"covariate_spec": spec.covariate_spec}
                    if spec.covariate_spec.multivariate
                    else {}
                ),
            ))
            table.require_coverage(self.validation_origins, steps)
            predict_duration = time.perf_counter() - predict_started
            prediction_path = trial_dir / "validation_predictions.csv"
            temporary_prediction = trial_dir / ".validation_predictions.csv.tmp"
            table.frame.to_csv(temporary_prediction, index=False)
            os.replace(temporary_prediction, prediction_path)
            metrics, valid_count, truth_hash = validation_truth(
                self.series, table, self.train_end, steps
            )
            elapsed = time.perf_counter() - started
            if self.timeout_seconds is not None and elapsed > self.timeout_seconds:
                record = TrialRecord(
                    spec=spec,
                    status="timed_out",
                    fit_duration_seconds=fit_duration,
                    predict_duration_seconds=predict_duration,
                    runtime=provider.runtime_fingerprint(),
                    error=f"trial exceeded {self.timeout_seconds} seconds",
                )
            else:
                record = TrialRecord(
                    spec=spec,
                    status="succeeded",
                    validation_metrics=metrics,
                    prediction_ref=str(prediction_path),
                    artifact_ref=artifact_ref,
                    truth_fingerprint=truth_hash,
                    valid_observations=valid_count,
                    fit_duration_seconds=fit_duration,
                    predict_duration_seconds=predict_duration,
                    runtime=provider.runtime_fingerprint(),
                )
        except _EXPECTED_TRIAL_ERRORS as exc:
            record = TrialRecord(
                spec=spec,
                status="failed",
                fit_duration_seconds=fit_duration,
                predict_duration_seconds=predict_duration,
                runtime=provider.runtime_fingerprint(),
                error=f"{type(exc).__name__}: {exc}",
            )
        _write_record(result_path, record)
        return record

    def run_plan(
        self,
        plan: ExperimentPlan,
        *,
        max_parallel_trials: int,
        cancellation: threading.Event | None = None,
    ) -> tuple[TrialRecord, ...]:
        """Run every planned trial with bounded parallelism and cancellation."""
        if max_parallel_trials <= 0:
            raise ValueError("max_parallel_trials must be positive")
        records: dict[str, TrialRecord] = {}
        with ThreadPoolExecutor(max_workers=max_parallel_trials) as executor:
            futures = {
                executor.submit(self.run_trial, spec, cancellation): spec.trial_id for spec in plan.trials
            }
            for future in as_completed(futures):
                records[futures[future]] = future.result()
        return tuple(records[spec.trial_id] for spec in plan.trials)
