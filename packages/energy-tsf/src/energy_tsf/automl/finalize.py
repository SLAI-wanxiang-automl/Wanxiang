"""Freeze validation selection, retrain members, and evaluate test once."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from energy_tsf.automl.trial import validation_truth
from energy_tsf.contracts.artifact import (
    EnsembleSpec,
    FinalEvaluationReport,
    FinalMemberResult,
)
from energy_tsf.contracts.modeling import TopKSelection
from energy_tsf.contracts.result import ForecastTable
from energy_tsf.data.schemas import PreparedSeries
from energy_tsf.data.splits import horizon_steps
from energy_tsf.providers.base import TrainableProvider
from energy_tsf.providers.registry import ProviderRegistry
from energy_tsf.tools.ensemble import combine_available_predictions, fit_ensemble
from energy_tsf.tools.constraints import constrain_forecast


def build_validation_ensemble(
    selection: TopKSelection,
    series: PreparedSeries,
) -> tuple[EnsembleSpec, dict[str, float | None]]:
    """Fit ensemble weights from the frozen Top-K validation predictions."""
    if selection.status != "complete":
        raise ValueError(f"{selection.requested} successful validation candidates are required for an ensemble")
    predictions: dict[str, np.ndarray] = {}
    indices = None
    for record in selection.records:
        frame = pd.read_csv(record.prediction_ref)
        member_indices = frame["target_index"].to_numpy(dtype=int)
        if indices is None:
            indices = member_indices
        elif not np.array_equal(indices, member_indices):
            raise ValueError("Top-K validation predictions are not aligned")
        predictions[record.spec.trial_id] = frame["y_pred"].to_numpy(dtype=float)
    assert indices is not None
    truth = series.frame[series.target].to_numpy(dtype=float)[indices]
    include = ~series.imputation_mask.iloc[indices].to_numpy(dtype=bool)
    filtered = {member: values[include] for member, values in predictions.items()}
    validation_rmse = {}
    for record in selection.records:
        value = record.validation_metrics.get("rmse")
        if value is None:
            raise ValueError("Top-K member lacks validation RMSE")
        validation_rmse[record.spec.trial_id] = float(value)
    spec, combined = fit_ensemble(
        truth[include],
        filtered,
        validation_rmse,
    )
    from energy_tsf.tools.metrics import point_metrics

    metrics = point_metrics(truth[include], combined)
    normalized = {name: value if np.isfinite(value) else None for name, value in metrics.items()}
    return spec, normalized


def final_evaluate(
    selection: TopKSelection,
    ensemble: EnsembleSpec,
    series: PreparedSeries,
    registry: ProviderRegistry,
    *,
    retrain_origins: tuple[int, ...],
    test_origins: tuple[int, ...],
    train_end: int,
    work_dir: str | Path,
) -> FinalEvaluationReport:
    """Retrain frozen Top-K on train+validation and evaluate test exactly once."""

    selected_ids = tuple(record.spec.trial_id for record in selection.records)
    if selection.status != "complete" or selected_ids != ensemble.member_trial_ids:
        raise ValueError("final evaluation requires the frozen complete Top-K and matching weights")
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    steps = horizon_steps(series.frequency)
    member_results: list[FinalMemberResult] = []
    member_predictions: dict[str, np.ndarray] = {}
    aligned_frame: pd.DataFrame | None = None
    truth_hash: str | None = None

    for record in selection.records:
        provider = registry.create(record.spec.provider)
        member_dir = root / record.spec.trial_id
        member_dir.mkdir(parents=True, exist_ok=True)
        model = None
        if isinstance(provider, TrainableProvider):
            origins = retrain_origins
            if record.spec.feature_spec is not None:
                origins = tuple(
                    origin for origin in origins if origin >= record.spec.feature_spec.required_history
                )
            model = provider.fit(
                series,
                origins,
                train_end=train_end,
                params=record.spec.params,
                feature_spec=record.spec.feature_spec,
            )
            artifact = provider.save(model, member_dir / "artifact.joblib")
            model = provider.load(artifact)
            artifact_ref = str(artifact)
        else:
            artifact_ref = str(record.spec.params["checkpoint_path"])
        table = constrain_forecast(provider.predict(
            model,
            series,
            test_origins,
            steps,
            params=record.spec.params,
            feature_spec=record.spec.feature_spec,
            **(
                {"covariate_spec": record.spec.covariate_spec}
                if record.spec.covariate_spec.multivariate
                else {}
            ),
        ))
        table.require_coverage(test_origins, steps)
        metrics, _, member_truth_hash = validation_truth(series, table, train_end, steps)
        if truth_hash is None:
            truth_hash = member_truth_hash
            aligned_frame = table.frame.drop(columns="y_pred").copy()
        elif member_truth_hash != truth_hash:
            raise ValueError("final member test predictions are not aligned")
        prediction_path = member_dir / "test_predictions.csv"
        table.frame.to_csv(prediction_path, index=False)
        member_predictions[record.spec.trial_id] = table.predictions
        member_results.append(
            FinalMemberResult(
                trial_id=record.spec.trial_id,
                provider=record.spec.provider,
                artifact_ref=artifact_ref,
                test_metrics=metrics,
                prediction_ref=str(prediction_path),
                runtime=provider.runtime_fingerprint(),
            )
        )

    assert aligned_frame is not None and truth_hash is not None
    combined = combine_available_predictions(ensemble, member_predictions)
    ensemble_frame = aligned_frame.copy()
    ensemble_frame["provider"] = "ensemble"
    ensemble_frame["y_pred"] = combined
    ensemble_table = ForecastTable(ensemble_frame)
    ensemble_metrics, _, ensemble_truth_hash = validation_truth(series, ensemble_table, train_end, steps)
    if ensemble_truth_hash != truth_hash:
        raise ValueError("ensemble test truth is not aligned with members")
    ensemble_path = root / "test_predictions_ensemble.csv"
    ensemble_table.frame.to_csv(ensemble_path, index=False)
    report_path = root / "test_report.json"
    report_payload = {
        "frozen_member_ids": list(selected_ids),
        "frozen_weights": list(ensemble.weights),
        "members": [member.model_dump(mode="json") for member in member_results],
        "ensemble_metrics": ensemble_metrics,
        "truth_fingerprint": truth_hash,
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return FinalEvaluationReport(
        members=tuple(member_results),
        ensemble_metrics=ensemble_metrics,
        ensemble_prediction_ref=str(ensemble_path),
        report_ref=str(report_path),
        truth_fingerprint=truth_hash,
    )
