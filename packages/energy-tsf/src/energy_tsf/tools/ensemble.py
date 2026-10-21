"""Validation-fitted non-negative ensemble and inference fallback rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

from energy_tsf.contracts.artifact import EnsembleSpec
from energy_tsf.contracts.forecast import ForecastPlan


def _prediction_matrix(member_ids: Sequence[str], predictions: Mapping[str, np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(predictions[member], dtype=float).reshape(-1) for member in member_ids]
    if not arrays or len({array.shape for array in arrays}) != 1:
        raise ValueError("all ensemble predictions must have the same non-empty shape")
    matrix = np.column_stack(arrays)
    if not np.isfinite(matrix).all():
        raise ValueError("ensemble predictions must be finite")
    return matrix


def inverse_rmse_weights(rmse: Sequence[float]) -> np.ndarray:
    """Convert validation RMSE values to normalized inverse-error weights."""
    errors = np.asarray(rmse, dtype=float)
    if errors.ndim != 1 or errors.size == 0 or np.any(errors < 0) or not np.isfinite(errors).all():
        raise ValueError("RMSE values must be finite non-negative numbers")
    perfect = errors == 0
    if perfect.any():
        return perfect.astype(float) / perfect.sum()
    inverse = 1.0 / errors
    return inverse / inverse.sum()


def fit_ensemble(
    truth,
    member_predictions: Mapping[str, np.ndarray],
    validation_rmse: Mapping[str, float],
    *,
    min_valid_members: int = 1,
) -> tuple[EnsembleSpec, np.ndarray]:
    """Fit weights from validation truth only; test truth is not an input."""

    member_ids = tuple(member_predictions)
    matrix = _prediction_matrix(member_ids, member_predictions)
    target = np.asarray(truth, dtype=float).reshape(-1)
    if target.shape[0] != matrix.shape[0] or not np.isfinite(target).all():
        raise ValueError("validation truth must align with every member prediction")
    method: Literal["nnls", "inverse_rmse"] = "nnls"
    try:
        from scipy.optimize import nnls

        weights, _ = nnls(matrix, target)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("degenerate NNLS weights")
        weights = weights / weights.sum()
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        method = "inverse_rmse"
        weights = inverse_rmse_weights([validation_rmse[member] for member in member_ids])
    weights[int(np.argmax(weights))] += 1.0 - float(weights.sum())

    fallback = tuple(sorted(member_ids, key=lambda member: (validation_rmse[member], member)))
    spec = EnsembleSpec(
        method=method,
        member_trial_ids=member_ids,
        weights=tuple(float(weight) for weight in weights),
        fallback_order=fallback,
        min_valid_members=min_valid_members,
    )
    return spec, matrix @ weights


def combine_available_predictions(
    spec: EnsembleSpec,
    predictions: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Combine available members, renormalizing only when the spec permits it."""
    available = [member for member in spec.member_trial_ids if member in predictions]
    if len(available) < spec.min_valid_members:
        raise ValueError("not enough valid ensemble members")
    if len(available) != len(spec.member_trial_ids) and not spec.renormalize_on_failure:
        raise ValueError("an ensemble member is missing and renormalization is disabled")
    matrix = _prediction_matrix(available, predictions)
    weights = np.asarray(
        [spec.weights[spec.member_trial_ids.index(member)] for member in available], dtype=float
    )
    if weights.sum() <= 0:
        champion = next((member for member in spec.fallback_order if member in predictions), None)
        if champion is None:
            raise ValueError("no fallback ensemble member is available")
        return np.asarray(predictions[champion], dtype=float)
    weights /= weights.sum()
    return matrix @ weights


def combine_forecast_sources(
    plan: ForecastPlan,
    predictions: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Combine completed source Solutions using the frozen outer weights."""

    source_ids = tuple(source.solution_id for source in plan.sources)
    available = [source_id for source_id in source_ids if source_id in predictions]
    if not available:
        raise ValueError("no forecast source completed")
    if len(available) != len(source_ids) and not plan.renormalize_on_source_failure:
        raise ValueError("a forecast source failed and source fallback is disabled")
    matrix = _prediction_matrix(available, predictions)
    weights_by_id = {source.solution_id: source.weight for source in plan.sources}
    weights = np.asarray([weights_by_id[source_id] for source_id in available], dtype=float)
    if weights.sum() <= 0:
        raise ValueError("available forecast sources have zero total weight")
    return matrix @ (weights / weights.sum())
