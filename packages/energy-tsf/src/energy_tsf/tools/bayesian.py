"""Transparent fractional-evidence Bayesian ensemble updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

from energy_tsf.contracts.backtest import BayesianWeightUpdate, WindowEvidence


def update_ensemble_weights(
    member_trial_ids: Sequence[str],
    previous_weights: Sequence[float],
    window_rmse: Mapping[str, Mapping[str, float]],
    *,
    scale: float,
    scale_source: Literal["capacity", "truth_q95"],
    epsilon: float = 1e-3,
    temperature: float = 0.10,
    prior_strength: float | None = None,
) -> BayesianWeightUpdate:
    """Update a Dirichlet prior with one fractional observation per window."""

    members = tuple(member_trial_ids)
    weights = np.asarray(previous_weights, dtype=float)
    if not members or len(members) != len(weights) or len(set(members)) != len(members):
        raise ValueError("members and previous weights must be aligned and unique")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("previous weights must be finite non-negative values")
    if not window_rmse:
        raise ValueError("at least one backtest window is required")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Bayesian update scale must be finite and positive")
    if epsilon <= 0 or temperature <= 0:
        raise ValueError("epsilon and temperature must be positive")

    weights /= weights.sum()
    prior_weights = (weights + epsilon) / (weights + epsilon).sum()
    strength = float(prior_strength if prior_strength is not None else len(window_rmse))
    if strength <= 0:
        raise ValueError("prior_strength must be positive")
    prior_alpha = strength * prior_weights
    evidence: list[WindowEvidence] = []
    total_responsibility = np.zeros(len(members), dtype=float)
    for window_id, values in window_rmse.items():
        if set(values) != set(members):
            raise ValueError(f"window {window_id} must contain RMSE for every member")
        losses = np.asarray([values[member] / scale for member in members], dtype=float)
        if not np.isfinite(losses).all() or np.any(losses < 0):
            raise ValueError(f"window {window_id} contains invalid RMSE")
        logits = -losses / temperature
        logits -= logits.max()
        responsibilities = np.exp(logits)
        responsibilities /= responsibilities.sum()
        total_responsibility += responsibilities
        evidence.append(
            WindowEvidence(
                window_id=window_id,
                normalized_rmse={member: float(value) for member, value in zip(members, losses)},
                responsibilities={
                    member: float(value) for member, value in zip(members, responsibilities)
                },
            )
        )
    posterior_alpha = prior_alpha + total_responsibility
    new_weights = posterior_alpha / posterior_alpha.sum()
    return BayesianWeightUpdate(
        member_trial_ids=members,
        previous_weights=tuple(float(value) for value in weights),
        prior_weights=tuple(float(value) for value in prior_weights),
        prior_alpha=tuple(float(value) for value in prior_alpha),
        window_evidence=tuple(evidence),
        posterior_alpha=tuple(float(value) for value in posterior_alpha),
        new_weights=tuple(float(value) for value in new_weights),
        epsilon=epsilon,
        temperature=temperature,
        prior_strength=strength,
        scale=scale,
        scale_source=scale_source,
    )
