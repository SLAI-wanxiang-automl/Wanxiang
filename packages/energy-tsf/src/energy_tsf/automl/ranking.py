"""Validation-only stable ranking and Top-K selection."""

from __future__ import annotations

import math

from energy_tsf.contracts.modeling import TopKSelection, TrialRecord


def rank_trials(records: list[TrialRecord] | tuple[TrialRecord, ...], top_k: int = 5) -> TopKSelection:
    """Rank eligible trials using validation metrics only.

    Trials must have complete artifacts and share the same validation truth
    fingerprint. Stable tie-breakers make the selected Top-K reproducible.
    """
    eligible = []
    for record in records:
        rmse = record.validation_metrics.get("rmse")
        mae = record.validation_metrics.get("mae")
        if (
            record.status == "succeeded"
            and rmse is not None
            and mae is not None
            and math.isfinite(rmse)
            and math.isfinite(mae)
            and record.prediction_ref is not None
            and record.artifact_ref is not None
            and record.truth_fingerprint is not None
        ):
            eligible.append(record)

    fingerprints = {record.truth_fingerprint for record in eligible}
    if len(fingerprints) > 1:
        raise ValueError("successful trials were not evaluated against the same validation truth")
    eligible.sort(
        key=lambda record: (
            record.validation_metrics["rmse"],
            record.validation_metrics["mae"],
            record.predict_duration_seconds,
            record.spec.provider,
            record.spec.trial_id,
        )
    )
    selected = tuple(eligible[:top_k])
    return TopKSelection(
        status="complete" if len(selected) == top_k else "insufficient",
        requested=top_k,
        records=selected,
        truth_fingerprint=next(iter(fingerprints), None),
    )
