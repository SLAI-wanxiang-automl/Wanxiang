"""Per-origin validation ranking and Top-k-union candidate pools."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from energy_tsf.contracts.modeling import TrialRecord
from energy_tsf.contracts.search_pool import SearchPoolArtifact, SearchPoolEntry, WindowTrialScore
from energy_tsf.data.schemas import PreparedSeries
from energy_tsf.data.splits import horizon_steps


def search_pool_fingerprint(value: SearchPoolArtifact | dict) -> str:
    """Return the stable content hash for a SearchPool artifact."""
    payload = value.model_dump(mode="json") if isinstance(value, SearchPoolArtifact) else dict(value)
    payload.pop("pool_id", None)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _eligible(record: TrialRecord) -> bool:
    rmse = record.validation_metrics.get("rmse")
    mae = record.validation_metrics.get("mae")
    return bool(
        record.status == "succeeded"
        and record.prediction_ref
        and record.truth_fingerprint
        and rmse is not None
        and mae is not None
        and math.isfinite(rmse)
        and math.isfinite(mae)
    )


def build_search_pool(
    records: tuple[TrialRecord, ...] | list[TrialRecord],
    series: PreparedSeries,
    *,
    request_id: str,
    station_id: str,
    horizon: str = "1d",
    top_k_per_window: int = 3,
) -> tuple[SearchPoolArtifact, tuple[WindowTrialScore, ...]]:
    """Keep the union of each validation window's Top-K eligible trials."""
    """Return the union of each validation origin's Top-k trials."""

    if top_k_per_window <= 0:
        raise ValueError("top_k_per_window must be positive")
    eligible = [record for record in records if _eligible(record)]
    if not eligible:
        raise ValueError("no successful trials are available for a search pool")
    specs = {(record.spec.data_snapshot_id, record.spec.split_id) for record in eligible}
    if len(specs) != 1:
        raise ValueError("search-pool trials must share one data snapshot and split")
    data_snapshot_id, split_id = next(iter(specs))
    if len({record.truth_fingerprint for record in eligible}) != 1:
        raise ValueError("search-pool trials were not evaluated against the same validation truth")

    values = series.frame[series.target].to_numpy(dtype=float)
    mask = series.imputation_mask.to_numpy(dtype=bool)
    expected_steps = horizon_steps(series.frequency, horizon)
    by_origin: dict[int, list[tuple[TrialRecord, float, float, int]]] = {}
    for record in eligible:
        prediction_ref = record.prediction_ref
        if prediction_ref is None:
            continue
        frame = pd.read_csv(Path(prediction_ref))
        required = {"origin", "horizon_step", "target_index", "y_pred"}
        if not required.issubset(frame.columns):
            raise ValueError(f"trial {record.spec.trial_id} prediction table is incomplete")
        for origin, group in frame.groupby("origin", sort=True):
            group = group.sort_values("horizon_step", kind="stable")
            if len(group) != expected_steps or not np.array_equal(
                group["horizon_step"].to_numpy(dtype=int), np.arange(1, expected_steps + 1)
            ):
                continue
            indices = group["target_index"].to_numpy(dtype=int)
            expected = int(origin) + group["horizon_step"].to_numpy(dtype=int) - 1
            if not np.array_equal(indices, expected) or indices.min() < 0 or indices.max() >= len(values):
                continue
            include = ~mask[indices]
            if not include.any():
                continue
            truth = values[indices][include]
            predicted = group["y_pred"].to_numpy(dtype=float)[include]
            errors = predicted - truth
            rmse = float(np.sqrt(np.mean(errors**2)))
            mae = float(np.mean(np.abs(errors)))
            if math.isfinite(rmse) and math.isfinite(mae):
                by_origin.setdefault(int(origin), []).append((record, rmse, mae, int(include.sum())))
    if not by_origin:
        raise ValueError("no validation window has observed truth")

    scores: list[WindowTrialScore] = []
    selected_by_trial: dict[str, list[tuple[str, int]]] = {}
    for origin in sorted(by_origin):
        window_id = f"origin-{origin}"
        ranked = sorted(
            by_origin[origin],
            key=lambda item: (
                item[1],
                item[2],
                item[0].predict_duration_seconds,
                item[0].spec.provider,
                item[0].spec.trial_id,
            ),
        )
        for rank, (record, rmse, mae, valid) in enumerate(ranked, 1):
            selected = rank <= top_k_per_window
            scores.append(
                WindowTrialScore(
                    window_id=window_id,
                    origin=origin,
                    trial_id=record.spec.trial_id,
                    provider=record.spec.provider,
                    rmse=rmse,
                    mae=mae,
                    valid_observations=valid,
                    rank=rank,
                    selected=selected,
                )
            )
            if selected:
                selected_by_trial.setdefault(record.spec.trial_id, []).append((window_id, rank))

    records_by_id = {record.spec.trial_id: record for record in eligible}
    entries = []
    for trial_id, windows in selected_by_trial.items():
        record = records_by_id[trial_id]
        entries.append(
            SearchPoolEntry(
                spec=record.spec,
                validation_metrics=record.validation_metrics,
                selected_windows=tuple(window_id for window_id, _ in windows),
                best_window_rank=min(rank for _, rank in windows),
                runtime=record.runtime,
            )
        )
    entries.sort(
        key=lambda entry: (
            entry.best_window_rank,
            -len(entry.selected_windows),
            entry.validation_metrics.get("rmse", math.inf),
            entry.spec.provider,
            entry.spec.trial_id,
        )
    )
    payload = SearchPoolArtifact(
        pool_id="0" * 64,
        request_id=request_id,
        station_id=station_id,
        data_snapshot_id=data_snapshot_id,
        split_id=split_id,
        horizon=horizon,
        top_k_per_window=top_k_per_window,
        window_count=len(by_origin),
        entries=tuple(entries),
    )
    return payload.model_copy(update={"pool_id": search_pool_fingerprint(payload)}), tuple(scores)
