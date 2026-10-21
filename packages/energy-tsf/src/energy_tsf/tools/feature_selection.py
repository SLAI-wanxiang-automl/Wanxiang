"""Small deterministic selectors used by multivariate trials."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from energy_tsf.contracts.features import (
    CovariateSpec,
    FeatureSelectionEvidence,
    FeatureSpec,
    StatisticalSelectionSpec,
    WrapperSelectionSpec,
)
from energy_tsf.data.schemas import PreparedSeries
from energy_tsf.tools.features import build_supervised_dataset


def select_statistical_features(
    series: PreparedSeries,
    origins: tuple[int, ...],
    *,
    train_end: int,
    feature_spec: FeatureSpec,
    selection: StatisticalSelectionSpec,
    seed: int,
) -> tuple[FeatureSpec, FeatureSelectionEvidence]:
    """Freeze variance/correlation/MI selection using training rows only."""

    source = feature_spec.model_copy(update={"selected_features": ()})
    dataset = build_supervised_dataset(series, origins, source)
    keep_rows = dataset.target_indices < train_end
    x = dataset.x[keep_rows]
    y = dataset.y[keep_rows]
    if not len(y):
        raise ValueError("statistical feature selection has no training rows")

    variances = np.var(x, axis=0)
    candidates = np.flatnonzero(variances > selection.variance_threshold)
    if not len(candidates):
        candidates = np.asarray([int(np.argmax(variances))])
    mi = mutual_info_regression(x[:, candidates], y, random_state=seed)
    ranked = candidates[np.argsort(-mi, kind="stable")]
    ranked = ranked[: selection.mutual_information_top_n]

    chosen: list[int] = []
    for index in ranked:
        if chosen:
            correlations = np.corrcoef(x[:, [*chosen, int(index)]], rowvar=False)
            if np.nanmax(np.abs(correlations[-1, :-1])) >= selection.correlation_threshold:
                continue
        chosen.append(int(index))
        if len(chosen) >= selection.max_selected_features:
            break
    if not chosen:
        chosen = [int(ranked[0])]

    names = tuple(dataset.feature_names[index] for index in chosen)
    score_by_name = {
        dataset.feature_names[int(index)]: float(score)
        for index, score in zip(candidates, mi)
        if int(index) in chosen
    }
    evidence = FeatureSelectionEvidence(
        method="statistical",
        selected=names,
        scores=score_by_name,
    )
    return source.model_copy(update={"selected_features": names}), evidence


def select_wrapper_covariates(
    candidates: CovariateSpec,
    evaluate: Callable[[CovariateSpec], float],
    selection: WrapperSelectionSpec,
    *,
    allowed: tuple[str, ...] | None = None,
) -> tuple[CovariateSpec, FeatureSelectionEvidence]:
    """Greedy validation-RMSE wrapper with a strict evaluation budget."""

    pool = tuple(name for name in candidates.columns if allowed is None or name in allowed)
    selected: list[str] = []
    evaluated: list[tuple[str, ...]] = []
    scores: dict[str, float] = {}

    def as_spec(names: tuple[str, ...]) -> CovariateSpec:
        chosen = set(names)
        return CovariateSpec(
            past_only=tuple(name for name in candidates.past_only if name in chosen),
            known_future=tuple(name for name in candidates.known_future if name in chosen),
        )

    baseline = float(evaluate(CovariateSpec()))
    evaluated.append(())
    scores["<univariate>"] = baseline
    best = baseline
    while pool and len(selected) < selection.max_selected_covariates:
        remaining = [name for name in pool if name not in selected]
        round_scores: list[tuple[float, str]] = []
        for name in remaining:
            if len(evaluated) >= selection.max_evaluations_per_provider:
                break
            subset = (*selected, name)
            score = float(evaluate(as_spec(subset)))
            evaluated.append(subset)
            scores["+".join(subset)] = score
            round_scores.append((score, name))
        if not round_scores:
            break
        score, name = min(round_scores, key=lambda item: (item[0], item[1]))
        if best - score < selection.minimum_improvement:
            break
        selected.append(name)
        best = score

    chosen = as_spec(tuple(selected))
    return chosen, FeatureSelectionEvidence(
        method="wrapper",
        selected=chosen.columns,
        scores=scores,
        evaluated_subsets=tuple(evaluated),
    )
