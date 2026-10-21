"""Deterministically allocate a global trial budget across providers."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Literal, Mapping, cast

from pydantic import Field

from energy_tsf.automl.search_space import SearchCandidate, SearchSpace
from energy_tsf.contracts.common import ContractModel
from energy_tsf.contracts.features import (
    CovariateSpec,
    FeatureSelectionEvidence,
    FeatureSpec,
    StatisticalSelectionSpec,
    WrapperSelectionSpec,
)
from energy_tsf.contracts.modeling import ExperimentPlan, TrialSpec
from energy_tsf.data.splits import horizon_steps
from energy_tsf.providers.registry import ProviderRegistry
from energy_tsf.providers.foundation.base import FoundationProvider


class PlanningInput(ContractModel):
    data_snapshot_id: str
    split_id: str
    train_end: int = Field(gt=0)
    validation_end: int = Field(gt=0)
    frequency: str
    providers: tuple[str, ...]
    max_trials: int = Field(gt=0)
    seed: int = Field(ge=0)
    foundation_checkpoints: dict[str, str] = Field(default_factory=dict)
    prediction_mode: Literal["univariate", "multivariate"] = "univariate"
    covariates: CovariateSpec = Field(default_factory=CovariateSpec)
    feature_options: dict[str, Any] = Field(default_factory=dict)
    statistical_selection: StatisticalSelectionSpec = Field(default_factory=StatisticalSelectionSpec)
    wrapper_selection: WrapperSelectionSpec = Field(default_factory=WrapperSelectionSpec)


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _trial_id(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()[:16]


def _candidate_to_spec(
    provider_name: str,
    candidate: SearchCandidate,
    inputs: PlanningInput,
    provider,
    input_mode: Literal["univariate", "multivariate"] = "univariate",
) -> TrialSpec | None:
    season = horizon_steps(inputs.frequency)
    params = dict(candidate.params)
    context_length = candidate.context_days * season if candidate.context_days else None
    feature_spec = None

    if input_mode == "multivariate" and provider_name not in {
        "ridge", "xgboost", "timesfm-2.5", "moirai-2", "chronos-2"
    }:
        return None

    covariate_spec = inputs.covariates if input_mode == "multivariate" else CovariateSpec()
    selection = FeatureSelectionEvidence()
    if provider.capabilities.requires_features:
        feature = {
            "season_length": season,
            "daily_lags": (1, 2, 3, 7),
            "recent_lags": 12,
            "rolling_windows": (6, 12, season),
            "calendar": True,
            **inputs.feature_options,
            **candidate.feature,
        }
        if input_mode == "multivariate":
            feature.update(
                covariates=covariate_spec,
                target_differences=tuple(dict.fromkeys((*feature.get("target_differences", (1,)), season))),
                covariate_lags=tuple(dict.fromkeys((*feature.get("covariate_lags", (1, 4)), season))),
                covariate_differences=tuple(
                    dict.fromkeys((*feature.get("covariate_differences", (1,)), season))
                ),
                covariate_rolling_windows=tuple(
                    dict.fromkeys((*feature.get("covariate_rolling_windows", (4, 16)), season))
                ),
            )
            selection = FeatureSelectionEvidence(method="statistical")
        feature_spec = FeatureSpec.model_validate(feature)
        if feature_spec.required_history + season > inputs.train_end:
            return None
    if provider_name in {"arima", "sarima"} and context_length is not None:
        if context_length > inputs.train_end:
            return None
        params["context_length"] = context_length
    if provider_name == "sarima":
        params["season_length"] = season
    if provider.capabilities.family == "foundation":
        checkpoint = inputs.foundation_checkpoints.get(provider_name)
        if checkpoint is None or context_length is None or context_length > inputs.train_end:
            return None
        params.update(checkpoint_path=checkpoint, context_length=context_length)
        if input_mode == "multivariate":
            selection = FeatureSelectionEvidence(method="wrapper")

    params = provider.validate_params(params)
    identity = {
        "provider": provider_name,
        "params": params,
        "feature_spec": feature_spec.model_dump(mode="json") if feature_spec else None,
        "input_mode": input_mode,
        "covariate_spec": covariate_spec.model_dump(mode="json"),
        "selection_method": selection.method,
        "context_length": context_length,
        "seed": inputs.seed,
        "data_snapshot_id": inputs.data_snapshot_id,
        "split_id": inputs.split_id,
    }
    return TrialSpec(
        trial_id=_trial_id(identity),
        provider=provider_name,
        params=params,
        input_mode=input_mode,
        covariate_spec=covariate_spec,
        feature_spec=feature_spec,
        feature_selection=selection,
        statistical_selection=inputs.statistical_selection,
        wrapper_selection=inputs.wrapper_selection,
        context_length=context_length,
        seed=inputs.seed,
        data_snapshot_id=inputs.data_snapshot_id,
        split_id=inputs.split_id,
    )


def build_experiment_plan(
    inputs: PlanningInput,
    spaces: Mapping[str, SearchSpace],
    registry: ProviderRegistry,
    *,
    availability: Mapping[str, tuple[bool, str | None]] | None = None,
) -> ExperimentPlan:
    """Expand a stable, budget-limited trial plan without using test data.

    Provider availability, checkpoint readiness, data length, and feature
    requirements are resolved before trials are scheduled. The returned plan
    contains all trial identities and unavailable-provider reasons.
    """

    availability = availability or registry.availability(inputs.providers)
    queues: dict[str, list[TrialSpec]] = {}
    unavailable: dict[str, str] = {}

    for name in inputs.providers:
        available, reason = availability.get(name, (False, "availability not reported"))
        if not available:
            unavailable[name] = reason or "provider unavailable"
            continue
        space = spaces.get(name)
        if space is None:
            unavailable[name] = "search space not configured"
            continue
        provider = registry.create(name)
        if provider.capabilities.family == "foundation":
            foundation = cast(FoundationProvider, provider)
            checkpoint_ready, checkpoint_reason = foundation.checkpoint_available(
                inputs.foundation_checkpoints.get(name)
            )
            if not checkpoint_ready:
                unavailable[name] = checkpoint_reason or "local checkpoint is unavailable"
                continue
        candidates = list(space.candidates)
        if len(candidates) > 1:
            tail = candidates[1:]
            random.Random(f"{inputs.seed}:{name}").shuffle(tail)
            candidates = [candidates[0], *tail]
        specs: list[TrialSpec] = []
        for candidate in candidates[: space.max_trials]:
            univariate = _candidate_to_spec(name, candidate, inputs, provider)
            if univariate is not None:
                specs.append(univariate)
            if inputs.prediction_mode == "multivariate":
                multivariate = _candidate_to_spec(
                    name, candidate, inputs, provider, input_mode="multivariate"
                )
                if multivariate is not None:
                    specs.append(multivariate)
        if specs:
            queues[name] = specs
        else:
            unavailable[name] = "no candidate is compatible with data length or runtime inputs"

    selected: list[TrialSpec] = []
    names = [name for name in inputs.providers if name in queues]
    while len(selected) < inputs.max_trials and any(queues[name] for name in names):
        for name in names:
            if queues[name] and len(selected) < inputs.max_trials:
                selected.append(queues[name].pop(0))

    plan_payload = {
        "trials": [trial.model_dump(mode="json") for trial in selected],
        "unavailable": unavailable,
        "max_trials": inputs.max_trials,
        "seed": inputs.seed,
    }
    plan_id = hashlib.sha256(_canonical(plan_payload)).hexdigest()
    return ExperimentPlan(plan_id=plan_id, trials=tuple(selected), unavailable=unavailable, max_trials=inputs.max_trials, seed=inputs.seed)
