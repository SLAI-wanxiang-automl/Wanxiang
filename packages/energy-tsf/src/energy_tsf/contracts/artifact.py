"""Immutable model, ensemble, solution, and publication contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from energy_tsf.contracts.common import ContractModel
from energy_tsf.contracts.features import CovariateSpec, FeatureSelectionEvidence, FeatureSpec


class ObjectRef(ContractModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str
    size: int = Field(ge=0)


class EnsembleSpec(ContractModel):
    method: Literal["nnls", "inverse_rmse", "bayesian_update"]
    member_trial_ids: tuple[str, ...]
    weights: tuple[float, ...]
    fallback_order: tuple[str, ...]
    renormalize_on_failure: bool = True
    min_valid_members: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_weights(self) -> "EnsembleSpec":
        if not self.member_trial_ids or len(self.member_trial_ids) != len(self.weights):
            raise ValueError("ensemble members and weights must have the same non-zero length")
        if set(self.fallback_order) != set(self.member_trial_ids):
            raise ValueError("fallback_order must contain every member exactly once")
        if any(weight < 0 for weight in self.weights) or sum(self.weights) <= 0:
            raise ValueError("ensemble weights must be non-negative with a positive sum")
        if abs(sum(self.weights) - 1.0) > 1e-8:
            raise ValueError("ensemble weights must sum to one")
        if self.min_valid_members > len(self.member_trial_ids):
            raise ValueError("min_valid_members cannot exceed member count")
        return self


class CompatibilitySpec(ContractModel):
    target: str
    frequency: str
    horizon: str
    timestamp_column: str | None = None
    prediction_mode: Literal["univariate", "multivariate"] = "univariate"
    covariates: CovariateSpec = Field(default_factory=CovariateSpec)


class SolutionMember(ContractModel):
    trial_id: str
    provider: str
    artifact_ref: str
    params: dict[str, Any]
    input_mode: Literal["univariate", "multivariate"] = "univariate"
    covariate_spec: CovariateSpec = Field(default_factory=CovariateSpec)
    feature_spec: FeatureSpec | None = None
    feature_selection: FeatureSelectionEvidence = Field(default_factory=FeatureSelectionEvidence)
    runtime: dict[str, str] = Field(default_factory=dict)
    validation_metrics: dict[str, float | None]
    test_metrics: dict[str, float | None] = Field(default_factory=dict)
    test_prediction_ref: str | None = None


class SolutionArtifact(ContractModel):
    solution_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str
    station_id: str
    data_snapshot_id: str
    compatibility: CompatibilitySpec
    members: tuple[SolutionMember, ...]
    ensemble: EnsembleSpec
    validation_report_ref: str
    test_report_ref: str | None = None
    validation_predictions_ref: str
    test_predictions_ref: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    code_fingerprint: str
    created_at: str
    search_pool_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parent_solution_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    update_evidence_ref: str | None = None

    @model_validator(mode="after")
    def validate_members(self) -> "SolutionArtifact":
        ids = tuple(member.trial_id for member in self.members)
        if ids != self.ensemble.member_trial_ids:
            raise ValueError("solution member order must match ensemble member order")
        return self


class CandidateRef(ContractModel):
    station_id: str
    solution_id: str
    path: str
    complete: bool


class ActiveRef(ContractModel):
    station_id: str
    target: str
    horizon: str
    solution_id: str
    solution_path: str
    previous_solution_id: str | None = None


class ReleaseDecision(ContractModel):
    approved: bool
    reasons: tuple[str, ...] = ()


class FinalMemberResult(ContractModel):
    trial_id: str
    provider: str
    artifact_ref: str
    test_metrics: dict[str, float | None]
    prediction_ref: str
    runtime: dict[str, str]


class FinalEvaluationReport(ContractModel):
    members: tuple[FinalMemberResult, ...]
    ensemble_metrics: dict[str, float | None]
    ensemble_prediction_ref: str
    report_ref: str
    truth_fingerprint: str
