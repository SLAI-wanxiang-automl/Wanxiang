"""Final full-data fitting after update selection is frozen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from energy_tsf.contracts.modeling import TopKSelection, TrialSpec
from energy_tsf.data.schemas import PreparedSeries
from energy_tsf.providers.base import TrainableProvider
from energy_tsf.providers.registry import ProviderRegistry


@dataclass(frozen=True)
class FinalizedUpdateMember:
    spec: TrialSpec
    artifact_ref: str
    runtime: dict[str, str]


def finalize_update_members(
    selection: TopKSelection,
    series: PreparedSeries,
    registry: ProviderRegistry,
    *,
    retrain_origins: tuple[int, ...],
    work_dir: str | Path,
) -> tuple[FinalizedUpdateMember, ...]:
    """Fit, persist, and reload selected update models after selection is frozen."""
    if selection.status != "complete":
        raise ValueError("a complete frozen Top-K is required for final update training")
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    completed = []
    for record in selection.records:
        provider = registry.create(record.spec.provider)
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
                train_end=len(series.frame),
                params=record.spec.params,
                feature_spec=record.spec.feature_spec,
            )
            artifact = provider.save(model, root / record.spec.trial_id / "artifact.joblib")
            provider.load(artifact)
            artifact_ref = str(artifact)
        else:
            artifact_ref = str(record.spec.params["checkpoint_path"])
        completed.append(
            FinalizedUpdateMember(
                spec=record.spec,
                artifact_ref=artifact_ref,
                runtime=provider.runtime_fingerprint(),
            )
        )
    return tuple(completed)
