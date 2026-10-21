"""LightGBM official sklearn-API provider."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from pydantic import Field

from energy_tsf.providers._tabular import FittedFeatureModel, TabularProvider
from energy_tsf.providers.base import ProviderCapabilities, ProviderParams
from energy_tsf.contracts.features import FeatureSpec


class LightGBMParams(ProviderParams):
    n_estimators: int = Field(default=300, gt=0)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    num_leaves: int = Field(default=31, gt=1)
    max_depth: int = Field(default=-1, ge=-1)
    min_child_samples: int = Field(default=20, gt=0)
    subsample: float = Field(default=0.9, gt=0, le=1)
    colsample_bytree: float = Field(default=0.9, gt=0, le=1)
    reg_lambda: float = Field(default=1.0, ge=0)
    random_state: int = Field(default=42, ge=0)
    n_jobs: int = 1


class LightGBMProvider(TabularProvider):
    name = "lightgbm"
    dependency = "lightgbm"
    params_model = LightGBMParams
    capabilities = ProviderCapabilities(
        name=name,
        family="ml",
        trainable=True,
        requires_features=True,
        artifact_kind="lightgbm_bundle",
    )

    def build_estimator(self, params: dict[str, Any]):
        from lightgbm import LGBMRegressor

        if params.get("subsample", 1.0) < 1.0:
            params = {**params, "subsample_freq": 1}
        return LGBMRegressor(objective="regression", verbosity=-1, **params)

    def save(self, model: FittedFeatureModel, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        booster = model.estimator.booster_ if hasattr(model.estimator, "booster_") else model.estimator
        metadata = {
            "params": model.params,
            "feature_spec": model.feature_spec.model_dump(mode="json"),
            "feature_names": list(model.feature_names),
        }
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("model.txt", booster.model_to_string())
            bundle.writestr("metadata.json", json.dumps(metadata, sort_keys=True))
        return target

    def load(self, path: str | Path) -> FittedFeatureModel:
        from lightgbm import Booster

        with zipfile.ZipFile(path, "r") as bundle:
            metadata = json.loads(bundle.read("metadata.json"))
            booster = Booster(model_str=bundle.read("model.txt").decode("utf-8"))
        return FittedFeatureModel(
            estimator=booster,
            feature_spec=FeatureSpec.model_validate(metadata["feature_spec"]),
            feature_names=tuple(metadata["feature_names"]),
            params=metadata["params"],
        )

    def runtime_fingerprint(self) -> dict[str, str]:
        import lightgbm

        return {**super().runtime_fingerprint(), "lightgbm": lightgbm.__version__}
