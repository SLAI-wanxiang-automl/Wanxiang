"""XGBoost official sklearn-API provider."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from pydantic import Field

from energy_tsf.providers._tabular import FittedFeatureModel, TabularProvider
from energy_tsf.providers.base import ProviderCapabilities, ProviderParams
from energy_tsf.contracts.features import FeatureSpec


class XGBoostParams(ProviderParams):
    n_estimators: int = Field(default=300, gt=0)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    max_depth: int = Field(default=6, gt=0)
    min_child_weight: float = Field(default=1.0, ge=0)
    subsample: float = Field(default=0.9, gt=0, le=1)
    colsample_bytree: float = Field(default=0.9, gt=0, le=1)
    reg_lambda: float = Field(default=1.0, ge=0)
    random_state: int = Field(default=42, ge=0)
    n_jobs: int = 1


class XGBoostProvider(TabularProvider):
    name = "xgboost"
    dependency = "xgboost"
    params_model = XGBoostParams
    capabilities = ProviderCapabilities(
        name=name,
        family="ml",
        trainable=True,
        requires_features=True,
        artifact_kind="xgboost_ubj_bundle",
    )

    def build_estimator(self, params: dict[str, Any]):
        from xgboost import XGBRegressor

        return XGBRegressor(objective="reg:squarederror", tree_method="hist", **params)

    def save(self, model: FittedFeatureModel, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "params": model.params,
            "feature_spec": model.feature_spec.model_dump(mode="json"),
            "feature_names": list(model.feature_names),
        }
        with tempfile.TemporaryDirectory() as directory:
            official = Path(directory) / "model.ubj"
            model.estimator.save_model(official)
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(official, "model.ubj")
                bundle.writestr("metadata.json", json.dumps(metadata, sort_keys=True))
        return target

    def load(self, path: str | Path) -> FittedFeatureModel:
        from xgboost import XGBRegressor

        with zipfile.ZipFile(path, "r") as bundle:
            metadata = json.loads(bundle.read("metadata.json"))
            with tempfile.TemporaryDirectory() as directory:
                official = Path(directory) / "model.ubj"
                official.write_bytes(bundle.read("model.ubj"))
                estimator = XGBRegressor(objective="reg:squarederror", tree_method="hist", **metadata["params"])
                estimator.load_model(official)
        return FittedFeatureModel(
            estimator=estimator,
            feature_spec=FeatureSpec.model_validate(metadata["feature_spec"]),
            feature_names=tuple(metadata["feature_names"]),
            params=metadata["params"],
        )

    def runtime_fingerprint(self) -> dict[str, str]:
        import xgboost

        return {**super().runtime_fingerprint(), "xgboost": xgboost.__version__}
