"""Offline Chronos 2 provider with a pinned upstream revision."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from energy_tsf.providers.foundation.base import FoundationParams, FoundationProvider, foundation_capabilities

CHRONOS_REPO: Literal["amazon/chronos-2"] = "amazon/chronos-2"
CHRONOS_REVISION: Literal["29ec3766d36d6f73f0696f85560a422f50e8498c"] = (
    "29ec3766d36d6f73f0696f85560a422f50e8498c"
)


class ChronosParams(FoundationParams):
    repo_id: Literal["amazon/chronos-2"] = CHRONOS_REPO
    revision: Literal["29ec3766d36d6f73f0696f85560a422f50e8498c"] = CHRONOS_REVISION
    local_files_only: Literal[True] = True


class _ChronosRuntime:
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def predict(self, contexts, horizon_steps, *, past_covariates=None, future_covariates=None):
        inputs = contexts
        if past_covariates:
            inputs = [
                {
                    "target": context,
                    "past_covariates": past,
                    "future_covariates": future,
                }
                for context, past, future in zip(contexts, past_covariates, future_covariates)
            ]
        quantiles, _mean = self.pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=horizon_steps,
            quantile_levels=[0.5],
            cross_learning=False,
        )
        if isinstance(quantiles, list):
            quantiles = [
                item.detach().cpu().numpy() if hasattr(item, "detach") else np.asarray(item)
                for item in quantiles
            ]
        rows = []
        for item in quantiles:
            array = item.detach().cpu().numpy() if hasattr(item, "detach") else np.asarray(item)
            array = np.asarray(array, dtype=float)
            rows.append(np.squeeze(array[0]))
        return np.asarray(rows, dtype=float)


class ChronosProvider(FoundationProvider):
    name = "chronos-2"
    dependency = "chronos"
    params_model = ChronosParams
    capabilities = foundation_capabilities(name)

    def default_runtime_factory(self, **kwargs: Any) -> _ChronosRuntime:
        import torch
        from chronos import Chronos2Pipeline

        pipeline = Chronos2Pipeline.from_pretrained(
            kwargs["checkpoint_path"],
            revision=kwargs["revision"],
            local_files_only=True,
            device_map=kwargs["device"],
            torch_dtype=torch.float32,
        )
        return _ChronosRuntime(pipeline)
