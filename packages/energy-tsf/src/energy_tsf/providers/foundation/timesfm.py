"""Offline TimesFM 2.5 provider with a pinned upstream revision."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from energy_tsf.providers.foundation.base import FoundationParams, FoundationProvider, foundation_capabilities

TIMESFM_REPO: Literal["google/timesfm-2.5-200m-pytorch"] = "google/timesfm-2.5-200m-pytorch"
TIMESFM_REVISION: Literal["1d952420fba87f3c6dee4f240de0f1a0fbc790e3"] = (
    "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
)


class TimesFMParams(FoundationParams):
    repo_id: Literal["google/timesfm-2.5-200m-pytorch"] = TIMESFM_REPO
    revision: Literal["1d952420fba87f3c6dee4f240de0f1a0fbc790e3"] = TIMESFM_REVISION
    local_files_only: Literal[True] = True


class _TimesFMRuntime:
    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, contexts, horizon_steps, *, past_covariates=None, future_covariates=None):
        if past_covariates:
            names = tuple(future_covariates[0])
            if not names:
                raise ValueError("TimesFM covariates must be known through the forecast horizon")
            dynamic = {
                name: [
                    np.concatenate((past[name], future[name])).astype(float, copy=False)
                    for past, future in zip(past_covariates, future_covariates)
                ]
                for name in names
            }
            point, _quantiles = self.model.forecast_with_covariates(
                inputs=contexts,
                dynamic_numerical_covariates=dynamic,
            )
            return np.asarray(point, dtype=float)
        _point, quantiles = self.model.forecast(horizon=horizon_steps, inputs=contexts)
        return np.asarray(quantiles, dtype=float)[..., 5]


class TimesFMProvider(FoundationProvider):
    name = "timesfm-2.5"
    dependency = "timesfm"
    params_model = TimesFMParams
    capabilities = foundation_capabilities(name)

    def default_runtime_factory(self, **kwargs: Any) -> _TimesFMRuntime:
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            kwargs["checkpoint_path"],
            revision=kwargs["revision"],
            local_files_only=True,
            torch_compile=False,
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=kwargs["context_length"],
                max_horizon=kwargs["horizon_steps"],
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                return_backcast=True,
            )
        )
        return _TimesFMRuntime(model)
