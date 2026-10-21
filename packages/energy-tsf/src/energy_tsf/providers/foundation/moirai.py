"""Offline Moirai 2 provider with a pinned upstream revision."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from energy_tsf.providers.foundation.base import FoundationParams, FoundationProvider, foundation_capabilities

MOIRAI_REPO: Literal["Salesforce/moirai-2.0-R-small"] = "Salesforce/moirai-2.0-R-small"
MOIRAI_REVISION: Literal["30f43ff08c8494f4943ae1521e9d4e94a0fbb389"] = (
    "30f43ff08c8494f4943ae1521e9d4e94a0fbb389"
)


class MoiraiParams(FoundationParams):
    repo_id: Literal["Salesforce/moirai-2.0-R-small"] = MOIRAI_REPO
    revision: Literal["30f43ff08c8494f4943ae1521e9d4e94a0fbb389"] = MOIRAI_REVISION
    local_files_only: Literal[True] = True


class _MoiraiRuntime:
    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, contexts, horizon_steps, *, past_covariates=None, future_covariates=None):
        import torch

        dynamic = None
        past_dynamic = None
        if past_covariates:
            future_names = tuple(future_covariates[0])
            past_names = tuple(name for name in past_covariates[0] if name not in future_names)
            if future_names:
                dynamic = [
                    np.column_stack(
                        [np.concatenate((past[name], future[name])) for name in future_names]
                    )
                    for past, future in zip(past_covariates, future_covariates)
                ]
            if past_names:
                past_dynamic = [
                    np.column_stack([past[name] for name in past_names])
                    for past in past_covariates
                ]
        if dynamic is None and past_dynamic is None:
            raw = np.asarray(self.model.predict(contexts), dtype=float)
        else:
            target = np.asarray(contexts, dtype=np.float32)[..., np.newaxis]
            device = self.model.device
            past_target = torch.as_tensor(target, device=device)
            observed_target = torch.as_tensor(np.isfinite(target), device=device)
            past_is_pad = torch.zeros(target.shape[:2], device=device, dtype=torch.bool)
            dynamic_tensor = (
                None
                if dynamic is None
                else torch.as_tensor(np.asarray(dynamic, dtype=np.float32), device=device)
            )
            past_dynamic_tensor = (
                None
                if past_dynamic is None
                else torch.as_tensor(np.asarray(past_dynamic, dtype=np.float32), device=device)
            )
            with torch.no_grad():
                prediction = self.model(
                    past_target=past_target,
                    past_observed_target=observed_target,
                    past_is_pad=past_is_pad,
                    feat_dynamic_real=dynamic_tensor,
                    observed_feat_dynamic_real=None
                    if dynamic_tensor is None
                    else torch.isfinite(dynamic_tensor),
                    past_feat_dynamic_real=past_dynamic_tensor,
                    past_observed_feat_dynamic_real=None
                    if past_dynamic_tensor is None
                    else torch.isfinite(past_dynamic_tensor),
                )
            raw = prediction.detach().cpu().numpy().astype(float, copy=False)
        if raw.ndim == 4 and raw.shape[-1] == 1:
            raw = raw[..., 0]
        levels = np.asarray(self.model.module.quantile_levels, dtype=float)
        median = int(np.argmin(np.abs(levels - 0.5)))
        return raw[:, median, :]


class MoiraiProvider(FoundationProvider):
    name = "moirai-2"
    dependency = "uni2ts"
    params_model = MoiraiParams
    capabilities = foundation_capabilities(name)

    def default_runtime_factory(self, **kwargs: Any) -> _MoiraiRuntime:
        import torch
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        module = Moirai2Module.from_pretrained(
            kwargs["checkpoint_path"],
            revision=kwargs["revision"],
            local_files_only=True,
        )
        model = Moirai2Forecast(
            module=module,
            prediction_length=kwargs["horizon_steps"],
            context_length=kwargs["context_length"],
            target_dim=1,
            feat_dynamic_real_dim=kwargs["known_future_covariates"],
            past_feat_dynamic_real_dim=kwargs["past_only_covariates"],
        ).to(torch.device(kwargs["device"]))
        return _MoiraiRuntime(model)
