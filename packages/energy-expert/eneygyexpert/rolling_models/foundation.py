"""Zero-shot foundation models: reuse checkpoints; never fine-tune in a window."""

from __future__ import annotations

from importlib import import_module

from ..data import KEY_COLUMNS
from ..window_models import ModelCapabilities
from .common import ModelAudit, horizon, model_history, point_output

FOUNDATION_MODELS = {
    "Chronos": "chronos",
    "TimesFM": "timesfm",
    "Moirai": "moirai",
    "Toto": "toto",
    "TiRex": "tirex",
    "Sundial": "sundial",
    "FlowState": "flowstate",
    "T0": "t0",
    "Tafsut": "tafsut",
    "PatchTSTFM": "patchtst_fm",
    "TabPFN": "tabpfn",
    "TimeGPT": "timegpt",
}


class FoundationAdapter:
    """Conservative first contract: only y; advanced covariates require new adapter.

    A preloaded or injected model is supported. Dependencies/checkpoints are
    loaded only on first use. The fitted object is a predictor preparation,
    not a training call. TimeGPT remains a remote service.
    """

    def __init__(
        self, name, *, model=None, model_kwargs=None, alias=None, context_length=None
    ):
        if name not in FOUNDATION_MODELS:
            raise ValueError(f"Unknown foundation model {name}")
        self.name, self.model = name, model
        self.model_kwargs = dict(model_kwargs or {})
        for key in ("finetuning_config", "finetune_steps"):
            if self.model_kwargs.get(key) not in (None, 0):
                raise ValueError(
                    "Rolling foundation adapter supports zero-shot inference only"
                )
        if model is not None:
            self._check_zero_shot(model)
        self.alias = alias or name
        self.capabilities = ModelCapabilities(context_length=context_length)

    @staticmethod
    def _check_zero_shot(model):
        if getattr(model, "finetuning_config", None) is not None or getattr(
            model, "finetune_steps", 0
        ):
            raise ValueError("Disable foundation fine-tuning before rolling inference")
        if hasattr(model, "reuse_loaded_model"):
            model.reuse_loaded_model = True

    def get_params(self):
        return dict(
            name=self.name,
            model_kwargs=dict(self.model_kwargs),
            context_length=self.capabilities.context_length,
        )

    def fit_frozen(self, window, freq):
        if self.model is None:
            module = import_module(
                f"foundationforecast.models.{FOUNDATION_MODELS[self.name]}"
            )
            self.model = getattr(module, self.name)(**self.model_kwargs)
        self._check_zero_shot(self.model)
        return FittedFoundation(self, self.model)

    def forecast(self, window, freq):
        return self.fit_frozen(window, freq).forecast(window, freq)


class FittedFoundation:
    def __init__(self, source, model):
        self.source, self.model = source, model
        self.alias, self.capabilities = source.alias, source.capabilities
        self.audit = ModelAudit()

    def forecast(self, window, freq):
        result = self.model.forecast(
            df=model_history(window), h=horizon(window), freq=freq
        )
        backend_alias = self.model.alias
        result = point_output(result, backend_alias).rename(
            columns={backend_alias: self.alias}
        )
        self.audit.predictions.append(dict(origin=window.origin))
        return result.loc[:, [*KEY_COLUMNS, self.alias]]
