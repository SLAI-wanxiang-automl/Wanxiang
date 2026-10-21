"""Median ensemble of already fixed member configurations and their policies."""

from __future__ import annotations

import numpy as np

from ..data import KEY_COLUMNS, ForecastWindow
from ..window_models import ModelCapabilities


class MedianEnsembleAdapter:
    def __init__(self, models, alias="MedianModelEnsemble"):
        # Prediction-level pipeline summaries reserve "MedianEnsemble".
        # Keep this fitted model candidate distinct so neither result is lost.
        if not models or len({m.alias for m in models}) != len(models):
            raise ValueError("Ensemble needs nonempty members with unique aliases")
        if any(m.capabilities.tuning_mode != "fixed" for m in models):
            raise ValueError(
                "Tune members first; ensemble accepts fixed configurations"
            )
        self.models, self.alias = list(models), alias
        state = any(m.capabilities.update_mode == "state_update" for m in models)
        mode = "state_update" if state else "frozen"
        self.capabilities = ModelCapabilities(
            min_history=max(m.capabilities.min_history for m in models),
            uses_historical_covariates=any(
                m.capabilities.uses_historical_covariates for m in models
            ),
            uses_future_covariates=any(
                m.capabilities.uses_future_covariates for m in models
            ),
            requires_future_covariates_in_history=any(
                m.capabilities.requires_future_covariates_in_history for m in models
            ),
            required_future_covariate_cols=tuple(
                dict.fromkeys(
                    col
                    for m in models
                    for col in m.capabilities.required_future_covariate_cols
                )
            ),
            update_mode=mode,
            supported_update_modes=(mode,),
        )

    def get_params(self):
        return {
            m.alias: m.get_params() if hasattr(m, "get_params") else {}
            for m in self.models
        }

    def minimum_training_history(self, h):
        return max(
            model.minimum_training_history(h)
            if hasattr(model, "minimum_training_history")
            else max(model.capabilities.min_history, getattr(model, "lags", 0) + h + 1)
            for model in self.models
        )

    def fit_frozen(self, window, freq):
        return FittedMedianEnsemble(
            self, [m.fit_frozen(window, freq) for m in self.models]
        )

    def forecast(self, window, freq):
        return self.fit_frozen(window, freq).forecast(window, freq)


class FittedMedianEnsemble:
    def __init__(self, source, models):
        self.models = models
        self.alias, self.capabilities = source.alias, source.capabilities

    def update(self, history, freq):
        for model in self.models:
            if model.capabilities.update_mode == "state_update":
                model.update(history, freq)

    def forecast(self, window, freq):
        result = window.future_covariates.loc[:, list(KEY_COLUMNS)].reset_index(
            drop=True
        )
        predictions = []
        for model in self.models:
            context = model.capabilities.context_length
            history = (
                window.history.groupby("unique_id", sort=False).tail(context)
                if context
                else window.history
            )
            member_window = ForecastWindow(
                window.origin, history, window.future_covariates, window.actuals
            )
            forecast = model.forecast(member_window, freq)
            if not forecast.loc[:, list(KEY_COLUMNS)].equals(result):
                raise ValueError("Ensemble member prediction timestamps do not match")
            predictions.append(forecast[model.alias].to_numpy(dtype=float))
        result[self.alias] = np.median(np.vstack(predictions), axis=0)
        return result
