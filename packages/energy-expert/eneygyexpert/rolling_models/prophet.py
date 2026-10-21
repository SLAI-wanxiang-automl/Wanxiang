"""Per-series Prophet objects fitted once and reused at rolling timestamps."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from ..data import KEY_COLUMNS
from ..window_models import ModelCapabilities
from .common import ModelAudit, model_future, model_history


class ProphetAdapter:
    fit_on_full_history = True

    def __init__(self, *, params=None, alias="Prophet", future_covariate_cols=()):
        self.params, self.alias = deepcopy(params or {}), alias
        self.future_covariate_cols = tuple(future_covariate_cols)
        self.historical_covariate_cols = ()
        self.capabilities = ModelCapabilities(
            uses_future_covariates=bool(future_covariate_cols),
            requires_future_covariates_in_history=bool(future_covariate_cols),
            required_future_covariate_cols=self.future_covariate_cols,
        )

    def get_params(self):
        return dict(
            params=deepcopy(self.params),
            future_covariate_cols=self.future_covariate_cols,
        )

    def fit_frozen(self, window, freq):
        from prophet import Prophet

        history = model_history(window, future=self.future_covariate_cols)
        models = {}
        for uid, frame in history.groupby("unique_id", sort=False):
            model = Prophet(**self.params)
            for col in self.future_covariate_cols:
                model.add_regressor(col)
            model.fit(frame.drop(columns="unique_id"))
            models[uid] = model
        return FittedProphet(self, models, history)

    def forecast(self, window, freq):
        return self.fit_frozen(window, freq).forecast(window, freq)


class FittedProphet:
    def __init__(self, source, models, history):
        self.source, self.models = source, models
        self.alias, self.capabilities = source.alias, source.capabilities
        self.audit = ModelAudit()
        self.audit.fitted(history)

    def forecast(self, window, freq):
        rows = []
        for uid, future in model_future(
            window, self.source.future_covariate_cols
        ).groupby("unique_id", sort=False):
            frame = future.loc[:, list(KEY_COLUMNS)].copy()
            frame[self.alias] = (
                self.models[uid]
                .predict(future.drop(columns="unique_id"))
                .yhat.to_numpy()
            )
            rows.append(frame)
        self.audit.predictions.append(dict(origin=window.origin))
        return pd.concat(rows, ignore_index=True)
