"""Explicit opt-in sktime update adapter; parameter snapshots prevent refitting."""

from __future__ import annotations

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd

from ..data import KEY_COLUMNS
from ..window_models import ModelCapabilities
from .common import ModelAudit, horizon, model_history


class SKTimeRollingAdapter:
    """Declare parameter_names from get_fitted_params() for state_update auditing.

    sktime update_params=False is not a universal guarantee for every model.
    Every declared learned parameter is checked after each update. A model
    whose fitted parameters cannot be inspected must use a specific adapter.
    """

    fit_on_full_history = True

    def __init__(
        self, model, *, alias="SKTime", update_mode="state_update", parameter_names=()
    ):
        if update_mode != "state_update":
            raise ValueError("Generic sktime adapter requires state_update")
        if not parameter_names:
            raise ValueError(
                "Declare learned parameter_names to audit update_params=False"
            )
        self.model, self.alias = model, alias
        self.parameter_names = tuple(parameter_names)
        self.capabilities = ModelCapabilities(
            update_mode="state_update", supported_update_modes=("state_update",)
        )

    def get_params(self):
        return self.model.get_params(deep=False)

    def fit_frozen(self, window, freq):
        history = model_history(window)
        models = {}
        for uid, group in history.groupby("unique_id", sort=False):
            model = deepcopy(self.model)
            model.fit(group.set_index("ds").y, fh=np.arange(1, horizon(window) + 1))
            models[uid] = model
        return FittedSKTime(self, models, history)

    def forecast(self, window, freq):
        return self.fit_frozen(window, freq).forecast(window, freq)


class FittedSKTime:
    def __init__(self, source, models, history):
        self.source, self.models = source, models
        self.alias, self.capabilities = source.alias, source.capabilities
        self.ends = history.groupby("unique_id").ds.max().to_dict()
        self.parameters = {uid: self._snapshot(model) for uid, model in models.items()}
        self.audit = ModelAudit()
        self.audit.fitted(history)

    def _snapshot(self, model):
        params = model.get_fitted_params()
        missing = set(self.source.parameter_names) - set(params)
        if missing:
            raise ValueError(f"Cannot audit fitted parameters: {sorted(missing)}")
        return pickle.dumps({key: params[key] for key in self.source.parameter_names})

    def update(self, history, freq):
        for uid, model in self.models.items():
            frame = history.loc[history.unique_id == uid].sort_values("ds")
            if frame.empty or frame.ds.max() < self.ends[uid]:
                raise ValueError("State origin must move forward")
            new = frame.loc[frame.ds > self.ends[uid]]
            if new.empty:
                continue
            expected = pd.date_range(self.ends[uid], periods=len(new) + 1, freq=freq)[
                1:
            ]
            if not pd.DatetimeIndex(new.ds).equals(expected):
                raise ValueError("New state observations must be contiguous")
            model.update(new.set_index("ds").y, update_params=False)
            if self._snapshot(model) != self.parameters[uid]:
                raise ValueError("sktime update changed learned parameters")
            self.ends[uid] = new.ds.max()
            self.audit.updates.append(dict(end=new.ds.max(), rows=len(new)))

    def forecast(self, window, freq):
        self.update(window.history, freq)
        rows = []
        for uid, future in window.future_covariates.groupby("unique_id", sort=False):
            values = self.models[uid].predict(fh=np.arange(1, len(future) + 1))
            frame = future.loc[:, list(KEY_COLUMNS)].copy()
            frame[self.alias] = np.asarray(values).reshape(-1)
            rows.append(frame)
        self.audit.predictions.append(dict(origin=window.origin))
        return pd.concat(rows, ignore_index=True)
