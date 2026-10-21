"""NeuralForecast adapters with native search spaces and common rolling scores."""

from __future__ import annotations

from copy import deepcopy

from ..data import KEY_COLUMNS, ForecastData
from ..window_models import ModelCapabilities
from .common import ModelAudit, horizon, model_future, model_history, point_output

NEURAL_MODELS = {
    "AutoPatchTST": "PatchTST",
    "AutoNBEATS": "NBEATS",
    "AutoNHITS": "NHITS",
    "AutoTFT": "TFT",
    "AutoDeepAR": "DeepAR",
}
HIST_MODELS = {"NHITS", "TFT"}
FUTURE_MODELS = {"NHITS", "TFT", "DeepAR"}


class NeuralForecastAdapter:
    fit_on_full_history = True

    def __init__(
        self,
        name,
        *,
        h,
        config,
        alias=None,
        historical_covariate_cols=(),
        future_covariate_cols=(),
        context_length=None,
        trainer_kwargs=None,
    ):
        name = NEURAL_MODELS.get(name, name)
        if name not in NEURAL_MODELS.values():
            raise ValueError(f"Unknown neural model {name}")
        if historical_covariate_cols and name not in HIST_MODELS:
            raise ValueError(f"{name} does not support historical covariates")
        if future_covariate_cols and name not in FUTURE_MODELS:
            raise ValueError(f"{name} does not support future covariates")
        if h < 1:
            raise ValueError("h must be positive")
        self.name, self.h, self.config = name, h, deepcopy(config)
        self.alias = alias or name
        self.historical_covariate_cols = tuple(historical_covariate_cols)
        self.future_covariate_cols = tuple(future_covariate_cols)
        self.trainer_kwargs = dict(trainer_kwargs or {})
        input_size = self.config.get("input_size", 3 * h)
        if not isinstance(input_size, int) or input_size < 1:
            raise ValueError("Declare a positive fixed input_size")
        context = input_size if context_length is None else context_length
        if context < input_size:
            raise ValueError("context_length cannot be shorter than input_size")
        # Static variables are intentionally outside the data contract.
        for key in ("hist_exog_list", "futr_exog_list", "stat_exog_list", "h", "alias"):
            if key in self.config:
                raise ValueError(
                    f"Configure {key} through the adapter, not model config"
                )
        if self.config.get("early_stop_patience_steps", -1) != -1:
            raise ValueError(
                "Use fixed max_steps; early stopping needs a training-only split"
            )
        self.capabilities = ModelCapabilities(
            uses_historical_covariates=bool(self.historical_covariate_cols),
            uses_future_covariates=bool(self.future_covariate_cols),
            requires_future_covariates_in_history=bool(self.future_covariate_cols),
            required_future_covariate_cols=self.future_covariate_cols,
            min_history=input_size,
            context_length=context,
        )

    def get_params(self):
        return dict(
            name=self.name,
            h=self.h,
            config=deepcopy(self.config),
            historical_covariate_cols=self.historical_covariate_cols,
            future_covariate_cols=self.future_covariate_cols,
            context_length=self.capabilities.context_length,
            trainer_kwargs=self.trainer_kwargs,
        )

    def minimum_training_history(self, h):
        return self.capabilities.min_history + h

    def fit_frozen(self, window, freq):
        from neuralforecast import NeuralForecast, models
        from neuralforecast.losses.pytorch import MAE

        if horizon(window) != self.h:
            raise ValueError("Neural model horizon differs from pipeline h")
        history = model_history(
            window, self.historical_covariate_cols, self.future_covariate_cols
        )
        config = deepcopy(self.config)
        config.setdefault("input_size", self.capabilities.min_history)
        if self.name != "DeepAR":
            config.setdefault("loss", MAE())
        config.setdefault("random_seed", 1)
        trainer = dict(
            logger=False, enable_checkpointing=False, enable_progress_bar=False
        )
        trainer.update(self.trainer_kwargs)
        params = dict(h=self.h, alias=self.alias, **config, **trainer)
        if self.historical_covariate_cols:
            params["hist_exog_list"] = list(self.historical_covariate_cols)
        if self.future_covariate_cols:
            params["futr_exog_list"] = list(self.future_covariate_cols)
        model = getattr(models, self.name)(**params)
        fitted = NeuralForecast(models=[model], freq=freq)
        fitted.fit(df=history, val_size=0)
        return FittedNeuralForecast(self, fitted, history)

    def forecast(self, window, freq):
        # Training uses the whole safe prefix even if prediction context is short.
        return self.fit_frozen(window, freq).forecast(window, freq)


class FittedNeuralForecast:
    def __init__(self, source, fitted, history):
        self.alias, self.capabilities = source.alias, source.capabilities
        self.source, self.fitted = source, fitted
        self.audit = ModelAudit()
        self.audit.fitted(history)

    def forecast(self, window, freq):
        if horizon(window) != self.source.h:
            raise ValueError("Cannot change horizon after fitting a neural model")
        history = model_history(
            window,
            self.source.historical_covariate_cols,
            self.source.future_covariate_cols,
        )
        prediction = self.fitted.predict(
            df=history, futr_df=model_future(window, self.source.future_covariate_cols)
        )
        self.audit.predictions.append(dict(origin=window.origin, h=self.source.h))
        return point_output(prediction, self.alias)


class AutoNeuralForecastAdapter:
    """Retain original Auto config spaces/Optuna, replace one-split objective.

    Each trial's fixed configuration is trained/scored on identical inner
    rolling folds. The winning ordinary model is returned without weights;
    validation and test each fit it once. No trial is trained on held-out data.
    """

    def __init__(
        self,
        name,
        *,
        h,
        config=None,
        num_samples=10,
        alias=None,
        historical_covariate_cols=(),
        future_covariate_cols=(),
        context_length=None,
        trainer_kwargs=None,
        random_seed=1,
    ):
        if name not in NEURAL_MODELS:
            raise ValueError(f"Unknown Auto neural model {name}")
        if h < 1 or num_samples < 1:
            raise ValueError("h and num_samples must be positive")
        base = NEURAL_MODELS[name]
        if historical_covariate_cols and base not in HIST_MODELS:
            raise ValueError(f"{name} does not support historical covariates")
        if future_covariate_cols and base not in FUTURE_MODELS:
            raise ValueError(f"{name} does not support future covariates")
        self.name, self.h, self.config = name, h, config
        self.alias = alias or name
        self.num_samples, self.random_seed = num_samples, random_seed
        self.historical_covariate_cols = tuple(historical_covariate_cols)
        self.future_covariate_cols = tuple(future_covariate_cols)
        self.context_length, self.trainer_kwargs = context_length, trainer_kwargs
        self.inner_n_windows, self.inner_step_size = 1, h
        self.metric, self.seasonality = "mae", 1
        self.selection_log = []
        # Default native spaces explore input_size up to 5*h.
        self.capabilities = ModelCapabilities(
            min_history=context_length or 5 * h,
            uses_historical_covariates=bool(historical_covariate_cols),
            uses_future_covariates=bool(future_covariate_cols),
            requires_future_covariates_in_history=bool(future_covariate_cols),
            required_future_covariate_cols=self.future_covariate_cols,
            tuning_mode="native",
        )

    def minimum_training_history(self, h):
        return self.capabilities.min_history + h

    def tune(self, window, freq):
        import optuna
        from neuralforecast import auto

        from ..backtesting import rolling_cross_validate

        auto_class = getattr(auto, self.name)
        if self.config is None:
            space = auto_class.get_default_config(h=self.h, backend="optuna")
        elif callable(self.config):
            space = self.config
        else:
            # Supports original Ray search dictionaries through the same converter.
            space = auto_class._ray_config_to_optuna(deepcopy(self.config))
        history = model_history(
            window, self.historical_covariate_cols, self.future_covariate_cols
        )
        covariates = tuple(
            dict.fromkeys(
                (*self.historical_covariate_cols, *self.future_covariate_cols)
            )
        )
        data = ForecastData(
            history,
            freq,
            historical_covariate_cols=covariates,
            future_covariate_cols=self.future_covariate_cols,
            future_covariates=(
                history[[*KEY_COLUMNS, *self.future_covariate_cols]]
                if self.future_covariate_cols
                else None
            ),
        )
        candidates = {}

        def objective(trial):
            config = deepcopy(space(trial))
            for key in ("h", "loss", "valid_loss"):
                if config.get(key) is None:
                    config.pop(key, None)
            candidate = NeuralForecastAdapter(
                self.name,
                h=self.h,
                config=config,
                alias=self.alias,
                historical_covariate_cols=self.historical_covariate_cols,
                future_covariate_cols=self.future_covariate_cols,
                context_length=self.context_length,
                trainer_kwargs=self.trainer_kwargs,
            )
            result = rolling_cross_validate(
                data,
                [candidate],
                validation_start=history.ds.min(),
                validation_end=history.ds.max(),
                h=self.h,
                n_windows=self.inner_n_windows,
                step_size=self.inner_step_size,
                seasonality=self.seasonality,
            )
            result.best_model(self.metric)  # Reject a partially successful trial.
            candidates[trial.number] = candidate
            return float(result.summary.iloc[0][f"mean_{self.metric}"])

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        study.optimize(objective, n_trials=self.num_samples)
        selected = candidates[study.best_trial.number]
        self.study = study
        self.selection_log = [
            dict(
                origin=window.origin,
                params=selected.get_params(),
                score=study.best_value,
                inner_n_windows=self.inner_n_windows,
            )
        ]
        return selected

    def forecast(self, window, freq):
        return self.tune(window, freq).forecast(window, freq)
