"""Backend-independent contracts and real small-model integration tests."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from eneygyexpert.data import ForecastData
from eneygyexpert.rolling_models import (
    FoundationAdapter,
    MedianEnsembleAdapter,
    SKTimeRollingAdapter,
    create_model,
    model_catalog,
)
from eneygyexpert.test_rolling import rolling_test_forecast
from eneygyexpert.window_models import forecast_from_data


def sample(n=100, covariates=False):
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    y = 5 + np.sin(np.arange(n) * np.pi / 4) + 0.01 * np.arange(n)
    frame = pd.DataFrame(dict(unique_id="pv", ds=dates, y=y))
    if not covariates:
        return ForecastData(frame, "h"), dates
    frame["lmd"] = np.arange(n) / n
    frame["nwp"] = np.cos(np.arange(n) / 3)
    return ForecastData(
        frame,
        "h",
        historical_covariate_cols=("lmd", "nwp"),
        future_covariate_cols=("nwp",),
        future_covariates=frame[["unique_id", "ds", "nwp"]],
    ), dates


def test_catalog_and_covariate_rejection():
    names = set(model_catalog().name)
    assert len(names) == 37  # 8 ML + 5 neural + 11 stats + Prophet + 12 foundation.
    assert {"AutoTFT", "AutoNHITS", "AutoPatchTST", "PatchTSTFM"} <= names
    with pytest.raises(ValueError, match="historical covariates"):
        create_model("AutoDeepAR", h=4, historical_covariate_cols=("lmd",))
    with pytest.raises(ValueError, match="future covariates"):
        create_model("AutoPatchTST", h=4, future_covariate_cols=("nwp",))


class FakeFoundation:
    alias = "backend"
    finetuning_config = None
    reuse_loaded_model = False

    def __init__(self):
        self.calls = []

    def forecast(self, df, h, freq):
        assert df.columns.tolist() == ["unique_id", "ds", "y"]
        self.calls.append(df.ds.max())
        return pd.DataFrame(
            dict(
                unique_id="pv",
                ds=pd.date_range(df.ds.max(), periods=h + 1, freq=freq)[1:],
                backend=1.0,
            )
        )


def test_foundation_zero_shot_and_ensemble():
    data, dates = sample(covariates=True)
    backend = FakeFoundation()
    model = FoundationAdapter(
        "Chronos", model=backend, alias="Chronos", context_length=16
    )
    result = rolling_test_forecast(
        data, model, test_start=dates[80], test_end=dates[-1], h=4
    )
    assert result.forecasts.origin.nunique() == 5
    assert backend.calls == list(dates[79:96:4])
    assert backend.reuse_loaded_model
    backend.finetuning_config = object()
    with pytest.raises(ValueError, match="fine-tuning"):
        model.fit_frozen(data.window(dates[79], 4).model_input(), "h")


@pytest.mark.parametrize(
    "name", ["AutoARIMA", "AutoETS", "AutoCES", "Theta", "DynamicOptimizedTheta"]
)
def test_real_state_models_no_refit_and_theta_season_phase(name, monkeypatch):
    pytest.importorskip("statsforecast")
    data, dates = sample()
    params = (
        {"order": (1, 0, 0), "seasonal_order": (0, 0, 0)}
        if name == "AutoARIMA"
        else (
            {"model": "AAA"}
            if name == "AutoETS"
            else ({"model": "S"} if name == "AutoCES" else {})
        )
    )
    model = create_model(name, tuning=False, params=params, season_length=8)
    # Non-season-aligned prefix verifies frozen seasonal pattern rotation.
    window = data.window(dates[74], 4).model_input()
    fitted = model.fit_frozen(window, "h")
    before = pickle.dumps(fitted.models["pv"].model_)
    initial = fitted.forecast(window, "h")[model.alias].to_numpy()
    expected = fitted.models["pv"].predict(h=4)["mean"]
    np.testing.assert_allclose(initial, expected, rtol=1e-6, atol=1e-6)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Parameter fitting was repeated during rolling state update"
        )

    monkeypatch.setattr(fitted.models["pv"], "fit", forbidden)
    for pos in [78, 78, 82]:
        fitted.forecast(data.window(dates[pos], 4).model_input(), "h")
    assert pickle.dumps(fitted.models["pv"].model_) == before
    assert len(fitted.audit.fits) == 1
    assert [event["rows"] for event in fitted.audit.updates] == [4, 4]
    with pytest.raises(ValueError, match="move forward"):
        fitted.forecast(window, "h")


@pytest.mark.parametrize(
    "name",
    [
        "ADIDA",
        "IMAPA",
        "CrostonClassic",
        "HistoricAverage",
        "SeasonalNaive",
        "ZeroModel",
    ],
)
def test_real_frozen_statistical_models(name):
    pytest.importorskip("statsforecast")
    data, dates = sample()
    model = create_model(name, tuning=False, season_length=8)
    fitted = model.fit_frozen(data.window(dates[79], 4).model_input(), "h")
    first = fitted.forecast(data.window(dates[79], 4).model_input(), "h")
    later = fitted.forecast(data.window(dates[83], 4).model_input(), "h")
    assert np.isfinite(first[model.alias]).all()
    assert len(fitted.audit.fits) == 1
    if name != "SeasonalNaive":
        np.testing.assert_array_equal(first[model.alias], later[model.alias])


def test_real_native_stats_structure_selection_and_mixed_ensemble():
    pytest.importorskip("statsforecast")
    from eneygyexpert.pipeline import train_validation_test

    data, dates = sample()
    native = create_model(
        "AutoARIMA",
        h=4,
        season_length=8,
        num_samples=2,
        params={"max_p": 1, "max_q": 1, "max_P": 0, "max_Q": 0, "seasonal": False},
    )
    native.inner_n_windows = 2
    native.inner_step_size = 4
    selected = native.tune(data.window(dates[59], 4).model_input(), "h")
    assert "order" in selected.per_series_params["pv"]
    ensemble = MedianEnsembleAdapter(
        [selected, create_model("SeasonalNaive", season_length=8)]
    )
    assert ensemble.capabilities.update_mode == "state_update"
    result = train_validation_test(data, [ensemble], h=4, min_train_size=16)
    assert result.test.summary.iloc[0].weight_policy == "state_update"


def test_sktime_guard_rejects_hidden_refit():
    class Fake:
        def fit(self, y, fh):
            self.coefficient = 1.0
            self.last = y.index[-1]

        def get_fitted_params(self):
            return {"coefficient": self.coefficient}

        def update(self, y, update_params):
            assert update_params is False
            self.coefficient += 1

        def predict(self, fh):
            return np.ones(len(fh))

    data, dates = sample()
    model = SKTimeRollingAdapter(Fake(), parameter_names=("coefficient",))
    fitted = model.fit_frozen(data.window(dates[79], 4).model_input(), "h")
    with pytest.raises(ValueError, match="changed learned parameters"):
        fitted.update(data.history.loc[data.history.ds <= dates[83]], "h")


@pytest.mark.parametrize(
    "name", ["AutoPatchTST", "AutoNBEATS", "AutoNHITS", "AutoTFT", "AutoDeepAR"]
)
def test_real_neural_frozen_weights_and_covariates(name, monkeypatch):
    pytest.importorskip("neuralforecast")
    import torch

    data, dates = sample(covariates=True)
    params = dict(
        input_size=16,
        max_steps=1,
        batch_size=4,
        windows_batch_size=8,
        random_seed=1,
        scaler_type="standard",
    )
    extras = {
        "AutoPatchTST": dict(
            hidden_size=8, n_heads=2, encoder_layers=1, patch_len=4, stride=2
        ),
        "AutoNBEATS": dict(n_blocks=[1, 1, 1], mlp_units=[[8, 8], [8, 8], [8, 8]]),
        "AutoNHITS": dict(n_blocks=[1, 1, 1], mlp_units=[[8, 8], [8, 8], [8, 8]]),
        "AutoTFT": dict(hidden_size=8, n_head=2),
        "AutoDeepAR": dict(lstm_hidden_size=8, lstm_n_layers=1, trajectory_samples=10),
    }
    params.update(extras[name])
    historical = ("lmd",) if name in ("AutoNHITS", "AutoTFT") else ()
    future = ("nwp",) if name in ("AutoNHITS", "AutoTFT", "AutoDeepAR") else ()
    model = create_model(
        name,
        h=4,
        tuning=False,
        params=params,
        historical_covariate_cols=historical,
        future_covariate_cols=future,
        trainer_kwargs={"accelerator": "cpu", "devices": 1},
    )
    fitted = model.fit_frozen(data.window(dates[79], 4).model_input(), "h")
    weights = {
        key: value.clone()
        for key, value in fitted.fitted.models[0].state_dict().items()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen neural model was retrained")

    monkeypatch.setattr(fitted.fitted, "fit", forbidden)
    first = forecast_from_data(fitted, data, dates[79], 4)
    later = forecast_from_data(fitted, data, dates[83], 4)
    assert first.shape == later.shape == (4, 3)
    assert np.isfinite(first[model.alias]).all()
    for key, value in fitted.fitted.models[0].state_dict().items():
        assert torch.equal(value, weights[key])
    assert len(fitted.audit.fits) == 1


def test_real_neural_native_space_common_folds():
    pytest.importorskip("neuralforecast")
    from eneygyexpert.pipeline import train_validation_test

    def space(trial):
        return dict(
            input_size=8,
            max_steps=1,
            n_blocks=[1, 1, 1],
            mlp_units=[[4, 4], [4, 4], [4, 4]],
            windows_batch_size=4,
            learning_rate=trial.suggest_categorical("learning_rate", [0.001, 0.01]),
        )

    model = create_model(
        "AutoNHITS",
        h=4,
        params=space,
        num_samples=1,
        context_length=8,
        trainer_kwargs={"accelerator": "cpu", "devices": 1},
    )
    data, dates = sample(n=60)
    result = train_validation_test(data, [model], h=4, step_size=8, min_train_size=20)
    assert result.inner_n_windows == 2
    assert result.selected_model == "AutoNHITS"
    assert result.fixed_models["AutoNHITS"].config["input_size"] == 8


def test_real_ml_native_and_frozen_lags(monkeypatch):
    pytest.importorskip("mlforecast")
    from mlforecast.auto import AutoRidge

    from eneygyexpert.native_automl import AutoMLForecastWindowAdapter
    from eneygyexpert.pipeline import train_validation_test

    model = AutoMLForecastWindowAdapter(
        AutoRidge(config=lambda trial: {"alpha": 0.1}),
        alias="AutoRidge",
        season_length=4,
        num_samples=1,
        context_length=24,
    )
    data, dates = sample(n=100)
    result = train_validation_test(data, [model], h=4, step_size=8, min_train_size=28)
    assert result.inner_n_windows == 4
    assert result.test.forecasts.shape[0] == 12
    assert result.fixed_models["AutoRidge"].get_params()["model_params"]["alpha"] == 0.1


def test_real_prophet_frozen_regressors(monkeypatch):
    pytest.importorskip("prophet")
    data, dates = sample(covariates=True)
    model = create_model(
        "Prophet",
        tuning=False,
        future_covariate_cols=("nwp",),
        params={
            "daily_seasonality": False,
            "weekly_seasonality": False,
            "yearly_seasonality": False,
            "n_changepoints": 0,
        },
    )
    fitted = model.fit_frozen(data.window(dates[79], 4).model_input(), "h")
    coefficients = deepcopy(fitted.models["pv"].params)

    def forbidden(*args, **kwargs):
        raise AssertionError("Prophet was refitted")

    monkeypatch.setattr(fitted.models["pv"], "fit", forbidden)
    for pos in [79, 83]:
        prediction = fitted.forecast(data.window(dates[pos], 4).model_input(), "h")
        assert np.isfinite(prediction[model.alias]).all()
    for name, value in coefficients.items():
        np.testing.assert_array_equal(value, fitted.models["pv"].params[name])


@pytest.mark.parametrize("name", ["AutoETS", "AutoCES"])
def test_real_native_smoothing_structure_is_frozen(name):
    pytest.importorskip("statsforecast")
    data, dates = sample()
    model = create_model(name, h=4, season_length=8, num_samples=2)
    model.inner_n_windows = 2
    model.inner_step_size = 4
    selected = model.tune(data.window(dates[59], 4).model_input(), "h")
    settings = selected.per_series_params["pv"]
    assert "Z" not in settings["model"]
    result = rolling_test_forecast(
        data, selected, test_start=dates[80], test_end=dates[-1], h=4
    )
    assert len(result.fitted_model.audit.fits) == 1
    assert len(result.fitted_model.audit.updates) == 4


@pytest.mark.parametrize(
    "name",
    [
        "AutoLinearRegression",
        "AutoRidge",
        "AutoLasso",
        "AutoElasticNet",
        "AutoRandomForest",
        "AutoLGBM",
        "AutoXGBoost",
        "AutoCatboost",
    ],
)
def test_real_all_ml_backends_frozen(name, monkeypatch):
    pytest.importorskip("mlforecast")
    optional = {
        "AutoLGBM": "lightgbm",
        "AutoXGBoost": "xgboost",
        "AutoCatboost": "catboost",
    }
    if name in optional:
        pytest.importorskip(optional[name])
    settings = {
        "AutoLinearRegression": {},
        "AutoRidge": {"alpha": 0.1},
        "AutoLasso": {"alpha": 0.01},
        "AutoElasticNet": {"alpha": 0.01, "l1_ratio": 0.5},
        "AutoRandomForest": {"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
        "AutoLGBM": {
            "n_estimators": 5,
            "num_leaves": 4,
            "min_child_samples": 2,
            "n_jobs": 1,
            "verbosity": -1,
        },
        "AutoXGBoost": {"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
        "AutoCatboost": {
            "iterations": 5,
            "depth": 2,
            "thread_count": 1,
            "verbose": False,
        },
    }
    model = create_model(name, season_length=4, num_samples=1, context_length=24)
    model.auto_model.config = lambda trial: settings[name]
    model.inner_n_windows = 2
    model.inner_step_size = 4
    data, dates = sample()
    selected = model.tune(data.window(dates[79], 4).model_input(), "h")
    fitted = selected.fit_frozen(data.window(dates[79], 4).model_input(), "h")

    def forbidden(*args, **kwargs):
        raise AssertionError("MLForecast model was retrained")

    monkeypatch.setattr(fitted.fitted, "fit", forbidden)
    for pos in [79, 83]:
        prediction = forecast_from_data(fitted, data, dates[pos], 4)
        assert prediction.shape == (4, 3)
        assert np.isfinite(prediction[name]).all()


def test_reports_and_checkpoint_roundtrip(tmp_path):
    import json

    import joblib

    from eneygyexpert.pipeline import train_validation_test
    from eneygyexpert.rolling_models.report import (
        save_fitted_model,
        save_pipeline_result,
    )

    pytest.importorskip("statsforecast")
    data, dates = sample()
    result = train_validation_test(data, [create_model("HistoricAverage")], h=4)
    save_pipeline_result(result, tmp_path)
    metadata = json.loads((tmp_path / "selection.json").read_text())
    assert metadata["selected_model"] == result.selected_model
    path = save_fitted_model(result.test.fitted_model, tmp_path / "model.joblib")
    restored = joblib.load(path)
    window = data.window(dates[95], 4).model_input()
    pd.testing.assert_frame_equal(
        restored.forecast(window, "h"), result.test.fitted_model.forecast(window, "h")
    )


def test_neural_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("neuralforecast")
    import joblib

    from eneygyexpert.rolling_models.report import save_fitted_model

    data, dates = sample()
    model = create_model(
        "AutoNHITS",
        h=4,
        tuning=False,
        params={
            "input_size": 8,
            "max_steps": 1,
            "n_blocks": [1, 1, 1],
            "mlp_units": [[4, 4], [4, 4], [4, 4]],
            "windows_batch_size": 4,
        },
        trainer_kwargs={"accelerator": "cpu", "devices": 1},
    )
    window = data.window(dates[79], 4).model_input()
    fitted = model.fit_frozen(window, "h")
    path = save_fitted_model(fitted, tmp_path / "neural.joblib")
    restored = joblib.load(path)
    pd.testing.assert_frame_equal(
        restored.forecast(window, "h"), fitted.forecast(window, "h")
    )


def test_only_consumed_future_variables_require_archives():
    from eneygyexpert.window_models import ModelCapabilities

    dates = pd.date_range("2024", periods=20, freq="h")
    history = pd.DataFrame(dict(unique_id="pv", ds=dates, y=1.0, nwp_a=2.0))
    future = pd.DataFrame(dict(unique_id="pv", ds=dates, nwp_a=2.0, nwp_b=3.0))
    data = ForecastData(
        history,
        "h",
        historical_covariate_cols=("nwp_a",),
        future_covariate_cols=("nwp_a", "nwp_b"),
        future_covariates=future,
    )
    capabilities = ModelCapabilities(
        uses_future_covariates=True,
        requires_future_covariates_in_history=True,
        required_future_covariate_cols=("nwp_a",),
    )
    capabilities.validate(data, 4)
