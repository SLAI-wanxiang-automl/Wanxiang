import sys
import types

import pandas as pd

from eneygyexpert.data import ForecastData
from eneygyexpert.native_automl import AutoMLForecastWindowAdapter
from eneygyexpert.window_models import forecast_from_data


def test_native_automl_adapter_preserves_search_and_drops_covariates(monkeypatch):
    class FakeAutoMLForecast:
        def __init__(self, models, freq, season_length):
            self.freq = freq

        def fit(self, df, n_windows, step_size, refit, h, num_samples, loss):
            assert df.columns.tolist() == ["unique_id", "ds", "y"]
            assert (n_windows, h, num_samples) == (2, 4, 3)
            assert step_size is None
            assert refit is True
            self.last_date = df.ds.max()
            trial = types.SimpleNamespace(
                user_attrs={
                    "config": {
                        "model_params": {"alpha": 0.5},
                        "mlf_init_params": {"lags": [1]},
                        "mlf_fit_params": {},
                    }
                },
                value=0.2,
            )
            self.results_ = {"AutoRidge": types.SimpleNamespace(best_trial=trial)}

        def predict(self, h):
            return pd.DataFrame(
                {
                    "unique_id": "pv",
                    "ds": pd.date_range(self.last_date, periods=h + 1, freq=self.freq)[1:],
                    "AutoRidge": 1.0,
                }
            )

    monkeypatch.setitem(sys.modules, "mlforecast", types.ModuleType("mlforecast"))
    auto_module = types.ModuleType("mlforecast.auto")
    auto_module.AutoMLForecast = FakeAutoMLForecast
    monkeypatch.setitem(sys.modules, "mlforecast.auto", auto_module)

    class FakeEstimator:
        def set_params(self, **params):
            assert params == {"alpha": 0.5}
            return self

    class FakeMLForecast:
        def __init__(self, models, freq, lags):
            assert lags == [1]
            self.freq = freq

        def fit(self, df):
            assert df.columns.tolist() == ["unique_id", "ds", "y"]
            self.last_date = df.ds.max()

        def predict(self, h, new_df=None):
            assert new_df is not None
            return pd.DataFrame(
                {
                    "unique_id": "pv",
                    "ds": pd.date_range(self.last_date, periods=h + 1, freq=self.freq)[1:],
                    "AutoRidge": 1.0,
                }
            )

    sys.modules["mlforecast"].MLForecast = FakeMLForecast
    sklearn_module = types.ModuleType("sklearn")
    sklearn_module.__path__ = []
    sklearn_base = types.ModuleType("sklearn.base")
    sklearn_base.clone = lambda estimator: FakeEstimator()
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_module)
    monkeypatch.setitem(sys.modules, "sklearn.base", sklearn_base)
    dates = pd.date_range("2025-01-01", periods=20, freq="h")
    frame = pd.DataFrame(
        {"unique_id": "pv", "ds": dates, "y": range(20), "observed_temp": range(20)}
    )
    data = ForecastData(frame, "h", historical_covariate_cols=("observed_temp",))
    auto_model = types.SimpleNamespace(model=FakeEstimator())
    model = AutoMLForecastWindowAdapter(
        auto_model, alias="AutoRidge", season_length=4, inner_n_windows=2, num_samples=3
    )
    result = forecast_from_data(model, data, dates[15], 4)
    assert result.shape == (4, 3)
    assert model.capabilities.tuning_mode == "native"
    assert model.selection_log[0].best_config["model_params"]["alpha"] == 0.5
    frozen = model.frozen_candidate()
    frozen_result = forecast_from_data(frozen, data, dates[15], 4)
    pd.testing.assert_frame_equal(result, frozen_result)
    assert frozen.capabilities.tuning_mode == "fixed"
