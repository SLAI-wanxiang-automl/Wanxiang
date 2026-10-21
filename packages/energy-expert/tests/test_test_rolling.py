import numpy as np
import pandas as pd

from eneygyexpert.data import ForecastData
from eneygyexpert.test_rolling import rolling_test_forecast
from eneygyexpert.window_models import ModelCapabilities


def test_test_origins_update_only_with_observed_values():
    dates = pd.date_range("2025-01-01", periods=28, freq="h")
    frame = pd.DataFrame({"unique_id": "pv", "ds": dates, "y": np.arange(28)})
    data = ForecastData(frame, "h")

    class LastValueModel:
        alias = "last_value"
        capabilities = ModelCapabilities()

        def forecast(self, window, freq):
            del freq
            last = window.history.y.iloc[-1]
            output = window.future_covariates.copy()
            output[self.alias] = last
            return output

    result = rolling_test_forecast(
        data,
        LastValueModel(),
        test_start=dates[16],
        test_end=dates[27],
        h=4,
        step_size=4,
        seasonality=4,
        weight_policy="refit",
    )
    assert result.summary.n_origins.iloc[0] == 3
    assert result.forecasts.groupby("origin").y_pred.first().tolist() == [15, 19, 23]
    assert result.forecasts.shape[0] == 12
    assert result.by_lead.lead.tolist() == [1, 2, 3, 4]


def test_bounds_clip_predictions_before_scoring():
    dates = pd.date_range("2025-01-01", periods=20, freq="h")
    frame = pd.DataFrame({"unique_id": "pv", "ds": dates, "y": np.abs(np.arange(20) - 10) * 1.0})
    data = ForecastData(frame, "h")

    class NegativeModel:
        alias = "negative"
        capabilities = ModelCapabilities()

        def forecast(self, window, freq):
            del freq
            output = window.future_covariates.copy()
            output[self.alias] = -5.0
            return output

    clipped = rolling_test_forecast(
        data,
        NegativeModel(),
        test_start=dates[12],
        test_end=dates[19],
        h=4,
        step_size=4,
        weight_policy="refit",
        bounds=(0.0, None),
    )
    assert (clipped.forecasts.y_pred == 0.0).all()
    raw = rolling_test_forecast(
        data,
        NegativeModel(),
        test_start=dates[12],
        test_end=dates[19],
        h=4,
        step_size=4,
        weight_policy="refit",
    )
    assert (raw.forecasts.y_pred == -5.0).all()
    assert clipped.forecasts.absolute_error.max() < raw.forecasts.absolute_error.max()


def test_overlapping_test_horizons_keep_origin_and_lead():
    dates = pd.date_range("2025-01-01", periods=28, freq="h")
    frame = pd.DataFrame({"unique_id": "pv", "ds": dates, "y": np.arange(28)})
    data = ForecastData(frame, "h")

    class LastValueModel:
        alias = "last_value"
        capabilities = ModelCapabilities()

        def forecast(self, window, freq):
            output = window.future_covariates.copy()
            output[self.alias] = window.history.y.iloc[-1]
            return output

    result = rolling_test_forecast(
        data,
        LastValueModel(),
        test_start=dates[16],
        test_end=dates[27],
        h=8,
        step_size=4,
        weight_policy="refit",
    )
    assert result.summary.n_origins.iloc[0] == 2
    assert result.forecasts.shape[0] == 16
    assert result.forecasts.ds.duplicated().sum() == 4


def test_frozen_weights_fit_once_and_use_fresh_history():
    dates = pd.date_range("2025-01-01", periods=28, freq="h")
    frame = pd.DataFrame({"unique_id": "pv", "ds": dates, "y": np.arange(28)})
    data = ForecastData(frame, "h")

    class FrozenPredictor:
        alias = "frozen"
        capabilities = ModelCapabilities()

        def forecast(self, window, freq):
            output = window.future_covariates.copy()
            output[self.alias] = window.history.y.iloc[-1]
            return output

    class FitOnceModel:
        alias = "frozen"
        capabilities = ModelCapabilities()
        fit_calls = 0

        def fit_frozen(self, window, freq):
            assert window.actuals.empty
            self.fit_calls += 1
            return FrozenPredictor()

    model = FitOnceModel()
    result = rolling_test_forecast(
        data, model, test_start=dates[16], test_end=dates[27], h=4
    )
    assert model.fit_calls == 1
    assert result.forecasts.groupby("origin").y_pred.first().tolist() == [15, 19, 23]
    assert result.summary.weight_policy.iloc[0] == "frozen"
