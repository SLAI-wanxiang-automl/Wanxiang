import numpy as np
import pandas as pd
import pytest

from eneygyexpert.data import ForecastData
from eneygyexpert.window_models import (
    DirectRidgeWindowModel,
    ModelCapabilities,
    UnivariateWindowAdapter,
    forecast_from_data,
)


def make_data() -> ForecastData:
    dates = pd.date_range("2025-01-01", periods=80, freq="h")
    nwp = np.sin(np.arange(80) / 5)
    history = pd.DataFrame(
        {
            "unique_id": "solar",
            "ds": dates,
            "y": 10 + 2 * nwp,
            "lmd_temp": np.arange(80) / 80,
            "nwp_irrad": nwp,
        }
    )
    return ForecastData(
        history,
        freq="h",
        historical_covariate_cols=("lmd_temp", "nwp_irrad"),
        future_covariate_cols=("nwp_irrad",),
        future_covariates=history[["unique_id", "ds", "nwp_irrad"]],
    )


def test_direct_ridge_uses_known_future_covariate_and_context():
    data = make_data()
    model = DirectRidgeWindowModel(
        historical_covariate_cols=("lmd_temp",),
        future_covariate_cols=("nwp_irrad",),
        lags=4,
        context_length=48,
        lower_bound=0,
    )
    result = forecast_from_data(model, data, data.history.ds.iloc[-9], h=8)
    assert result.shape == (8, 3)
    assert np.isfinite(result[model.alias]).all()
    assert np.mean(np.abs(result[model.alias] - data.history.y.iloc[-8:].to_numpy())) < 0.2


def test_direct_ridge_requires_archived_future_covariates():
    data = make_data()
    data.historical_covariate_cols = ("lmd_temp",)
    model = DirectRidgeWindowModel(
        historical_covariate_cols=("lmd_temp",),
        future_covariate_cols=("nwp_irrad",),
    )
    with pytest.raises(ValueError, match="past values of future covariates"):
        forecast_from_data(model, data, data.history.ds.iloc[-9], h=8)


def test_univariate_adapter_excludes_covariates():
    class SpyModel:
        alias = "spy"

        def forecast(self, df, h, freq):
            assert df.columns.tolist() == ["unique_id", "ds", "y"]
            dates = pd.date_range(df.ds.max(), periods=h + 1, freq=freq)[1:]
            return pd.DataFrame({"unique_id": "solar", "ds": dates, "spy": 1.0})

    data = make_data()
    result = forecast_from_data(
        UnivariateWindowAdapter(SpyModel(), context_length=12),
        data,
        data.history.ds.iloc[-9],
        h=8,
    )
    assert result.shape == (8, 3)


def test_model_cannot_read_held_out_targets():
    class VisibilitySpy:
        alias = "spy"
        capabilities = ModelCapabilities()

        def forecast(self, window, freq):
            assert window.actuals.empty
            output = window.future_covariates[["unique_id", "ds"]].copy()
            output[self.alias] = 1.0
            return output

    data = make_data()
    result = forecast_from_data(VisibilitySpy(), data, data.history.ds.iloc[-9], 8)
    assert result.shape == (8, 3)
