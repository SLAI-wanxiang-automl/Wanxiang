import numpy as np
import pandas as pd
import pytest

from eneygyexpert.backtesting import SeasonalNaiveWindowModel, rolling_cross_validate
from eneygyexpert.data import ForecastData
from eneygyexpert.window_models import DirectRidgeWindowModel


def test_multiple_windows_select_model_from_validation_only():
    dates = pd.date_range("2025-01-01", periods=100, freq="h")
    irradiance = np.sin(np.arange(100) / 5)
    frame = pd.DataFrame(
        {
            "unique_id": "pv",
            "ds": dates,
            "y": 10 + 2 * irradiance,
            "lmd_temp": np.arange(100) / 100,
            "nwp_irrad": irradiance,
        }
    )
    data = ForecastData(
        history=frame,
        freq="h",
        historical_covariate_cols=("lmd_temp", "nwp_irrad"),
        future_covariate_cols=("nwp_irrad",),
        future_covariates=frame[["unique_id", "ds", "nwp_irrad"]],
    )
    models = [
        SeasonalNaiveWindowModel(seasonality=24),
        DirectRidgeWindowModel(
            historical_covariate_cols=("lmd_temp",),
            future_covariate_cols=("nwp_irrad",),
            lags=4,
            context_length=48,
        ),
    ]
    result = rolling_cross_validate(
        data,
        models,
        validation_start=dates[68],
        validation_end=dates[91],
        h=8,
        n_windows=3,
        step_size=8,
        seasonality=24,
    )
    assert result.scores.shape[0] == 6  # two models x three origins x one series
    assert result.forecasts.shape[0] == 48  # two models x three origins x eight leads
    assert result.failures.empty
    assert result.best_model("mae") == "DirectRidge"
    assert result.forecasts.ds.max() == dates[91]
    assert dates[92] not in set(result.forecasts.ds)


def test_invalid_validation_span_rejected():
    dates = pd.date_range("2025-01-01", periods=40, freq="h")
    frame = pd.DataFrame({"unique_id": "pv", "ds": dates, "y": np.arange(40)})
    data = ForecastData(frame, "h")
    with pytest.raises(ValueError, match="extend before validation_start"):
        rolling_cross_validate(
            data,
            [SeasonalNaiveWindowModel(4)],
            validation_start=dates[35],
            validation_end=dates[39],
            h=4,
            n_windows=2,
            step_size=4,
        )
