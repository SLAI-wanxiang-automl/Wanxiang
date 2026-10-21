import numpy as np
import pandas as pd

from eneygyexpert.data import ForecastData
from eneygyexpert.tuning import GridTunedWindowModel, direct_ridge_grid
from eneygyexpert.window_models import forecast_from_data


def make_data(future_target_shift: float = 0.0) -> ForecastData:
    dates = pd.date_range("2025-01-01", periods=100, freq="h")
    nwp = np.sin(np.arange(100) / 5)
    target = 10 + 2 * nwp
    target[92:] += future_target_shift
    frame = pd.DataFrame(
        {
            "unique_id": "pv",
            "ds": dates,
            "y": target,
            "lmd_temp": np.arange(100) / 100,
            "nwp_irrad": nwp,
        }
    )
    return ForecastData(
        frame,
        "h",
        ("lmd_temp", "nwp_irrad"),
        ("nwp_irrad",),
        frame[["unique_id", "ds", "nwp_irrad"]],
    )


def make_tuned_model() -> GridTunedWindowModel:
    grid = direct_ridge_grid(
        historical_covariate_cols=("lmd_temp",),
        future_covariate_cols=("nwp_irrad",),
        lags=(2, 4),
        context_lengths=(48,),
        alphas=(0.1, 1.0),
    )
    return GridTunedWindowModel(
        grid,
        alias="TunedDirectRidge",
        inner_h=8,
        inner_n_windows=2,
        inner_step_size=8,
        metric="mae",
    )


def test_nested_tuning_uses_only_past_targets():
    first = make_data()
    changed_test_targets = make_data(future_target_shift=1000)
    model_a = make_tuned_model()
    model_b = make_tuned_model()
    origin = first.history.ds.iloc[91]
    forecast_a = forecast_from_data(model_a, first, origin, 8)
    forecast_b = forecast_from_data(model_b, changed_test_targets, origin, 8)
    pd.testing.assert_frame_equal(forecast_a, forecast_b)
    assert model_a.selection_log[0].selected_params == model_b.selection_log[0].selected_params
    assert model_a.selection_log[0].inner_last_target <= origin
    assert model_a.capabilities.tuning_mode == "framework"
    assert model_a.frozen_candidate(origin).get_params() == model_a.selection_log[0].selected_params
