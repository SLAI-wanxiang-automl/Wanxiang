import pandas as pd
import pytest

from eneygyexpert.data import ForecastData


def sample_data() -> ForecastData:
    history = pd.DataFrame(
        {
            "unique_id": ["wind", "wind", "wind", "wind"],
            "ds": pd.date_range("2025-01-01", periods=4, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0],
            "observed_wind": [3.0, 4.0, 5.0, 6.0],
        }
    )
    future = pd.DataFrame(
        {
            "unique_id": ["wind", "wind"],
            "ds": pd.to_datetime(["2025-01-03", "2025-01-04"]),
            "weather_forecast": [7.0, 8.0],
        }
    )
    return ForecastData(
        history=history,
        freq="D",
        historical_covariate_cols=("observed_wind",),
        future_covariate_cols=("weather_forecast",),
        future_covariates=future,
    )


def test_window_separates_future_targets_from_model_inputs():
    window = sample_data().window("2025-01-02", h=2)
    assert window.history["y"].tolist() == [1.0, 2.0]
    assert window.history["observed_wind"].tolist() == [3.0, 4.0]
    assert window.future_covariates["weather_forecast"].tolist() == [7.0, 8.0]
    assert "y" not in window.future_covariates
    assert window.actuals["y"].tolist() == [3.0, 4.0]


def test_window_requires_complete_future_covariates():
    data = sample_data()
    data.future_covariates = data.future_covariates.iloc[:1]
    with pytest.raises(ValueError, match="cover every series"):
        data.window("2025-01-02", h=2)


def test_duplicate_keys_rejected():
    data = sample_data()
    duplicated = pd.concat([data.history, data.history.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        ForecastData(history=duplicated, freq="D")


def test_univariate_window_remains_available():
    data = sample_data()
    univariate = ForecastData(history=data.history[["unique_id", "ds", "y"]], freq="D")
    window = univariate.window("2025-01-02", h=2)
    assert window.future_covariates.columns.tolist() == ["unique_id", "ds"]
    assert len(window.actuals) == 2


def test_same_covariate_name_can_be_historical_and_future_known():
    data = sample_data()
    data.history = data.history.rename(columns={"observed_wind": "wind_speed"})
    data.future_covariates = data.future_covariates.rename(
        columns={"weather_forecast": "wind_speed"}
    )
    data.historical_covariate_cols = ("wind_speed",)
    data.future_covariate_cols = ("wind_speed",)
    data = ForecastData(
        history=data.history,
        freq="D",
        historical_covariate_cols=data.historical_covariate_cols,
        future_covariate_cols=data.future_covariate_cols,
        future_covariates=data.future_covariates,
    )
    window = data.window("2025-01-02", h=2)
    assert window.history["wind_speed"].tolist() == [3.0, 4.0]
    assert window.future_covariates["wind_speed"].tolist() == [7.0, 8.0]


def test_context_length_limits_each_series_history():
    window = sample_data().window("2025-01-02", h=2, context_length=1)
    assert window.history["y"].tolist() == [2.0]
    assert window.future_covariates["weather_forecast"].tolist() == [7.0, 8.0]
    assert window.actuals["y"].tolist() == [3.0, 4.0]


def test_invalid_context_length_rejected():
    with pytest.raises(ValueError, match="context_length must be positive"):
        sample_data().window("2025-01-02", h=2, context_length=0)


@pytest.mark.parametrize("invalid_target", [float("inf"), float("-inf"), "bad"])
def test_nonfinite_or_nonnumeric_targets_rejected(invalid_target):
    frame = sample_data().history.astype({"y": object})
    frame.loc[0, "y"] = invalid_target
    with pytest.raises(ValueError, match="finite numeric values"):
        ForecastData(history=frame, freq="D")
