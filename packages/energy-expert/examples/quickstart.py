"""Run a synthetic rolling experiment without LLM or pretrained weights."""
import numpy as np
import pandas as pd

from eneygyexpert.backtesting import SeasonalNaiveWindowModel
from eneygyexpert.data import ForecastData
from eneygyexpert.pipeline import train_validation_test
from eneygyexpert.rolling_models.report import save_pipeline_result


def main():
    frame = pd.DataFrame({
        "unique_id": "synthetic",
        "ds": pd.date_range("2025-01-01", periods=200, freq="D"),
        "y": np.tile([0., 1., 3., 5., 4., 2., 1.], 29)[:200],
    })
    result = train_validation_test(
        ForecastData(frame, freq="D"),
        [SeasonalNaiveWindowModel(7, "weekly"),
         SeasonalNaiveWindowModel(1, "daily")],
        h=7, step_size=7, seasonality=7, min_train_size=28,
    )
    print(result.leaderboard.to_string(index=False))
    print("Selected:", result.selected_model)
    print("Reports:", save_pipeline_result(result, "outputs/quickstart"))


if __name__ == "__main__":
    main()
