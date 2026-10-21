"""Recompute AutoPatchTST test forecasts from its saved tuned configuration.

Needed because the clip-consistent validation ranking promotes AutoPatchTST
into the top-5, but the original run only saved test forecasts for the
original top-5. Same protocol as the pipeline test phase: fit_frozen once on
full history at the first test origin, then predict at every origin.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/run_patchtst_test.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "examples")
from pvod_full_experiment import build_data  # noqa: E402
from pvod_split_pipeline import load_station  # noqa: E402
from eneygyexpert.test_rolling import rolling_test_forecast  # noqa: E402
from eneygyexpert.rolling_models.neural import NeuralForecastAdapter  # noqa: E402

CSV = "dataset/PVOD/station00.csv"
OUT = Path("outputs/pvod_full/main/test_AutoPatchTST.csv")


def main() -> None:
    saved = json.loads(
        Path("outputs/pvod_full/main/selection.json").read_text()
    )["models"]["AutoPatchTST"]["params"]
    config = saved["config"]
    model = NeuralForecastAdapter(
        "AutoPatchTST",
        h=saved["h"],
        config=config,
        alias="AutoPatchTST",
        historical_covariate_cols=saved["historical_covariate_cols"],
        future_covariate_cols=saved["future_covariate_cols"],
        context_length=saved["context_length"],
        trainer_kwargs=saved["trainer_kwargs"],
    )
    data, _, _ = build_data(CSV)
    log = json.loads(Path("outputs/pvod_full/experiment_log.json").read_text())
    boundaries = log["boundaries"]

    result = rolling_test_forecast(
        data,
        model,
        test_start=boundaries["test_start"],
        test_end=boundaries["test_end"],
        h=96,
        step_size=96,
        seasonality=96,
    )
    result.forecasts.to_csv(OUT, index=False)
    print(f"saved {OUT} | mae={result.summary.mae.iloc[0]:.4f} "
          f"rmse={result.summary.rmse.iloc[0]:.4f}")


if __name__ == "__main__":
    main()
