"""PVOD 60/20/20 with original model families and declared rolling policies."""

import argparse
from pathlib import Path

import pandas as pd

from eneygyexpert.data import ForecastData
from eneygyexpert.pipeline import train_validation_test
from eneygyexpert.rolling_models import create_model
from eneygyexpert.rolling_models.report import save_fitted_model, save_pipeline_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--models", nargs="+", default=["SeasonalNaive", "AutoRidge"])
    parser.add_argument("--h", type=int, default=96)
    parser.add_argument("--step", type=int, default=96)
    parser.add_argument("--context", type=int, default=672)
    parser.add_argument("--season-length", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--min-train-size", type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--output", default="outputs/pvod_models")
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot validation top-k models and their test ensembles",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of validation winners to evaluate on test",
    )
    parser.add_argument("--plot-windows", type=int, default=5)
    parser.add_argument("--plot-seed", type=int, default=42)
    parser.add_argument(
        "--plot-context",
        type=int,
        default=None,
        help="Displayed historical points; defaults to --context",
    )
    args = parser.parse_args()
    display_context = args.context if args.plot_context is None else args.plot_context
    if args.plot and (args.plot_windows < 1 or display_context < 1):
        parser.error("plot-windows and displayed context must be positive")
    from pvod_split_pipeline import load_station

    frame = load_station(args.csv)
    if "unique_id" not in frame:
        frame["unique_id"] = "PVOD"
    historical = tuple(c for c in frame if c.startswith("lmd_"))
    future = tuple(c for c in frame if c.startswith("nwp_"))
    data = ForecastData(
        frame,
        "15min",
        historical_covariate_cols=historical + future,
        future_covariate_cols=future,
        future_covariates=frame[["unique_id", "ds", *future]] if future else None,
    )
    models = []
    for name in args.models:
        kwargs = {}
        if name in ("AutoNHITS", "AutoTFT"):
            kwargs.update(
                historical_covariate_cols=historical, future_covariate_cols=future
            )
        elif name == "AutoDeepAR":
            kwargs["future_covariate_cols"] = future
        if name in ("AutoPatchTST", "AutoNBEATS", "AutoNHITS", "AutoTFT", "AutoDeepAR"):
            kwargs.update(
                context_length=args.context,
                num_samples=args.num_samples,
                trainer_kwargs={"accelerator": args.accelerator, "devices": 1},
            )
        elif name in ("AutoARIMA", "AutoETS", "AutoCES"):
            kwargs["num_samples"] = args.num_samples
        elif name in (
            "AutoLinearRegression",
            "AutoRidge",
            "AutoLasso",
            "AutoElasticNet",
            "AutoRandomForest",
            "AutoLGBM",
            "AutoXGBoost",
            "AutoCatboost",
        ):
            kwargs.update(context_length=args.context, num_samples=args.num_samples)
        models.append(
            create_model(name, h=args.h, season_length=args.season_length, **kwargs)
        )
    result = train_validation_test(
        data,
        models,
        h=args.h,
        step_size=args.step,
        min_train_size=args.min_train_size,
        seasonality=args.season_length,
        top_k=args.top_k,
    )
    output = save_pipeline_result(result, args.output)
    if args.plot:
        from eneygyexpert.rolling_models.plotting import plot_test_windows

        plot_test_windows(
            data,
            result.test_results,
            output,
            n_windows=args.plot_windows,
            seed=args.plot_seed,
            context_length=args.plot_context
            if args.plot_context is not None
            else args.context,
        )
        print("Test comparison:")
        print(
            pd.concat([r.summary for r in result.test_results.values()]).to_string(
                index=False
            )
        )
    if args.save_weights:
        save_fitted_model(
            result.test.fitted_model, Path(output) / "selected_model.joblib"
        )
    print("Boundaries:", result.boundaries)
    print("Inner windows:", result.inner_n_windows)
    print(result.leaderboard.to_string(index=False))
    print("Selected:", result.selected_models)
    print("Ensemble weights:", result.ensemble_weights)
    print(result.test.summary.to_string(index=False))
    print("Reports:", output)


if __name__ == "__main__":
    main()
