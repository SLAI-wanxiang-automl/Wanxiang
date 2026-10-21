"""PVOD full experiment: AutoARIMA baseline + all-candidate search + ensembles.

Stages
------
1. Baseline: AutoARIMA alone through the same 60/20/20 split.
2. Main: all candidate families (stats / ML / neural / Prophet) with train-only
   tuning, validation leaderboard, top-5 test, Median + inverse-MAE weighted
   prediction ensembles.
3. Plots: 5 random test windows with top-5, ensembles, and the AutoARIMA
   baseline overlaid.
4. A consolidated experiment_log.json for downstream LLM reporting.

Excluded by design: intermittent-demand models (Zero/Croston/ADIDA/IMAPA --
for sparse demand, not continuous solar power) and foundation models (need
zero-shot checkpoint downloads; add later if wanted).

Usage:
    python examples/pvod_full_experiment.py dataset/PVOD/station00.csv
"""

import argparse
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd

from eneygyexpert.data import ForecastData
from eneygyexpert.pipeline import train_validation_test
from eneygyexpert.rolling_models import create_model
from eneygyexpert.rolling_models.report import save_pipeline_result

from pvod_split_pipeline import load_station

ML_NAMES = (
    "AutoLinearRegression", "AutoRidge", "AutoLasso", "AutoElasticNet",
    "AutoRandomForest", "AutoLGBM", "AutoXGBoost", "AutoCatboost",
)
NEURAL_NAMES = ("AutoPatchTST", "AutoNBEATS", "AutoNHITS", "AutoTFT", "AutoDeepAR")
STATS_NAMES = ("SeasonalNaive", "HistoricAverage", "AutoETS", "AutoCES",
               "Theta", "DynamicOptimizedTheta")
CANDIDATES = STATS_NAMES + ML_NAMES + NEURAL_NAMES + ("Prophet",)


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return repr(value)


def build_data(csv: str):
    frame = load_station(csv)
    frame["unique_id"] = "PVOD"
    historical = tuple(c for c in frame if c.startswith("lmd_"))
    future = tuple(c for c in frame if c.startswith("nwp_"))
    data = ForecastData(
        frame, "15min",
        historical_covariate_cols=historical + future,
        future_covariate_cols=future,
        future_covariates=frame[["unique_id", "ds", *future]],
    )
    return data, historical, future


def make_model(name, *, h, season_length, context, num_samples, accelerator,
               historical, future):
    kwargs = {}
    if name in ("AutoNHITS", "AutoTFT"):
        kwargs.update(historical_covariate_cols=historical,
                      future_covariate_cols=future)
    elif name in ("AutoDeepAR", "Prophet"):
        kwargs["future_covariate_cols"] = future
    if name in NEURAL_NAMES:
        kwargs.update(context_length=context, num_samples=num_samples,
                      trainer_kwargs={"accelerator": accelerator, "devices": 1})
    elif name in ("AutoARIMA", "AutoETS", "AutoCES"):
        kwargs["num_samples"] = num_samples
    elif name in ML_NAMES:
        kwargs.update(context_length=context, num_samples=num_samples)
    return create_model(name, h=h, season_length=season_length, **kwargs)


def model_params(model):
    return jsonable(model.get_params()) if hasattr(model, "get_params") else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--h", type=int, default=96)
    parser.add_argument("--step", type=int, default=96)
    parser.add_argument("--context", type=int, default=672)
    parser.add_argument("--season-length", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--inner-windows", type=int, default=8,
                        help="Tuning windows inside train (controls runtime)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--output", default="outputs/pvod_full")
    parser.add_argument("--plot-windows", type=int, default=5)
    parser.add_argument("--plot-seed", type=int, default=42)
    parser.add_argument("--plot-context", type=int, default=192)
    args = parser.parse_args()

    data, historical, future = build_data(args.csv)
    dates = pd.DatetimeIndex(sorted(data.history.ds.unique()))
    train_n = int(len(dates) * 0.6)
    # 把内层调参窗口数压到 inner_windows：min_train_size = train_n - h - (w-1)*step
    min_train_size = max(
        train_n - args.h - (args.inner_windows - 1) * args.step,
        args.context + args.h,
    )

    log = {
        "meta": {
            "csv": args.csv, "n_points": len(dates), "freq": "15min",
            "h": args.h, "step": args.step, "season_length": args.season_length,
            "num_samples": args.num_samples, "inner_windows": args.inner_windows,
            "context": args.context, "top_k": args.top_k,
            "candidates": list(CANDIDATES) + ["AutoARIMA (baseline)"],
            "excluded": {
                "intermittent": ["ZeroModel", "CrostonClassic", "ADIDA", "IMAPA"],
                "foundation": "all (zero-shot checkpoints not downloaded)",
            },
            "started": pd.Timestamp.now().isoformat(),
        }
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- stage 1: AutoARIMA baseline ----------------
    print(f"[1/4] AutoARIMA baseline (inner windows={args.inner_windows})",
          flush=True)
    t0 = time.perf_counter()
    baseline_model = make_model(
        "AutoARIMA", h=args.h, season_length=args.season_length,
        context=args.context, num_samples=args.num_samples,
        accelerator=args.accelerator, historical=historical, future=future,
    )
    baseline = train_validation_test(
        data, [baseline_model], h=args.h, step_size=args.step,
        min_train_size=min_train_size, seasonality=args.season_length, top_k=1,
    )
    save_pipeline_result(baseline, out / "baseline")
    print(f"      done in {time.perf_counter() - t0:.0f}s | "
          f"validation mae={baseline.leaderboard.mae.iloc[0]:.4f} | "
          f"test mae={baseline.test.summary.mae.iloc[0]:.4f}", flush=True)
    log["baseline"] = {
        "model": "AutoARIMA",
        "best_params": model_params(baseline.fixed_models["AutoARIMA"]),
        "tuning_log": jsonable(baseline.tuning_logs.get("AutoARIMA", [])),
        "validation_summary": baseline.leaderboard.iloc[0].to_dict(),
        "test_summary": baseline.test.summary.iloc[0].to_dict(),
        "test_by_lead": baseline.test.by_lead.to_dict("records"),
    }

    # ---------------- stage 2: all candidates ----------------
    print(f"[2/4] {len(CANDIDATES)} candidates: tuning on train only "
          f"({args.inner_windows} inner windows x {args.num_samples} samples)",
          flush=True)
    models = [
        make_model(name, h=args.h, season_length=args.season_length,
                   context=args.context, num_samples=args.num_samples,
                   accelerator=args.accelerator, historical=historical,
                   future=future)
        for name in CANDIDATES
    ]
    t0 = time.perf_counter()
    main_result = train_validation_test(
        data, models, h=args.h, step_size=args.step,
        min_train_size=min_train_size, seasonality=args.season_length,
        top_k=args.top_k,
    )
    print(f"      done in {time.perf_counter() - t0:.0f}s", flush=True)
    log["boundaries"] = jsonable(main_result.boundaries)
    log["inner_n_windows"] = main_result.inner_n_windows
    log["leaderboard"] = main_result.leaderboard.to_dict("records")
    log["candidates"] = {
        alias: {
            "best_params": model_params(model),
            "tuning_log": jsonable(main_result.tuning_logs.get(alias, [])),
        }
        for alias, model in main_result.fixed_models.items()
    }
    log["selected_top_k"] = main_result.selected_models
    log["ensemble_weights"] = jsonable(main_result.ensemble_weights)

    # ---------------- stage 3: combined test + plots ----------------
    print("[3/4] test scores (top-k + ensembles + baseline)", flush=True)
    combined = dict(main_result.test_results)
    combined["AutoARIMA"] = baseline.test
    log["test_results"] = {
        alias: {
            "summary": result.summary.iloc[0].to_dict(),
            "by_lead": result.by_lead.to_dict("records"),
        }
        for alias, result in combined.items()
    }
    from eneygyexpert.rolling_models.plotting import plot_test_windows

    plot_path = plot_test_windows(
        data, combined, out / "plots", n_windows=args.plot_windows,
        seed=args.plot_seed, context_length=args.plot_context,
    )
    print(f"      plot saved: {plot_path}", flush=True)
    plotted = pd.read_csv(plot_path.parent / "plotted_test_windows.csv")
    log["plot"] = {
        "file": str(plot_path), "windows": plotted.to_dict("records"),
        "digest": {
            f"{row.unique_id}@{row.origin}": {
                alias: {
                    "mae": float(result.forecasts.loc[
                        (result.forecasts.unique_id == row.unique_id)
                        & (result.forecasts.origin == pd.Timestamp(row.origin)),
                        "absolute_error",
                    ].mean()),
                }
                for alias, result in combined.items()
            }
            for _, row in plotted.iterrows()
        },
    }
    save_pipeline_result(main_result, out / "main")

    # ---------------- stage 4: experiment log ----------------
    log["meta"]["finished"] = pd.Timestamp.now().isoformat()
    (out / "experiment_log.json").write_text(
        json.dumps(jsonable(log), ensure_ascii=False, indent=2)
    )
    print(f"[4/4] experiment log: {out / 'experiment_log.json'}", flush=True)

    print("\nleaderboard (validation):")
    print(main_result.leaderboard.to_string(index=False))
    print(f"\nselected top-{args.top_k}: {main_result.selected_models}")
    print(f"ensemble weights: {main_result.ensemble_weights}")
    print("\ntest comparison:")
    print(pd.concat(
        [r.summary for r in combined.values()], ignore_index=True
    ).to_string(index=False))


if __name__ == "__main__":
    main()
