"""Recompute PVOD results with a global non-negative clip, without retraining.

All model forecasts were saved as CSVs. Clipping only changes negative
predictions, so every metric is re-derived post-hoc:

1. Clip each model's validation and test ``y_pred`` at zero.
2. Recompute MAE / RMSE / MASE and by-lead tables.
3. Recompute ensemble weights from clipped validation MAEs (inverse-MAE rule
   from ``prediction_ensemble``) and rebuild Median / Weighted ensembles
   from clipped member forecasts.
4. Verify the validation top-5 ranking is unchanged.
5. Write an updated experiment_log.json + forecast CSVs to a new directory,
   preserving the original (unclipped) results.

Usage:
    python scripts/apply_nonneg_clip.py [--src outputs/pvod_full] [--dst outputs/pvod_full_clipped]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SEASONALITY = 96


def load_raw_y(csv_path: str) -> pd.Series:
    raw = pd.read_csv(csv_path).rename(columns={"date_time": "ds", "power": "y"})
    raw["ds"] = pd.to_datetime(raw["ds"])
    return (
        raw.drop_duplicates(subset=["ds"], keep="first")
        .sort_values("ds")
        .set_index("ds")["y"]
    )


class MaseScale:
    """scale = mean |y[t] - y[t-S]| over history <= origin (same as pipeline)."""

    def __init__(self, raw_y: pd.Series):
        self.dates = raw_y.index
        self.values = raw_y.to_numpy(dtype=float)
        self._cache: dict = {}

    def __call__(self, origin: pd.Timestamp) -> float:
        key = pd.Timestamp(origin)
        if key in self._cache:
            return self._cache[key]
        pos = self.dates.searchsorted(key, side="right")
        history = self.values[:pos]
        scale = (
            float(np.abs(history[SEASONALITY:] - history[:-SEASONALITY]).mean())
            if len(history) > SEASONALITY
            else np.nan
        )
        self._cache[key] = scale
        return scale


def recompute(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["y_pred"] = frame.y_pred.clip(lower=0.0)
    frame["absolute_error"] = (frame.y - frame.y_pred).abs()
    frame["squared_error"] = (frame.y - frame.y_pred) ** 2
    return frame


def summarize(frame: pd.DataFrame, model: str, scale_fn, weight_policy="frozen"):
    by_lead = frame.groupby("lead", as_index=False).agg(
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        n=("absolute_error", "size"),
    )
    by_lead["rmse"] = np.sqrt(by_lead.pop("mse"))
    mase_scores = []
    for origin, group in frame.groupby("origin"):
        scale = scale_fn(origin)
        mase_scores.append(group.absolute_error.mean() / scale
                           if np.isfinite(scale) and scale > 0 else np.nan)
    summary = dict(
        model=model,
        weight_policy=weight_policy,
        n_origins=int(frame.origin.nunique()),
        n_predictions=len(frame),
        mae=float(frame.absolute_error.mean()),
        rmse=float(np.sqrt(frame.squared_error.mean())),
        mean_mase=(float(np.mean(mase_scores))
                   if np.isfinite(mase_scores).all() else np.nan),
    )
    return summary, by_lead.to_dict("records")


def ensemble_weights(validation_mae: dict[str, float]) -> dict[str, float]:
    aliases = list(validation_mae)
    errors = np.array([validation_mae[a] for a in aliases], dtype=float)
    if (errors == 0).any():
        weights = (errors == 0).astype(float)
    else:
        weights = errors.min() / errors
    weights /= weights.sum()
    return dict(zip(aliases, weights.tolist(), strict=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="outputs/pvod_full")
    parser.add_argument("--dst", default="outputs/pvod_full_clipped")
    args = parser.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    log = json.loads((src / "experiment_log.json").read_text())
    scale_fn = MaseScale(load_raw_y(log["meta"]["csv"]))

    # ---------- validation: all candidates + baseline ----------
    val_summary = {}
    val_frames = {}
    for csv in sorted((src / "main").glob("validation_*.csv")):
        loaded = pd.read_csv(csv)
        if "y_pred" not in loaded.columns:
            continue  # skip summary files such as validation_scores.csv
        frame = recompute(loaded)
        model = frame.model.iloc[0]
        val_frames[model] = frame
        val_summary[model] = summarize(frame, model, scale_fn)[0]
    base_val = recompute(pd.read_csv(src / "baseline" / "validation_AutoARIMA.csv"))
    val_summary["AutoARIMA"] = summarize(base_val, "AutoARIMA", scale_fn,
                                         "state_update")[0]

    # ---------- clip-consistent selection (candidates exclude the baseline) ----------
    candidates_rank = [m for m, _ in sorted(val_summary.items(),
                                            key=lambda kv: kv[1]["mae"])
                       if m != "AutoARIMA"]
    new_top = candidates_rank[:5]
    print(f"clipped candidate ranking: {candidates_rank[:6]} ...")
    if new_top != log["selected_top_k"]:
        print(f"[note] top-5 changed after clipping: "
              f"{log['selected_top_k']} -> {new_top}")
        print("       AutoPatchTST test forecasts recomputed separately "
              "(scripts/run_patchtst_test.py).")
    log["selected_top_k"] = new_top

    # ---------- test: members + baseline ----------
    members = new_top
    test = pd.read_csv(src / "main" / "test_comparison_forecasts.csv")
    test = test.loc[~test.model.isin(["MedianEnsemble", "WeightedEnsemble"])]
    extra = pd.read_csv(src / "main" / "test_AutoPatchTST.csv")
    test = pd.concat([test, extra], ignore_index=True)
    clipped_members = {m: recompute(test.loc[test.model == m]) for m in members}
    baseline_test = recompute(pd.read_csv(src / "baseline" / "test_forecasts.csv"))
    old_mae = {alias: entry["summary"]["mae"]
               for alias, entry in log["test_results"].items()}
    old_mae.setdefault("AutoPatchTST",
                       float(extra.loc[extra.model == "AutoPatchTST",
                                      "absolute_error"].mean()))

    # ---------- ensembles from clipped members ----------
    weights = ensemble_weights({m: val_summary[m]["mae"] for m in members})
    keys = ["unique_id", "origin", "ds", "lead"]
    base = clipped_members[members[0]].sort_values(keys).reset_index(drop=True)
    matrix = np.stack([
        clipped_members[m].sort_values(keys).y_pred.to_numpy(dtype=float)
        for m in members
    ])
    weight_vector = np.array([weights[m] for m in members])
    ensembles = {}
    for alias, preds in (("MedianEnsemble", np.median(matrix, axis=0)),
                         ("WeightedEnsemble", weight_vector @ matrix)):
        frame = base.copy()
        frame["model"] = alias
        frame["y_pred"] = preds
        frame["absolute_error"] = (frame.y - frame.y_pred).abs()
        frame["squared_error"] = (frame.y - frame.y_pred) ** 2
        ensembles[alias] = frame

    all_test = {**clipped_members, **ensembles, "AutoARIMA": baseline_test}

    # ---------- update experiment log ----------
    log["meta"]["non_negative_clip"] = "y_pred clipped at 0 for all models post-hoc"
    log["ensemble_weights"] = weights
    log["leaderboard"] = [
        {**row, **{k: val_summary[row["model"]][k]
                   for k in ("mae", "rmse", "mean_mase")}}
        for row in log["leaderboard"]
    ]
    log["baseline"]["validation_summary"] = val_summary["AutoARIMA"]
    log["baseline"]["test_summary"] = summarize(
        baseline_test, "AutoARIMA", scale_fn, "state_update")[0]
    log["test_results"] = {
        alias: dict(zip(("summary", "by_lead"), summarize(
            frame, alias, scale_fn,
            "prediction_ensemble" if alias in ensembles else
            ("state_update" if alias == "AutoARIMA" else "frozen"))))
        for alias, frame in all_test.items()
    }
    log["plot"] = dict(log["plot"])
    windows = pd.DataFrame(log["plot"]["windows"])
    log["plot"]["digest"] = {
        f"{row.unique_id}@{row.origin}": {
            alias: {"mae": float(frame.loc[
                (frame.unique_id == row.unique_id)
                & (frame.origin == row.origin), "absolute_error"
            ].mean())}
            for alias, frame in all_test.items()
        }
        for row in windows.itertuples()
    }

    # ---------- write outputs ----------
    for sub in ("main", "baseline", "plots"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    for model, frame in val_frames.items():
        frame.to_csv(dst / "main" / f"validation_{model}.csv", index=False)
    pd.concat(
        [*clipped_members.values(), *ensembles.values()], ignore_index=True
    ).to_csv(dst / "main" / "test_comparison_forecasts.csv", index=False)
    baseline_test.to_csv(dst / "baseline" / "test_forecasts.csv", index=False)
    for extra in ("selection.json", "model_catalog.csv"):
        if (src / "main" / extra).exists():
            shutil.copy(src / "main" / extra, dst / "main" / extra)
    (dst / "experiment_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2, default=str)
    )

    # ---------- before/after report ----------
    orig_cmp = pd.read_csv(src / "main" / "test_comparison_forecasts.csv")
    extra_cmp = pd.read_csv(src / "main" / "test_AutoPatchTST.csv")
    neg_counts = {
        alias: int((grp.y_pred < 0).sum())
        for alias, grp in pd.concat(
            [orig_cmp, extra_cmp], ignore_index=True
        ).groupby("model")
    }
    neg_counts["AutoARIMA"] = int(
        (pd.read_csv(src / "baseline" / "test_forecasts.csv").y_pred < 0).sum()
    )
    rows = [
        dict(model=alias, mae_before=old_mae.get(alias, np.nan),
             mae_after=float(frame.absolute_error.mean()),
             neg_clipped=neg_counts.get(alias, 0))
        for alias, frame in all_test.items()
    ]
    report = pd.DataFrame(rows).sort_values("mae_after")
    print("\ntest MAE before/after clipping (neg_clipped = rows lifted to 0):")
    print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nnew ensemble weights: "
          f"{ {k: round(v, 4) for k, v in weights.items()} }")
    print(f"saved: {dst / 'experiment_log.json'}")


if __name__ == "__main__":
    main()
