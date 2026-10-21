"""Plot existing test predictions without fitting or changing model selection."""

from pathlib import Path

import numpy as np
import pandas as pd


def plot_test_windows(
    data, test_results, directory, *, n_windows=5, seed=42, context_length=672
):
    """Sample common (series, origin) windows; save PNG and sampled keys.

    context_length controls the displayed target history only, independently
    of each model's actual input length. Fewer available windows are all shown.
    """
    if n_windows < 1 or context_length < 1 or not test_results:
        raise ValueError("Positive window/context counts and test results are required")
    common = None
    for result in test_results.values():
        keys = set(
            zip(result.forecasts.unique_id, result.forecasts.origin, strict=True)
        )
        common = keys if common is None else common & keys
    keys = sorted(common, key=lambda key: (str(key[0]), key[1]))
    if not keys:
        raise ValueError("No common test windows to plot")
    indices = np.random.default_rng(seed).choice(
        len(keys), min(n_windows, len(keys)), replace=False
    )
    selected = [keys[i] for i in sorted(indices)]

    import matplotlib.dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(14, 3.4 * len(selected)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(selected), 1, squeeze=False)[:, 0]
    for ax, (uid, origin) in zip(axes, selected, strict=True):
        history = (
            data.history.loc[
                (data.history.unique_id == uid) & (data.history.ds <= origin)
            ]
            .sort_values("ds")
            .tail(context_length)
        )
        ax.plot(
            history.ds, history.y, color="0.4", label="Observed history", linewidth=1.2
        )
        truth = None
        for alias, result in test_results.items():
            forecast = result.forecasts.loc[
                (result.forecasts.unique_id == uid)
                & (result.forecasts.origin == origin)
            ].sort_values("ds")
            if truth is None:
                truth = forecast
            elif (
                not forecast[["ds", "y"]]
                .reset_index(drop=True)
                .equals(truth[["ds", "y"]].reset_index(drop=True))
            ):
                raise ValueError("Compared models must share horizons and test targets")
            ax.plot(forecast.ds, forecast.y_pred, label=alias, linewidth=1.4)
        ax.plot(
            truth.ds,
            truth.y,
            color="black",
            linestyle="--",
            label="Future actual",
            linewidth=1.5,
        )
        ax.axvline(origin, color="0.5", linestyle=":")
        ax.axvspan(truth.ds.iloc[0], truth.ds.iloc[-1], color="tab:blue", alpha=0.05)
        ax.set_title(
            f"{uid} | origin {pd.Timestamp(origin)} | "
            f"displayed history: {len(history)} points"
        )
        ax.set_ylabel("Power")
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.legend(loc="upper left", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time")
    path = directory / "test_windows.png"
    figure.savefig(path, dpi=160)
    pd.DataFrame(selected, columns=["unique_id", "origin"]).to_csv(
        directory / "plotted_test_windows.csv", index=False
    )
    return path
