"""Rolling backtest on the PVOD dataset.

Uses the last 10% of the timeline as the validation span. Windows forecast
h=96 points (one day ahead at 15-min frequency) with step_size=96 (daily
origins), matching ``rolling_cross_validate`` semantics.

Usage:
    python scripts/backtest_pvod.py [path/to/station00.csv]
"""

import sys
import time
from pathlib import Path

import pandas as pd

from eneygyexpert.backtesting import SeasonalNaiveWindowModel, rolling_cross_validate
from eneygyexpert.data import ForecastData
from eneygyexpert.window_models import DirectRidgeWindowModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_pvod import load_pvod_station  # noqa: E402

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "dataset" / "PVOD" / "station00.csv"

H = 96          # 预测步长：96 个 15min 点 = 1 天
STEP = 96       # 滚动步长：逐天一个 origin
VALID_FRACTION = 0.10  # 最后 10% 时间轴作为验证段
CONTEXT_LENGTH = 96 * 7  # DirectRidge 上下文：1 周


def build_validation_span(data: ForecastData) -> tuple[int, int, int]:
    """Return (validation_start_pos, end_pos, n_windows) over the last 10%."""
    dates = pd.DatetimeIndex(sorted(data.history.ds.unique()))
    end_pos = len(dates) - 1
    start_pos = int(len(dates) * (1 - VALID_FRACTION))
    # origin_pos = end_pos - h - i * step >= start_pos - 1
    n_windows = (end_pos - H - (start_pos - 1)) // STEP + 1
    return start_pos, end_pos, max(n_windows, 1)


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    data = load_pvod_station(csv_path)
    dates = pd.DatetimeIndex(sorted(data.history.ds.unique()))
    start_pos, end_pos, n_windows = build_validation_span(data)
    validation_start, validation_end = dates[start_pos], dates[end_pos]

    print(f"backtest on {csv_path.name}")
    print(f"  validation : {validation_start} -> {validation_end}  "
          f"({end_pos - start_pos + 1} points = last {VALID_FRACTION:.0%})")
    print(f"  windows    : {n_windows} x h={H}, step_size={STEP}")

    models = [
        SeasonalNaiveWindowModel(seasonality=H),
        DirectRidgeWindowModel(
            historical_covariate_cols=tuple(data.historical_covariate_cols),
            future_covariate_cols=tuple(data.future_covariate_cols),
            lags=4,
            context_length=CONTEXT_LENGTH,
            lower_bound=0.0,
        ),
    ]

    t0 = time.perf_counter()
    result = rolling_cross_validate(
        data,
        models,
        validation_start=validation_start,
        validation_end=validation_end,
        h=H,
        n_windows=n_windows,
        step_size=STEP,
        seasonality=H,
    )
    elapsed = time.perf_counter() - t0

    print(f"[OK] rolling_cross_validate finished in {elapsed:.1f}s")
    print(f"  forecasts  : {len(result.forecasts)} rows")
    print(f"  scores     : {len(result.scores)} rows")
    if result.failures.empty:
        print("  failures   : none")
    else:
        print("  failures   :")
        print(result.failures.to_string(index=False))

    print("\nsummary (per model):")
    print(result.summary.to_string(index=False))
    print(f"\nbest model by MAE : {result.best_model('mae')}")
    print(f"best model by RMSE: {result.best_model('rmse')}")
    try:
        print(f"best model by MASE: {result.best_model('mase')}")
    except ValueError as exc:
        print(f"best model by MASE: unavailable ({exc})")

    out_dir = Path(__file__).resolve().parent.parent / "experiments" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.forecasts.to_csv(out_dir / "pvod_backtest_forecasts.csv", index=False)
    result.scores.to_csv(out_dir / "pvod_backtest_scores.csv", index=False)
    print(f"\nsaved: {out_dir / 'pvod_backtest_forecasts.csv'}")
    print(f"saved: {out_dir / 'pvod_backtest_scores.csv'}")


if __name__ == "__main__":
    main()
