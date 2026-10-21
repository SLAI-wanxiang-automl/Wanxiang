"""Load PVOD station CSV into the ForecastData contract and verify it works.

Usage:
    python scripts/load_pvod.py [path/to/station00.csv]
"""

import sys
from pathlib import Path

import pandas as pd

from eneygyexpert.data import ForecastData

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "dataset" / "PVOD" / "station00.csv"


def load_pvod_station(csv_path: Path, station_id: str | None = None) -> ForecastData:
    """Read one PVOD station CSV and wrap it in ``ForecastData``.

    CSV schema: ``date_time`` + ``nwp_*`` / ``lmd_*`` covariates + ``power``.
    Mapped to contract:
    - ``y`` = power
    - ``lmd_*`` = historical covariates (实测值，仅历史可得)
    - ``nwp_*`` = future covariates (数值天气预报，预测时点已知)，
      同时归档进 ``historical_covariate_cols``——DirectRidge 类模型
      训练时需要历史帧中的 nwp 存档值 (requires_future_covariates_in_history)
    """
    raw = pd.read_csv(csv_path)
    station_id = station_id or csv_path.stem  # e.g. "station00"

    raw = raw.rename(columns={"date_time": "ds", "power": "y"})
    raw.insert(0, "unique_id", station_id)
    raw["ds"] = pd.to_datetime(raw["ds"])

    # PVOD 原始数据存在极少量重复时间戳（如 station00 的 2019-01-01 01:15:00）：
    # 契约要求 (unique_id, ds) 唯一，这里保留首次出现并记录丢弃数量
    n_before = len(raw)
    raw = raw.drop_duplicates(subset=["unique_id", "ds"], keep="first")
    if n_before != len(raw):
        print(f"[warn] dropped {n_before - len(raw)} duplicate timestamp rows")

    historical_cols = tuple(
        c for c in raw.columns if c.startswith(("lmd_", "nwp_"))
    )
    future_cols = tuple(c for c in raw.columns if c.startswith("nwp_"))

    history = raw.loc[:, ["unique_id", "ds", "y", *historical_cols]]
    future_covariates = raw.loc[:, ["unique_id", "ds", *future_cols]]

    return ForecastData(
        history=history,
        freq="15min",
        historical_covariate_cols=historical_cols,
        future_covariate_cols=future_cols,
        future_covariates=future_covariates,
    )


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    data = load_pvod_station(csv_path)

    h = data.history
    print(f"[OK] loaded: {csv_path.name}")
    print(f"     series      : {sorted(h.unique_id.unique())}")
    print(f"     rows        : {len(h)}")
    print(f"     time range  : {h.ds.min()}  ->  {h.ds.max()}")
    print(f"     freq        : {data.freq}")
    print(f"     historical  : {len(data.historical_covariate_cols)} cols (lmd_*)")
    print(f"     future      : {len(data.future_covariate_cols)} cols (nwp_*)")
    print(f"     y stats     : min={h.y.min():.4f} max={h.y.max():.4f} nan={h.y.isna().sum()}")

    # 通过 window() 做一次无泄漏切分，验证读取的数据可完整走通契约
    origin = h.ds.iloc[-49]  # 留最后 48 个点 (12h) 做预测期
    window = data.window(origin, h=48)
    print(f"[OK] window()    : origin={window.origin}, "
          f"history={len(window.history)}, actuals={len(window.actuals)}")


if __name__ == "__main__":
    main()
