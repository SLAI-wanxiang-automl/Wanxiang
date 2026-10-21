"""Run PVOD with train-only tuning and frozen validation/test weights."""
import argparse
import pandas as pd
from eneygyexpert.data import ForecastData
from eneygyexpert.backtesting import SeasonalNaiveWindowModel
from eneygyexpert.tuning import GridTunedWindowModel, direct_ridge_grid
from eneygyexpert.pipeline import train_validation_test


def load_station(csv_path: str) -> pd.DataFrame:
    """Read a PVOD station CSV into the contract schema on a complete grid.

    station00 原始数据有三处问题：列名为 ``date_time``（非 ``date``）、
    存在 1 个重复时间戳、时间轴在 2018-12 有 2 处缺口（缺 97 点）。
    ``train_validation_test`` 要求完整规则网格，这里去重后从最后一个
    缺口之后截取，不插值、不伪造数据。
    """
    frame = pd.read_csv(csv_path).rename(columns={'date_time': 'ds', 'power': 'y'})
    frame['ds'] = pd.to_datetime(frame['ds'])
    frame = frame.sort_values('ds').drop_duplicates(subset=['ds'], keep='first')
    step = pd.Timedelta('15min')
    after_gap = frame.loc[frame['ds'].diff() > step, 'ds']
    if len(after_gap):
        frame = frame.loc[frame['ds'] >= after_gap.iloc[-1]].reset_index(drop=True)
    grid = pd.date_range(frame['ds'].min(), frame['ds'].max(), freq='15min')
    assert pd.DatetimeIndex(frame['ds']).equals(grid), 'grid is not complete'
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--h', type=int, default=96)
    parser.add_argument('--step', type=int, default=96)
    args = parser.parse_args()
    frame = load_station(args.csv)
    frame['unique_id'] = 'PVOD'
    historical = tuple(c for c in frame if c.startswith('lmd_'))
    future = tuple(c for c in frame if c.startswith('nwp_'))
    data = ForecastData(frame, freq='15min',
                        historical_covariate_cols=historical + future,
                        future_covariate_cols=future,
                        future_covariates=frame[['unique_id', 'ds', *future]])
    tuned = GridTunedWindowModel(
        direct_ridge_grid(historical_covariate_cols=historical,
                          future_covariate_cols=future, lags=(2, 4),
                          context_lengths=(672,), alphas=(0.1, 1.0), lower_bound=0),
        alias='DirectRidgeTuned', inner_h=args.h, inner_step_size=args.step,
        seasonality=96,
    )
    result = train_validation_test(data, [SeasonalNaiveWindowModel(96), tuned],
                                   h=args.h, step_size=args.step, seasonality=96)
    print('Boundaries:', result.boundaries)
    print('Inner windows:', result.inner_n_windows)
    print(result.leaderboard.to_string(index=False))
    print('Selected:', result.selected_model)
    print(result.test.summary.to_string(index=False))


if __name__ == '__main__':
    main()
