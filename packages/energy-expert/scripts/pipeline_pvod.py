"""Full PVOD pipeline: outer selection -> frozen params -> one fit -> rolling test.

Stages
------
1. Outer model selection + framework hyperparameter search on a validation span
   (the 10% of the timeline just before the test span).
2. Freeze the selected parameters at the last validation origin.
3. Test phase via ``rolling_test_forecast`` with ``weight_policy='frozen'``:
   weights are fitted ONCE at the first test origin, then every test origin
   updates the history inputs only.

Audits
------
A. fit-once  : count ``fit_frozen`` / ``_fit_ridge`` calls during the test
               phase (expect exactly 1 / h calls, all inside that one fit).
B. fit border: the single training window must end at the first test origin
               and must not carry held-out actuals.
C. no future y: perturbing true power AFTER the first test origin must not
               change the fitted weights nor the first origin's predictions
               (later origins may legitimately differ: observed test targets
               enter their input histories).

Usage:
    python scripts/pipeline_pvod.py [path/to/station00.csv]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eneygyexpert.backtesting import SeasonalNaiveWindowModel, rolling_cross_validate
from eneygyexpert.data import ForecastData
from eneygyexpert.test_rolling import rolling_test_forecast
from eneygyexpert.tuning import GridTunedWindowModel, direct_ridge_grid
from eneygyexpert.window_models import DirectRidgeWindowModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_pvod import load_pvod_station  # noqa: E402

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "dataset" / "PVOD" / "station00.csv"

H = 96                 # 1 天 @15min
STEP = 96              # 逐天滚动
SEASONALITY = 96
CONTEXT_LENGTH = 96 * 7  # DirectRidge 上下文 1 周
OUTER_WINDOWS = 5
INNER_WINDOWS = 3


class FitAuditor:
    """Count fit_frozen/_fit_ridge calls and capture the fitted weights."""

    def __init__(self, candidate: DirectRidgeWindowModel):
        self.fit_calls = 0
        self.ridge_calls = 0
        self.training_windows: list[pd.DataFrame] = []
        self.fitted_weights = None
        self._install(candidate)

    def _install(self, candidate: DirectRidgeWindowModel) -> None:
        auditor = self
        orig_fit = candidate.fit_frozen

        def counting_fit(window, freq):
            auditor.fit_calls += 1
            assert window.actuals.empty, "fit_frozen received held-out actuals"
            auditor.training_windows.append(window.history)
            fitted = orig_fit(window, freq)
            auditor.fitted_weights = fitted.fitted
            return fitted

        candidate.fit_frozen = counting_fit
        orig_ridge = DirectRidgeWindowModel._fit_ridge

        @staticmethod
        def counting_ridge(x, y, alpha):
            auditor.ridge_calls += 1
            return orig_ridge(x, y, alpha)

        DirectRidgeWindowModel._fit_ridge = counting_ridge


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    data = load_pvod_station(csv_path)
    dates = pd.DatetimeIndex(sorted(data.history.ds.unique()))
    n = len(dates)
    test_start_pos = int(n * 0.90)
    test_start, test_end = dates[test_start_pos], dates[-1]
    validation_end = dates[test_start_pos - 1]
    validation_start = dates[int(n * 0.80)]
    first_test_origin = dates[test_start_pos - 1]

    print(f"pipeline on {csv_path.name}  ({n} points @15min)")
    print(f"  validation span : {validation_start} -> {validation_end}")
    print(f"  test span       : {test_start} -> {test_end}")

    # ------------------------------------------------------------------
    # Stage 1: outer selection with framework-internal tuning (nested CV)
    # ------------------------------------------------------------------
    grid = direct_ridge_grid(
        historical_covariate_cols=tuple(data.historical_covariate_cols),
        future_covariate_cols=tuple(data.future_covariate_cols),
        lags=(2, 4),
        context_lengths=(CONTEXT_LENGTH,),
        alphas=(0.1, 1.0),
        lower_bound=0.0,
    )
    tuned = GridTunedWindowModel(
        grid,
        alias="TunedDirectRidge",
        inner_h=H,
        inner_n_windows=INNER_WINDOWS,
        inner_step_size=STEP,
        metric="mae",
        seasonality=SEASONALITY,
    )
    candidates = [SeasonalNaiveWindowModel(SEASONALITY), tuned]

    t0 = time.perf_counter()
    outer = rolling_cross_validate(
        data,
        candidates,
        validation_start=validation_start,
        validation_end=validation_end,
        h=H,
        n_windows=OUTER_WINDOWS,
        step_size=STEP,
        seasonality=SEASONALITY,
    )
    print(f"\n[stage 1] outer selection done in {time.perf_counter() - t0:.1f}s")
    print(outer.summary.to_string(index=False))
    assert outer.failures.empty, f"outer failures:\n{outer.failures}"
    best = outer.best_model("mae")
    print(f"  best model on validation (MAE): {best}")

    decision = tuned.selection_log[-1]
    print(f"  final tuning origin : {decision.outer_origin}")
    print(f"  selected candidate  : {decision.selected_candidate}")
    print(f"  selected params     : {decision.selected_params}")

    # ------------------------------------------------------------------
    # Stage 2: freeze parameters at the last validation origin
    # ------------------------------------------------------------------
    fixed = tuned.frozen_candidate(decision.outer_origin)
    print(f"\n[stage 2] parameters frozen at {decision.outer_origin}")

    # ------------------------------------------------------------------
    # Stage 3+4: rolling test with frozen weights + audits
    # ------------------------------------------------------------------
    auditor = FitAuditor(fixed.selected)
    t0 = time.perf_counter()
    clean = rolling_test_forecast(
        data,
        fixed,
        test_start=test_start,
        test_end=test_end,
        h=H,
        step_size=STEP,
        seasonality=SEASONALITY,
        weight_policy="frozen",
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[stage 3+4] rolling frozen test done in {elapsed:.1f}s")
    print(clean.summary.to_string(index=False))

    # ---- audit A: fitted exactly once ----
    assert auditor.fit_calls == 1, f"fit_frozen called {auditor.fit_calls} times"
    assert auditor.ridge_calls == H, f"_fit_ridge called {auditor.ridge_calls} times"
    print(f"\n[audit A] fit_frozen x{auditor.fit_calls}, _fit_ridge x{auditor.ridge_calls} "
          f"(= 1 fit x {H} leads) -> weights fitted exactly once")

    # ---- audit B: training window border ----
    train_window = auditor.training_windows[0]
    assert train_window.ds.max() == first_test_origin, "fit window extends past first origin"
    assert train_window.ds.max() <= validation_end
    print(f"[audit B] training window: [{train_window.ds.min()} -> {train_window.ds.max()}] "
          f"({len(train_window)} rows, ends at first test origin, no held-out actuals)")

    # ---- audit C: future true power cannot leak ----
    perturbed_history = data.history.copy()
    future_mask = perturbed_history.ds > first_test_origin
    perturbed_history.loc[future_mask, "y"] = (
        perturbed_history.loc[future_mask, "y"] * 1000.0 + 100.0
    )
    perturbed = ForecastData(
        history=perturbed_history,
        freq=data.freq,
        historical_covariate_cols=data.historical_covariate_cols,
        future_covariate_cols=data.future_covariate_cols,
        future_covariates=data.future_covariates,
    )
    auditor2 = FitAuditor(fixed.selected)
    leaked = rolling_test_forecast(
        perturbed,
        fixed,
        test_start=test_start,
        test_end=test_end,
        h=H,
        step_size=STEP,
        seasonality=SEASONALITY,
        weight_policy="frozen",
    )

    uid = data.history.unique_id.iloc[0]
    a = auditor.fitted_weights[uid]
    b = auditor2.fitted_weights[uid]
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, b)), "weights changed"
    print("[audit C] weights identical under future-y perturbation")

    first_origin_rows = clean.forecasts.loc[clean.forecasts.origin == first_test_origin]
    first_origin_leak = leaked.forecasts.loc[leaked.forecasts.origin == first_test_origin]
    assert np.array_equal(
        first_origin_rows.y_pred.to_numpy(), first_origin_leak.y_pred.to_numpy()
    ), "first-origin predictions changed"
    print("[audit C] first-origin predictions identical under future-y perturbation "
          "(no future true power was read)")

    later = clean.forecasts.loc[clean.forecasts.origin > first_test_origin, "y_pred"]
    later_leak = leaked.forecasts.loc[leaked.forecasts.origin > first_test_origin, "y_pred"]
    changed = (~np.isclose(later.to_numpy(), later_leak.to_numpy())).mean()
    assert changed > 0, "later origins did not update with observed history"
    print(f"[audit C] {changed:.0%} of later-origin predictions changed as expected "
          "(observed test targets legitimately enter later input histories)")

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    print("\nper-lead error (head/tail):")
    print(clean.by_lead.head(3).to_string(index=False))
    print(clean.by_lead.tail(3).to_string(index=False))
    print(f"\nper-origin MAE: mean={clean.scores.mae.mean():.4f} "
          f"std={clean.scores.mae.std():.4f} origins={clean.scores.origin.nunique()}")
    print("\n[OK] full pipeline verified: selection -> frozen params -> one fit -> "
          "multi-window prediction")


if __name__ == "__main__":
    main()
