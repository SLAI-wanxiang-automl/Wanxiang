# Validation — 2026-10-10

Two independent code-review subagents examined the alias collision and data/rolling/tuning boundaries.

Fixes:
- MedianEnsembleAdapter defaults to MedianModelEnsemble, avoiding reserved prediction-ensemble result names. Regression verifies that candidate and generated ensemble results coexist.
- ForecastData rejects nonnumeric and nonfinite targets, preventing invalid targets from contaminating evaluation.

Local validation (/opt/anaconda3/bin/python):
- data/window/backtesting/tuning/test_rolling/pipeline/native_automl: 29 passed.
- rolling_model_adapters excluding real Prophet: 4 passed, 31 skipped, 1 deselected (missing optional backends).
- Originally reported native StatsForecast integration case skipped because statsforecast is unavailable in this environment.
- Real Prophet test crashes in the local cmdstanpy native dependency; full backend verification is incomplete.
- Synthetic examples/quickstart.py completed and exported reports.

Initial publication excluded local datasets, experiment outputs, copied virtual environments and secrets. Original checkout has corrupt Git objects; publication uses a new source snapshot and preserves upstream attribution. Full legacy history is not recovered.

## Documentation, PVOD and LLM API update

- Included the owner-provided dataset/PVOD/station00.csv and documented its loader behavior (28,896 original rows; 15,802 continuous rows after cleaning).
- Ran the cleaned PVOD data through SeasonalNaiveWindowModel and the full train/validation/test pipeline: 32 test origins, 3,072 test predictions, test MAE 0.372719.
- Four isolated report tests passed: remote request URL/model/authentication, authenticated model discovery, explicit model without discovery, final prompt containing generated image paths.
- Remote API behavior was tested with mocked transport; no paid external model invocation was made.
