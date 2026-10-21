---
name: modeling
description: Convert requests to build, train, compare, or select time-series models into an auditable configuration and run the deterministic modeling workflow.
---

# Modeling

## When to use

Use this skill when the user wants to build, train, compare, select, or publish a forecasting model from a renewable-energy time series. Use `forecast` when the user only wants a forecast from an already published model.

The repository does not register an internal Agent. The external host Agent interprets the request and creates structured input. Deterministic code performs data preparation, training, metrics, ranking, and publication.

## Required inputs

- Input CSV or Parquet path.
- One target column. Ask if multiple targets are plausible; do not guess.
- Sampling frequency and station ID.
- Optional timestamp column. Without one, preserve input row order.

Unless the user specifies otherwise, use the configured defaults: one-day horizon, chronological 70/20/10 split, Top-5, seed 42, up to 64 trials, two parallel trials, and all configured Providers. Publishing is enabled by default; use `--no-publish` when the user asks only to inspect candidates.

## Procedure

1. Confirm this is modeling, not forecasting, backtesting, or updating.
2. Resolve required fields; ask for missing target or frequency when ambiguous.
3. Create a request ID using letters, digits, dots, underscores, or hyphens.
4. Prepare the request and TOML configuration:

   ```text
   energy-tsf prepare-modeling --request-id <id> --station-id <station> --data <path> --target <column> --timestamp-column <column> --frequency <frequency> --output-dir work_dirs/train/<id>
   ```

5. Review `config_hash`, `resolved_config`, and warnings. Never bypass validation.
6. Run `inspect-modeling --config <resolved.toml> --workspace-root <project>` to inspect data, splits, available Providers, and the deterministic trial plan. Inspect does not train or create a Candidate.
7. When the user requests execution, run `run-modeling --config <resolved.toml> --workspace-root <project>`. Report data quality, trial successes/failures, validation ranking, test metrics, ensemble, and Candidate/Active status. Use `report.en.md` for English and `report.md` for Chinese.

## Constraints

- Select Top-K and fit ensemble weights using validation only. Test data must not influence selection.
- Deterministic code handles missing values and features; never invent numeric values.
- Training writes a Candidate first. Active changes only after the release gate passes.
- Foundation Providers must record model identity, fixed revision, context length, and local weight reference; do not copy large weights into the asset store.

## Completion

Preparation produces `request.json`, task TOML, and `resolved.toml` with a stable reload hash. A completed run is determined by workflow results and reports, not by the Agent's assertion.
