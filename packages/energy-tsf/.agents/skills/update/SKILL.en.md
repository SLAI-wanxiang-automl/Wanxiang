---
name: update
description: Convert a request to rerun AutoML and retrain an existing station model from recent data into a pool-constrained update configuration.
---

# Update

## When to use

Use this skill when the user wants to retrain an existing station model using recent actual output, reselect members, and update the forecast solution. Use `modeling` to create the first model, `backtest` to evaluate and reweight the existing members, and `forecast` for inference only.

The host Agent turns natural language into `UpdateRequest`. Windowing, pool lookup, training, metrics, ensembling, and publication are handled by deterministic code.

## Required inputs

- Existing station ID with an Active solution for the same target, frequency, and horizon.
- Recent data path, target, timestamp column, frequency, and IANA timezone.
- The initial implementation fixes the latest seven days as validation and earlier data as training; horizon and stride are `1d`.

Publication is enabled by default. Use `--no-publish` for demonstration or evaluation without release. The Agent must not invent Providers, hyperparameters, or search-space entries; candidates come from the SearchPool bound to Active.

## Procedure

1. Build `UpdateRequest` and run `prepare-update`.
2. Run `inspect-update` to freeze Active, its SearchPool, the data snapshot, and the UpdatePlan.
3. Explain pool size, unavailable candidates, and the seven validation windows.
4. Run `run-update`; retrain only candidates from the bound pool and produce the next pool, Top-K, ensemble, and solution.
5. Report comparable parent/new metrics, gate outcome, and whether Active changed. Use `report.en.md` for English and `report.md` for Chinese.

## Constraints

- Do not search similar stations or read a station's latest pool; use only the pool bound to Active.
- Each validation window uses records strictly before its origin. Missing actuals do not contribute to metrics.
- The latest seven days are validation, not a hidden test set.
- If Active lacks a `search_pool_id`, stop and recommend running Modeling first.
- Before release, Active must still point to the plan's parent. If the new solution's seven-day RMSE is worse, retain the Candidate but do not publish.

## Completion

Produce a frozen plan, trial records, per-window leaderboard, next SearchPool, new Solution, report, and result. Switch Active atomically only after the release gate passes.
