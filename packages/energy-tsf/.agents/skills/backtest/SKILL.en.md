---
name: backtest
description: Convert a request to evaluate an existing station model on recent actual output and update ensemble weights into a structured backtest configuration.
---

# Backtest

## When to use

Use this skill to evaluate an existing station model on recent actuals and update its ensemble weights. Use `forecast` for future predictions, `modeling` for initial training, and `update` for retraining or changing model members/search.

The host Agent interprets the request and calls the workflow. Deterministic code constructs windows and contexts, computes metrics, updates Bayesian weights, and applies release gates.

## Required inputs

- Station ID with an Active solution for the same target and frequency.
- CSV/Parquet actuals, target, timestamp column, frequency, and timezone.
- Treat all supplied values as test actuals; do not mix in unlabeled context data.

The initial horizon and stride are both fixed to `1d`. Publication is enabled by default. Use `--no-publish` when the user requests evaluation only.

## Procedure

1. Convert the request to `BacktestRequest`.
2. Run `prepare-backtest` to create the request and resolved configuration.
3. Run `inspect-backtest` to freeze the same-station Active solution and check the training snapshot and rolling windows. Stop if no same-station Active exists; do not search similar stations.
4. Run `run-backtest` for read-only window inference, metrics, and Bayesian weight update.
5. Report member metrics, before/after ensemble metrics, old/new weights, and publication result. Use `report.en.md` for English and `report.md` for Chinese.

## Constraints

- Each window context contains only records strictly earlier than its origin.
- Do not call fit/save or change model, parameters, features, or member set.
- Missing actuals do not contribute to metrics.
- New weights create a new Solution; never overwrite the parent.
- Before publication, Active must still point to the parent frozen in the plan.
- If the updated ensemble has worse overall RMSE, keep the report but do not switch Active.

## Completion

Produce a frozen plan, point predictions, metrics, weight evidence, machine and human reports, and result file. When publication is requested, create a new Solution/Candidate and update Active only after the gate passes.
