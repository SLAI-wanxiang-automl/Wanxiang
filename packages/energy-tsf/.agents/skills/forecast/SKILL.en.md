---
name: forecast
description: Turn a request for a future renewable-energy forecast into an auditable configuration and run the read-only forecasting workflow.
---

# Forecast

## When to use

Use this skill when the user wants to forecast future output from recent observations using an existing solution. Use `modeling` if they ask to train, compare, or publish a model.

The host Agent converts natural language into a structured request. Data cleaning, asset retrieval, similarity scoring, scale conversion, inference, ensembling, and constraints are performed by deterministic code.

## Required inputs

- Data path, target column, and sampling frequency.
- Station ID, energy type, timezone, and target unit.
- Optional timestamp column; without one, deliver relative horizon steps only.
- For cross-station absolute-power transfer, target capacity and unit.
- Similarity-search metadata must be supplied by the user or registered station record; never guess it.

The initial horizon is fixed to `1d`. Similar-station retrieval is allowed by default when no same-station Active solution exists. Do not silently fall back across stations when a same-station solution exists but is not executable.

## Procedure

1. Confirm the task is forecasting, not modeling, backtesting, or updating.
2. Build a `ForecastRequest`; do not pass natural language to numeric code.
3. Prepare the request. Example:

   ```text
   energy-tsf prepare-forecast --request-id <id> --station-id <station> --energy-type solar --timezone Asia/Shanghai --data <path> --target power --target-unit MW --timestamp-column timestamp --frequency 10min --capacity <mw> --capacity-unit MW --output-dir work_dirs/forecast/<id>
   ```

4. Check `config_hash`, `resolved_config`, and warnings. Do not bypass validation.
5. Run `inspect-forecast`. Prefer a compatible same-station Active solution. If none exists, review compatible candidates, their evidence and rejection reasons, then select one to three sources and record short selection reasons.
6. Same-station plans are created automatically. For similar-station plans, write `selection.json` and run `energy-tsf plan-forecast --config <resolved.toml> --selection <selection.json> --workspace-root <project>`.
7. Run `energy-tsf run-forecast --config <resolved.toml> --workspace-root <project>`. Report the forecast, data quality, sources, degradation, constraints, and report path. Use `report.en.md` for English and `report.md` for Chinese.

## Constraints

- Forecast may read Active, immutable Solutions, model objects, and station metadata; it must not read Candidate.
- Do not fit, save a model, retune, refit ensemble weights, download weights, or modify Active.
- Similarity ranks only hard-compatible candidates; it cannot override frequency, target, unit, capacity, or context incompatibility.
- Cross-station absolute-power transfer requires explicit capacity conversion.
- Freeze concrete Solution IDs in the plan. An Active update during execution does not change the current forecast.
- Record selection facts and concise reasons, not hidden reasoning.

## Completion

Preparation creates `request.json`, task TOML, and `resolved.toml` whose hash is stable after reload. A complete run also produces the frozen plan, forecast table, machine report, human report, and traceable events. Use workflow status to determine completion.
