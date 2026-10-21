# Energy TSF Harness Agent Guide

This repository is operated by an external host Agent through natural language. It does not register a primary or child Agent in code.

## Language selection

- Use the language explicitly requested by the user. Otherwise, reply in the user's language.
- The canonical Chinese task instructions are `.agents/skills/<name>/SKILL.md`; English counterparts are `.agents/skills/<name>/SKILL.en.md`. Load the matching version for the active task. The English sidecar must keep the same rules and command names as the canonical file.
- Use `README.md` for Chinese and `README.en.md` for English. Keep code identifiers, CLI commands, serialized field names, enum values, and machine-readable JSON stable in English; never translate them.
- Human-readable Markdown run reports are written as `report.md` (Chinese) and `report.en.md` (English). Use the version matching the user's language; both files describe the same execution. Machine-readable JSON/CSV schemas and field names remain language-neutral.

## Task routing

The repository supports four top-level tasks:

- When the user asks to build, compare, or publish a model, load the `modeling` skill in the selected language.
- When the user asks for a future forecast, load the `forecast` skill in the selected language.
- When the user asks to evaluate an existing model on recent actual output and update ensemble weights, load the `backtest` skill in the selected language.
- When the user asks to rerun AutoML and retrain using recent data, load the `update` skill in the selected language.

All natural-language requests must first become structured Requests. Deterministic code performs numeric calculations, training, inference, metrics, constraints, and asset publication. The forecast track may read only Active assets. Modeling, backtest, and update publication must write a Candidate and pass the corresponding gate. Backtest only adjusts weights of existing members; update retrains from the SearchPool bound to Active and does not expand the search space beyond that pool.
