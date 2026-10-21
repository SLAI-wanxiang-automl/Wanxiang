[Chinese](README.md) | [English](README.en.md)

<div align="center">

# 🌙 Mingyue Wanxiang

### Agent Harness Framework for Renewable Energy Forecasting

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
![Renewable Energy Forecasting](https://img.shields.io/badge/task-renewable%20energy%20forecasting-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Mingyue Wanxiang** is an agentic framework for time-series forecasting at real-world renewable energy sites.
It combines natural-language Agent workflows with auditable AutoML execution to support model development, power forecasting, rolling backtests, and model updates.

</div>

---

## 🧭 Project overview

Renewable energy data arrives continuously, while site conditions and forecast errors change over time. A model trained once and left unchanged may not meet the needs of long-running operations. The Mingyue Wanxiang renewable energy forecasting framework is designed for this setting: an Agent turns user intent into a structured request, and deterministic workflows run models, validate results, and maintain traceable forecasting solutions.

Models and forecasting solutions can evolve as new data arrives. Rolling backtests evaluate the current solution and adjust ensemble weights; when retraining is needed, constrained updates use the existing candidate search pool. Every release must pass its gate, and a failed candidate does not replace the current Active solution.

> This framework uses a harness architecture. An external Agent such as Codex understands the request, fills in required details, invokes workflows, and uses Skills and tools to complete the task.

## ✨ Highlights

- 🤖 **Natural-language task entry** — start a workflow in plain language through a host Agent; workflow contracts constrain subsequent execution.
- 🔁 **Designed for long-running operation** — repeat forecasts, evaluate recent performance, and update models within a bounded search pool.
- 📊 **Four workflows** — modeling, forecasting, backtesting, and updating cover initial training through ongoing maintenance.
- 🧾 **Auditable execution** — configurations, input fingerprints, plans, model assets, release gates, and reports make each run traceable.
- 🛡️ **Controlled publication** — a candidate updates Active only after passing its release gate; forecasts read published solutions only.
- 🧩 **Multiple model Providers** — ARIMA, SARIMA, Ridge, LightGBM, XGBoost, TimesFM 2.5, Moirai 2, and Chronos 2.

## 🧰 Four workflows

| Workflow | When to use it | Execution boundary |
|---|---|---|
| **Modeling** | Build an initial forecasting solution for a site or compare models and ensembles | Bounded AutoML search; selects on validation and evaluates test after selection is frozen; release gate required |
| **Forecast** | Forecast future output using the currently published solution | Reads Active only; does not train or change published solutions |
| **Backtest** | Evaluate the current solution over rolling windows and recalibrate ensemble weights as new actual output arrives | Keeps existing model members and updates weights only; new solution is published after passing the release gate |
| **Update** | Retrain and reselect models when recent data indicates an update is needed | Retrains only candidates in the SearchPool bound to Active without expanding the search space; release gate required |

## 🏁 Quick start

### 1. Install

Install Git and Python 3.12 or later, and set up an Agent host such as Codex, Claude Code, Pi, or DeepSeek Harness.
For example, clone the repository and start Codex:

```bash
git clone https://github.com/keyingyang206/energyTSF.git
cd energyTSF
codex
```

For first-time use or to run the CLI directly, install the Python environment and project dependencies as described in the project configuration.

### 2. Open the repository in an Agent

Open the project directory in a host that supports repository-level Agent instructions, such as Codex. Then describe the task in plain language and provide the requested data. The examples below use the PVOD dataset and show **inputs to the Agent**. The Agent follows the repository's task instructions to turn each request into structured configuration and invoke the appropriate workflow.

```text
Modeling: Use pvod.csv to build a multivariate one-day forecasting solution for pvod-solar.
The target is power, the timestamp column is date, and the sampling frequency is 15 minutes. Compare Ridge and XGBoost, keep the Top-5, and publish if the release gate passes.

Forecast: Use the currently published solution for pvod-solar to forecast power for the next day, based on historical output through 2019-02-24 16:00 and future NWP available at that time.
The timestamp column is date and the sampling frequency is 15 minutes. Read Active only; do not retrain.

Backtest: Use actual output from 2019-02-17 through 2019-02-24 in pvod.csv to backtest the currently published solution for pvod-solar with daily rolling windows. The target is power, the timestamp column is date, and the frequency is 15 minutes.
Update the existing ensemble weights based on these 7 days and publish the new solution if it passes the release gate.

Update: Use the most recent 28 days of pvod.csv through 2019-03-03 to update the forecasting solution for pvod-solar. The target is power, the timestamp column is date, and the frequency is 15 minutes.
Use the last 7 days for validation, retrain only candidates in the SearchPool bound to Active, keep the Top-5, and try to publish through the release gate.
```

For real forecasting, future covariates must be forecasts actually available at the forecast issue time; observations that become available later must not be used as known-future inputs. Forecast, backtest, and update also require the relevant station configuration and Active assets. If no Active solution exists, complete modeling first.

Workflows can also be invoked directly through the CLI. Each task typically follows `prepare → inspect → run`; see `energy-tsf <command> --help` for options.

## 🗂️ Code map

```text
src/energy_tsf/contracts/   Request, plan, asset, and result contracts
src/energy_tsf/config/      Configuration loading, writing, and validation
src/energy_tsf/data/        Data input, cleaning, splitting, and snapshots
src/energy_tsf/providers/   Unified model Provider interfaces
src/energy_tsf/automl/      Bounded search, trial ranking, and finalization
src/energy_tsf/runtime/     Forecast, backtest, and update execution tracks
src/energy_tsf/workflows/   Orchestration for the four workflows
src/energy_tsf/assets/      Candidate, Active, and publication
src/energy_tsf/memory/      Filesystem asset store and retrieval
src/energy_tsf/gates/       Preflight checks and release gates
```

## 📖 Workflow guides

- [Modeling guide](.agents/skills/modeling/SKILL.en.md)
- [Forecast guide](.agents/skills/forecast/SKILL.en.md)
- [Backtest guide](.agents/skills/backtest/SKILL.en.md)
- [Update guide](.agents/skills/update/SKILL.en.md)

Generated requests, configurations, intermediate outputs, and reports are stored under `work_dirs/`, `runs/`, and `asset_store/`.

## 📜 License

This project is released under the [MIT License](LICENSE).

Copyright © 2026 **Shenzhen Loop Area Institute**.

Third-party dependencies, Foundation software packages, and model weights are governed by their respective licenses; review those terms before use.
