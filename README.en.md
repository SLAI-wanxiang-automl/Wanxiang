[中文](README.md) | [English](README.en.md)

<div align="center">

# 🌙 Mingyue · Wanxiang

### Renewable Energy Forecasting Agent: Two Parallel Routes — Harness and Agentic Workflow

[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Renewable Energy Forecasting](https://img.shields.io/badge/task-renewable%20energy%20forecasting-blue.svg)](docs/architecture.md)

**Mingyue Wanxiang** explores one question along **two parallel routes**: how an Agent can
genuinely own renewable energy output forecasting. `energy-tsf` takes the
**Harness route**; `energy-expert` takes the
**Agentic Workflow route**. They are not upstream and downstream of each other — they are two
competing answers.

</div>

---

## Two parallel routes

One goal, two agent architecture paradigms.

```
                Renewable energy forecasting (same problem)
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
     Route A · Harness                  Route B · Agentic Workflow
     packages/energy-tsf                packages/energy-expert
     ──────────────────────              ────────────────────────────
     Agent lives outside                 Agent lives inside
     host: Codex / Claude / Pi …         agent.py + llm_reports.py
             │                                   │
             ▼                                   ▼
     SKILL contracts parse intent        LLM tool calls: feature analysis /
             │                           cross-validation / forecast
             ▼                                   │
     natural language → Request → TOML            ▼
             │                          train_validation_test()
             ▼                          tune on train → rank on validation
     deterministic CLI workflow                  │
     all numeric work is code                    ▼
             │                          LLM writes experience / test reports
             ▼                          (metrics still decide the ranking)
     four task tracks
     modeling / forecast / backtest / update
             │
             ▼
     track permissions → Candidate → release gate → Active
     asset store · ledger · bounded updates
```

| | Route A · **Harness** | Route B · **Agentic Workflow** |
|---|---|---|
| Component | `energy-tsf` | `energy-expert` |
| Where the agent lives | Outside the repo, host agent | Inside the repo, `agent.py` |
| Orchestration | SKILL contracts + deterministic CLI | LLM tool calls + pipeline |
| Numeric decisions | Entirely deterministic code | Validation metrics (LLM explains only) |
| Model catalog | 8 providers, focused | Broad: stats / ML / deep / Prophet / sktime / foundation |
| Long-term operation | ✅ forecast / backtest / bounded update | ❌ stops after evaluation |
| Release governance | ✅ gates / per-track permissions / ledger | ❌ |
| Reproducibility | Strong (config hash + input fingerprints) | Medium (locked deps, LLM non-determinism) |
| Dependency weight | Light (pandas / sklearn / lgbm / xgb) | Heavy (torch / ray / prophet / …) |
| Best for | Running long-term and being accountable | Comparing models and producing conclusions |

> The two routes overlap and differ in coverage. We keep both to compare them and to port
> proven capabilities across. See [Why two routes](docs/why-two-routes.md).

---

## 30-second quickstart

```bash
git clone https://github.com/SLAI-wanxiang-automl/Wanxiang.git
cd Wanxiang
```

**① Route B · Agentic Workflow** — run a rolling evaluation (PVOD photovoltaic data included)

```bash
cd packages/energy-expert
uv sync --locked --group dev
uv run python examples/quickstart.py          # synthetic data, verifies the environment
uv run python examples/pvod_models_pipeline.py dataset/PVOD/station00.csv \
  --models SeasonalNaive AutoRidge --h 96 --step 96 \
  --season-length 96 --num-samples 3 --top-k 2 --plot
```

**② Route A · Harness** — install and inspect the four task tracks

```bash
cd ../energy-tsf
pip install -e .
energy-tsf --help
```

> Route A registers no agent of its own: it is driven by an external host (Codex, Claude Code,
> Pi, DeepSeek, …) through natural language. See
> [`packages/energy-tsf/AGENTS.md`](packages/energy-tsf/AGENTS.md).

---

## Using a single component

No need to clone the whole repository, and no need to install the other half's dependencies.

### Option 1: install with pip (recommended)

```bash
# Route A · Harness (Python 3.12+, light deps: pandas/sklearn/lightgbm/xgboost)
pip install "git+https://github.com/SLAI-wanxiang-automl/Wanxiang.git#subdirectory=packages/energy-tsf"

# Route B · Agentic Workflow (Python 3.10+, heavy deps: torch/neuralforecast/prophet, …)
pip install "git+https://github.com/SLAI-wanxiang-automl/Wanxiang.git#subdirectory=packages/energy-expert"
```

Pin a tag or commit with `...Wanxiang.git@<tag-or-sha>#subdirectory=packages/energy-tsf`.

In a `uv` project:

```toml
dependencies = [
  "energy-tsf-harness @ git+https://github.com/SLAI-wanxiang-automl/Wanxiang.git@v0.1.0#subdirectory=packages/energy-tsf",
]
```

### Option 2: checkout only that directory (reading or editing source)

```bash
git clone --filter=blob:none --sparse https://github.com/SLAI-wanxiang-automl/Wanxiang.git
cd Wanxiang
git sparse-checkout set packages/energy-tsf
```

`--filter=blob:none` fetches file contents on demand and `sparse-checkout` writes only the
selected directory into the working tree. Add the other component at any time:

```bash
git sparse-checkout add packages/energy-expert
```

> In cone mode a few repository-root files (README, LICENSE, …) are still checked out.
> That is expected.

### Option 3: pick only the dependencies you need

| Need | Do this | Not pulled in |
|---|---|---|
| Take the Harness route | install `energy-tsf` only | torch / prophet / ray / pyspark |
| Take the Agentic Workflow route | install `energy-expert` only | — |
| Compare both | clone the repo, `pip install -e` each | — |

One reason the two routes stay separate packages: the Harness route must be deployable at the
generation site, where shipping torch + ray is not realistic. See
[Why two routes](docs/why-two-routes.md).

> **Note**: GitHub's web UI has no official "download a single folder" button. Third-party
> DownGit-style tools drop commit history and are not recommended for production use —
> use one of the three options above.
>

---

## Key capabilities

**Shared by both routes**

- 🧩 **Energy-domain data protocol**: target `y`, historical covariates `H`, known-future covariates `F`. Future covariates must be forecasts actually available at the forecast origin — never post-hoc observations.
- 📊 **Chronologically strict evaluation**: time-ordered splits, hyperparameters searched inside train only, validation used solely for ranking, test used solely for reporting.
- 🤖 **LLM never makes numeric decisions**: it handles intent parsing and explanation; rankings and releases are decided by metrics and gates.

**Route A · Harness (energy-tsf)**

- 🔁 **Built for long-term operation**: repeated forecasting, rolling backtest that recalibrates ensemble weights, and updates constrained to the bound `SearchPool`.
- 🛡️ **Controlled release**: a candidate must pass the release gate before it atomically replaces Active; the forecast track is read-only on Active.
- 🧾 **Traceable execution**: config hashes, input fingerprints, plans, assets, and gate decisions are persisted for post-hoc review.
- 🔒 **Per-track permissions**: what each track may do is declared in config — the backtest track sets `allow_fit=false`, so it cannot retrain models under the cover of a backtest.

**Route B · Agentic Workflow (energy-expert)**

- 🧩 **Broad model catalog**: statistical, machine learning, deep learning, Prophet, sktime and time-series foundation models, with Ray / Dask / Spark backends.
- 🔬 **Multi-window rolling evaluation**: `step_size` advances the forecast origin, producing per-window forecasts, metrics and plots, with three ensemble strategies.
- 📝 **LLM experiment reports**: reads `experiment_log.json` and writes a training-experience report and a final test report with tuning and selection advice.
- 🗂️ **Ships real data**: the PVOD photovoltaic dataset (15-minute resolution, with measured weather `lmd_*` and NWP forecasts `nwp_*`).

---

## Repository layout

```text
Wanxiang/
├── packages/
│   ├── energy-expert/     Route B · Agentic Workflow: in-repo agent, rolling evaluation,
│   │                      broad model catalog, LLM reports
│   └── energy-tsf/        Route A · Harness: external host agent, task tracks, release
│                          gates, audit ledger
├── docs/
│   ├── architecture.md          Architecture of the two routes
│   ├── why-two-routes.md        Why both routes are kept in parallel
│   ├── terminology.md           Concept mapping between the two routes
│   └── roadmap.md               Roadmap
├── examples/              Workflow of both routes side by side
└── scripts/               Upstream sync helpers (subtree pull / push)
```

---

## Documentation

- [Architecture](docs/architecture.md) — each route's control flow, plus a comparison
- [Why two routes](docs/why-two-routes.md) — answers the most common question up front
- [Terminology mapping](docs/terminology.md) — concept correspondence between the routes
- [Roadmap](docs/roadmap.md) — near-term and long-term plans

---

## License and attribution

- This project: [MIT](LICENSE), Copyright © 2026 Shenzhen Loop Area Institute.
- `packages/energy-expert` extends [TimeCopilot](https://github.com/TimeCopilot/timecopilot)
  and retains the upstream MIT notice (Copyright © Azul Garza).
- Third-party dependencies, foundation model weights, and datasets are governed by their own
  licenses. This repository's MIT license does not replace their terms.

---

<div align="center">

**Mingyue · Wanxiang**
*Two parallel routes — Harness and Agentic Workflow — for agentic renewable energy forecasting.*

</div>
