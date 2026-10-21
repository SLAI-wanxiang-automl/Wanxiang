<div align="center">

# 🌙 Mingyue · Wanxiang

### Renewable Energy Forecasting Agent: From Model Selection to Long-Term Operation

[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Renewable Energy Forecasting](https://img.shields.io/badge/task-renewable%20energy%20forecasting-blue.svg)](docs/architecture.md)

**Mingyue Wanxiang** is a renewable energy output forecasting system built for real
generation sites. It splits the problem in two: a *modeling engine* answers
**which model this site should use**, and a *runtime system* answers
**how the model keeps running after it ships**.

</div>

---

## Why two layers

Renewable energy data keeps arriving, and site conditions, equipment state, and error
distributions drift over time. A model trained once and never revisited cannot carry
long-term operation. At the same time, a benchmark-only platform stops right after
evaluation and never answers "what happens when new data shows up".

Mingyue Wanxiang assigns these to two components joined by explicit contracts:

```
                        User / Host Agent (natural language)
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │  packages/energy-tsf     Runtime System           │
        │  ─────────────────────────────────────────────── │
        │  Four task tracks  modeling / forecast /          │
        │                    backtest / update              │
        │  Candidate → release gate → Active (atomic)       │
        │  Asset store · ledger · per-track permissions     │
        └───────────────────────┬──────────────────────────┘
                                │  SearchPool / Solution contract
                                ▼
        ┌──────────────────────────────────────────────────┐
        │  packages/energy-expert  Modeling Engine          │
        │  ─────────────────────────────────────────────── │
        │  y / H / F data protocol · chronological splits   │
        │  Tune on train → rank on validation → report test │
        │  Rolling multi-window evaluation · model catalog  │
        │  LLM explains; metrics decide                     │
        └──────────────────────────────────────────────────┘
```

| | Component | What it solves | Use it alone when |
|---|---|---|---|
| **L2** | [`energy-tsf`](packages/energy-tsf) · Runtime | Life after release: repeat forecasting, rolling backtest, constrained updates, auditability | You already have a model and must operate it at real sites |
| **L0–L1** | [`energy-expert`](packages/energy-expert) · Engine | Life before release: data protocol, rolling evaluation, model catalog, ensembling | You have a new site and need to compare and select models |

> These are not duplicate implementations — they are the upper and lower layers of one
> product. See [Why two components](docs/why-two-components.md).

---

## 30-second quickstart

```bash
git clone https://github.com/SLAI-wanxiang-automl/Wanxiang.git
cd Wanxiang
```

**① Modeling engine** — run a rolling evaluation (PVOD photovoltaic data included)

```bash
cd packages/energy-expert
uv sync --locked --group dev
uv run python examples/quickstart.py          # synthetic data, verifies the environment
uv run python examples/pvod_models_pipeline.py dataset/PVOD/station00.csv \
  --models SeasonalNaive AutoRidge --h 96 --step 96 \
  --season-length 96 --num-samples 3 --top-k 2 --plot
```

**② Runtime system** — install and inspect the four task tracks

```bash
cd ../energy-tsf
pip install -e .
energy-tsf --help
```

> The runtime is designed to be driven by an external host Agent (Codex, Claude Code, Pi,
> DeepSeek, …) through natural language. See
> [`packages/energy-tsf/AGENTS.md`](packages/energy-tsf/AGENTS.md).

---

## Using a single component

No need to clone the whole repository, and no need to install the other half's dependencies.

### Option 1: install with pip (recommended)

```bash
# Runtime (Python 3.12+, light deps: pandas/sklearn/lightgbm/xgboost)
pip install "git+https://github.com/SLAI-wanxiang-automl/Wanxiang.git#subdirectory=packages/energy-tsf"

# Modeling engine (Python 3.10+, heavy deps: torch/neuralforecast/prophet, …)
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
| Long-term operation and audit only | install `energy-tsf` only | torch / prophet / ray / pyspark |
| Model comparison only | install `energy-expert` only | — |
| Both | clone the repo, `pip install -e` each | — |

This is the core reason the two components are separate packages: the runtime has to be
deployed at the generation site, where shipping torch + ray is not realistic. See
[Why two components](docs/why-two-components.md).

> **Note**: GitHub's web UI has no official "download a single folder" button. Third-party
> DownGit-style tools drop commit history and are not recommended for production use —
> use one of the three options above.
>
> The original component repositories (`EnergyTSF`, `EnergyExpert`) are still individually
> cloneable, but they are archived read-only and will drift behind this repository. Do not
> treat them as the primary source.

---

## Key capabilities

- 🤖 **Natural-language entry** (runtime): the host Agent interprets intent; [SKILL contracts](packages/energy-tsf/.agents/skills/) turn it into structured requests, and all numeric work is done by deterministic code.
- 🔬 **Reproducible rolling evaluation** (engine): 60/20/20 chronological split, hyperparameters searched inside train only, validation used solely for ranking, test used solely for reporting.
- 🔁 **Built for long-term operation** (runtime): repeated forecasting, rolling backtest that recalibrates ensemble weights, and updates constrained to the bound `SearchPool`.
- 🛡️ **Controlled release**: a candidate must pass the release gate before it atomically replaces Active; the forecast track is read-only on Active.
- 🧾 **Traceable execution**: config hashes, input fingerprints, plans, assets, and gate decisions are persisted for post-hoc review.
- 🧩 **Broad model catalog** (engine): statistical, machine learning, deep learning, Prophet, sktime, and time-series foundation models (TimesFM 2.5, Moirai 2, Chronos 2).

---

## Repository layout

```text
Wanxiang/
├── packages/
│   ├── energy-expert/     Modeling engine (L0–L1): data protocol, rolling evaluation,
│   │                      model catalog, LLM reports
│   └── energy-tsf/        Runtime system (L2): task tracks, asset governance,
│                          release gates, audit ledger
├── docs/
│   ├── architecture.md          Two-layer architecture and data flow
│   ├── why-two-components.md    Why this is not a single component
│   ├── terminology.md           Concept mapping between the two components
│   └── roadmap.md               Roadmap
├── examples/              Cross-component end-to-end workflow
└── scripts/               Upstream sync helpers (subtree pull / push)
```

---

## Documentation

- [Architecture](docs/architecture.md) — how the layers cooperate, and how permissions and gates constrain execution
- [Why two components](docs/why-two-components.md) — answers the most common question up front
- [Terminology mapping](docs/terminology.md) — concept correspondence between engine and runtime
- [Roadmap](docs/roadmap.md) — near-term and long-term plans

---

## Current status

This repository consolidates two pre-existing repositories. Component code was imported
as-is with full commit history preserved. As of 2026-10:

- The modeling engine ships a real dataset, examples, and unit tests; end-to-end regression
  tests for the runtime are still being filled in.
- The two components are currently **invoked separately**: the engine produces rankings and
  ensemble weights, the runtime owns release and operation. Automatic hand-off (engine
  results becoming runtime Candidates) is in progress — see the [roadmap](docs/roadmap.md).
- Foundation model adapters in the engine support zero-shot inference only. Foundation
  providers in the runtime require locally materialized weights with pinned revisions and
  do not download at run time.

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
*Renewable energy forecasting, from model selection to long-term operation.*

</div>
