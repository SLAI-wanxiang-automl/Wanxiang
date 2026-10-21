<div align="center">

# 🌙 明月 · 万象

### 新能源出力预测 Agent：从建模到长期运行

[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Renewable Energy Forecasting](https://img.shields.io/badge/task-renewable%20energy%20forecasting-blue.svg)](docs/architecture.md)

**明月·万象** 是面向真实新能源场站的出力预测系统。它把「选模型」和「跑模型」拆成两层：
建模引擎负责回答*这个场站该用哪个模型*，运行系统负责回答*模型上线之后怎么一直跑下去*。

</div>

---

## 为什么需要两层

新能源数据持续抵达，场站的运行条件、设备状态和误差分布都会随时间变化。
一次训练、长期不变的模型撑不住长期运行；而只有评测、没有发布治理的实验平台，评完就结束，
没有回答「新数据到了怎么办」。

明月·万象把这两件事交给两个组件，中间用明确的契约连接：

```
                        用户 / 宿主 Agent（自然语言）
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │  packages/energy-tsf     运行系统 Runtime         │
        │  ─────────────────────────────────────────────── │
        │  四类任务轨道  建模 / 预测 / 回测 / 更新            │
        │  Candidate → 发布门禁 → Active（原子发布）         │
        │  资产库 · 审计日志 · 分轨权限 · 有界更新            │
        └───────────────────────┬──────────────────────────┘
                                │  SearchPool / Solution 契约
                                ▼
        ┌──────────────────────────────────────────────────┐
        │  packages/energy-expert  建模引擎 Engine           │
        │  ─────────────────────────────────────────────── │
        │  y / H / F 数据协议 · 时序三段划分                 │
        │  训练内调参 → 验证集选型 → 测试集只报告             │
        │  多窗口滚动评测 · 模型谱系 · 三种集成               │
        │  LLM 只做解释，不做决策                            │
        └──────────────────────────────────────────────────┘
```

| | 组件 | 解决什么 | 单独适用场景 |
|---|---|---|---|
| **L2** | [`energy-tsf`](packages/energy-tsf) · 运行系统 | 模型发布之后：持续预测、滚动回测、受约束更新、可审计 | 已有模型，需要长期在真实场站运行与追责 |
| **L0–L1** | [`energy-expert`](packages/energy-expert) · 建模引擎 | 模型确定之前：数据协议、滚动评测、模型谱系与集成 | 拿到新场站数据，要横向对比选模型、出实验结论 |

> 两者不是重复实现，而是一个产品的上下两层。详见 [为什么是两个组件](docs/why-two-components.md)。

---

## 30 秒上手

```bash
git clone https://github.com/SLAI-wanxiang-automl/Wanxiang.git
cd Wanxiang
```

**① 建模引擎**：跑通一次滚动评测（自带 PVOD 光伏数据）

```bash
cd packages/energy-expert
uv sync --locked --group dev
uv run python examples/quickstart.py          # 合成数据，验证环境
uv run python examples/pvod_models_pipeline.py dataset/PVOD/station00.csv \
  --models SeasonalNaive AutoRidge --h 96 --step 96 \
  --season-length 96 --num-samples 3 --top-k 2 --plot
```

**② 运行系统**：安装并查看四类任务

```bash
cd ../energy-tsf
pip install -e .
energy-tsf --help
```

> 运行系统设计为由外部宿主 Agent（Codex / Claude Code / Pi / DeepSeek 等）通过自然语言驱动，
> 详见 [`packages/energy-tsf/AGENTS.md`](packages/energy-tsf/AGENTS.md)。

---

## 单独使用某个组件

不需要 clone 整个仓库，也不需要装另一半的依赖。

### 方式一：pip 直接安装（推荐）

```bash
# 运行系统（Python 3.12+，依赖轻：pandas/sklearn/lightgbm/xgboost）
pip install "git+https://github.com/SLAI-wanxiang-automl/Wanxiang.git#subdirectory=packages/energy-tsf"

# 建模引擎（Python 3.10+，依赖较重：torch/neuralforecast/prophet 等）
pip install "git+https://github.com/SLAI-wanxiang-automl/Wanxiang.git#subdirectory=packages/energy-expert"
```

可钉版本或提交：`...Wanxiang.git@<tag-or-sha>#subdirectory=packages/energy-tsf`。

`uv` 项目里同样可用：

```toml
dependencies = [
  "energy-tsf-harness @ git+https://github.com/SLAI-wanxiang-automl/Wanxiang.git@v0.1.0#subdirectory=packages/energy-tsf",
]
```

### 方式二：只检出该目录（需要读源码或改代码）

```bash
git clone --filter=blob:none --sparse https://github.com/SLAI-wanxiang-automl/Wanxiang.git
cd Wanxiang
git sparse-checkout set packages/energy-tsf
```

`--filter=blob:none` 表示文件内容按需拉取，`sparse-checkout` 只把指定目录写进工作区。
需要另一个组件时随时补上：

```bash
git sparse-checkout add packages/energy-expert
```

> cone 模式下仓库根目录的少量文件（README、LICENSE 等）仍会检出，这是正常行为。

### 方式三：按需求选依赖，避免互相拖累

| 需求 | 做法 | 不会引入 |
|---|---|---|
| 只要长期运行与审计 | 只装 `energy-tsf` | torch / prophet / ray / pyspark |
| 只要横向对比选模型 | 只装 `energy-expert` | — |
| 两个都要 | clone 整仓，各自 `pip install -e` | — |

这也是当初把两个组件拆成独立包的核心理由：运行系统要部署到场站侧，
装一套 torch + ray 是不现实的。详见 [为什么是两个组件](docs/why-two-components.md)。

> **注意**：GitHub 网页端没有「下载单个文件夹」的官方入口。第三方 DownGit 类工具会丢失提交历史，
> 不建议用于生产环境，请用上述三种方式之一。
>
> 两个组件的原仓库（`EnergyTSF`、`EnergyExpert`）仍可单独 clone，但它们已归档为只读，
> 会逐渐落后于本仓库，不建议作为主要来源。

---

## 核心能力

- 🤖 **自然语言任务入口**（运行系统）：宿主 Agent 理解需求，[SKILL 契约](packages/energy-tsf/.agents/skills/)把自然语言转成结构化请求，数值计算全部交给确定性代码。
- 🔬 **可复现的滚动评测**（建模引擎）：60/20/20 时序切分，超参只在训练集内搜索，验证集只做排名，测试集只做报告。
- 🔁 **面向长期运行**（运行系统）：支持重复预测、滚动回测校准集成权重、受 `SearchPool` 约束的模型更新。
- 🛡️ **受控发布**：候选方案必须过发布门禁才会原子替换 Active；预测轨道只读 Active。
- 🧾 **可追溯执行**：配置哈希、输入指纹、计划、资产与门禁记录全量落盘，便于复核每次运行。
- 🧩 **宽模型谱系**（建模引擎）：统计 / 机器学习 / 深度学习 / Prophet / sktime / 时间序列基础模型（TimesFM 2.5、Moirai 2、Chronos 2）。

---

## 项目结构

```text
Wanxiang/
├── packages/
│   ├── energy-expert/     建模引擎（L0–L1）：数据协议、滚动评测、模型谱系、LLM 报告
│   └── energy-tsf/        运行系统（L2）：任务轨道、资产治理、发布门禁、审计日志
├── docs/
│   ├── architecture.md           两层架构与数据流
│   ├── why-two-components.md     为什么不是一个仓库 / 一个组件
│   ├── terminology.md            两个组件之间的术语映射
│   └── roadmap.md                路线图
├── examples/              跨组件端到端流程
└── scripts/               上游同步脚本（subtree pull / push）
```

---

## 文档

- [架构设计](docs/architecture.md) — 两层如何协作、权限与门禁如何约束
- [为什么是两个组件](docs/why-two-components.md) — 主动回答最常见的质疑
- [术语映射表](docs/terminology.md) — 建模引擎与运行系统的概念对照
- [路线图](docs/roadmap.md) — 近期与中长期计划

---

## 现状说明

本项目由两个已有仓库合并而来，组件代码保持原样导入、提交历史完整保留。
当前状态（2026-10）：

- 建模引擎已有真实数据集、示例与单元测试；运行系统的端到端回归测试仍在补齐中。
- 两个组件目前**分别调用**：建模引擎产出排名与集成权重，运行系统负责发布与运维，
  中间的自动贯通（建模结果自动成为运行系统的 Candidate）正在推进，见 [路线图](docs/roadmap.md)。
- 建模引擎中的时间序列基础模型适配器仅支持 zero-shot 推理；
  运行系统中的基础模型要求本地已物化的权重与钉死的版本，不支持运行时下载。

---

## 许可证与署名

- 本项目：[MIT](LICENSE)，Copyright © 2026 Shenzhen Loop Area Institute。
- `packages/energy-expert` 基于 [TimeCopilot](https://github.com/TimeCopilot/timecopilot) 扩展，
  保留上游 MIT 版权声明（Copyright © Azul Garza）。
- 第三方依赖、基础模型权重与数据集适用各自许可证，本项目 MIT 不替代其授权条件。

---

<div align="center">

**明月 · 万象** · Mingyue Wanxiang
*Renewable energy forecasting, from model selection to long-term operation.*

</div>
