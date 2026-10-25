<div align="center">

# 🌙 明月 · 万象

### 新能源出力预测 Agent：Harness 与 Agentic Workflow 两条并行路线

[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Renewable Energy Forecasting](https://img.shields.io/badge/task-renewable%20energy%20forecasting-blue.svg)](docs/architecture.md)

**明月·万象** 用**两条并行技术路线**探索同一个问题：如何让 Agent 真正承担新能源场站的出力预测。
[`energy-tsf`](packages/energy-tsf) 走 **Harness 路线**，[`energy-expert`](packages/energy-expert) 走
**Agentic Workflow 路线**。两者不是上下游，而是并列的两套解法。

</div>

---

## 两条并行路线

同一个目标，两种 Agent 架构范式。

```
                    新能源出力预测（同一问题）
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
    路线 A · Harness                   路线 B · Agentic Workflow
    packages/energy-tsf                packages/energy-expert
    ──────────────────────              ────────────────────────────
    Agent 在仓库外                       Agent 在仓库内
    宿主 Codex / Claude / Pi …           agent.py + llm_reports.py
             │                                   │
             ▼                                   ▼
    读 SKILL 契约理解意图                  LLM 工具调用：时序特征分析 /
             │                            交叉验证 / 预测
             ▼                                   │
    自然语言 → 结构化 Request → TOML                ▼
             │                          train_validation_test()
             ▼                          训练内调参 → 验证选型 → 测试报告
    确定性 CLI 工作流                              │
    数值计算全部由代码完成                           ▼
             │                          LLM 生成《训练经验报告》《最终测试报告》
             ▼                          （排名仍由验证集指标决定）
    四类任务轨道
    建模 / 预测 / 回测 / 更新
             │
             ▼
    轨道权限 → Candidate → 发布门禁 → Active
    资产库 · 审计日志 · 有界更新
```

| | 路线 A · **Harness** | 路线 B · **Agentic Workflow** |
|---|---|---|
| 组件 | [`energy-tsf`](packages/energy-tsf) | [`energy-expert`](packages/energy-expert) |
| Agent 位置 | 仓库外，宿主 Agent | 仓库内，`agent.py` |
| 编排方式 | SKILL 契约 + 确定性 CLI | LLM 工具调用 + 流水线 |
| 数值决策 | 全部由确定性代码完成 | 验证集指标（LLM 只做解释） |
| 模型谱系 | 8 个 Provider，聚焦 | 宽：统计 / ML / 深度 / Prophet / sktime / 基础模型 |
| 长期运行 | ✅ 预测 / 回测 / 受约束更新 | ❌ 评测完即止 |
| 发布治理 | ✅ 门禁 / 分轨权限 / 审计日志 | ❌ |
| 可复现性 | 强（配置哈希 + 输入指纹） | 中（依赖锁定，LLM 环节有随机性） |
| 依赖重量 | 轻（pandas / sklearn / lgbm / xgb） | 重（torch / ray / prophet / …） |
| 更适合 | 要长期跑、要出事能追责 | 要横向比、要出实验结论 |

> 两条路线的能力覆盖有重叠也有差异。我们同时保留两者，用于对照与取长补短。
> 选型建议见 [为什么是两条路线](docs/why-two-routes.md)。

---

## 30 秒上手

```bash
git clone https://github.com/SLAI-wanxiang-automl/Wanxiang.git
cd Wanxiang
```

**① 路线 B · Agentic Workflow**：跑通一次滚动评测（自带 PVOD 光伏数据）

```bash
cd packages/energy-expert
uv sync --locked --group dev
uv run python examples/quickstart.py          # 合成数据，验证环境
uv run python examples/pvod_models_pipeline.py dataset/PVOD/station00.csv \
  --models SeasonalNaive AutoRidge --h 96 --step 96 \
  --season-length 96 --num-samples 3 --top-k 2 --plot
```

**② 路线 A · Harness**：安装并查看四类任务

```bash
cd ../energy-tsf
pip install -e .
energy-tsf --help
```

> 路线 A 设计为不由仓库自己注册 Agent，而是由外部宿主（Codex / Claude Code / Pi / DeepSeek 等）
> 通过自然语言驱动，详见 [`packages/energy-tsf/AGENTS.md`](packages/energy-tsf/AGENTS.md)。

---

## 单独使用某个组件

不需要 clone 整个仓库，也不需要装另一半的依赖。

### 方式一：pip 直接安装（推荐）

```bash
# 路线 A · Harness（Python 3.12+，依赖轻：pandas/sklearn/lightgbm/xgboost）
pip install "git+https://github.com/SLAI-wanxiang-automl/Wanxiang.git#subdirectory=packages/energy-tsf"

# 路线 B · Agentic Workflow（Python 3.10+，依赖较重：torch/neuralforecast/prophet 等）
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
| 走 Harness 路线 | 只装 `energy-tsf` | torch / prophet / ray / pyspark |
| 走 Agentic Workflow 路线 | 只装 `energy-expert` | — |
| 两条都要对比着用 | clone 整仓，各自 `pip install -e` | — |

这也是两条路线保持独立包的核心理由之一：Harness 路线要部署到场站侧，
装一套 torch + ray 是不现实的。详见 [为什么是两条路线](docs/why-two-routes.md)。

> **注意**：GitHub 网页端没有「下载单个文件夹」的官方入口。第三方 DownGit 类工具会丢失提交历史，
> 不建议用于生产环境，请用上述三种方式之一。
>
> 两个组件的原仓库（`EnergyTSF`、`EnergyExpert`）仍可单独 clone，但它们已归档为只读，
> 会逐渐落后于本仓库，不建议作为主要来源。

---

## 核心能力

**两条路线共有**

- 🧩 **新能源场景的数据协议**：统一目标 `y`、历史协变量 `H`、未来已知协变量 `F`；未来协变量必须是预测时点可获得的预报，不能用事后实测冒充。
- 📊 **时序严格的评测**：按时间顺序切分，超参只在训练集内搜索，验证集只做排名，测试集只做报告。
- 🤖 **LLM 不参与数值决策**：只负责意图理解与结果解释，排行榜与发布由指标和门禁决定。

**路线 A · Harness（energy-tsf）**

- 🔁 **面向长期运行**：重复预测、滚动回测校准集成权重、受 `SearchPool` 约束的模型更新。
- 🛡️ **受控发布**：候选方案必须过发布门禁才会原子替换 Active；预测轨道只读 Active。
- 🧾 **可追溯执行**：配置哈希、输入指纹、计划、资产与门禁记录全量落盘，便于复核每次运行。
- 🔒 **分轨权限**：每条轨道能做什么由配置写死，例如回测轨道 `allow_fit=false`，禁止借回测之名重训模型。

**路线 B · Agentic Workflow（energy-expert）**

- 🧩 **宽模型谱系**：统计 / 机器学习 / 深度学习 / Prophet / sktime / 时间序列基础模型，支持 Ray / Dask / Spark 分布式。
- 🔬 **多窗口滚动评测**：`step_size` 推进预测起点，逐窗口出预测、指标与图表，三种集成方式。
- 📝 **LLM 实验报告**：读取 `experiment_log.json` 生成《训练经验报告》《最终测试报告》，给出调参与选型建议。
- 🗂️ **自带真实数据**：PVOD 光伏场站数据（15 分钟粒度，含实测气象 `lmd_*` 与 NWP 预报 `nwp_*`）。

---

## 项目结构

```text
Wanxiang/
├── packages/
│   ├── energy-expert/     路线 B · Agentic Workflow：库内 Agent、滚动评测、宽模型谱系、LLM 报告
│   └── energy-tsf/        路线 A · Harness：外部宿主 Agent、任务轨道、发布门禁、审计日志
├── docs/
│   ├── architecture.md           两条路线的架构对照
│   ├── why-two-routes.md          为什么并行保留两条路线
│   ├── terminology.md             两条路线之间的术语映射
│   └── roadmap.md                 路线图
├── examples/              两条路线的对照流程
└── scripts/               上游同步脚本（subtree pull / push）
```

---

## 文档

- [架构设计](docs/architecture.md) — 两条路线各自的控制流与对照
- [为什么是两条路线](docs/why-two-routes.md) — 主动回答最常见的质疑
- [术语映射表](docs/terminology.md) — 两条路线的概念对照
- [路线图](docs/roadmap.md) — 近期与中长期计划

---

## 现状说明

本项目由两个已有仓库合并而来，组件代码保持原样导入、提交历史完整保留。
两条路线**独立演进、互不依赖**，可以单独安装与使用。当前状态（2026-10）：

- `energy-expert` 已有真实数据集、示例与单元测试；`energy-tsf` 的端到端回归测试仍在补齐中。
- 两条路线的能力覆盖并不相同：`energy-tsf` 独有发布治理与长期运行，
  `energy-expert` 独有宽模型谱系与滚动评测。对齐与取长补短见 [路线图](docs/roadmap.md)。
- `energy-expert` 中的基础模型适配器仅支持 zero-shot 推理；
  `energy-tsf` 中的基础模型要求本地已物化的权重与钉死的版本，不支持运行时下载。

---

## 许可证与署名

- 本项目：[MIT](LICENSE)，Copyright © 2026 Shenzhen Loop Area Institute。
- `packages/energy-expert` 基于 [TimeCopilot](https://github.com/TimeCopilot/timecopilot) 扩展，
  保留上游 MIT 版权声明（Copyright © Azul Garza）。
- 第三方依赖、基础模型权重与数据集适用各自许可证，本项目 MIT 不替代其授权条件。

---

<div align="center">

**明月 · 万象** · Mingyue Wanxiang
*Two parallel routes — Harness and Agentic Workflow — for agentic renewable energy forecasting.*

</div>
