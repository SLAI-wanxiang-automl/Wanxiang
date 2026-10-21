[Chinese](README.md) | [English](README.en.md)

<div align="center">

# 🌙 明月·万象

### 新能源出力预测 Agent Harness 框架

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
![Renewable Energy Forecasting](https://img.shields.io/badge/task-renewable%20energy%20forecasting-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**明月·万象**框架：面向真实新能源场站的时间序列预测智能体。
本项目将自然语言驱动的 Agent 工作流与可审计的 AutoML 执行结合起来，支持模型构建、出力预测、滚动回测和模型更新。

</div>

---

## 🧭 项目简介

新能源数据会持续抵达，场站运行条件和预测误差也会随时间变化。一次训练、长期不变的模型难以覆盖持续运行的需求。明月·万象新能源出力预测框架围绕这一特点设计：Agent 将用户意图整理成结构化请求，框架用确定性工作流运行模型、核验结果，并维护可追溯的预测方案。

模型和预测方案可以随新数据持续演进：通过滚动回测评估当前方案、调整集成权重；需要重新训练时，在既有候选搜索池内执行受约束更新。每次发布都要经过相应门禁，失败的候选不会替换当前 Active 方案。

> 本框架基于harness架构实现，由外部 Agent 如codex 理解需求、补齐必要信息并调用工作流，执行SKILLs和工具完成任务；

## ✨ 核心能力

- 🤖 **自然语言任务入口**：通过宿主 Agent 用日常语言发起任务，由工作流契约约束后续执行。
- 🔁 **面向长期运行**：支持重复预测、近期表现回测和受搜索池约束的模型更新。
- 📊 **四类任务**：建模、预测、回测和更新覆盖从初次训练到持续维护的流程。
- 🧾 **可追溯执行**：记录配置、输入指纹、计划、模型资产、门禁和报告，便于复核每次运行。
- 🛡️ **受控模型发布**：候选方案通过发布门禁后才会更新 Active；预测只读取已发布方案。
- 🧩 **多类模型 Provider**：包括 ARIMA、SARIMA、Ridge、LightGBM、XGBoost、TimesFM 2.5、Moirai 2 和 Chronos 2。

## 🧰 四类任务

| 任务 | 适用场景 | 执行边界 |
|---|---|---|
| **Modeling（建模）** | 首次为场站建立预测方案，或比较模型与集成 | 受界定的 AutoML 搜索；按 validation 选择，冻结后评估 test；通过门禁才发布 |
| **Forecast（预测）** | 使用当前已发布方案生成未来出力预测 | 只读 Active，不训练或更改已发布方案 |
| **Backtest（回测）** | 新的真实出力到达后，滚动评估当前方案并校准集成权重 | 保留既有模型成员，只更新权重；新方案通过门禁后发布 |
| **Update（更新）** | 近期数据表明需要重训和重新选择模型 | 仅在 Active 绑定的 SearchPool 内重训，不扩大候选搜索范围；通过门禁后发布 |

## 🏁 快速开始

### 1. 安装

需要安装 Git、Python 3.12 或更高版本，并已配置如 Codex, Claude Code, Pi, 或 DeepSeek Harness等软件。
示例：
克隆仓库并启动 Codex：

```bash
git clone https://github.com/SLAI-wanxiang-automl/Wanxiang.git
cd Wanxiang/packages/energy-tsf
codex
```
首次使用或需要直接运行 CLI 时，再按项目依赖说明安装 Python 环境。

### 2. 在 Agent 中打开仓库

在支持仓库级 Agent 指引的宿主（例如 Codex）中打开项目目录。然后用自然语言描述任务，并上传给定要求的数据。下面以pvod数据集为例，展示的是**发给 Agent 的输入**；Agent 会根据仓库内的任务指引，将请求转换为对应的结构化配置和工作流调用。

```text
建模：使用 pvod.csv，为 pvod-solar 建立未来 1 天的多变量出力预测方案。
目标列是 power，时间列是 date，采样频率为 15 分钟；比较 Ridge 和 XGBoost，保留 Top-5，并在发布门禁通过后发布。

预测：使用 pvod-solar 当前已发布的方案，基于截至 2019-02-24 16:00 的历史出力和该时点可用的未来 NWP，预测接下来 1 天的 power。
时间列为 date，采样频率为 15 分钟；只读取 Active，不重新训练。

回测：使用 pvod.csv 中 2019-02-17 至 2019-02-24 的真实出力，对 pvod-solar 当前已发布方案按天滚动回测；目标列 power，时间列 date，频率 15 分钟。
根据这 7 天的结果更新现有集成权重，并在发布门禁通过后发布新方案。

更新：使用 pvod.csv 中截至 2019-03-03 的最近 28 天数据更新 pvod-solar 的预测方案；目标列 power，时间列 date，频率 15 分钟。
最后 7 天用于 validation，只重训当前 Active 绑定的 SearchPool 中的候选，保留 Top-5，并尝试通过门禁发布。
```

开展真实预测时，未来协变量必须是在预测时点实际可获得的预报数据，不能把事后观测当作未来已知输入。预测、回测和更新还需要相应的场站配置及 Active 资产；若尚无 Active，请先完成建模。

Agent 也可以直接通过 CLI 执行工作流。每类任务通常遵循 `prepare → inspect → run`，详细参数可运行 `energy-tsf <command> --help` 查看。


## 🗂️ 代码导航

```text
src/energy_tsf/contracts/   请求、计划、资产与结果契约
src/energy_tsf/config/      配置加载、写入与校验
src/energy_tsf/data/        数据输入、清理、切分与快照
src/energy_tsf/providers/   统一模型 Provider 接口
src/energy_tsf/automl/      有界搜索、试验排序与最终化
src/energy_tsf/runtime/     预测、回测与更新执行轨道
src/energy_tsf/workflows/   四类任务的工作流编排
src/energy_tsf/assets/      Candidate、Active 与发布
src/energy_tsf/memory/      文件系统资产库与检索
src/energy_tsf/gates/       执行前检查与发布门禁
```

## 📖 任务指引

- [建模指引](.agents/skills/modeling/SKILL.md)
- [预测指引](.agents/skills/forecast/SKILL.md)
- [回测指引](.agents/skills/backtest/SKILL.md)
- [更新指引](.agents/skills/update/SKILL.md)

运行生成的请求、配置、中间结果和报告位于 `work_dirs/`、`runs/` 与 `asset_store/`。
## 📜 许可证

本项目基于 [MIT License](LICENSE) 发布。

Copyright © 2026 **Shenzhen Loop Area Institute**.

第三方依赖、Foundation 软件包和模型权重适用其各自的许可证；使用前请查阅对应条款。
