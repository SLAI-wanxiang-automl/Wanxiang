# 术语映射表

两条路线各自有一套术语，因为它们诞生于不同阶段、面向不同问题。本文给出对照，避免读文档时混淆。

> 注意：下表是**概念对照**，不代表数据流上的先后关系。两条路线是并列的，见 [architecture.md](architecture.md)。

## 核心概念对照

| 路线 B `energy-expert`<br>Agentic Workflow | 路线 A `energy-tsf`<br>Harness | 含义 |
|---|---|---|
| `ForecastData` | `PreparedData` / `PreparedHistory` | 经过清洗与校验的建模输入 |
| `y` | `target` | 预测目标（出力功率） |
| historical covariates `H` | `past_only_covariates` | 历史协变量，如实测气象 `lmd_*` |
| future covariates `F` | `known_future_covariates` | 未来已知协变量，如 NWP 预报 `nwp_*` |
| `unique_id` | `station_id` | 序列 / 场站标识 |
| `h` | `horizon_steps` | 预测步长数 |
| `step_size` | `step` | 滚动窗口推进步长 |
| `n_windows` / `inner_n_windows` | `windows` | 滚动窗口数 |
| `leaderboard` | validation ranking / Top-K | 候选模型排名 |
| `selected_models` + `ensemble_weights` | `Solution.members` + `Solution.ensemble` | 入选模型与集成权重 |
| `model_catalog()` | `ProviderCapabilities` | 模型能力的显式声明 |
| `experiment_log.json` | `runs/` + `report.md` | 单次执行的可追溯产物 |
| `train_validation_test()` | `modeling` 轨道 | 首次建模与选型 |
| — | `forecast` 轨道 | 用已发布方案生成未来预测（路线 B 无对应） |
| 滚动重估（脚本级） | `backtest` 轨道 | 新真实出力到达后的滚动评估与权重校准 |
| 重新训练 | `update` 轨道 | 在 `SearchPool` 内重训并重新选择 |
| — | `Active` / `Candidate` | 已发布方案 / 待发布方案（路线 B 无对应） |
| — | `SearchPool` | 与 Active 绑定的候选搜索池（路线 B 无对应） |
| — | `gates/release.py` | 发布门禁（路线 B 无对应） |

## 任务类型对照

| 用户意图 | 路线 B `energy-expert` | 路线 A `energy-tsf` |
|---|---|---|
| 为这个场站建个模型 | `train_validation_test()` | `energy-tsf prepare/run-modeling` |
| 预测接下来一天 | 需自行调用模型 | `energy-tsf prepare/run-forecast` |
| 用最近真实出力评估一下 | `scripts/backtest_pvod.py` | `energy-tsf prepare/run-backtest` |
| 最近数据变了，重训一下 | 需自行重跑流水线 | `energy-tsf prepare/run-update` |
| 解释为什么选这个模型 | `scripts/llm_reports.py` | `report.md`（审计视角） |

## Agent 相关概念对照

| 概念 | 路线 B | 路线 A |
|---|---|---|
| Agent 在哪 | 库内 `agent.py` | 仓库外宿主 Agent |
| 意图入口 | `EneygyExpert.analyze()` | `.agents/skills/<task>/SKILL.md` |
| LLM 配置 | `OPENAI_BASE_URL` / `EE_LLM` / `EE_REPORT_MODEL` | 由宿主 Agent 自行配置 |
| LLM 产出 | 训练经验报告、最终测试报告 | 结构化 Request（不产出数值） |
| LLM 能否编排流程 | 能（工具调用） | 否（只做意图理解） |

## 「未来已知」的语义

两条路线对 `F` / `known_future_covariates` 的约定一致，且都把它当作硬约束：

> 未来协变量必须是在**对应预测起点已经发布、可获得**的预报数据，
> 不能用事后实测观测替代。

需要注意：路线 B 目前**不自动核验**天气预报的发布时间 / 版本，需要数据准备阶段按预测起点自行对齐。
这是已知缺口，见 [路线图](roadmap.md)。

## 时点对齐相关

| 概念 | 路线 B `energy-expert` | 路线 A `energy-tsf` |
|---|---|---|
| 未来时间索引 | `ForecastData` 内部按 `freq` 生成 | `tools/point_in_time.future_time_index()` |
| 无时间戳序列 | — | `tools/point_in_time.relative_horizon_index()` |
| 输入指纹 | `experiment_log.json` 记录配置 | `tools/fingerprints` + `config_hash` |
| 跨场站迁移 | — | `tools/transfer`（按装机容量归一化到 per-unit） |

## 命名不一致的已知项

以下命名问题已记录，将在后续版本统一（不影响当前使用）：

- 路线 B 的 Python 包名为 `eneygyexpert`（历史拼写），与其目录名 `energy-expert` 不一致。
- 路线 B 内部存在 `models/` 与 `rolling_models/` 两套模型抽象（继承自上游 TimeCopilot）。
- 路线 B 的 `MedianEnsembleAdapter` 默认别名是 `MedianModelEnsemble`；
  `MedianEnsemble` 与 `WeightedEnsemble` 保留给流水线生成的预测级集成。
