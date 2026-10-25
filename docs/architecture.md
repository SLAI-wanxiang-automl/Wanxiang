# 架构设计

> 明月·万象对「Agent 如何承担新能源出力预测」这个问题，同时维护**两条并行路线**。
> 本文分别说明两条路线的控制流，再给出对照与设计取舍。

## 1. 总览：两条路线，不是两层

```
                      新能源出力预测（同一问题）
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
      路线 A · Harness                   路线 B · Agentic Workflow
      packages/energy-tsf               packages/energy-expert
              │                                   │
              │  （并列探索，互不依赖）              │
              ▼                                   ▼
     可审计的长期运行系统                 可解释的建模评测系统
```

**两条路线是并列关系，不是上下游。** 它们各自端到端地解决自己的目标场景，
不存在「先用 B 选模型、再交给 A 上线」的必然顺序。

| | 路线 A · **Harness** | 路线 B · **Agentic Workflow** |
|---|---|---|
| 核心命题 | Agent 如何**安全、可追责地**驱动预测系统 | Agent 如何**自主、可解释地**完成建模评测 |
| Agent 位置 | 仓库**外**，宿主 Agent | 仓库**内**，`agent.py` |
| 编排机制 | SKILL 契约 + 确定性 CLI | LLM 工具调用 + 流水线 |
| 产物 | Active 方案、审计日志 | leaderboard、集成权重、LLM 报告 |
| 终点 | 长期运行 | 一份结论 |

---

## 2. 路线 A · Harness（`energy-tsf`）

### 2.1 控制流

```
宿主 Agent（Codex / Claude Code / Pi / DeepSeek …）
      │  读 .agents/skills/<task>/SKILL.md
      ▼
用户自然语言 → 结构化 Request → TOML
      │
      ▼
prepare-*   生成权威 request.json / *.toml / resolved.toml（带 config_hash）
      │
      ▼
inspect-*   检查数据、切分、可用 Provider、trial 计划
      │      ⚠ 不训练、不写 Candidate
      ▼
run-*       确定性执行：清洗 → 训练 → 指标 → 排名 → 写 Candidate
      │
      ▼
发布门禁    构件完整性 · 可重载性 · 冒烟推理 · 元数据钉死
      │
      ├── 通过 → 原子替换 Active
      └── 不通过 → 保留原 Active，记录原因
```

**关键设计**：代码库内**不注册**任何 primary / child Agent。宿主 Agent 只负责理解意图、
补齐缺失信息、生成结构化请求；清洗、训练、指标、排名、发布全部由确定性代码完成。
这样 LLM 的不可确定性被隔离在数值计算之外。

### 2.2 四类任务轨道

| 任务 | 做什么 | 边界 |
|---|---|---|
| **modeling** | 首次建模或比较模型与集成 | 有界 AutoML 搜索；按 validation 选择；过门禁才发布 |
| **forecast** | 用已发布方案生成未来预测 | 只读 Active，不训练、不改资产 |
| **backtest** | 新真实出力到达后滚动评估并校准集成权重 | **只调权重，不重训** |
| **update** | 近期数据表明需要重训时重新选择 | 只在 Active 绑定的 `SearchPool` 内，**不扩大搜索空间** |

### 2.3 权限与治理

`configs/tracks/*.toml` 显式声明每条轨道的能力，而不是靠约定：

| 能力 | modeling | forecast | backtest | update |
|---|:--:|:--:|:--:|:--:|
| 读 Active | — | ✅ | ✅ | ✅ |
| 写 Candidate | ✅ | ❌ | ✅ | ✅ |
| 写 Active | ✅ | ❌ | ✅ | ✅ |
| 允许拟合 / 存模型 | ✅ | ❌ | **❌** | ✅ |
| 允许下载权重 | 受策略约束 | ❌ | ❌ | 受策略约束 |
| 允许发布 | ✅ | ❌ | ✅ | ✅ |

资产与审计：

```text
asset_store/
├── active/          当前已发布方案
├── candidate/       待发布方案
├── solutions/       方案定义（成员 + 集成权重）
├── trials/          试验记录
├── search_pools/    Active 绑定的候选搜索池
├── objects/         内容寻址对象
└── ledger/          append-only 事件日志
```

系统策略（`configs/policies/system.toml`）：`content_addressed = true`、
`ledger.append_only = true`、基础模型 `local_files_only = true` + `allow_download = false` +
`require_pinned_revision = true`。

### 2.4 路线 A 的取舍

- ✅ 可复现（config_hash + 输入指纹）、可追责（审计日志）、可回归测试。
- ✅ 依赖轻，能部署到场站侧。
- ❌ 依赖宿主 Agent 的可用性，行为边界由 SKILL 契约约束，灵活度低于路线 B。
- ❌ 目前没有端到端回归测试。

---

## 3. 路线 B · Agentic Workflow（`energy-expert`）

### 3.1 控制流

```
                  LLM（OpenAI 兼容：vLLM 本地 / 远程 API）
                        │
        ┌───────────────┴────────────────┐
        │                                │
   ① 交互式分析与工具调用              ② 实验报告生成
   EneygyExpert.analyze()           scripts/llm_reports.py
   时序特征分析 / 交叉验证 / 预测       读 experiment_log.json
        │                                │
        │                                ▼
        │                     training_experience.md
        │                     final_report.md
        ▼
   ┌──────────────────────────────────────────────┐
   │  确定性流水线 train_validation_test()         │
   │  ─────────────────────────────────────────── │
   │  ForecastData(y, H, F) · 时序 60/20/20        │
   │  训练内调参 → 验证集选型 → 测试集只报告         │
   │  多窗口滚动 · 三种集成                        │
   └──────────────────────────────────────────────┘
        │
        ▼
   leaderboard · 逐窗口预测 · 指标 · 图表
```

**关键设计**：LLM 在环内做分析与编排，但**排行榜由验证集指标决定，不由 LLM 决定**。
LLM 的价值在于解释「为什么是这个模型」和「下一步怎么调」，这是路线 A 目前没有的能力。

### 3.2 评测协议

- **数据协议**：`y` 目标出力、`H` 历史协变量（实测气象 `lmd_*`）、`F` 未来已知协变量（NWP 预报 `nwp_*`）、`unique_id` 多序列。
- **时序切分**：默认 60% / 20% / 20%。
- **训练内调参**：所有候选模型在 train 内部用统一的 inner `h` / `step` / `n_windows` 搜索超参；验证集只排名，测试集只报告。
- **多窗口滚动**：`step_size=96`（15 分钟数据即每天推进一次起点），逐窗口出预测、指标与图表。
- **三种集成**：模型级中位数、预测级中位数、验证 MAE 加权。
- **能力显式声明**：`model_catalog()` 查询，不假设每个模型都支持 H/F。

### 3.3 路线 B 的取舍

- ✅ 模型谱系最宽（statsforecast / mlforecast / neuralforecast / prophet / sktime / 基础模型），支持 Ray / Dask / Spark。
- ✅ 自带 PVOD 真实数据、示例与单元测试；LLM 报告能给出可读的选型理由。
- ❌ **没有发布治理与长期运行**：评完即止，不回答「新数据到了怎么办」。
- ❌ 依赖重（torch / ray / prophet …）；LLM 环节引入不可复现性。
- ❌ 目前不自动核验 NWP 的发布时间 / 版本，需在数据准备阶段自行对齐。

---

## 4. 两条路线共享的约定

无论走哪条路线，以下约束一致：

1. **未来协变量必须是预测时点可获得的预报**，不得使用事后实测观测。
2. **超参搜索与模型选择不允许发生在验证集/测试集上**。
3. **LLM 不参与数值决策**，只做意图理解与结果解释。
4. **模型能力必须显式声明**，不能假设任意模型都支持历史/未来协变量。

---

## 5. 设计取舍

| 取舍 | 路线 A 的选择 | 路线 B 的选择 |
|---|---|---|
| Agent 放哪 | 仓库外，宿主驱动 | 仓库内，库自带 |
| 编排确定性 | 契约 + CLI，完全确定 | LLM 工具调用，灵活 |
| 模型谱系 | 8 个 Provider，聚焦可控 | 尽量宽，支持分布式 |
| 长期运行 | 一等公民（轨道 / 门禁 / 审计） | 不涉及 |
| 依赖策略 | 轻量，可上场站 | 重量，面向离线实验 |
| 基础模型 | 本地权重 + 钉死版本，禁止运行时下载 | zero-shot 推理，不微调 |
| 可解释性 | 靠审计报告追溯 | 靠 LLM 报告解释 |

我们不预设哪条路线更优。两条路线并行推进，是为了在真实场景里对照，
再把验证有效的能力互相移植——详见 [路线图](roadmap.md)。
