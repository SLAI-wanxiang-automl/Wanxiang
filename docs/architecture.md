# 架构设计

> 本文描述明月·万象的两层架构、组件职责边界，以及运行系统如何用权限与门禁约束执行。

## 1. 分层

```
L2  运行与治理   packages/energy-tsf
    ─────────────────────────────────────────────
    任务轨道 · Candidate/Active · 发布门禁 · 审计日志 · 分轨权限

L1  模型与评测   packages/energy-expert
    ─────────────────────────────────────────────
    训练内调参 · 验证集选型 · 测试集报告 · 多窗口滚动 · 集成

L0  数据与协议   packages/energy-expert
    ─────────────────────────────────────────────
    y / H / F 协议 · 时序切分 · 时点对齐 · 场站元数据
```

L0–L1 回答「**用什么模型**」，L2 回答「**怎么一直用下去**」。
两层之间不共享内存状态，只通过文件系统上的契约构件（SearchPool、Solution、Candidate）连接。

## 2. 建模引擎（L0–L1）内部数据流

```
原始 CSV / Parquet
      │
      ▼
ForecastData(y, H, F, unique_id)
      │   · y   目标出力
      │   · H   历史协变量（实测气象 lmd_*）
      │   · F   未来已知协变量（数值天气预报 nwp_*）
      ▼
按时间顺序切分  train 60% / validation 20% / test 20%
      │
      ├─ 训练内调参：所有候选模型在 train 内部用统一的 inner h / step / n_windows 搜索超参
      │
      ├─ 验证集：按 MAE / RMSE / MASE 排名，取 top-k，确定集成权重
      │
      └─ 测试集：只做最终报告，不参与任何选择
      │
      ▼
多窗口滚动（step_size = 96，15min 数据即每天推进一次起点）
      │
      ▼
leaderboard · 逐窗口预测 · 指标 · 图表 · experiment_log.json
      │
      ▼
（可选）LLM 读取 experiment_log.json 生成《训练经验报告》《最终测试报告》
        LLM 只解释结果；排名由验证指标决定
```

**关键约束**：超参搜索不允许发生在验证集或测试集上；最终测试阶段会用已固定配置在
train+validation 历史上重新拟合，因此测试权重不是训练权重的原样复用。

## 3. 运行系统（L2）内部数据流

```
宿主 Agent（自然语言）
      │
      ▼
.agents/skills/{modeling,forecast,backtest,update}/SKILL.md
      │   自然语言 → 结构化 Request → TOML
      ▼
prepare-*   生成权威 request.json / *.toml / resolved.toml（带 config_hash）
      │
      ▼
inspect-*   检查数据、切分、可用 Provider、trial 计划（不训练、不写 Candidate）
      │
      ▼
run-*       确定性执行：清洗 → 训练 → 指标 → 排名 → 写 Candidate
      │
      ▼
发布门禁    构件完整性 · 可重载性 · smoke inference · 元数据钉死
      │
      ├── 通过 → 原子替换 Active
      └── 不通过 → 保留原 Active，记录原因
```

## 4. 四类任务轨道与权限

运行系统用 `configs/tracks/*.toml` 显式声明每条轨道能做什么。这是防止「回测时偷练模型」
这类越界行为的机制，而不是靠约定：

| 能力 | modeling | forecast | backtest | update |
|---|:--:|:--:|:--:|:--:|
| 读 Active | — | ✅ | ✅ | ✅ |
| 写 Candidate | ✅ | ❌ | ✅ | ✅ |
| 写 Active | ✅ | ❌ | ✅ | ✅ |
| 允许拟合 / 存模型 | ✅ | ❌ | **❌** | ✅ |
| 允许下载权重 | 受策略约束 | ❌ | ❌ | 受策略约束 |
| 允许发布 | ✅ | ❌ | ✅ | ✅ |

语义边界同样写死在任务定义里：

- **backtest** 只调整既有成员的集成权重，**不重训模型**。
- **update** 只在 Active 绑定的 `SearchPool` 内重训，**不扩大搜索空间**，避免每次更新都漂移、无法归因。
- **forecast** 只读 Active，不训练、不改动已发布方案。

## 5. 资产与审计

```
asset_store/
├── active/          当前已发布方案
├── candidate/       待发布方案
├── solutions/       方案定义（成员 + 集成权重）
├── trials/          试验记录
├── search_pools/    Active 绑定的候选搜索池
├── objects/         内容寻址对象
└── ledger/          append-only 事件日志
```

系统策略（`configs/policies/system.toml`）：

- `content_addressed = true`：资产按内容寻址，天然去重与防篡改。
- `ledger.enabled = true`, `append_only = true`：事件只追加。
- `foundation.local_files_only = true`, `allow_download = false`, `require_pinned_revision = true`
  —— 基础模型权重必须本地化、版本钉死，不复制权重进资产库，也不在运行时下载。

## 6. 两层如何连接（当前状态与目标）

**当前**：两个组件分别调用。建模引擎产出 leaderboard 与集成权重；运行系统负责发布与运维。
跨层交接目前由人工 / 脚本完成。

**目标**：建模引擎的一次评测结果直接物化为运行系统的 Candidate，走同一套发布门禁：

```
energy-expert: train_validation_test(...)
        │  leaderboard + ensemble_weights + 模型构件
        ▼
   适配层（进行中）
        ▼
energy-tsf: Candidate → 发布门禁 → Active
```

进度见 [路线图](roadmap.md)。

## 7. 设计取舍

| 取舍 | 选择 | 原因 |
|---|---|---|
| 数值决策由谁做 | 确定性代码，不由 LLM 做 | 可复现、可审计、可回归测试 |
| 模型谱系放在哪 | 全部放在建模引擎 | 运行系统不需要知道具体模型实现 |
| 基础模型权重 | 本地化 + 钉死版本 | 场站环境常无外网；合规可追责 |
| 组件耦合方式 | 文件系统契约，不共享内存 | 可独立演进、可独立测试 |
| 代码是否合并 | 否，保持两个包独立 | 职责与依赖差异大，强行合并会互相拖累 |
