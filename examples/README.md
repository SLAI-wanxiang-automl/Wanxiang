# 两条路线的流程对照

> **两条路线是并列的**，各自端到端解决自己的目标场景。本文并列给出两条路线的完整流程，
> 不存在「先跑 B 再把结果交给 A」的必然顺序。

## 路线 A · Harness（`energy-tsf`）

```
宿主 Agent 读 SKILL.md 理解意图
        │
        ▼
prepare-*   自然语言 → 结构化 Request → TOML（config_hash）
        │
        ▼
inspect-*   检查数据 / 切分 / Provider / trial 计划（不训练）
        │
        ▼
run-*       确定性执行 → 写 Candidate
        │
        ▼
发布门禁 → 通过则 Active，不通过则保留原版本
```

### 一步到位：建模

在路线 A 中，宿主 Agent（Codex / Claude Code 等）会把自然语言转成结构化请求：

```text
建模：使用 pvod.csv，为 pvod-solar 建立未来 1 天的多变量出力预测方案。
目标列是 power，时间列是 date，采样频率为 15 分钟；比较 Ridge 和 XGBoost，
保留 Top-5，并在发布门禁通过后发布。
```

等价的 CLI 流程：

```bash
cd packages/energy-tsf
pip install -e .
energy-tsf prepare-modeling --request-id pvod-001 --station-id pvod-solar \
  --data <path/to/pvod.csv> --target power --timestamp-column date \
  --frequency 15min --output-dir work_dirs/train/pvod-001
energy-tsf inspect-modeling --config work_dirs/train/pvod-001/resolved.toml --workspace-root .
energy-tsf run-modeling    --config work_dirs/train/pvod-001/resolved.toml --workspace-root .
```

### 长期运行：预测 / 回测 / 更新

```bash
energy-tsf prepare-forecast --request-id pvod-fc-001 --station-id pvod-solar ...
energy-tsf run-forecast     --config work_dirs/forecast/pvod-fc-001/resolved.toml --workspace-root .

energy-tsf prepare-backtest --request-id pvod-bt-001 --station-id pvod-solar ...
energy-tsf run-backtest     --config work_dirs/backtest/pvod-bt-001/resolved.toml --workspace-root .

energy-tsf prepare-update --request-id pvod-up-001 --station-id pvod-solar ...
energy-tsf run-update     --config work_dirs/update/pvod-up-001/resolved.toml --workspace-root .
```

回测只调整既有成员的集成权重，不重训模型；更新只在 Active 绑定的 `SearchPool` 内进行。
两者同样需过发布门禁。

---

## 路线 B · Agentic Workflow（`energy-expert`）

```
ForecastData(y, H, F)
        │
        ▼
train_validation_test()   60/20/20 → 训练内调参 → 验证选型 → 测试只报告
        │                          ↑
        │                  多窗口滚动（step_size 推进起点）
        ▼
leaderboard · 逐窗口预测 · 指标 · 图表 · experiment_log.json
        │
        ▼
（可选）LLM 读 experiment_log.json → 《训练经验报告》《最终测试报告》
```

### 一次完整评测

```bash
cd packages/energy-expert
uv sync --locked --group dev
uv run python examples/quickstart.py          # 合成数据，先验证环境
uv run python examples/pvod_full_experiment.py dataset/PVOD/station00.csv \
  --accelerator cpu --num-samples 3 --inner-windows 8 \
  --top-k 5 --output outputs/pvod_full
```

产出 `outputs/pvod_full/experiment_log.json`、评测 CSV 与逐窗口图表。

### LLM 报告（可选）

配置 OpenAI 兼容服务后生成解释性报告：

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
export EE_REPORT_MODEL=qwen3
uv run python scripts/llm_reports.py --mode experience --log outputs/pvod_full/experiment_log.json
uv run python scripts/llm_reports.py --mode final      --log outputs/pvod_full/experiment_log.json
```

### 交互式分析入口

```python
result = train_validation_test(data, models, h=96, step_size=96, seasonality=96, metric="mae")
print(result.leaderboard, result.selected_model, result.test.summary)
# EneygyExpert.analyze() 会调用时序特征分析 / 交叉验证 / 预测工具，给出分析与选型说明
```

---

## 两条路线并用（可选，非必需）

如果你既想要宽模型谱系选出好模型，又想要可追责的长期运行，可以组合使用——
但这是**人工衔接**，不是系统内的自动贯通（自动贯通见 [路线图](../docs/roadmap.md)）：

| 阶段 | 用哪条 | 做什么 |
|---|---|---|
| 1. 选型 | 路线 B | 用宽模型谱系跑滚动评测，确定模型家族与超参区间 |
| 2. 收窄 | 路线 B → A | 把选出的少量候选写进路线 A 的 `SearchPool` |
| 3. 上线 | 路线 A | 建模 → 发布门禁 → Active，之后走预测 / 回测 / 更新 |

---

## 注意事项

- 真实预测时，未来协变量必须是**预测时点实际可获得**的预报数据，不能用事后观测。
- 路线 A 的预测 / 回测 / 更新都需要场站配置与 Active 资产；尚无 Active 时请先完成建模。
- 路线 B 首次运行耗时较长（依赖安装、内核编译、基础模型权重下载、多窗口搜索）。
  建议先用 `examples/quickstart.py` 与 SeasonalNaive 验证流程，再扩大规模。
- 路线 B 当前不自动核验 NWP 的发布时间 / 版本，需在数据准备阶段按预测起点自行对齐。
