# 跨组件端到端流程

> **当前状态**：两个组件**分别调用**。本目录描述目标流程，并给出现阶段可逐步执行的命令。
> 自动贯通（引擎结果直接成为运行系统的 Candidate）见 [路线图](../docs/roadmap.md)。

## 目标流程

```
① 建模引擎选型          energy-expert: train_validation_test()
                              │
                              │  leaderboard + ensemble_weights + 模型构件
                              ▼
② 物化为 Candidate      （适配层，进行中）
                              │
                              ▼
③ 发布门禁              energy-tsf: gates/release.py
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              通过 → Active        不通过 → 保留原 Active
                    │
                    ▼
④ 日常预测              energy-tsf: run-forecast（只读 Active）
                    │
                    ▼
⑤ 滚动回测              energy-tsf: run-backtest（只调权重，不重训）
                    │
                    ▼
⑥ 受约束更新            energy-tsf: run-update（只在 SearchPool 内重训）
```

## 现阶段分步执行

### ① 用建模引擎在 PVOD 上完成一次完整评测

```bash
cd packages/energy-expert
uv sync --locked --group dev
uv run python examples/pvod_full_experiment.py dataset/PVOD/station00.csv \
  --accelerator cpu --num-samples 3 --inner-windows 8 \
  --top-k 5 --output outputs/pvod_full
```

产出 `outputs/pvod_full/experiment_log.json`、评测 CSV 与逐窗口图表。

可选：配置 OpenAI 兼容的 LLM 服务后生成解释性报告。

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
export EE_REPORT_MODEL=qwen3
uv run python scripts/llm_reports.py --mode experience --log outputs/pvod_full/experiment_log.json
uv run python scripts/llm_reports.py --mode final      --log outputs/pvod_full/experiment_log.json
```

### ② 用运行系统建立并发布方案

在运行系统中，宿主 Agent（Codex / Claude Code 等）会把自然语言转成结构化请求：

```text
建模：使用 pvod.csv，为 pvod-solar 建立未来 1 天的多变量出力预测方案。
目标列是 power，时间列是 date，采样频率为 15 分钟；比较 Ridge 和 XGBoost，
保留 Top-5，并在发布门禁通过后发布。
```

等价的 CLI 流程：

```bash
cd ../energy-tsf
pip install -e .
energy-tsf prepare-modeling --request-id pvod-001 --station-id pvod-solar \
  --data <path/to/pvod.csv> --target power --timestamp-column date \
  --frequency 15min --output-dir work_dirs/train/pvod-001
energy-tsf inspect-modeling --config work_dirs/train/pvod-001/resolved.toml --workspace-root .
energy-tsf run-modeling    --config work_dirs/train/pvod-001/resolved.toml --workspace-root .
```

### ③④ 预测与回测

```bash
energy-tsf prepare-forecast --request-id pvod-fc-001 --station-id pvod-solar ...
energy-tsf run-forecast     --config work_dirs/forecast/pvod-fc-001/resolved.toml --workspace-root .

energy-tsf prepare-backtest --request-id pvod-bt-001 --station-id pvod-solar ...
energy-tsf run-backtest     --config work_dirs/backtest/pvod-bt-001/resolved.toml --workspace-root .
```

回测只调整既有成员的集成权重，不重训模型；新方案同样需过发布门禁。

### ⑤ 受约束更新

```bash
energy-tsf prepare-update --request-id pvod-up-001 --station-id pvod-solar ...
energy-tsf run-update     --config work_dirs/update/pvod-up-001/resolved.toml --workspace-root .
```

更新只在 Active 绑定的 `SearchPool` 内进行，不扩大搜索空间。

## 注意事项

- 真实预测时，未来协变量必须是**预测时点实际可获得**的预报数据，不能用事后观测。
- 预测、回测、更新都需要场站配置与 Active 资产；尚无 Active 时请先完成建模。
- 首次运行建模引擎耗时较长（依赖安装、内核编译、基础模型权重下载、多窗口搜索）。
  建议先用 `examples/quickstart.py` 与 SeasonalNaive 验证流程，再扩大规模。
