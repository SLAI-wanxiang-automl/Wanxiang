# EneygyExpert

面向光伏、风电等**新能源出力预测**的 Agentic AutoML 系统。项目以 LLM 辅助分析和候选模型探索，以可复现的滚动评测作为模型选型依据，统一统计模型、机器学习、深度学习与时间序列基础模型的使用方式。

> **第一次运行可能比较长。** 安装科学计算与深度学习依赖、首次编译计算内核、下载基础模型权重，以及多个滚动窗口中的超参数搜索都可能耗时。建议先用小规模数据和 SeasonalNaive 验证流程，再增加模型、搜索次数和预测窗口。

## 方法与特点

- **新能源场景的数据协议**：统一目标 `y`、历史协变量 `H` 和未来已知协变量 `F`。支持单变量预测、带多变量历史信息的预测，以及使用天气预报/NWP 等未来外生信息的预测；`unique_id` 支持多条序列。这里的“多元”主要指协变量输入，不代表所有适配器都支持联合多目标输出。
- **跨模型家族适配**：StatsForecast、MLForecast、NeuralForecast、Prophet、sktime 与基础模型通过统一窗口接口参与评测。能力按模型显式声明，不能假设每个模型都支持 H/F；可用 `model_catalog()` 查询。
- **训练内调参，验证集选型**：按时间顺序划分训练/验证/测试集（默认 60%/20%/20%）。在训练集内部按统一 horizon、step 和窗口数调参，确定各候选模型的超参数；在验证集上通过 MAE、RMSE 或 MASE 排名，选择 top-k；测试集只用于最终报告。模型差异由指标客观衡量。
- **明确的滚动更新语义**：区分固定参数预测与状态更新，历史观测可以进入后续预测窗口。框架调参和原生 AutoML 均遵守训练内搜索约束，并记录配置与日志。
- **可追踪的集成和产物**：支持模型级中位数集成、预测级中位数集成，以及使用验证集 MAE 确定权重的预测集成，输出排名、逐窗口预测、指标与图表。


### LLM 在系统中的作用

LLM 已有两类入口：

1. **交互式分析与工具调用**：`EneygyExpert.analyze()` 调用时序特征分析、交叉验证和预测工具，基于评测结果给出分析与模型选择说明。
2. **实验报告生成**：`scripts/llm_reports.py` 读取完整实验的 `experiment_log.json`，生成训练经验报告和最终测试报告。训练报告汇总各候选模型的最优超参数、训练内调参得分、验证排名与集成权重，并提出后续训练建议；最终报告分析测试指标、相对 AutoARIMA 基线的提升、不同预测步长误差与测试窗口图表。

H/F 滚动实验由 `train_validation_test()` 执行，候选模型与搜索预算目前由实验脚本配置，原生超参数搜索由 Optuna/模型后端执行。LLM 报告解释这些实验结果，模型排名仍由验证指标确定。交互式 Agent 与 H/F 实验流水线目前是分别调用的入口。

**后续方向**：从训练报告中提取可复用的训练经验，结合新场站的数据特征，辅助跨场站候选模型选择与搜索空间设计；最终仍通过滚动验证检验建议效果。目前已有训练经验报告生成，跨场站经验提取与自动选型尚未贯通。

验证阶段从训练区间拟合；最终测试阶段会用已固定的配置在训练+验证历史上重新拟合。因此测试阶段的权重并非训练阶段权重的原封不动复用，超参数搜索不会在验证/测试集上重复执行。

## 安装

建议新建 Python 3.11 或 3.12 环境。

```bash
git clone git@github.com:humble-gambler/EneygyExpert.git
cd EneygyExpert
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

也可使用 uv 安装锁定依赖与开发工具：

```bash
uv sync --locked --group dev
uv run python examples/quickstart.py
```

`pip install -e .` 会解析 pyproject.toml 中的版本范围；`uv sync --locked` 使用 uv.lock。不要直接复用其他机器复制来的 `.venv`。当前依赖较重，部分基础模型需要网络与模型访问权限；深度模型建议使用适配的 GPU/PyTorch 环境。仅运行确定性流水线不需要 LLM API key。项目尚未承诺已发布到 PyPI，请从源码安装。

## 快速教程：不需要 LLM 的完整滚动实验

```bash
python examples/quickstart.py
```

该示例生成合成日序列，比较两个季节性基线，按验证 MAE 选择模型并保存报告到 `outputs/quickstart/`。它适合首先检查数据划分和运行环境。

```python
import pandas as pd
from eneygyexpert.data import ForecastData
from eneygyexpert.pipeline import train_validation_test
from eneygyexpert.rolling_models import create_model

# CSV 至少包含 unique_id、ds、y；ds 为时间，y 为功率。
frame = pd.read_csv("power.csv")
data = ForecastData(frame, freq="15min")
models = [
    create_model("SeasonalNaive", season_length=96),
    create_model("AutoARIMA", season_length=96, num_samples=3),
]
result = train_validation_test(
    data, models, h=96, step_size=96, seasonality=96, metric="mae",
)
print(result.leaderboard)
print(result.selected_model)
print(result.test.summary)
```

保持时间戳规则、无重复、无缺失，且各序列共享同一时间网格。确保各区间足够容纳训练历史和完整预测窗口。`h=96` 对 15 分钟数据表示预测未来一天；`step_size=96` 表示每天推进一次预测起点。

## 使用历史天气与未来天气预报

```python
from eneygyexpert.data import ForecastData
from eneygyexpert.rolling_models import create_model, model_catalog

# history 含 unique_id、ds、y、measured_temperature。
# nwp 含 unique_id、ds、forecast_temperature，预测 horizon 必须完整覆盖。
data = ForecastData(
    history, freq="15min",
    historical_covariate_cols=("measured_temperature",),
    future_covariate_cols=("forecast_temperature",),
    future_covariates=nwp,
)
model = create_model(
    "AutoTFT", h=96, context_length=672, num_samples=3,
    historical_covariate_cols=("measured_temperature",),
    future_covariate_cols=("forecast_temperature",),
)
# 将 data 和 [model] 传入上述 train_validation_test()。
print(model_catalog())
```

未来外生变量必须是**在对应预测起点已经发布、可获得的预报**，不能用事后实测天气替代。当前数据结构不自动核验天气预报的发布时间/版本，需要在数据准备时按预测起点对齐。

## 使用仓库中的 PVOD 数据

仓库包含 `dataset/PVOD/station00.csv`（约 2.8 MB），是当前提供的 PVOD 单场站数据文件，并非完整多场站数据集。原始文件包含 28,896 行，覆盖 2018-08-15 16:00 至 2019-06-13 15:45，采样间隔为 15 分钟。

| 原始列 | 用途 |
| --- | --- |
| `date_time` | 时间，加载时映射为 `ds` |
| `power` | 目标功率，映射为 `y` |
| `lmd_*` | 本地历史气象观测，例如辐照度、温度、风速 |
| `nwp_*` | 数值天气预报，作为未来外生信息 |

示例的 `load_station()` 会排序、去除重复时间戳，并保留最后一个时间缺口之后的连续区间，不进行插值。当前保留 15,802 行，覆盖 2018-12-31 01:30 至 2019-06-13 15:45；随后实验脚本添加 `unique_id=PVOD` 并构建 H/F 输入。数据单位与来源授权见 [数据说明](dataset/PVOD/README.md)。

先运行季节性基线，检查数据和报告输出：

```bash
python examples/pvod_models_pipeline.py dataset/PVOD/station00.csv \
  --models SeasonalNaive --h 96 --step 96 --season-length 96 \
  --top-k 1 --plot --output outputs/pvod_baseline
```

增加 AutoRidge，比较多个模型：

```bash
python examples/pvod_models_pipeline.py dataset/PVOD/station00.csv \
  --models SeasonalNaive AutoRidge --h 96 --step 96 \
  --season-length 96 --num-samples 3 --top-k 2 --plot
```

生成供 LLM 使用的完整训练与测试日志：

```bash
python examples/pvod_full_experiment.py dataset/PVOD/station00.csv \
  --accelerator cpu --num-samples 3 --inner-windows 8 \
  --top-k 5 --output outputs/pvod_full
```

完整实验覆盖统计、ML、神经网络和 Prophet，计算量明显较大；有可用 GPU 时将 `--accelerator cpu` 改为 `--accelerator gpu`。它会保存 `experiment_log.json`、主实验/基线评测 CSV 和测试窗口图。只有完整实验入口生成下述 LLM 报告需要的日志格式；简化多模型示例的报告 CSV 不能直接替代。

更完整的参数见 `examples/pvod_models_pipeline.py`、`examples/pvod_split_pipeline.py` 与 `docs/rolling_models.md`。不同模型对 H/F 的支持不同，默认 AutoRidge 示例不使用天气外生变量；AutoTFT/AutoNHITS 等具备相应能力的适配器可使用 H/F。

## LLM 接入：本地服务与远程 API

交互式 Agent 与报告生成均支持 OpenAI 兼容协议，通过环境变量配置服务地址、API key 与模型名。

### 本地 vLLM

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
export EE_LLM=openai:qwen3
export EE_REPORT_MODEL=qwen3
python agent_local_smoke.py
```

`qwen3` 应替换为服务实际部署的模型名。`EE_LLM` 用于 Agent，带 `openai:` 提供商前缀；`EE_REPORT_MODEL` 用于报告脚本，填写 API 接受的原始模型名。不设置后者时，报告脚本会通过 `/models` 获取第一个模型。

### 远程 API

将地址改为供应商提供的 OpenAI 兼容 API 基础地址，填写对应模型名和凭据：

```bash
export OPENAI_BASE_URL=https://your-provider.example/v1
export OPENAI_API_KEY='YOUR_API_KEY'
export EE_LLM=openai:your-model-name
export EE_REPORT_MODEL=your-model-name
python agent_local_smoke.py
```

以上地址和模型名为占位符，请替换后执行。显式设置 `EE_REPORT_MODEL` 后无需调用供应商的 `/models` 接口。报告脚本使用 `Authorization: Bearer` 鉴权，请选择支持 `/chat/completions` 的兼容服务。凭据通过环境变量传入，不写入代码或提交到仓库。

### 生成训练经验与最终测试报告

先完成上面的完整 PVOD 实验，再在配置好本地或远程 LLM 后执行：

```bash
python scripts/llm_reports.py --mode experience \
  --log outputs/pvod_full/experiment_log.json
python scripts/llm_reports.py --mode final \
  --log outputs/pvod_full/experiment_log.json
```

默认输出：

- `outputs/pvod_full/training_experience.md`：训练设置、超参数与验证表现、家族规律、调参建议。
- `outputs/pvod_full/final_report.md`：最终测试比较、基线提升、预测步长误差和窗口分析。
- `outputs/pvod_full/plots/window_*.png`：最终报告引用的逐窗口图表，与 Markdown 一起保留。

可用 `--output /path/to/report.md` 自定义报告路径；最终报告默认图片路径相对于实验输出目录，移动 Markdown 时应一并保留或调整图片路径。远程报告调用会将模型配置、评测指标和窗口数值摘要发送给所配置的 API 服务；`analyze()` 则会根据工具调用发送时序特征和评测/预测摘要。LLM 建议属于对实验的解释，天气成因与跨场站泛化判断仍需额外证据。

## 测试与开发

```bash
uv run pytest -o addopts='' tests/test_data.py tests/test_window_models.py \
  tests/test_backtesting.py tests/test_tuning.py tests/test_test_rolling.py \
  tests/test_pipeline.py tests/test_native_automl.py
uv run pytest
```

默认测试配置排除 docs/live/distributed，原生模型测试需要相应后端依赖；完整运行可能耗时较长。

`MedianEnsembleAdapter` 的默认模型别名为 `MedianModelEnsemble`；`MedianEnsemble` 与 `WeightedEnsemble` 保留给流水线生成的预测级集成。

## 项目结构

- `eneygyexpert/rolling_models/`：模型能力、适配、集成、报告与绘图。
- `eneygyexpert/data.py`：窗口数据协议与时间划分。
- `eneygyexpert/pipeline.py`：训练内调参、验证选型与测试。
- `eneygyexpert/agent.py`：LLM 工具调用与分析。
- `examples/`、`scripts/`、`tests/`：教程、实验入口与测试。

## 来源与许可证

项目基于 [TimeCopilot](https://github.com/TimeCopilot/timecopilot) 扩展，保留上游 MIT 版权声明。代码采用 [MIT License](LICENSE)，允许使用、修改、分发和商业使用，分发时需保留版权与许可证声明。依赖库、第三方模型权重和数据集遵循其各自许可证；本项目 MIT 不替代它们的授权条件。
