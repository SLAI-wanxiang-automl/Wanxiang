# 原有模型的滚动预测适配

代码入口：`eneygyexpert.rolling_models.create_model()`。
适配层与原 `models` 分开，原有 forecast API 保留。默认所有数据划分和调参都通过 `train_validation_test()` 执行。

## 数据与更新协议

- `y`：目标功率；`H`：显式声明的历史协变量，例如 lmd；`F`：显式声明的未来已知协变量，例如 nwp。
- 支持 F 的神经网络及 Prophet，需要历史中也有相同列的归档 F，用于训练；H 和 F 可以同名。
- 不接收静态变量、不建模 issued_at。不支持协变量的适配器只读取 y。
- `frozen`：保持权重/系数固定，每个起点更新预测输入。规则模型可以读取新历史；常数水平模型保持拟合时的水平。
- `state_update`：保持模型结构和系数固定，摄入截至 origin 的已观测数据。当前统计适配器通过完整历史重新过滤/递推重建状态，不调用参数优化；不是常数时间的增量实现。
- 更新不能读取当前起点之后的目标，重叠窗口不会重复摄入数据。历史修改、倒退或时间缺口会报错。
- 验证与测试使用相同更新策略；第一版工厂不提供逐窗口 refit。旧 rolling_test_forecast 的 refit 参数仅为兼容保留。

## 模型表

下面“支持”指适配器已实现，不表示所有大型 checkpoint 已下载或运行。

| 模型 | 家族 | 接收变量 | 重点超参数/配置 | 选参 | 更新 |
|---|---|---|---|---|---|
| AutoLinearRegression | MLForecast | y | lags, lag_transforms, fit_intercept | native | frozen |
| AutoRidge | MLForecast | y | lags, lag_transforms, alpha | native | frozen |
| AutoLasso | MLForecast | y | lags, lag_transforms, alpha | native | frozen |
| AutoElasticNet | MLForecast | y | lags, lag_transforms, alpha, l1_ratio | native | frozen |
| AutoRandomForest | MLForecast | y | lags, n_estimators, max_depth, min_samples_leaf | native | frozen |
| AutoLGBM | MLForecast | y | lags, learning_rate, num_leaves, min_child_samples, n_estimators | native | frozen |
| AutoXGBoost | MLForecast | y | lags, learning_rate, max_depth, n_estimators, reg_lambda | native | frozen |
| AutoCatboost | MLForecast | y | lags, learning_rate, depth, iterations, l2_leaf_reg | native | frozen |
| AutoPatchTST | NeuralForecast | y | input_size, patch_len, stride, hidden_size, n_heads, encoder_layers, dropout, learning_rate, max_steps | native | frozen |
| AutoNBEATS | NeuralForecast | y | input_size, n_blocks, mlp_units, learning_rate, max_steps | native | frozen |
| AutoNHITS | NeuralForecast | y + H + F | input_size, n_pool_kernel_size, n_freq_downsample, mlp_units, learning_rate, max_steps | native | frozen |
| AutoTFT | NeuralForecast | y + H + F | input_size, hidden_size, n_head, dropout, learning_rate, max_steps | native | frozen |
| AutoDeepAR | NeuralForecast | y + F | input_size, lstm_hidden_size, lstm_n_layers, lstm_dropout, learning_rate, max_steps | native | frozen |
| AutoARIMA | StatsForecast | y | season_length, order, seasonal_order, include_mean, include_drift | native | state_update |
| AutoETS | StatsForecast | y | season_length, model, damped | native | state_update |
| AutoCES | StatsForecast | y | season_length, model | native | state_update |
| Theta | StatsForecast | y | season_length, decomposition_type | framework | state_update |
| DynamicOptimizedTheta | StatsForecast | y | season_length, decomposition_type | framework | state_update |
| SeasonalNaive | StatsForecast | y | season_length | fixed | frozen |
| HistoricAverage | StatsForecast | y | 无外部搜索参数 | fixed | frozen |
| ZeroModel | StatsForecast | y | 无外部搜索参数 | fixed | frozen |
| CrostonClassic | StatsForecast | y | 无外部搜索参数 | fixed | frozen |
| ADIDA | StatsForecast | y | 无外部搜索参数 | fixed | frozen |
| IMAPA | StatsForecast | y | 无外部搜索参数 | fixed | frozen |
| Prophet | Prophet | y + F | changepoint_prior_scale, seasonality_prior_scale, seasonality_mode | framework | frozen |
| Chronos | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |
| TimesFM | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |
| Moirai | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |
| Toto | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |
| TiRex | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |
| Sundial | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |
| FlowState | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |
| T0 | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |
| Tafsut | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |
| PatchTSTFM | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |
| TabPFN | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |
| TimeGPT | Foundation | y | checkpoint/model, context_length, inference configuration | fixed | frozen |

补充：已有 `DirectRidgeWindowModel` 支持 y/H/F、lags/alpha、frozen。`MedianEnsembleAdapter` 支持固定配置成员，继承各成员变量要求和更新策略；不重新调参。`SKTimeRollingAdapter` 只接受声明可审计参数的 state_update 模型，更新后校验参数未改变。

## 各家族选参

1. **MLForecast**：沿用 AutoMLForecast 的原生搜索空间及 Optuna；统一 inner h、step、窗口数、refit=True 和评分指标。最佳配置同时保留模型参数、lag/变换配置和拟合配置。第一版仅 y。
2. **NeuralForecast**：沿用原 Auto 类默认空间和 Optuna 引擎；用统一多窗口评分替换原先单验证段目标。每 trial 在各内层起点重新拟合，取共同指标平均值。固定 max_steps；不额外切出外部验证段做 early stopping。自定义空间可传 Optuna callable 或原 Ray 字典；这一滚动适配层执行 Optuna。
3. **AutoARIMA/ETS/CES**：原生 IC 搜索在若干训练前缀提出结构（num_samples 控制提案数量）；所有不同结构在相同内层窗口评分，冻结优胜结构。验证/测试重新估计该结构的系数一次，不再次自动选阶/选结构。各序列可以有各自结构。
4. **Theta/Prophet**：提供默认小型框架参数网格，可用 tune_grid() 自定义。Theta 的平滑参数由底层在拟合阶段估计；滚动时固定平滑参数及训练季节模式。
5. **Foundation**：第一版仅目标序列的 zero-shot 推理；关闭滚动微调，复用模型对象/底层缓存。TimeGPT 需要服务凭据及远程请求。不是每个模型都需要在前80%上训练，zero-shot 模型仅准备预测器。

## 6:2:2 执行

前60%内调参 → 最佳参数在前60%拟合一次 → 中间20%滚动验证选模 → 优胜参数在前80%拟合一次 → 最后20%滚动测试。
内层窗口数由 min_train_size、h、step 自动推算；验证和测试省略不足 h 的末尾窗口。context_length 只限定预测输入，拟合默认读取完整训练前缀。
内层仍然需要每候选/每trial/每窗口训练；frozen/state_update 节约的是验证及测试的逐窗口重新训练。

```python
from utilsforecast.data import generate_series

from eneygyexpert.data import ForecastData
from eneygyexpert.pipeline import train_validation_test
from eneygyexpert.rolling_models import create_model

# 历史协变量列名（例如 lmd）与未来已知协变量列名（例如 nwp）；
# 需要协变量的模型（如 AutoTFT）通过 historical_covariate_cols /
# future_covariate_cols 显式声明，这里演示仅使用 y。
lmd_cols = []
nwp_cols = []
df = generate_series(n_series=1, freq="h", min_length=400, seed=0)
data = ForecastData(history=df, freq="h")
models = [
    create_model("SeasonalNaive", season_length=24),
    create_model("AutoARIMA", h=24, season_length=24, num_samples=3),
]
result = train_validation_test(data, models, h=24, step_size=24, seasonality=24)
```

在真实 PVOD 数据（15 分钟粒度）上使用 ML / 神经网络模型时，通常这样配置：

```py
models = [
    create_model("SeasonalNaive", season_length=96),
    create_model("AutoRidge", season_length=96, num_samples=3, context_length=672),
    create_model("AutoTFT", h=96, num_samples=3, context_length=672,
                 historical_covariate_cols=lmd_cols, future_covariate_cols=nwp_cols),
    create_model("AutoARIMA", h=96, season_length=96, num_samples=3),
]
result = train_validation_test(data, models, h=96, step_size=96, seasonality=96)
```

完整 PVOD CLI：

```bash
PYTHONPATH=. python examples/pvod_models_pipeline.py dataset/pvod.csv \
  --models SeasonalNaive AutoRidge AutoTFT AutoARIMA \
  --h 96 --step 96 --num-samples 3 --output outputs/pvod_models
```

## 结果和权重

`result.fixed_models` 保存固定配置；`result.tuning_logs` 保存选择过程；`result.validation[alias].fitted_model` 和 `result.test.fitted_model` 保存实际拟合对象。
`save_pipeline_result()` 导出 CSV 和 selection.json；`save_fitted_model()` 可保存本地模型（CLI --save-weights）。状态模型保存的是评估结束时的状态与固定系数。基础模型/远程客户端仅保存配置引用，避免序列化 checkpoint 或服务凭据。

## 验证范围

测试文件：`tests/test_rolling_model_adapters.py`。覆盖真实 StatsForecast、MLForecast 和五种 NeuralForecast 模型的微型训练、变量支持、系数/权重保持不变、重复更新及共同窗口；原有流水线测试继续保留。
Foundation 使用注入预测器验证统一数据和生命周期协议，实际 checkpoint、依赖版本和服务连接需要在目标环境验证。Sktime 使用能审计参数的预测器验证守卫，具体模型需要声明自身参数名。
第一版输出点预测；概率区间及依赖版本变化需单独适配。

## PVOD script parameters and test plots

For 15-minute PVOD data, `--h 96` predicts 24 hours, `--step 96` advances
24 hours, `--context 672` caps supported rolling inputs at seven days, and
`--season-length 96` represents daily seasonality (also the MASE scale lag).
Context does not truncate final fitting history; state-update models retain
full observed history. Neural input_size is selected within the context cap.
`--min-train-size` sets the earliest inner training prefix; omitted means the
common minimum required by the candidates. Window counts are automatic.

`--num-samples` defaults to 3: ML/neural models use it as the search trial
budget; AutoARIMA/AutoETS/AutoCES use it for native structure proposal prefixes,
not the number of internal ARIMA/ETS configurations. Fixed models and explicit
framework grids do not use this budget. Three is useful for a smoke test; a
larger trial budget must be chosen against inner-window count and compute cost.

`--accelerator auto` is passed only to neural Lightning trainers, with one
selected device. Use `gpu` on a CUDA-enabled GPU server and `cpu` for CPU.
It does not enable GPU boosting or configure foundation backend devices.
`--output` chooses the report directory. `--save-weights` writes the selected
fitted model to selected_model.joblib; state-update checkpoints include their
state at the last forecast origin. Foundation clients cannot be serialized by
this option.

```bash
.venv/bin/python examples/pvod_models_pipeline.py /path/to/pvod.csv \
  --models SeasonalNaive AutoRidge AutoPatchTST \
  --h 96 --step 96 --context 672 --season-length 96 \
  --num-samples 3 --accelerator gpu --output outputs/pvod_models \
  --top-k 3 --plot --plot-windows 5 --plot-seed 42 --plot-context 192
```

`--top-k` (default 1) selects the best k eligible models by validation metric.
Only these models are fitted once on train+validation and evaluated on test,
regardless of whether plotting is enabled. k must not exceed the candidate
count; insufficient eligible validation scores raise an error. Ties are
resolved by alias. Test results never change selection.

Their aligned test predictions also form `MedianEnsemble` (pointwise median)
and `WeightedEnsemble` (pointwise weighted sum). Weights are fixed using
validation MAE: w_i = (1 / MAE_i) / sum_j(1 / MAE_j). If any validation MAE is
exactly zero, only zero-error models receive weight, equally divided. These
ensembles reuse predictions and require no additional fitting. Selection JSON
records selected_models and ensemble_weights. Existing selected_model and
--save-weights continue to refer to the first-ranked individual model.
For k=1, both ensembles equal the sole selected model.

`--plot` displays only these k models and the two ensembles. Plotting itself
never fits or expands the set of tested models.
Random common (series, origin) windows are sampled reproducibly without
replacement; fewer than five available windows are all plotted. Each panel
shows observed target history, future actual targets, and all predictions.
`--plot-context` controls display only and defaults to `--context`; models may
have different actual input lengths.

Outputs: `test_windows.png`, `plotted_test_windows.csv`,
`test_comparison_scores.csv`, `test_comparison_forecasts.csv`. Existing
`test_scores.csv` and `test_forecasts.csv` continue to contain the winner only.
