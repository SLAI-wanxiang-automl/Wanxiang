---
name: modeling
description: 将“建立、训练、比较或选择最佳时间序列模型”的自然语言请求转换为可审计的建模配置，并启动确定性建模流程。
---

# Modeling

## 何时使用

当用户要求基于一个新能源时间序列数据集建立、训练、比较、筛选或发布预测模型时使用本 Skill。若用户只要求用已发布模型生成未来预测，应改用 `forecast` Skill。

代码库内不调用 LLM 做意图识别，也不注册新的智能体。你是外部宿主 Agent，负责理解自然语言、检查必要信息并生成结构化请求；数据清洗、训练、指标、排名和发布均交给确定性代码。

## 必须明确的输入

- 数据文件路径：MVP 支持本地 CSV 或 Parquet；
- 目标列：只允许一个，存在多个可能目标时必须询问，不能猜测；
- 采样频率：使用 `10min`、`1h` 等格式；
- 场站标识；
- 时间戳列可选；未提供时按原始行顺序处理。

除非用户明确限制，否则使用以下默认值：未来 `1d`、单目标、按时间顺序 `70%/20%/10%` 切分、返回 Top-5、seed 42、最多 64 个 trial、并发 2，并允许全部八个 Provider。用户可显式覆盖 Top-K。首版发布默认开启；用户要求只看候选时传入 `--no-publish`。

## 执行步骤

1. 确认请求属于建模，而不是预测、回测或更新。
2. 收集必填输入；目标列或频率不明确时先询问用户。
3. 为本次请求生成只含字母、数字、点、下划线或短横线的 `request_id`。
4. 调用准备入口生成权威请求和 TOML：

   ```text
   energy-tsf prepare-modeling --request-id <id> --station-id <station> --data <path> --target <column> --timestamp-column <column> --frequency <frequency> --output-dir work_dirs/train/<id>
   ```

5. 检查返回值中的 `config_hash`、`resolved_config` 和 warnings。准备失败时向用户说明具体缺失或非法字段，不要绕过校验手工启动训练。
6. 先运行 `energy-tsf inspect-modeling --config <resolved.toml> --workspace-root <project>` 检查数据、切分、可用 Provider 和完整 trial 计划；该命令不训练，也不写 Candidate。
7. 用户要求正式执行后，运行 `energy-tsf run-modeling --config <resolved.toml> --workspace-root <project>`。完成后向用户汇报数据质量、trial 数量与失败、validation Top-K、一次性 test 结果、集成方案以及 Candidate/Active 引用。

## 关键约束

- 只能由 validation 指标选择 Top-K 和拟合集成权重；test 不能反向参与选择。
- 缺失值处理和辅助特征必须由代码完成，不能由 Agent 编造数值。
- 预测长度和切分比例在首版固定；Top-K 默认 5，可由自然语言请求显式覆盖。
- 训练轨先写 Candidate；只有发布门禁通过后才能原子更新 Active。
- 三个 TSFM 必须记录模型标识、固定 revision、上下文长度和本地权重引用，不复制大模型权重到资产库。

## 完成标准

准备阶段完成必须同时存在 `request.json`、`modeling.toml` 和 `resolved.toml`，且 resolved 配置可以重新加载并得到相同哈希。完整建模任务的完成标准由工作流报告判定，不以 Agent 的自然语言判断替代。
