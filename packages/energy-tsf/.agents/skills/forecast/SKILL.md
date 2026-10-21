---
name: forecast
description: 将“使用近期新能源出力生成未来预测”的自然语言请求转换为可审计的预测配置，并调用只读预测流程。
---

# Forecast

## 何时使用

当用户要求依据某场站近期出力，使用已有方案预测未来数据时使用本 Skill。若用户要求训练、比较或发布模型，应改用 `modeling` Skill。

你是外部宿主 Agent，负责理解自然语言并形成结构化请求。仓库内不注册预测子智能体；数据清洗、资产检索、相似度评分、尺度换算、模型推理、集成和约束均由确定性代码完成。

## 必须明确的输入

- 数据文件路径、目标列、采样频率；
- 场站 ID、能源类型、时区和目标单位；
- 可选时间戳列；未提供时只能交付相对预测步；
- 跨站绝对功率迁移需要目标场站容量和容量单位；
- 地点、技术类型等相似检索信息在用户提供或场站资料已有时使用，不能猜测。

首版预测长度固定为未来 `1d`。默认允许在没有同站 Active 时检索相似站；同站方案存在但不可执行时，默认不静默跨站回退。

## 执行步骤

1. 确认请求属于预测任务，而不是建模、回测或更新。
2. 将用户输入转换为 `ForecastRequest`，不要把自然语言直接传入数值代码。
3. 调用准备入口：

   ```text
   energy-tsf prepare-forecast --request-id <id> --station-id <station> --energy-type solar --timezone Asia/Shanghai --data <path> --target power --target-unit MW --timestamp-column timestamp --frequency 10min --capacity <mw> --capacity-unit MW --output-dir work_dirs/forecast/<id>
   ```

4. 检查返回的 `config_hash`、`resolved_config` 和 warnings。准备失败时说明缺失字段，不绕过校验。
5. 调用 `inspect-forecast` 清洗近期数据并解析资产：
   - 有同站 Active 时优先冻结该 Solution；
   - 无同站 Active 时读取兼容候选的公开事实、拒绝原因和分项评分；
   - 从兼容候选中选择 1～3 个来源，写入 `ForecastPlan`、权重和简短选择理由。
6. 同站路径会自动生成 `plan.json`。相似站路径需先把选择写成 `selection.json`，再调用：

   ```text
   energy-tsf plan-forecast --config <resolved.toml> --selection <selection.json> --workspace-root <project>
   ```

7. 调用 `run-forecast --config <resolved.toml> --workspace-root <project>` 执行冻结计划。向用户交付预测表、数据质量、方案来源、降级、约束和报告路径。

## 关键约束

- 预测轨只能读取 Active、不可变 Solution、模型对象和场站资料；不能读取 Candidate。
- 不得调用 `fit`、保存新模型、重新调参、重新拟合集成权重、下载权重或修改 Active。
- 相似度只对硬兼容候选排序；高分不能覆盖频率、目标、单位、容量或上下文不兼容。
- 跨站绝对功率必须使用明确容量换算；缺少容量时不能猜测。
- 计划必须固定具体 Solution ID；运行中 Active 更新不改变本次预测。
- 只记录选择事实和简短理由，不保存隐藏推理过程。

## 完成标准

准备阶段必须生成 `request.json`、`forecast.toml`、`resolved.toml`，且 resolved 配置重新加载后 hash 不变。完整任务还必须生成冻结计划、预测表、机器报告、人类报告和可追溯事件记录；是否完成由工作流结果判定。
