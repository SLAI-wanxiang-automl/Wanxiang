---
name: update
description: 将“用近期数据重新执行已有场站 AutoML 和模型训练”的请求转换为结构化更新配置，调用候选池约束的重训与发布流程。
---

# Update

## 何时使用

当用户要求已有场站使用近期真实出力重新训练模型、重新选择成员并更新预测方案时使用。首次建立模型用 `modeling`；只评估并调整原成员权重用 `backtest`；只生成未来预测用 `forecast`。

仓库内不注册更新智能体。外部宿主 Agent 负责把自然语言转换为 `UpdateRequest`，窗口切分、候选池检索、训练、指标、集成和发布全部由确定性代码完成。

## 必须明确的输入

- 已有场站 ID，且必须存在同目标、同频率、同 horizon 的 Active；
- 最近数据文件、目标列、时间戳列、采样频率和 IANA 时区；
- 第一版固定最后 7 天为 validation、此前数据为 train、horizon/stride 均为 `1d`；
- 默认发布；用户只要求演练时使用 `--no-publish`。

Agent 不生成 provider、超参数或搜索空间。它们必须从 Active Solution 绑定的 `search_pool_id` 检索。

## 执行步骤

1. 生成 `UpdateRequest`，调用 `prepare-update`。
2. 调用 `inspect-update`，精确固定 Active、SearchPool、数据快照和 UpdatePlan。
3. 向用户说明池大小、不可执行候选和七个验证窗口。
4. 调用 `run-update`，只重训池内候选，产生下一代池、Top-K、集成和最终模型。
5. 报告父/新方案同口径指标、发布门禁和 Active 是否切换。

## 关键约束

- 不检索相似场站，不读取场站“最新池”，只读取 Active Solution 绑定的池；
- validation 每个窗口只读取 origin 之前的数据；缺失真值不计指标；
- 最后 7 天是 validation，不伪造 test；
- 旧 Active 没有 `search_pool_id` 时停止并提示重新执行 Modeling；
- 发布前 Active 必须仍是计划冻结的父版本；新方案七日 RMSE 变差时保留 Candidate 但不发布。

## 完成标准

必须生成冻结 plan、候选训练记录、逐窗口榜单、下一代 SearchPool、新 Solution、报告和结果文件。请求发布时，只有门禁通过才原子切换 Active。
