---
name: release-candidate
description: 解释建模工作流产生的 Candidate、门禁结果和 Active 发布状态；不绕过确定性发布代码。
---

# Release Candidate

建模配置 `publish=true` 时，`run-modeling` 会在内部依次保存不可变 Solution、完成 Candidate、执行 artifact reload 与 smoke 门禁，再原子更新 Active。外部 Agent 不直接写 pointer，也不把“指标看起来不错”当成门禁通过。

完成后检查并报告：solution ID、Candidate 路径、门禁是否批准、拒绝原因、Active solution ID 和 rollback 前一版本。发布失败时必须说明旧 Active 未变化；Candidate 和报告保留用于诊断。`publish=false` 时只交付 Candidate，不应声称已发布。
