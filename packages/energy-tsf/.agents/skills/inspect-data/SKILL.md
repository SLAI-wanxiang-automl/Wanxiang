---
name: inspect-data
description: 在正式训练前检查时间序列数据、切分、缺失情况、可用 Provider 和确定性 trial 计划，不执行训练或发布。
---

# Inspect Data

本 Skill 是 modeling 的准备步骤，不是独立智能体。先使用 modeling Skill 生成并校验 `resolved.toml`，再调用：

```text
energy-tsf inspect-modeling --config <resolved.toml> --workspace-root <project>
```

向用户解释：总行数、原始缺失和插值数量、70%/20%/10% 边界、一天对应的步数、三个 split 的有效 origins、计划 trial 数、不可用 Provider 及原因。该命令必须保持只读：不训练、不写 Candidate、不更新 Active。若 validation 或 test 没有完整的一天窗口，应停止并说明数据长度不足。
