# Configs

- `workflows/modeling.toml`：建模工作流默认值；
- `workflows/forecast.toml`、`workflows/backtest.toml`、`workflows/update.toml`：预测、回测与更新默认值；
- `providers/*.toml`：八个 Provider 的显式、有限搜索候选；
- `tracks/*.toml`：训练、预测、回测与更新轨的显式权限边界；
- `stations/*.toml`：场站配置样例；
- `policies/system.toml`：系统级安全与存储策略预留。

任务级 TOML 只声明用户约束，不复制实现细节。正式执行始终保存展开后的 `resolved.toml` 和稳定配置哈希。
