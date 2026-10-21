# 贡献指南

## 仓库结构

这是一个 monorepo，但**两个包是独立的**：

| 目录 | 说明 | 语言 | 测试 |
|---|---|---|---|
| `packages/energy-expert` | 建模引擎（L0–L1） | Python 3.10–3.13 | `uv run pytest` |
| `packages/energy-tsf` | 运行系统（L2） | Python 3.12+ | 待补齐 |

伞仓根目录只放品牌文档、跨组件示例、CI 与同步脚本，**不放业务代码**。
改动业务代码请进对应的 `packages/` 子目录。

## 提交到哪里

- 改建模引擎 → 在 `packages/energy-expert/` 下改，直接提交到本仓库即可。
- 改运行系统 → 在 `packages/energy-tsf/` 下改，直接提交到本仓库即可。
- 需要把改动回推到子项目的独立仓库 → `./scripts/sync-upstream.sh push <component>`。
- 子项目独立仓库有更新 → `./scripts/sync-upstream.sh pull <component>`。

## 开发约定

1. **数值决策必须由确定性代码完成**，不由 LLM 输出决定。LLM 只做解释与报告。
2. **不允许在验证集或测试集上调参或选型**。超参搜索只发生在训练集内部。
3. **未来协变量必须是预测时点可获得的预报数据**，不得使用事后观测。
4. 新增模型时，必须显式声明其能力（是否支持历史/未来协变量、上下文长度）。
5. 运行系统的变更若涉及发布或资产写入，必须同时考虑对应的门禁与权限配置。

## 本地验证

```bash
# 建模引擎
cd packages/energy-expert
uv sync --locked --group dev
uv run python examples/quickstart.py
uv run pytest -o addopts='' tests/test_data.py tests/test_pipeline.py

# 运行系统
cd ../energy-tsf
pip install -e .
energy-tsf --help
```

完整测试（含深度学习与分布式后端）耗时较长，CI 中设为手动触发。

## 文档改动

- 中英文档成对维护：`README.md` ↔ `README.en.md`，两侧规则与命令名必须一致。
- 代码标识符、CLI 命令、序列化字段名、枚举值保持英文，不翻译。
