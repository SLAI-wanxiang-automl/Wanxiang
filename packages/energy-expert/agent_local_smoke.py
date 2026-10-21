"""EneygyExpert Agent + 本地 vLLM 端点的最小冒烟测试。

用法:
    OPENAI_BASE_URL=http://localhost:8001/v1 OPENAI_API_KEY=dummy \
        .venv/bin/python agent_local_smoke.py
"""

import os

os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from utilsforecast.data import generate_series  # noqa: E402

from eneygyexpert import EneygyExpert  # noqa: E402
from eneygyexpert.models.stats import AutoARIMA, SeasonalNaive, Theta  # noqa: E402

freq = "D"
df = generate_series(n_series=1, freq=freq, min_length=180, seed=0)
print(f"[data] 合成数据 {df.shape}", flush=True)

expert = EneygyExpert(
    llm=os.environ.get("EE_LLM", "openai:qwen3"),
    forecasters=[AutoARIMA(), Theta(), SeasonalNaive()],
    retries=2,
)

print("[agent] 开始 analyze ...", flush=True)
result = expert.analyze(df=df, freq=freq, h=12)

print("\n===== 结果 =====", flush=True)
print("selected_model:", result.output.selected_model, flush=True)
print("better_than_naive:", result.output.is_better_than_seasonal_naive, flush=True)
print("reason:", (result.output.reason_for_selection or "")[:300], flush=True)
print("tsfeatures_analysis:", (result.output.tsfeatures_analysis or "")[:300], flush=True)
if result.eval_df is not None:
    print("MASE:\n", result.eval_df.to_string(), flush=True)
if result.fcst_df is not None:
    print("forecast:\n", result.fcst_df.tail(5).to_string(), flush=True)
