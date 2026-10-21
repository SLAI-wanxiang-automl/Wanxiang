"""Generate LLM-written markdown reports from the PVOD experiment log.

Modes
-----
experience : training-log experience report (task 3). Inputs: every
             candidate's best hyperparameters, inner tuning score, and
             validation metrics.
final      : final forecast analysis report (task 4). Inputs: top-k test
             metrics, ensemble results, AutoARIMA baseline, per-lead errors,
             and a numeric digest of the plotted test windows.

Supports local vLLM and remote OpenAI-compatible APIs via environment variables.
Output is markdown.

Usage:
    python scripts/llm_reports.py --mode experience
    python scripts/llm_reports.py --mode final
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_ID = os.environ.get("EE_REPORT_MODEL", "")


def api_headers():
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


EXPERIENCE_SYSTEM = """你是一位资深的电力负荷/光伏功率预测算法工程师，负责审阅时间序列预测实验的训练日志并沉淀经验。
你会收到一份 JSON 格式的实验日志，包含：数据集信息、数据划分边界、每个候选模型在训练段内调参选出的最优超参数、内层交叉验证得分、验证集评测指标（MAE/RMSE/MASE）、top-k 入选名单与集成权重。
请基于数据本身推理，不要编造日志中不存在的数字。输出为简体中文 markdown。"""

EXPERIENCE_PROMPT = """请根据下面的实验日志，撰写一份《训练日志经验》报告，markdown 格式，结构如下：

# PVOD 光伏功率预测训练日志经验

## 1. 实验设置摘要
（数据、划分、调参协议；2-4 句）

## 2. 各候选模型调参结果与验证表现
用一张 markdown 表格列出全部候选模型：模型名 | 所属家族 | 最优超参数（紧凑列出）| 验证 MAE | 验证 RMSE | 验证 MASE。按验证 MAE 升序。

## 3. 家族级规律
（统计/ML/神经网络/Prophet 各家族在验证集上的相对表现，哪些类型的模型和超参组合在本数据上有效，哪些无效并给出可能原因）

## 4. 调参经验与建议
（哪些超参方向值得继续探索、哪些可以剪枝、采样预算是否充足、下次实验的具体可执行建议）

## 5. 风险与注意事项
（例如过拟合验证集、外推风险、数据特性对结论的限制）

要求：所有结论必须与日志数字一致；表格数字保留 4 位小数；全文 800-1500 字。

实验日志：
```json
{payload}
```"""

FINAL_SYSTEM = """你是一位资深的光伏功率预测分析师，负责撰写最终预测实验分析报告。
你会收到测试集评测结果（top-k 模型、两种集成、AutoARIMA baseline）、按 lead 时间的误差、以及被可视化测试窗口的数值摘要。
请基于数据本身推理，不要编造数字。输出为简体中文 markdown。"""

FINAL_PROMPT = """请根据下面的实验结果，撰写一份《最终预测分析报告》，markdown 格式，结构如下：

# PVOD 光伏功率预测最终分析报告

## 1. 实验概览
（任务、数据划分、模型选择协议；2-4 句。注意区分：全序列点数、测试段点数、测试窗口数不要混淆）

## 2. 测试集总榜
markdown 表格，列依次为：模型 | 测试 MAE | 测试 RMSE | 测试 MASE | 相对 baseline 的 MAE 提升 | 集成成员权重。按测试 MAE 升序排列。规则：
- "相对 baseline 的 MAE 提升"= (baseline_MAE - 模型MAE) / baseline_MAE * 100%，保留 2 位小数并带 %，正值表示优于 baseline，负值表示劣于；AutoARIMA 自身此列填 0%。
- "集成成员权重"只对 top-5 入选的单模型填写（来自 ensemble_member_weight 字段）；MedianEnsemble、WeightedEnsemble、AutoARIMA 此列填 "—"。
表格后用一段话解读排名（集成是否优于最优单模型？是否超过 baseline？提升了多少？）。

## 3. 误差随预测步长（lead）的演变
基于 by_lead 数据描述：短程（lead 1-16）/中程（17-48）/长程（49-96）各模型的误差量级与衰减形态，指出哪个模型在哪个区间占优、集成是否平滑了误差。

## 4. 被可视化测试窗口解读
基于 plot digest（每个绘图窗口各模型的窗口内 MAE），指出哪些日期的窗口最难预测（全体模型误差都大的窗口）、可能的原因（如天气突变），以及各模型在难易窗口上的稳定性差异。
本节必须为每个窗口插入对应的图片：先写 2-3 句该窗口的解读，随后紧接着一行插入该窗口的图片引用，格式严格为：
![测试窗口 {{i}}: origin {{origin}}](plots/window_{{i}}.png)
其中 {{i}} 为 payload 中 plot_images 列表的下标（从 1 开始），文件名与相对路径必须与 plot_images 中给出的完全一致，不要改写、不要省略、不要虚构其他图片。

## 5. 结论与建议
（最终推荐的生产配置、模型淘汰建议、后续改进方向）

要求：所有数字与输入一致；MAE/RMSE/MASE 保留 4 位小数；全文 1000-1800 字。

实验结果：
```json
{payload}
```"""


def chat(messages, max_tokens=6000, retries=3):
    for attempt in range(retries):
        try:
            body = json.dumps({
                "model": MODEL_ID, "messages": messages,
                "temperature": 0.2, "max_tokens": max_tokens,
            }).encode()
            request = urllib.request.Request(
                f"{BASE_URL}/chat/completions", data=body,
                headers=api_headers(),
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)["choices"][0]["message"]["content"]
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1} after error: {exc}", flush=True)
            time.sleep(10 * (attempt + 1))


def compact_round(obj):
    if isinstance(obj, dict):
        return {k: compact_round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [compact_round(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def lead_buckets(by_lead):
    buckets = {}
    for lead_row in by_lead:
        lead = int(lead_row["lead"])
        name = "1-16" if lead <= 16 else ("17-48" if lead <= 48 else "49-96")
        slot = buckets.setdefault(name, {"n": 0, "mae": 0.0, "rmse_sq": 0.0})
        slot["n"] += 1
        slot["mae"] += lead_row["mae"]
        slot["rmse_sq"] += lead_row["rmse"] ** 2
    return {
        name: {
            "mean_mae": round(slot["mae"] / slot["n"], 4),
            "mean_rmse": round((slot["rmse_sq"] / slot["n"]) ** 0.5, 4),
        }
        for name, slot in buckets.items()
    }


def build_payload(log, mode):
    if mode == "experience":
        leaderboard = sorted(log["leaderboard"], key=lambda r: r["mae"])
        return {
            "meta": log["meta"],
            "boundaries": log["boundaries"],
            "inner_n_windows": log["inner_n_windows"],
            "candidates": {
                alias: {
                    "best_params": entry["best_params"],
                    "inner_tuning_log": entry["tuning_log"],
                    "validation": next(
                        (row for row in leaderboard if row["model"] == alias), None),
                }
                for alias, entry in log["candidates"].items()
            },
            "leaderboard_sorted_by_mae": leaderboard,
            "selected_top_k": log["selected_top_k"],
            "ensemble_weights": log["ensemble_weights"],
        }
    baseline_mae = log["baseline"]["test_summary"]["mae"]
    test = {
        alias: {
            "summary": {k: v for k, v in entry["summary"].items()
                        if k in ("model", "weight_policy", "n_origins", "mae",
                                 "rmse", "mean_mase")},
            "ensemble_member_weight": log["ensemble_weights"].get(alias),
            "improvement_vs_baseline_mae": {
                "delta": round(baseline_mae - entry["summary"]["mae"], 6),
                "pct": round(
                    (baseline_mae - entry["summary"]["mae"]) / baseline_mae * 100, 4
                ),
            },
            "lead_buckets": lead_buckets(entry["by_lead"]),
        }
        for alias, entry in log["test_results"].items()
    }
    return {
        "meta": log["meta"],
        "boundaries": log["boundaries"],
        "test_span": {
            "n_origins": next(iter(test.values()))["summary"]["n_origins"],
            "n_predictions": next(iter(log["test_results"].values()))["summary"][
                "n_predictions"
            ],
        },
        "selected_top_k": log["selected_top_k"],
        "ensemble_weights": log["ensemble_weights"],
        "test_results_sorted_by_mae": dict(
            sorted(test.items(), key=lambda kv: kv[1]["summary"]["mae"])
        ),
        "plot_digest": log["plot"]["digest"],
        "plot_images": log.get("plot_images", []),
        "plot_file": log["plot"]["file"],
    }


def generate_window_plots(log, log_path, context=192):
    """Render one PNG per sampled test window from the saved forecast CSVs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    out_dir = Path(log_path).parent
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    forecasts = pd.read_csv(out_dir / "main" / "test_comparison_forecasts.csv")
    forecasts["ds"] = pd.to_datetime(forecasts.ds)
    forecasts["origin"] = pd.to_datetime(forecasts.origin)
    baseline_csv = out_dir / "baseline" / "test_forecasts.csv"
    if baseline_csv.exists():
        base = pd.read_csv(baseline_csv)
        base["ds"] = pd.to_datetime(base.ds)
        base["origin"] = pd.to_datetime(base.origin)
        forecasts = pd.concat(
            [forecasts, base.loc[:, forecasts.columns]], ignore_index=True
        )
    raw = pd.read_csv(log["meta"]["csv"]).rename(
        columns={"date_time": "ds", "power": "y"}
    )
    raw["ds"] = pd.to_datetime(raw["ds"])
    raw = raw.drop_duplicates(subset=["ds"], keep="first").sort_values("ds")

    entries = []
    for i, window in enumerate(log["plot"]["windows"], start=1):
        origin = pd.Timestamp(window["origin"])
        uid = window["unique_id"]
        history = raw.loc[raw.ds <= origin].tail(context)
        figure, ax = plt.subplots(figsize=(12, 4.2), layout="constrained")
        ax.plot(history.ds, history.y, color="0.4", lw=1.1, label="Observed history")
        sub = forecasts.loc[forecasts.origin == origin].sort_values(["model", "ds"])
        for model, group in sub.groupby("model", sort=True):
            ax.plot(group.ds, group.y_pred, lw=1.2, label=model)
        first = sub.loc[sub.model == sub.model.iloc[0]].sort_values("ds")
        ax.plot(first.ds, first.y, "k--", lw=1.6, label="Future actual")
        ax.axvline(origin, color="0.5", ls=":")
        ax.set_title(f"Test window {i}: {uid} | origin {origin}")
        ax.set_ylabel("Power")
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.legend(loc="upper left", ncol=3, fontsize=7)
        path = plots_dir / f"window_{i}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        entries.append({"i": i, "file": f"plots/window_{i}.png",
                        "unique_id": uid, "origin": str(origin)})
        print(f"      saved {path}", flush=True)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("experience", "final"), required=True)
    parser.add_argument("--log", default="outputs/pvod_full/experiment_log.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    global MODEL_ID
    if not MODEL_ID:
        request = urllib.request.Request(f"{BASE_URL}/models", headers=api_headers())
        with urllib.request.urlopen(request, timeout=30) as response:
            MODEL_ID = json.load(response)["data"][0]["id"]

    log = json.loads(Path(args.log).read_text())
    if args.mode == "experience":
        system, template = EXPERIENCE_SYSTEM, EXPERIENCE_PROMPT
        default_out = Path(args.log).parent / "training_experience.md"
    else:
        print("[plots] rendering per-window images", flush=True)
        log["plot_images"] = generate_window_plots(log, args.log)
        system, template = FINAL_SYSTEM, FINAL_PROMPT
        default_out = Path(args.log).parent / "final_report.md"
    payload = json.dumps(compact_round(build_payload(log, args.mode)),
                         ensure_ascii=False, indent=1)
    print(f"mode={args.mode} model={MODEL_ID} payload={len(payload)} chars",
          flush=True)
    output = Path(args.output) if args.output else default_out
    output.parent.mkdir(parents=True, exist_ok=True)

    content = chat([
        {"role": "system", "content": system},
        {"role": "user", "content": template.format(payload=payload)},
    ])
    output.write_text(content, encoding="utf-8")
    print(f"saved: {output} ({len(content)} chars)")


if __name__ == "__main__":
    main()
