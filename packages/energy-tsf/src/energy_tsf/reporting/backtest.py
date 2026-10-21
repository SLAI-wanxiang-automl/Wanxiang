"""Backtest prediction, metric, posterior, and publication reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd

from energy_tsf.contracts.artifact import ActiveRef, CandidateRef, ReleaseDecision
from energy_tsf.contracts.backtest import (
    BacktestInputProfile,
    BacktestPlan,
    BacktestRequest,
    BacktestResult,
    BayesianWeightUpdate,
)


def write_backtest_reports(
    output_dir: str | Path,
    *,
    request: BacktestRequest,
    plan: BacktestPlan,
    profile: BacktestInputProfile,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    update: BayesianWeightUpdate,
    config_hash: str,
    new_solution_id: str | None,
    candidate: CandidateRef | None,
    active: ActiveRef | None,
    decision: ReleaseDecision | None,
    failures: dict[str, str],
) -> BacktestResult:
    """Persist backtest predictions, metrics, weight evidence, and release status."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.csv"
    metrics_path = output / "metrics.csv"
    weights_path = output / "weights.json"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    weights_path.write_text(update.model_dump_json(indent=2) + "\n", encoding="utf-8")

    published = active is not None
    status: Literal["evaluated", "released", "release_failed"] = (
        "released" if published else ("release_failed" if candidate is not None else "evaluated")
    )
    reasons = () if decision is None else decision.reasons
    result = BacktestResult(
        request_id=request.request_id,
        status=status,
        plan_id=plan.plan_id,
        parent_solution_id=plan.parent_solution_id,
        new_solution_id=new_solution_id,
        candidate=candidate,
        active=active,
        published=published,
        predictions_path=str(predictions_path),
        metrics_path=str(metrics_path),
        weights_path=str(weights_path),
        report_path=str(output / "report.md"),
        release_reasons=reasons,
    )
    overall = metrics.loc[metrics["scope"] == "overall"].set_index("trial_id")
    payload = {
        "result": result.model_dump(mode="json"),
        "request": request.model_dump(mode="json"),
        "config_hash": config_hash,
        "input": profile.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "weight_update": update.model_dump(mode="json"),
        "overall_metrics": overall.reset_index().to_dict(orient="records"),
        "failures": failures,
        "release_decision": None if decision is None else decision.model_dump(mode="json"),
    }
    (output / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "result.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")

    member_lines = []
    for member, old, new in zip(
        update.member_trial_ids, update.previous_weights, update.new_weights
    ):
        row = overall.loc[member]
        member_lines.append(
            f"- `{member}`：RMSE {row['rmse']:.6g}，MAE {row['mae']:.6g}，"
            f"权重 {old:.6f} → {new:.6f}"
        )
    before = overall.loc["ensemble_before"]
    after = overall.loc["ensemble_after"]
    release_text = (
        f"已发布为 `{active.solution_id}`"
        if active is not None
        else ("仅评测，未请求发布" if candidate is None else "未通过发布门禁：" + "；".join(reasons))
    )
    markdown = f"""# 回测报告

## 请求与数据

- Request：`{request.request_id}`
- 场站：`{request.station_id}`
- 真值范围：{profile.start_time.isoformat()} ～ {profile.end_time.isoformat()}
- 窗口：{profile.window_count} 个，每窗 {profile.horizon_steps} 步
- 原始缺失：{profile.original_missing}
- 输入指纹：`{profile.input_fingerprint}`
- Config hash：`{config_hash}`
- 父 Solution：`{plan.parent_solution_id}`
- Plan：`{plan.plan_id}`

## 成员近期表现与权重

{chr(10).join(member_lines)}

## 集成对比

- 更新前：RMSE {before['rmse']:.6g}，MAE {before['mae']:.6g}，WAPE {before['wape']:.6g}
- 更新后：RMSE {after['rmse']:.6g}，MAE {after['mae']:.6g}，WAPE {after['wape']:.6g}
- 贝叶斯尺度：{update.scale:.6g}（{update.scale_source}）
- Prior strength：{update.prior_strength:.6g}

## 发布

{release_text}

逐点结果、逐窗指标和完整贝叶斯证据分别见 `predictions.csv`、`metrics.csv` 和 `weights.json`。
"""
    (output / "report.md").write_text(markdown, encoding="utf-8")
    member_lines_en = []
    for member, old, new in zip(
        update.member_trial_ids, update.previous_weights, update.new_weights
    ):
        row = overall.loc[member]
        member_lines_en.append(
            f"- `{member}`: RMSE {row['rmse']:.6g}, MAE {row['mae']:.6g}; "
            f"weight {old:.6f} → {new:.6f}"
        )
    release_text_en = (
        f"Published as `{active.solution_id}`"
        if active is not None
        else ("Evaluated only; publication was not requested" if candidate is None else "Release gate failed: " + "; ".join(reasons))
    )
    markdown_en = f"""# Backtest report

## Request and data

- Request: `{request.request_id}`
- Station: `{request.station_id}`
- Actual range: {profile.start_time.isoformat()} to {profile.end_time.isoformat()}
- Windows: {profile.window_count}, {profile.horizon_steps} steps each
- Original missing values: {profile.original_missing}
- Input fingerprint: `{profile.input_fingerprint}`
- Config hash: `{config_hash}`
- Parent Solution: `{plan.parent_solution_id}`
- Plan: `{plan.plan_id}`

## Recent member performance and weights

{chr(10).join(member_lines_en)}

## Ensemble comparison

- Before update: RMSE {before['rmse']:.6g}, MAE {before['mae']:.6g}, WAPE {before['wape']:.6g}
- After update: RMSE {after['rmse']:.6g}, MAE {after['mae']:.6g}, WAPE {after['wape']:.6g}
- Bayesian scale: {update.scale:.6g} ({update.scale_source})
- Prior strength: {update.prior_strength:.6g}

## Publication

{release_text_en}

Point predictions, per-window metrics, and complete Bayesian evidence are in `predictions.csv`, `metrics.csv`, and `weights.json`.
"""
    (output / "report.en.md").write_text(markdown_en, encoding="utf-8")
    return result
