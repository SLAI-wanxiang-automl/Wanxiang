"""Concise machine and human reports for model updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd

from energy_tsf.contracts.artifact import ActiveRef, CandidateRef, ReleaseDecision
from energy_tsf.contracts.modeling import TopKSelection, TrialRecord
from energy_tsf.contracts.search_pool import SearchPoolArtifact, WindowTrialScore
from energy_tsf.contracts.update import UpdateInputProfile, UpdatePlan, UpdateResult


def write_update_reports(
    work_dir: str | Path,
    *,
    profile: UpdateInputProfile,
    plan: UpdatePlan,
    records: tuple[TrialRecord, ...],
    selection: TopKSelection,
    next_pool: SearchPoolArtifact | None,
    window_scores: tuple[WindowTrialScore, ...],
    parent_rmse: float | None,
    updated_rmse: float | None,
    new_solution_id: str | None,
    candidate: CandidateRef | None,
    active: ActiveRef | None,
    decision: ReleaseDecision | None,
) -> UpdateResult:
    """Persist candidate rankings, pool lineage, metrics, and gate outcome."""
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    leaderboard = root / "leaderboard.csv"
    pd.DataFrame(
        [
            {
                "trial_id": record.spec.trial_id,
                "provider": record.spec.provider,
                "status": record.status,
                "validation_rmse": record.validation_metrics.get("rmse"),
                "validation_mae": record.validation_metrics.get("mae"),
                "fit_seconds": record.fit_duration_seconds,
                "predict_seconds": record.predict_duration_seconds,
                "error": record.error,
            }
            for record in records
        ]
    ).to_csv(leaderboard, index=False)
    window_path = root / "window_leaderboard.csv"
    pd.DataFrame([score.model_dump(mode="json") for score in window_scores]).to_csv(window_path, index=False)

    status: Literal["evaluated", "released", "release_failed", "insufficient_candidates"] = (
        "insufficient_candidates" if selection.status != "complete" else "evaluated"
    )
    if active is not None:
        status = "released"
    elif decision is not None and not decision.approved:
        status = "release_failed"
    report_payload = {
        "request_id": plan.request_id,
        "plan_id": plan.plan_id,
        "parent_solution_id": plan.parent_solution_id,
        "source_pool_id": plan.source_pool_id,
        "next_pool_id": None if next_pool is None else next_pool.pool_id,
        "input": profile.model_dump(mode="json"),
        "planned_trials": len(plan.trials),
        "succeeded_trials": sum(record.status == "succeeded" for record in records),
        "top_k": [record.spec.trial_id for record in selection.records],
        "parent_rmse": parent_rmse,
        "updated_rmse": updated_rmse,
        "new_solution_id": new_solution_id,
        "release": None if decision is None else decision.model_dump(mode="json"),
    }
    report_json = root / "report.json"
    report_json.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_md = root / "report.md"
    lines = [
        "# 模型更新报告",
        "",
        f"- 场站：`{plan.station_id}`；Plan：`{plan.plan_id}`。",
        f"- 数据：{profile.rows} 行；train {profile.train_rows} 行；validation {profile.validation_rows} 行、7 个逐日窗口。",
        f"- 候选池：`{plan.source_pool_id}`；计划/成功：{len(plan.trials)}/{sum(record.status == 'succeeded' for record in records)}。",
        f"- 新 Top-{selection.requested}：{', '.join(record.spec.trial_id for record in selection.records) or '不足'}。",
        f"- 父方案同口径 RMSE：{'N/A' if parent_rmse is None else f'{parent_rmse:.6f}'}。",
        f"- 新方案 validation RMSE：{'N/A' if updated_rmse is None else f'{updated_rmse:.6f}'}。",
        f"- 下一代候选池：`{next_pool.pool_id if next_pool else 'N/A'}`。",
        f"- 状态：`{status}`。",
    ]
    if decision is not None and decision.reasons:
        lines.extend(["", "## 发布门禁", "", *[f"- {reason}" for reason in decision.reasons]])
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    english_lines = [
        "# Model update report",
        "",
        f"- Station: `{plan.station_id}`; plan: `{plan.plan_id}`.",
        f"- Data: {profile.rows} rows; train {profile.train_rows}; validation {profile.validation_rows}; {profile.window_count} daily windows.",
        f"- SearchPool: `{plan.source_pool_id}`; planned/succeeded trials: {len(plan.trials)}/{sum(record.status == 'succeeded' for record in records)}.",
        f"- New Top-{selection.requested}: {', '.join(record.spec.trial_id for record in selection.records) or 'insufficient candidates'}.",
        f"- Parent comparable RMSE: {'N/A' if parent_rmse is None else f'{parent_rmse:.6f}' }.",
        f"- Updated validation RMSE: {'N/A' if updated_rmse is None else f'{updated_rmse:.6f}' }.",
        f"- Next SearchPool: `{next_pool.pool_id if next_pool else 'N/A'}`.",
        f"- Status: `{status}`.",
    ]
    if decision is not None and decision.reasons:
        english_lines.extend(["", "## Release gate", "", *[f"- {reason}" for reason in decision.reasons]])
    report_md.with_name("report.en.md").write_text("\n".join(english_lines) + "\n", encoding="utf-8")
    result_path = root / "result.json"
    result = UpdateResult(
        request_id=plan.request_id,
        status=status,
        plan_id=plan.plan_id,
        parent_solution_id=plan.parent_solution_id,
        source_pool_id=plan.source_pool_id,
        next_pool_id=None if next_pool is None else next_pool.pool_id,
        new_solution_id=new_solution_id,
        candidate=candidate,
        active=active,
        published=active is not None,
        report_path=str(report_md),
        leaderboard_path=str(leaderboard),
        window_leaderboard_path=str(window_path),
        result_path=str(result_path),
        release_reasons=() if decision is None else decision.reasons,
    )
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result
