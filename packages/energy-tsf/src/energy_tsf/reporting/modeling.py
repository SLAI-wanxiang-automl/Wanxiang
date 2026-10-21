"""Persist leaderboard, Top-K, aligned predictions, and concise report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from energy_tsf.contracts.artifact import EnsembleSpec, FinalEvaluationReport
from energy_tsf.contracts.data import OriginSet
from energy_tsf.contracts.modeling import ExperimentPlan, TopKSelection, TrialRecord
from energy_tsf.contracts.search_pool import SearchPoolArtifact, WindowTrialScore
from energy_tsf.data.schemas import PreparedSeries


def write_search_reports(
    plan: ExperimentPlan,
    records: tuple[TrialRecord, ...],
    selection: TopKSelection,
    output_dir: str | Path,
    *,
    search_pool: SearchPoolArtifact | None = None,
    window_scores: tuple[WindowTrialScore, ...] = (),
) -> dict[str, Path]:
    """Write trial rankings, validation evidence, and the modeling summary."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    leaderboard = root / "leaderboard.csv"
    rows = [
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
    pd.DataFrame(rows).to_csv(leaderboard, index=False)

    top_k = root / "top_k.json"
    top_k.write_text(json.dumps(selection.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation_predictions = root / "validation_predictions.csv"
    frames = []
    for record in selection.records:
        frame = pd.read_csv(record.prediction_ref)
        frame.insert(0, "trial_id", record.spec.trial_id)
        frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(validation_predictions, index=False)
    else:
        pd.DataFrame().to_csv(validation_predictions, index=False)

    window_leaderboard = root / "window_leaderboard.csv"
    pd.DataFrame([score.model_dump(mode="json") for score in window_scores]).to_csv(
        window_leaderboard, index=False
    )
    pool_summary = root / "search_pool.json"
    pool_summary.write_text(
        json.dumps(
            None if search_pool is None else search_pool.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report_json = root / "validation_report.json"
    report = {
        "plan_id": plan.plan_id,
        "planned_trials": len(plan.trials),
        "unavailable": plan.unavailable,
        "status_counts": pd.Series([record.status for record in records]).value_counts().to_dict(),
        "top_k_status": selection.status,
        "top_k": [record.spec.trial_id for record in selection.records],
        "search_pool_id": None if search_pool is None else search_pool.pool_id,
        "search_pool_size": 0 if search_pool is None else len(search_pool.entries),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md = root / "report.md"
    report_md.write_text(
        "# Modeling report\n\n"
        f"- Plan: `{plan.plan_id}`\n"
        f"- Planned trials: {len(plan.trials)}\n"
        f"- Successful trials: {sum(record.status == 'succeeded' for record in records)}\n"
        f"- Top-{selection.requested} status: `{selection.status}`\n"
        f"- Unavailable providers: {', '.join(plan.unavailable) or 'none'}\n",
        encoding="utf-8",
    )
    report_md.with_name("report.en.md").write_text(
        "# Modeling report\n\n"
        f"- Plan: `{plan.plan_id}`\n"
        f"- Planned trials: {len(plan.trials)}\n"
        f"- Successful trials: {sum(record.status == 'succeeded' for record in records)}\n"
        f"- Top-{selection.requested} status: `{selection.status}`\n"
        f"- Unavailable providers: {', '.join(plan.unavailable) or 'none'}\n",
        encoding="utf-8",
    )
    return {
        "leaderboard": leaderboard,
        "top_k": top_k,
        "validation_predictions": validation_predictions,
        "validation_report": report_json,
        "report_md": report_md,
        "window_leaderboard": window_leaderboard,
        "search_pool": pool_summary,
    }


def write_final_modeling_report(
    path: str | Path,
    *,
    prepared: PreparedSeries,
    origins: OriginSet,
    plan: ExperimentPlan,
    records: tuple[TrialRecord, ...],
    selection: TopKSelection,
    ensemble: EnsembleSpec,
    ensemble_validation_metrics: dict[str, float | None],
    final: FinalEvaluationReport,
    search_pool: SearchPoolArtifact,
) -> Path:
    """Replace the preliminary report with the completed, selection-safe summary."""

    target = Path(path)
    final_by_id = {member.trial_id: member for member in final.members}

    def metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.6f}"

    lines = [
        "# 建模运行报告",
        "",
        "## 数据与切分",
        "",
        f"- 行数：{len(prepared.frame)}；目标：`{prepared.target}`；频率：`{prepared.frequency}`；预测步数：{origins.horizon_steps}。",
        f"- 边界：train `[0, {prepared.split.train_end})`，validation `[{prepared.split.train_end}, {prepared.split.validation_end})`，test `[{prepared.split.validation_end}, {prepared.split.n_rows})`。",
        f"- 逐日 origins：train {len(origins.train)}，validation {len(origins.validation)}，test {len(origins.test)}。",
        f"- 原始缺失/插值：{prepared.quality.original_missing}/{prepared.imputation.total_imputed}；清洗后非有限值：{prepared.quality.non_finite_after_cleaning}。",
        "",
        "## AutoML",
        "",
        f"- Plan：`{plan.plan_id}`；计划/成功/失败：{len(plan.trials)}/{sum(item.status == 'succeeded' for item in records)}/{sum(item.status != 'succeeded' for item in records)}。",
        f"- 不可用 Provider：{'; '.join(f'{name}: {reason}' for name, reason in plan.unavailable.items()) or '无'}。",
        "- 排名只使用 validation：RMSE、MAE、预测耗时、Provider、trial ID 依次作为排序键；test 不参与选择。",
        f"- 候选搜索池：`{search_pool.pool_id}`；逐窗口 Top-{search_pool.top_k_per_window} 并集共 {len(search_pool.entries)} 个 Trial。",
        "",
        f"## 冻结 Top-{selection.requested}",
        "",
        "| 排名 | Trial | Provider | 上下文 | Validation RMSE | Validation MAE | Test RMSE | Test MAE |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, record in enumerate(selection.records, start=1):
        member = final_by_id[record.spec.trial_id]
        lines.append(
            f"| {rank} | `{record.spec.trial_id}` | {record.spec.provider} | {record.spec.context_length or '-'} | "
            f"{metric(record.validation_metrics.get('rmse'))} | {metric(record.validation_metrics.get('mae'))} | "
            f"{metric(member.test_metrics.get('rmse'))} | {metric(member.test_metrics.get('mae'))} |"
        )
    lines.extend(
        [
            "",
            "## 集成与发布前结果",
            "",
            f"- 方法：`{ensemble.method}`；权重：{list(ensemble.weights)}；失败时按验证排名回退并对可用权重归一化。",
            f"- Validation ensemble：RMSE {metric(ensemble_validation_metrics.get('rmse'))}，MAE {metric(ensemble_validation_metrics.get('mae'))}。",
            f"- Test ensemble：RMSE {metric(final.ensemble_metrics.get('rmse'))}，MAE {metric(final.ensemble_metrics.get('mae'))}。",
            "- 预测后处理：所有成员在计算指标和写出结果前应用最小值 0 的非负约束。",
            f"- Top-{selection.requested} 与集成权重在 test 运行前冻结；即使某成员 test 更好，也不会使用 test 重新选择。",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")
    english_lines = [
        "# Modeling run report",
        "",
        "## Data and splits",
        "",
        f"- Rows: {len(prepared.frame)}; target: `{prepared.target}`; frequency: `{prepared.frequency}`; horizon steps: {origins.horizon_steps}.",
        f"- Boundaries: train `[0, {prepared.split.train_end})`; validation `[{prepared.split.train_end}, {prepared.split.validation_end})`; test `[{prepared.split.validation_end}, {prepared.split.n_rows})`.",
        f"- Daily origins: train {len(origins.train)}; validation {len(origins.validation)}; test {len(origins.test)}.",
        f"- Original missing/imputed: {prepared.quality.original_missing}/{prepared.imputation.total_imputed}; non-finite after cleaning: {prepared.quality.non_finite_after_cleaning}.",
        "",
        "## AutoML",
        "",
        f"- Plan: `{plan.plan_id}`; planned/succeeded/failed trials: {len(plan.trials)}/{sum(item.status == 'succeeded' for item in records)}/{sum(item.status != 'succeeded' for item in records)}.",
        f"- Unavailable Providers: {'; '.join(f'{name}: {reason}' for name, reason in plan.unavailable.items()) or 'none'}.",
        "- Ranking uses validation only: RMSE, MAE, prediction time, Provider, and trial ID are the sort keys. Test does not participate in selection.",
        f"- SearchPool: `{search_pool.pool_id}`; top-{search_pool.top_k_per_window} per window; {len(search_pool.entries)} trials in the pool.",
        "",
        f"## Frozen Top-{selection.requested}",
        "",
        "| Rank | Trial | Provider | Context | Validation RMSE | Validation MAE | Test RMSE | Test MAE |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, record in enumerate(selection.records, start=1):
        member = final_by_id[record.spec.trial_id]
        english_lines.append(
            f"| {rank} | `{record.spec.trial_id}` | {record.spec.provider} | {record.spec.context_length or '-'} | "
            f"{metric(record.validation_metrics.get('rmse'))} | {metric(record.validation_metrics.get('mae'))} | "
            f"{metric(member.test_metrics.get('rmse'))} | {metric(member.test_metrics.get('mae'))} |"
        )
    english_lines.extend(
        [
            "",
            "## Ensemble and pre-release results",
            "",
            f"- Method: `{ensemble.method}`; weights: {list(ensemble.weights)}; on failure, fall back by validation ranking and renormalize available weights.",
            f"- Validation ensemble: RMSE {metric(ensemble_validation_metrics.get('rmse'))}; MAE {metric(ensemble_validation_metrics.get('mae'))}.",
            f"- Test ensemble: RMSE {metric(final.ensemble_metrics.get('rmse'))}; MAE {metric(final.ensemble_metrics.get('mae'))}.",
            "- Post-processing: apply the non-negative lower bound of 0 to all members before metrics and output.",
            f"- Top-{selection.requested} and ensemble weights were frozen before test evaluation; test results never trigger reselection.",
            "",
        ]
    )
    target.with_name("report.en.md").write_text("\n".join(english_lines), encoding="utf-8")
    return target
