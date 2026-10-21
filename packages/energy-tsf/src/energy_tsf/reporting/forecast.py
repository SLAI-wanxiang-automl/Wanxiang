"""Forecast tables plus machine-readable and human-readable reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from energy_tsf.contracts.forecast import ConstraintSummary, ForecastPlan, ForecastRequest, ForecastResult
from energy_tsf.data.schemas import ForecastContext
from energy_tsf.runtime.forecast_track import TrackForecast


def write_forecast_reports(
    output_dir: str | Path,
    *,
    request: ForecastRequest,
    plan: ForecastPlan,
    context: ForecastContext,
    track: TrackForecast,
    constrained_table,
    constraints: ConstraintSummary,
    config_hash: str,
) -> ForecastResult:
    """Persist forecast outputs, provenance tables, and delivery reports."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw = track.table.frame.copy(deep=True)
    final = constrained_table.frame.copy(deep=True)
    forecast = pd.DataFrame(
        {
            "request_id": request.request_id,
            "station_id": request.station.station_id,
            "horizon_step": final["horizon_step"].to_numpy(),
            "y_pred_raw": raw["y_pred"].to_numpy(dtype=float),
            "y_pred": final["y_pred"].to_numpy(dtype=float),
            "unit": request.station.target_unit,
            "degraded": track.degraded,
        }
    )
    if request.issue_time is not None:
        forecast.insert(2, "issue_time", request.issue_time.isoformat())
    if "timestamp" in final:
        position = 3 if "issue_time" in forecast else 2
        forecast.insert(position, "valid_time", final["timestamp"].to_numpy())
    forecast_path = output / "forecast.csv"
    forecast.to_csv(forecast_path, index=False)

    member_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    plan_sources = {source.solution_id: source for source in plan.sources}
    for source in track.sources:
        for trial_id, values in source.member_predictions.items():
            detail = source.member_details[trial_id]
            member_rows.extend(
                {
                    "station_id": source.station_id,
                    "solution_id": source.solution_id,
                    "trial_id": trial_id,
                    "provider": detail["provider"],
                    "internal_weight": detail["weight"],
                    "horizon_step": step,
                    "y_pred_source_scale": float(value),
                }
                for step, value in enumerate(values, 1)
            )
        source_rows.extend(
            {
                "station_id": source.station_id,
                "solution_id": source.solution_id,
                "outer_weight": plan_sources[source.solution_id].weight,
                "scale_method": plan_sources[source.solution_id].scale_method,
                "horizon_step": step,
                "y_pred_target_scale": float(value),
            }
            for step, value in enumerate(source.predictions, 1)
        )
    pd.DataFrame(member_rows).to_csv(output / "member_predictions.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output / "source_predictions.csv", index=False)

    failed_members = {
        f"{source.solution_id}:{trial_id}": reason
        for source in track.sources
        for trial_id, reason in source.failed_members.items()
    }
    result = ForecastResult(
        request_id=request.request_id,
        status="delivered",
        plan_id=plan.plan_id,
        input_fingerprint=context.profile.input_fingerprint,
        forecast_path=str(forecast_path),
        report_path=str(output / "report.md"),
        source_solution_ids=tuple(source.solution_id for source in track.sources),
        succeeded_members=tuple(
            trial_id for source in track.sources for trial_id in source.member_predictions
        ),
        failed_members=failed_members,
        degraded=track.degraded,
        warnings=tuple(track.failed_sources.values()),
    )
    forecast_minimum = float(forecast["y_pred"].min())
    forecast_maximum = float(forecast["y_pred"].max())
    payload = {
        "result": result.model_dump(mode="json"),
        "request": request.model_dump(mode="json"),
        "config_hash": config_hash,
        "plan": plan.model_dump(mode="json"),
        "context": context.profile.model_dump(mode="json"),
        "constraints": constraints.model_dump(mode="json"),
        "sources": [
            {
                "station_id": source.station_id,
                "solution_id": source.solution_id,
                "successful_members": list(source.member_predictions),
                "failed_members": source.failed_members,
                "member_details": source.member_details,
                "degraded": source.degraded,
            }
            for source in track.sources
        ],
        "failed_sources": track.failed_sources,
        "forecast_summary": {
            "rows": len(forecast),
            "minimum": forecast_minimum,
            "maximum": forecast_maximum,
            "mean": float(forecast["y_pred"].mean()),
        },
    }
    (output / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "result.json").write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    completed_by_id = {source.solution_id: source for source in track.sources}
    source_lines = "\n".join(
        f"- `{source.station_id}` / `{source.solution_id}`：外层权重 {source.weight:.6g}，"
        f"尺度 `{source.scale_method}`，理由 {'；'.join(source.selection_reasons)}；"
        f"{len(completed_by_id[source.solution_id].member_predictions)} 个成员成功，"
        f"{len(completed_by_id[source.solution_id].failed_members)} 个失败"
        for source in plan.sources
        if source.solution_id in completed_by_id
    )
    warnings = "无" if not result.warnings and not failed_members else "存在降级，详见 report.json"
    markdown = f"""# 预测报告

## 请求

- Request：`{request.request_id}`
- 目标场站：`{request.station.station_id}`
- 预测长度：{context.horizon_steps} 步（{request.horizon}）
- 输入行数：{context.profile.rows}
- 输入指纹：`{context.profile.input_fingerprint}`
- Config hash：`{config_hash}`
- Plan：`{plan.plan_id}`（{plan.mode}）

## 数据质量

- 原始缺失：{context.profile.original_missing}
- 已插补：{context.profile.imputed}
- 不规则间隔：{context.profile.irregular_intervals}
- 历史范围：{context.profile.start_time or '相对顺序'} ～ {context.profile.end_time or '相对顺序'}
- 预测范围：{context.profile.future_start or 'horizon step 1'} ～ {context.profile.future_end or context.horizon_steps}

## 方案来源

{source_lines}

## 约束与结果

- 输出单位：{request.station.target_unit}
- 预测范围：{forecast_minimum:.6g} ～ {forecast_maximum:.6g}
- 约束修改点数：{constraints.modified_points}
- 最大约束调整量：{constraints.max_abs_adjustment:.6g}
- 降级：{warnings}

完整逐点结果见 `forecast.csv`，成员和来源结果分别见 `member_predictions.csv` 与 `source_predictions.csv`。本报告陈述执行事实和历史资产来源，不对未来实际精度作保证。
"""
    (output / "report.md").write_text(markdown, encoding="utf-8")
    source_lines_en = "\n".join(
        f"- `{source.station_id}` / `{source.solution_id}`: outer weight {source.weight:.6g}; "
        f"scale `{source.scale_method}`; reasons {', '.join(source.selection_reasons)}; "
        f"{len(completed_by_id[source.solution_id].member_predictions)} members succeeded, "
        f"{len(completed_by_id[source.solution_id].failed_members)} failed"
        for source in plan.sources
        if source.solution_id in completed_by_id
    )
    warning_text_en = "None" if not result.warnings and not failed_members else "Degraded; see report.json"
    markdown_en = f"""# Forecast report

## Request

- Request: `{request.request_id}`
- Target station: `{request.station.station_id}`
- Horizon: {context.horizon_steps} steps ({request.horizon})
- Input rows: {context.profile.rows}
- Input fingerprint: `{context.profile.input_fingerprint}`
- Config hash: `{config_hash}`
- Plan: `{plan.plan_id}` ({plan.mode})

## Data quality

- Original missing values: {context.profile.original_missing}
- Imputed values: {context.profile.imputed}
- Irregular intervals: {context.profile.irregular_intervals}
- History range: {context.profile.start_time or 'relative order'} to {context.profile.end_time or 'relative order'}
- Forecast range: {context.profile.future_start or 'horizon step 1'} to {context.profile.future_end or context.horizon_steps}

## Sources

{source_lines_en}

## Constraints and results

- Output unit: {request.station.target_unit}
- Forecast range: {forecast_minimum:.6g} to {forecast_maximum:.6g}
- Points modified by constraints: {constraints.modified_points}
- Maximum absolute adjustment: {constraints.max_abs_adjustment:.6g}
- Degradation: {warning_text_en}

Point-level results are in `forecast.csv`; member and source predictions are in `member_predictions.csv` and `source_predictions.csv`. This report records execution facts and historical asset provenance; it does not guarantee future accuracy.
"""
    (output / "report.en.md").write_text(markdown_en, encoding="utf-8")
    return result
