"""Cheap request checks that run before creating a modeling task."""

from __future__ import annotations

from pathlib import Path

from energy_tsf.contracts.backtest import BacktestRequest
from energy_tsf.contracts.forecast import ForecastRequest
from energy_tsf.contracts.modeling import ModelingRequest, PreflightIssue, PreflightReport
from energy_tsf.contracts.update import UpdateRequest
from energy_tsf.data.ingestion import inspect_target_schema, load_future_covariates

_SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq"}


def preflight_request(request: ModelingRequest) -> PreflightReport:
    """Check the local dataset reference without loading the full dataset."""

    errors: list[PreflightIssue] = []
    warnings: list[PreflightIssue] = []
    path = Path(request.data_uri).expanduser()

    if not path.is_file():
        errors.append(PreflightIssue(code="data_not_found", message=f"dataset not found: {path}"))
    elif path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        errors.append(
            PreflightIssue(
                code="unsupported_data_format",
                message="dataset must be CSV or Parquet",
            )
        )
    else:
        try:
            inspect_target_schema(path, request.target, request.timestamp_column, request.covariates)
        except ValueError as exc:
            errors.append(PreflightIssue(code="invalid_data_schema", message=str(exc)))
    if request.timestamp_column is None:
        warnings.append(
            PreflightIssue(
                code="timestamp_not_provided",
                message="row order will be treated as chronological order",
            )
        )

    return PreflightReport(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))


def preflight_forecast_request(request: ForecastRequest) -> PreflightReport:
    """Check a forecast input file without loading its complete history."""

    errors: list[PreflightIssue] = []
    warnings: list[PreflightIssue] = []
    path = Path(request.data_uri).expanduser()
    station = request.station

    if not path.is_file():
        errors.append(PreflightIssue(code="data_not_found", message=f"dataset not found: {path}"))
    elif path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        errors.append(
            PreflightIssue(code="unsupported_data_format", message="dataset must be CSV or Parquet")
        )
    else:
        try:
            inspect_target_schema(path, station.target, station.timestamp_column, request.covariates)
        except ValueError as exc:
            errors.append(PreflightIssue(code="invalid_data_schema", message=str(exc)))
    if station.timestamp_column is None:
        warnings.append(
            PreflightIssue(
                code="timestamp_not_provided",
                message="forecast will use relative horizon steps instead of valid timestamps",
            )
        )
    if request.future_covariates_uri is not None:
        future_path = Path(request.future_covariates_uri).expanduser()
        if not future_path.is_file():
            errors.append(
                PreflightIssue(code="future_data_not_found", message=f"future covariates not found: {future_path}")
            )
        elif future_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            errors.append(
                PreflightIssue(code="unsupported_future_data_format", message="future covariates must be CSV or Parquet")
            )
        elif station.timestamp_column is not None:
            try:
                load_future_covariates(future_path, station.timestamp_column, request.covariates)
            except ValueError as exc:
                errors.append(PreflightIssue(code="invalid_future_schema", message=str(exc)))
    if request.allow_similar and station.target_unit.lower() not in {"w", "kw", "mw", "gw", "p.u."}:
        warnings.append(
            PreflightIssue(
                code="unrecognized_transfer_unit",
                message="similar-station transfer may be unavailable for this target unit",
            )
        )
    return PreflightReport(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))


def preflight_backtest_request(request: BacktestRequest) -> PreflightReport:
    """Check timestamped backtest truth before creating a task directory."""

    errors: list[PreflightIssue] = []
    path = Path(request.data_uri).expanduser()
    if not path.is_file():
        errors.append(PreflightIssue(code="data_not_found", message=f"dataset not found: {path}"))
    elif path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        errors.append(PreflightIssue(code="unsupported_data_format", message="dataset must be CSV or Parquet"))
    else:
        try:
            inspect_target_schema(path, request.target, request.timestamp_column, request.covariates)
        except ValueError as exc:
            errors.append(PreflightIssue(code="invalid_data_schema", message=str(exc)))
    return PreflightReport(valid=not errors, errors=tuple(errors))


def preflight_update_request(request: UpdateRequest) -> PreflightReport:
    """Check the timestamped update dataset before creating task files."""

    errors: list[PreflightIssue] = []
    path = Path(request.data_uri).expanduser()
    if not path.is_file():
        errors.append(PreflightIssue(code="data_not_found", message=f"dataset not found: {path}"))
    elif path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        errors.append(PreflightIssue(code="unsupported_data_format", message="dataset must be CSV or Parquet"))
    else:
        try:
            inspect_target_schema(path, request.target, request.timestamp_column, request.covariates)
        except ValueError as exc:
            errors.append(PreflightIssue(code="invalid_data_schema", message=str(exc)))
    return PreflightReport(valid=not errors, errors=tuple(errors))
