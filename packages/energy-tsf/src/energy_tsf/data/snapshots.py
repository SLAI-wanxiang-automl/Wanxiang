"""Content-addressed, immutable prepared-data snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from energy_tsf.contracts.data import (
    DataQualityReport,
    DataSnapshotRef,
    ImputationReport,
    TemporalSplit,
)
from energy_tsf.contracts.features import CovariateSpec
from energy_tsf.data.schemas import PreparedSeries


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _frame_bytes(prepared: PreparedSeries) -> bytes:
    return prepared.frame.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S.%f").encode("utf-8")


def _mask_bytes(prepared: PreparedSeries) -> bytes:
    return pd.DataFrame({"is_imputed": prepared.imputation_mask.astype(int)}).to_csv(index=False).encode("utf-8")


def _metadata(prepared: PreparedSeries, lineage: Mapping[str, Any]) -> dict[str, Any]:
    split = prepared.split.model_dump(mode="json")
    if split.get("partition_kind") == "modeling":
        split.pop("partition_kind")
    return {
        "target": prepared.target,
        "timestamp_column": prepared.timestamp_column,
        "frequency": prepared.frequency,
        "covariates": prepared.covariates.model_dump(mode="json"),
        "rows": len(prepared.frame),
        "split": split,
        "quality": prepared.quality.model_dump(mode="json"),
        "imputation": prepared.imputation.model_dump(mode="json"),
        "lineage": dict(lineage),
        "files": {"cleaned": "cleaned.csv", "imputation_mask": "imputation_mask.csv"},
    }


def snapshot_fingerprint(prepared: PreparedSeries, lineage: Mapping[str, Any] | None = None) -> str:
    """Hash prepared values, missingness mask, metadata, and lineage together."""
    digest = hashlib.sha256()
    for payload in (_canonical_json(_metadata(prepared, lineage or {})), _frame_bytes(prepared), _mask_bytes(prepared)):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _reference(path: Path, manifest: Mapping[str, Any]) -> DataSnapshotRef:
    return DataSnapshotRef(
        snapshot_id=manifest["snapshot_id"],
        path=str(path),
        rows=manifest["rows"],
        target=manifest["target"],
        frequency=manifest["frequency"],
    )


def create_snapshot(
    prepared: PreparedSeries,
    memory_root: str | os.PathLike[str],
    *,
    lineage: Mapping[str, Any] | None = None,
) -> DataSnapshotRef:
    """Write an immutable snapshot and publish it with a completion marker."""

    lineage = lineage or {}
    snapshot_id = snapshot_fingerprint(prepared, lineage)
    data_root = Path(memory_root) / "data" / "snapshots"
    destination = data_root / snapshot_id
    complete = destination / ".complete"
    if complete.is_file():
        return load_snapshot_ref(destination)
    if destination.exists():
        raise ValueError(f"incomplete snapshot already exists: {destination}")

    data_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id[:12]}.", dir=data_root))
    try:
        manifest = {"snapshot_id": snapshot_id, **_metadata(prepared, lineage)}
        prepared.frame.to_csv(temporary / "cleaned.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
        pd.DataFrame({"is_imputed": prepared.imputation_mask.astype(int)}).to_csv(
            temporary / "imputation_mask.csv", index=False
        )
        manifest["checksums"] = {
            name: hashlib.sha256((temporary / name).read_bytes()).hexdigest()
            for name in manifest["files"].values()
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / ".complete").write_text(snapshot_id + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _reference(destination, manifest)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not (path / ".complete").is_file():
        raise ValueError(f"snapshot is incomplete: {path}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if path.name != manifest["snapshot_id"]:
        raise ValueError("snapshot directory and manifest id do not match")
    marker = (path / ".complete").read_text(encoding="utf-8").strip()
    if marker != manifest["snapshot_id"]:
        raise ValueError("snapshot completion marker does not match its manifest")
    expected_files = {"cleaned": "cleaned.csv", "imputation_mask": "imputation_mask.csv"}
    if manifest.get("files") != expected_files:
        raise ValueError("snapshot manifest does not declare the required attachments")
    expected_checksums = set(expected_files.values())
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict) or set(checksums) != expected_checksums:
        raise ValueError("snapshot manifest checksums do not cover all required attachments")
    for name, digest in checksums.items():
        if Path(name).name != name:
            raise ValueError("snapshot attachment names cannot contain directories")
        source = path / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"snapshot attachment checksum mismatch: {name}")

    split = manifest.get("split")
    metadata = {
        key: manifest[key]
        for key in (
            "target", "timestamp_column", "frequency", "covariates", "rows",
            "split", "quality", "imputation", "lineage", "files",
        )
    }
    if split.get("partition_kind") == "modeling":
        metadata["split"] = {key: value for key, value in split.items() if key != "partition_kind"}
    digest = hashlib.sha256()
    for payload in (
        _canonical_json(metadata),
        (path / expected_files["cleaned"]).read_bytes(),
        (path / expected_files["imputation_mask"]).read_bytes(),
    ):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    if digest.hexdigest() != manifest["snapshot_id"]:
        raise ValueError("snapshot content fingerprint does not match its directory")
    return manifest


def load_snapshot_ref(path: str | os.PathLike[str]) -> DataSnapshotRef:
    """Load a snapshot reference after validating its manifest and content hash."""
    source = Path(path)
    return _reference(source, _load_manifest(source))


def load_snapshot(reference: DataSnapshotRef | str | os.PathLike[str]) -> PreparedSeries:
    """Reload a completed snapshot into its prepared-series representation."""

    source = Path(reference.path if isinstance(reference, DataSnapshotRef) else reference)
    manifest = _load_manifest(source)
    frame = pd.read_csv(source / manifest["files"]["cleaned"])
    timestamp_column = manifest["timestamp_column"]
    if timestamp_column is not None:
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="raise")
    else:
        frame.index = pd.RangeIndex(len(frame), name="relative_time")
    mask = pd.read_csv(source / manifest["files"]["imputation_mask"])["is_imputed"].astype(bool)
    mask.index = frame.index
    mask.name = manifest["target"]
    return PreparedSeries(
        frame=frame,
        target=manifest["target"],
        timestamp_column=timestamp_column,
        frequency=manifest["frequency"],
        imputation_mask=mask,
        split=TemporalSplit.model_validate(manifest["split"]),
        quality=DataQualityReport.model_validate(manifest["quality"]),
        imputation=ImputationReport.model_validate(manifest["imputation"]),
        covariates=CovariateSpec.model_validate(manifest.get("covariates", {})),
    )


def load_snapshot_by_id(memory_root: str | os.PathLike[str], snapshot_id: str) -> PreparedSeries:
    """Resolve an immutable training snapshot referenced by a Solution."""

    return load_snapshot(Path(memory_root) / "data" / "snapshots" / snapshot_id)
