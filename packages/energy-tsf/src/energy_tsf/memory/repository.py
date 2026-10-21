"""Content-addressed Memory Repository with immutable trial and solution views."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from energy_tsf.contracts.artifact import CandidateRef, ObjectRef, SolutionArtifact
from energy_tsf.contracts.data import DataSnapshotRef
from energy_tsf.contracts.modeling import TrialRecord
from energy_tsf.contracts.search_pool import SearchPoolArtifact, WindowTrialScore
from energy_tsf.data.snapshots import load_snapshot_ref
from energy_tsf.memory.search_pool import load_search_pool, save_search_pool
from energy_tsf.memory.solution import solution_fingerprint


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class MemoryRepository:
    """Persist immutable run assets in a content-addressed filesystem store.

    Large files are stored once under their SHA-256 digest. Higher-level
    records retain stable references and manifests are checked when read.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Create a repository rooted at the supplied asset directory."""
        self.root = Path(root)

    def put_bytes(self, payload: bytes) -> ObjectRef:
        """Store bytes by digest, rejecting incomplete or conflicting objects."""
        digest = hashlib.sha256(payload).hexdigest()
        objects_root = self.root / "objects"
        directory = objects_root / digest
        blob = directory / "blob"
        if directory.exists():
            if not (directory / ".complete").is_file() or not blob.is_file() or blob.read_bytes() != payload:
                raise ValueError(f"immutable object collision or incomplete object: {digest}")
            return ObjectRef(sha256=digest, uri=f"object://{digest}", size=len(payload))
        objects_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}.", dir=objects_root))
        try:
            (temporary / "manifest.json").write_text(
                _json({"sha256": digest, "size": len(payload)}), encoding="utf-8"
            )
            (temporary / "blob").write_bytes(payload)
            (temporary / ".complete").write_text(digest + "\n", encoding="utf-8")
            try:
                os.replace(temporary, directory)
            except OSError:
                if not (directory / ".complete").is_file() or (directory / "blob").read_bytes() != payload:
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return ObjectRef(sha256=digest, uri=f"object://{digest}", size=len(payload))

    def put_file(self, path: str | os.PathLike[str]) -> ObjectRef:
        return self.put_bytes(Path(path).read_bytes())

    def resolve_reference(self, reference: str) -> Path:
        """Resolve a local or content-addressed URI and verify object checksums."""
        for prefix in ("object://", "runtime-ref://"):
            if reference.startswith(prefix):
                digest = reference[len(prefix) :]
                directory = self.root / "objects" / digest
                path = directory / "blob"
                if not (directory / ".complete").is_file() or not path.is_file():
                    raise FileNotFoundError(f"object reference does not resolve: {reference}")
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"object checksum does not match reference: {reference}")
                return path
        path = Path(reference)
        if not path.exists():
            raise FileNotFoundError(f"reference does not resolve: {reference}")
        return path

    def register_data_snapshot(self, reference: DataSnapshotRef) -> DataSnapshotRef:
        loaded = load_snapshot_ref(reference.path)
        if loaded.snapshot_id != reference.snapshot_id:
            raise ValueError("data snapshot reference does not match its manifest")
        return loaded

    def save_search_pool(
        self,
        pool: SearchPoolArtifact,
        scores: tuple[WindowTrialScore, ...] = (),
    ) -> Path:
        return save_search_pool(self.root, pool, scores)

    def load_search_pool(self, pool_id: str) -> SearchPoolArtifact:
        return load_search_pool(self.root, pool_id)

    def save_trial(self, task_id: str, record: TrialRecord) -> TrialRecord:
        prediction_ref = record.prediction_ref
        artifact_ref = record.artifact_ref
        if prediction_ref and Path(prediction_ref).is_file():
            prediction_ref = self.put_file(prediction_ref).uri
        if artifact_ref and Path(artifact_ref).is_file():
            artifact_ref = self.put_file(artifact_ref).uri
        elif record.spec.provider in {"timesfm-2.5", "moirai-2", "chronos-2"}:
            runtime_reference = {
                "provider": record.spec.provider,
                "repo_id": record.spec.params.get("repo_id"),
                "revision": record.spec.params.get("revision"),
                "checkpoint_path": record.spec.params.get("checkpoint_path"),
                "context_length": record.spec.params.get("context_length"),
            }
            ref = self.put_bytes(_json(runtime_reference).encode("utf-8"))
            artifact_ref = f"runtime-ref://{ref.sha256}"
        stored = record.model_copy(update={"prediction_ref": prediction_ref, "artifact_ref": artifact_ref})
        path = self.root / "trials" / task_id / record.spec.trial_id / "record.json"
        payload = _json(stored.model_dump(mode="json"))
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise ValueError("cannot overwrite an immutable trial record")
        _atomic_text(path, payload)
        return stored

    def save_solution(
        self,
        solution: SolutionArtifact,
        attachments: Mapping[str, str | os.PathLike[str] | bytes] | None = None,
    ) -> Path:
        """Persist an immutable Solution after validating its attachments."""
        if solution_fingerprint(solution) != solution.solution_id:
            raise ValueError("solution_id does not match solution contents")
        destination = self.root / "solutions" / solution.solution_id
        if (destination / ".complete").is_file():
            existing = self.load_solution(solution.solution_id)
            if existing != solution:
                raise ValueError("cannot overwrite an immutable solution")
            return destination
        if destination.exists():
            raise ValueError("incomplete solution directory already exists")

        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{solution.solution_id[:12]}.", dir=parent))
        try:
            (temporary / "solution.json").write_text(
                _json(solution.model_dump(mode="json")), encoding="utf-8"
            )
            (temporary / "solution.toml").write_text(
                "[solution]\n"
                f'id = "{solution.solution_id}"\n'
                f'request_id = "{solution.request_id}"\n'
                f'station_id = "{solution.station_id}"\n'
                f'members = {json.dumps(list(solution.ensemble.member_trial_ids))}\n'
                f'weights = {json.dumps(list(solution.ensemble.weights))}\n',
                encoding="utf-8",
            )
            checksums = {}
            for name, source in (attachments or {}).items():
                if Path(name).name != name:
                    raise ValueError("solution attachment names cannot contain directories")
                target = temporary / name
                payload = source if isinstance(source, bytes) else Path(source).read_bytes()
                target.write_bytes(payload)
                checksums[name] = hashlib.sha256(payload).hexdigest()
            (temporary / "manifest.json").write_text(
                _json({"solution_id": solution.solution_id, "attachments": checksums}), encoding="utf-8"
            )
            manifest_digest = hashlib.sha256((temporary / "manifest.json").read_bytes()).hexdigest()
            (temporary / ".complete").write_text(
                f"{solution.solution_id}\n{manifest_digest}\n", encoding="utf-8"
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination

    def load_solution(self, solution_id: str) -> SolutionArtifact:
        """Load a Solution after checking its identity and referenced assets."""
        path = self.root / "solutions" / solution_id
        if not (path / ".complete").is_file():
            raise ValueError(f"solution is incomplete: {solution_id}")
        marker_lines = (path / ".complete").read_text(encoding="utf-8").splitlines()
        if not marker_lines or marker_lines[0] != solution_id:
            raise ValueError("solution completion marker does not match its directory")
        solution = SolutionArtifact.model_validate_json((path / "solution.json").read_text(encoding="utf-8"))
        if solution_fingerprint(solution) != solution_id:
            raise ValueError("solution content fingerprint does not match its directory")
        manifest_bytes = (path / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("solution_id") != solution_id:
            raise ValueError("solution manifest does not match its directory")
        if len(marker_lines) > 1 and marker_lines[1] != hashlib.sha256(manifest_bytes).hexdigest():
            raise ValueError("solution manifest checksum does not match its completion marker")
        attachments = manifest.get("attachments")
        if not isinstance(attachments, dict):
            raise ValueError("solution manifest attachments must be a mapping")
        reserved = {"solution.json", "solution.toml", "manifest.json", ".complete"}
        actual_attachments = {item.name for item in path.iterdir() if item.is_file()} - reserved
        if actual_attachments != set(attachments):
            raise ValueError("solution manifest does not list exactly the stored attachments")
        for name, digest in attachments.items():
            if Path(name).name != name:
                raise ValueError("solution attachment names cannot contain directories")
            attachment = path / name
            if not attachment.is_file() or hashlib.sha256(attachment.read_bytes()).hexdigest() != digest:
                raise ValueError(f"solution attachment checksum mismatch: {name}")
        return solution

    def create_candidate(self, solution: SolutionArtifact) -> CandidateRef:
        solution_path = self.root / "solutions" / solution.solution_id
        if not (solution_path / ".complete").is_file():
            raise ValueError("solution must be stored before creating a Candidate")
        path = self.root / "candidate" / solution.station_id / solution.solution_id
        manifest = path / "manifest.json"
        payload = _json(
            {
                "station_id": solution.station_id,
                "solution_id": solution.solution_id,
                "solution_path": str(solution_path),
            }
        )
        if manifest.exists() and manifest.read_text(encoding="utf-8") != payload:
            raise ValueError("cannot overwrite an immutable Candidate")
        _atomic_text(manifest, payload)
        marker = f"{solution.solution_id}\n{hashlib.sha256(payload.encode('utf-8')).hexdigest()}\n"
        _atomic_text(path / ".complete", marker)
        return CandidateRef(
            station_id=solution.station_id,
            solution_id=solution.solution_id,
            path=str(path),
            complete=True,
        )

    def load_candidate(self, path: str | os.PathLike[str]) -> CandidateRef:
        source = Path(path)
        if not (source / ".complete").is_file():
            raise ValueError("Candidate is incomplete")
        manifest_bytes = (source / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        station_id = manifest.get("station_id")
        solution_id = manifest.get("solution_id")
        marker_lines = (source / ".complete").read_text(encoding="utf-8").splitlines()
        if not isinstance(station_id, str) or not isinstance(solution_id, str):
            raise ValueError("Candidate manifest lacks station or solution identity")
        if not marker_lines or marker_lines[0] != solution_id:
            raise ValueError("Candidate completion marker does not match its manifest")
        if len(marker_lines) > 1 and marker_lines[1] != hashlib.sha256(manifest_bytes).hexdigest():
            raise ValueError("Candidate manifest checksum does not match its completion marker")
        expected_dir = (self.root / "candidate" / station_id / solution_id).resolve()
        if source.resolve() != expected_dir:
            raise ValueError("Candidate directory does not match its manifest identity")
        expected_solution_path = (self.root / "solutions" / solution_id).resolve()
        try:
            recorded_solution_path = Path(manifest["solution_path"]).resolve()
        except (KeyError, TypeError) as exc:
            raise ValueError("Candidate manifest lacks a valid solution path") from exc
        if recorded_solution_path != expected_solution_path:
            raise ValueError("Candidate solution path does not match its solution identity")
        solution = self.load_solution(solution_id)
        if solution.station_id != station_id:
            raise ValueError("Candidate station does not match its Solution")
        return CandidateRef(
            station_id=station_id,
            solution_id=solution_id,
            path=str(source),
            complete=True,
        )

    def list_active(self, *, target: str | None = None, horizon: str | None = None):
        from energy_tsf.assets.active import list_active

        return list_active(self.root, target=target, horizon=horizon)
