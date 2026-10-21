"""Content-addressed SearchPool persistence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from energy_tsf.automl.window_ranking import search_pool_fingerprint
from energy_tsf.contracts.search_pool import SearchPoolArtifact, WindowTrialScore


def save_search_pool(
    root: str | os.PathLike[str],
    pool: SearchPoolArtifact,
    scores: tuple[WindowTrialScore, ...] = (),
) -> Path:
    """Persist an immutable SearchPool and its per-window score evidence."""
    if search_pool_fingerprint(pool) != pool.pool_id:
        raise ValueError("pool_id does not match search-pool contents")
    destination = Path(root) / "search_pools" / pool.pool_id
    if (destination / ".complete").is_file():
        if load_search_pool(root, pool.pool_id) != pool:
            raise ValueError("cannot overwrite an immutable search pool")
        return destination
    if destination.exists():
        raise ValueError("incomplete search-pool directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{pool.pool_id[:12]}.", dir=destination.parent))
    try:
        pool_bytes = (json.dumps(pool.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (temporary / "pool.json").write_bytes(pool_bytes)
        leaderboard = pd.DataFrame([score.model_dump(mode="json") for score in scores])
        leaderboard.to_csv(temporary / "window_leaderboard.csv", index=False)
        attachments = {
            "pool.json": hashlib.sha256(pool_bytes).hexdigest(),
            "window_leaderboard.csv": hashlib.sha256((temporary / "window_leaderboard.csv").read_bytes()).hexdigest(),
        }
        (temporary / "manifest.json").write_text(
            json.dumps({"pool_id": pool.pool_id, "attachments": attachments}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / ".complete").write_text(pool.pool_id + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def load_search_pool(root: str | os.PathLike[str], pool_id: str) -> SearchPoolArtifact:
    """Load a SearchPool and verify its content ID and attachment checksums."""
    source = Path(root) / "search_pools" / pool_id
    if not (source / ".complete").is_file():
        raise ValueError(f"search pool is incomplete: {pool_id}")
    pool = SearchPoolArtifact.model_validate_json((source / "pool.json").read_text(encoding="utf-8"))
    if pool.pool_id != pool_id or search_pool_fingerprint(pool) != pool_id:
        raise ValueError("search-pool fingerprint does not match its directory")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["attachments"].items():
        if Path(name).name != name:
            raise ValueError("search-pool attachment names cannot contain directories")
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"search-pool attachment checksum mismatch: {name}")
    return pool
