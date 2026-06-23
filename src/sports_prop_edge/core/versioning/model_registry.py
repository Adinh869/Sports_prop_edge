"""Filesystem model version registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sports_prop_edge.core.versioning.versioning_types import (
    TRACKED_COMPONENTS,
    ModelVersion,
    VersionStatus,
)

REGISTRY_FILENAME = "registry.json"


def registry_path(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[4]
    return base / "data" / "versioning" / REGISTRY_FILENAME


def _load_registry(root: Path | None = None) -> list[dict[str, Any]]:
    path = registry_path(root)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("versions", []))
    except (json.JSONDecodeError, OSError, TypeError):
        return []


def _save_registry(versions: list[dict[str, Any]], root: Path | None = None) -> None:
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"versions": versions}, indent=2), encoding="utf-8")


def register_version(
    name: str,
    config_hash: str,
    metadata: dict[str, Any] | None = None,
    *,
    component: str | None = None,
    status: VersionStatus = "EXPERIMENTAL",
    snapshot_id: str = "",
    root: Path | None = None,
) -> ModelVersion:
    """Register a component version in the filesystem registry."""
    meta = dict(metadata or {})
    comp = component or str(meta.get("component", "")).strip()
    if not comp:
        raise ValueError("component is required (pass component= or metadata['component'])")
    if comp not in TRACKED_COMPONENTS:
        raise ValueError(f"unsupported component {comp!r}; expected one of {TRACKED_COMPONENTS}")

    version = ModelVersion(
        version_id=str(meta.get("version_id") or uuid4().hex[:12]),
        component=comp,
        name=str(name),
        config_hash=str(config_hash),
        status=status,
        snapshot_id=str(snapshot_id or meta.get("snapshot_id", "")),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        metadata=meta,
    )

    versions = _load_registry(root)
    versions.append(version.to_dict())
    _save_registry(versions, root)
    return version


def get_latest_version(component: str, root: Path | None = None) -> ModelVersion | None:
    """Return the most recently registered version for a component."""
    versions = list_versions(component, root)
    return versions[0] if versions else None


def list_versions(component: str, root: Path | None = None) -> list[ModelVersion]:
    """List all registered versions for a component (newest first)."""
    raw = _load_registry(root)
    matches: list[tuple[int, ModelVersion]] = []
    for idx, item in enumerate(raw):
        if str(item.get("component")) == component:
            matches.append((idx, ModelVersion.from_dict(item)))
    matches.sort(key=lambda pair: pair[0], reverse=True)
    return [v for _, v in matches]


def get_version(version_id: str, root: Path | None = None) -> ModelVersion | None:
    """Lookup a version by id."""
    for raw in _load_registry(root):
        if str(raw.get("version_id")) == version_id:
            return ModelVersion.from_dict(raw)
    return None


def find_last_stable_version(root: Path | None = None) -> ModelVersion | None:
    """Return the newest STABLE registered version across all components."""
    stable = [
        ModelVersion.from_dict(v)
        for v in _load_registry(root)
        if str(v.get("status")) == "STABLE"
    ]
    if not stable:
        return None
    stable.sort(key=lambda v: v.created_at, reverse=True)
    return stable[0]


def update_version_status(
    version_id: str,
    status: VersionStatus,
    root: Path | None = None,
) -> ModelVersion | None:
    """Update version status (e.g. mark FAILED after bad run)."""
    versions = _load_registry(root)
    updated: ModelVersion | None = None
    for item in versions:
        if str(item.get("version_id")) == version_id:
            item["status"] = status
            updated = ModelVersion.from_dict(item)
            break
    if updated is not None:
        _save_registry(versions, root)
    return updated
