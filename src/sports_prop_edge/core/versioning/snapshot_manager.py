"""Filesystem persistence for full system state snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sports_prop_edge.core.versioning.versioning_types import (
    COMPONENT_CALIBRATION,
    COMPONENT_CORRELATION,
    COMPONENT_GOVERNANCE,
    COMPONENT_RISK,
    SnapshotBundle,
    SnapshotMetadata,
    VersionStatus,
)
from sports_prop_edge.strategy.correlation import CorrelationCalibrationConfig
from sports_prop_edge.strategy.learning_feedback import LearningOverlay, load_learning_overlay
from sports_prop_edge.strategy.learning_governance import (
    GovernanceConfig,
    GovernanceState,
    load_governance_state,
)
from sports_prop_edge.strategy.risk_positioning import RiskPositioningConfig
from sports_prop_edge.strategy.system_observability import SystemStateSnapshot

SNAPSHOTS_DIR = "data/versioning/snapshots"
MANIFEST_FILE = "manifest.json"
SYSTEM_STATE_FILE = "system_state.json"
LEARNING_OVERLAY_FILE = "learning_overlay.json"
GOVERNANCE_STATE_FILE = "governance_state.json"
CORRELATION_CONFIG_FILE = "correlation_config.json"
GOVERNANCE_CONFIG_FILE = "governance_config.json"
RISK_PARAMETERS_FILE = "risk_parameters.json"


def snapshots_root(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[4]
    return base / SNAPSHOTS_DIR


def config_hash(data: Any) -> str:
    """Stable short hash for config dictionaries."""
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def collect_config_hashes(
    *,
    learning_overlay: LearningOverlay | None = None,
    governance_state: GovernanceState | None = None,
    correlation_config: CorrelationCalibrationConfig | None = None,
    governance_config: GovernanceConfig | None = None,
    risk_config: RiskPositioningConfig | None = None,
    root: Path | None = None,
) -> dict[str, str]:
    """Compute config hashes for tracked components."""
    overlay = learning_overlay or load_learning_overlay(root)
    gov_state = governance_state or load_governance_state(root)
    corr = correlation_config or CorrelationCalibrationConfig.from_env()
    gov_cfg = governance_config or GovernanceConfig.from_env()
    risk = risk_config or RiskPositioningConfig.from_env()

    calibration_subset = {"calibration_drift": overlay.calibration_drift}
    return {
        COMPONENT_CORRELATION: config_hash(asdict(corr)),
        COMPONENT_CALIBRATION: config_hash(calibration_subset),
        COMPONENT_GOVERNANCE: config_hash(
            {"config": asdict(gov_cfg), "state": gov_state.to_dict()}
        ),
        COMPONENT_RISK: config_hash(asdict(risk)),
    }


def save_system_snapshot(
    snapshot: SystemStateSnapshot,
    *,
    learning_overlay: LearningOverlay | None = None,
    governance_state: GovernanceState | None = None,
    correlation_config: CorrelationCalibrationConfig | None = None,
    governance_config: GovernanceConfig | None = None,
    risk_config: RiskPositioningConfig | None = None,
    version_status: VersionStatus = "EXPERIMENTAL",
    snapshot_id: str | None = None,
    root: Path | None = None,
) -> SnapshotMetadata:
    """Persist observability snapshot and related overlays to disk."""
    sid = snapshot_id or uuid4().hex[:12]
    out_dir = snapshots_root(root) / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay = learning_overlay or load_learning_overlay(root)
    gov_state = governance_state or load_governance_state(root)
    corr = correlation_config or CorrelationCalibrationConfig.from_env()
    gov_cfg = governance_config or GovernanceConfig.from_env()
    risk = risk_config or RiskPositioningConfig.from_env()
    hashes = collect_config_hashes(
        learning_overlay=overlay,
        governance_state=gov_state,
        correlation_config=corr,
        governance_config=gov_cfg,
        risk_config=risk,
        root=root,
    )

    if snapshot.system_health_score >= 0.72 and snapshot.stability_status == "STABLE":
        status: VersionStatus = "STABLE"
    elif version_status == "FAILED":
        status = "FAILED"
    else:
        status = version_status

    metadata = SnapshotMetadata(
        snapshot_id=sid,
        slate_id=snapshot.slate_id,
        captured_at=snapshot.captured_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        version_status=status,
        config_hashes=hashes,
        system_health_score=float(snapshot.system_health_score),
    )

    _write_json(out_dir / MANIFEST_FILE, metadata.to_dict())
    _write_json(out_dir / SYSTEM_STATE_FILE, snapshot.to_dict())
    _write_json(out_dir / LEARNING_OVERLAY_FILE, overlay.to_dict())
    _write_json(out_dir / GOVERNANCE_STATE_FILE, gov_state.to_dict())
    _write_json(out_dir / CORRELATION_CONFIG_FILE, asdict(corr))
    _write_json(out_dir / GOVERNANCE_CONFIG_FILE, asdict(gov_cfg))
    _write_json(out_dir / RISK_PARAMETERS_FILE, asdict(risk))
    return metadata


def load_system_snapshot(snapshot_id: str, root: Path | None = None) -> SnapshotBundle:
    """Load a persisted snapshot bundle by id."""
    out_dir = snapshots_root(root) / snapshot_id
    manifest_path = out_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"snapshot not found: {snapshot_id}")

    metadata = SnapshotMetadata.from_dict(_read_json(manifest_path))
    bundle = SnapshotBundle(
        metadata=metadata,
        system_state=_read_json_optional(out_dir / SYSTEM_STATE_FILE) or {},
        learning_overlay=_read_json_optional(out_dir / LEARNING_OVERLAY_FILE),
        governance_state=_read_json_optional(out_dir / GOVERNANCE_STATE_FILE),
        correlation_config=_read_json_optional(out_dir / CORRELATION_CONFIG_FILE),
        risk_parameters=_read_json_optional(out_dir / RISK_PARAMETERS_FILE),
        governance_config=_read_json_optional(out_dir / GOVERNANCE_CONFIG_FILE),
    )
    return bundle


def list_snapshots(root: Path | None = None) -> list[SnapshotMetadata]:
    """List snapshot metadata entries (newest directories first)."""
    base = snapshots_root(root)
    if not base.exists():
        return []
    items: list[SnapshotMetadata] = []
    for path in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        manifest = path / MANIFEST_FILE
        if manifest.exists():
            try:
                items.append(SnapshotMetadata.from_dict(_read_json(manifest)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return items


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except (json.JSONDecodeError, OSError):
        return None
