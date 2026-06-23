"""Filesystem rollback for learning, governance, and config overlays."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from sports_prop_edge.core.versioning.model_registry import (
    find_last_stable_version,
    get_version,
)
from sports_prop_edge.core.versioning.snapshot_manager import (
    GOVERNANCE_CONFIG_FILE,
    GOVERNANCE_STATE_FILE,
    LEARNING_OVERLAY_FILE,
    CORRELATION_CONFIG_FILE,
    RISK_PARAMETERS_FILE,
    load_system_snapshot,
    snapshots_root,
)
from sports_prop_edge.core.versioning.versioning_types import RollbackPlan, RollbackResult
from sports_prop_edge.strategy.learning_feedback import LearningOverlay, save_learning_overlay
from sports_prop_edge.strategy.learning_governance import GovernanceState, save_governance_state

CONFIG_RESTORE_DIR = "data/config/versioning_restore"


def _restore_path(root: Path, filename: str) -> Path:
    return root / CONFIG_RESTORE_DIR / filename


def _write_restore_file(root: Path, filename: str, data: dict[str, Any]) -> Path:
    path = _restore_path(root, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _build_plan(version_id: str, snapshot_id: str, bundle_exists: bool) -> RollbackPlan:
    return RollbackPlan(
        version_id=version_id,
        snapshot_id=snapshot_id,
        restore_learning_overlay=bundle_exists,
        restore_governance_state=bundle_exists,
        restore_correlation_config=bundle_exists,
        restore_calibration_overlay=bundle_exists,
        restore_risk_parameters=bundle_exists,
        partial=not bundle_exists,
        steps=[
            "restore learning_overlay.json",
            "restore governance_state.json",
            "restore correlation_config.json sidecar",
            "restore risk_parameters.json sidecar",
            "write governance_config.json sidecar",
        ],
    )


def _execute_rollback_plan(
    plan: RollbackPlan,
    bundle,
    *,
    root: Path | None = None,
) -> RollbackResult:
    base = root or Path(__file__).resolve().parents[4]
    restored: list[str] = []
    skipped: list[str] = []
    warnings: list[str] = []

    try:
        if plan.restore_learning_overlay and bundle.learning_overlay:
            overlay = LearningOverlay.from_dict(bundle.learning_overlay)
            save_learning_overlay(overlay, base)
            _write_restore_file(base, LEARNING_OVERLAY_FILE, bundle.learning_overlay)
            restored.append("learning_overlay")
        elif plan.restore_learning_overlay:
            skipped.append("learning_overlay")

        if plan.restore_governance_state and bundle.governance_state:
            state = GovernanceState.from_dict(bundle.governance_state)
            save_governance_state(state, base)
            _write_restore_file(base, GOVERNANCE_STATE_FILE, bundle.governance_state)
            restored.append("governance_state")
        elif plan.restore_governance_state:
            skipped.append("governance_state")

        if plan.restore_correlation_config and bundle.correlation_config:
            _write_restore_file(base, CORRELATION_CONFIG_FILE, bundle.correlation_config)
            restored.append("correlation_config")
        else:
            skipped.append("correlation_config")

        if plan.restore_calibration_overlay and bundle.learning_overlay:
            calibration = {
                "calibration_drift": bundle.learning_overlay.get("calibration_drift", {}),
            }
            _write_restore_file(base, "calibration_overlay.json", calibration)
            restored.append("calibration_overlay")
        else:
            skipped.append("calibration_overlay")

        if plan.restore_risk_parameters and bundle.risk_parameters:
            _write_restore_file(base, RISK_PARAMETERS_FILE, bundle.risk_parameters)
            restored.append("risk_parameters")
        else:
            skipped.append("risk_parameters")

        if bundle.governance_config:
            _write_restore_file(base, GOVERNANCE_CONFIG_FILE, bundle.governance_config)
            restored.append("governance_config")
        else:
            skipped.append("governance_config")

        if skipped:
            plan.partial = True
            warnings.append(f"partial rollback; skipped: {', '.join(skipped)}")

        success = bool(restored)
        return RollbackResult(
            success=success,
            plan=plan,
            restored=restored,
            skipped=skipped,
            warnings=warnings,
        )
    except Exception as exc:
        return RollbackResult(
            success=False,
            plan=plan,
            restored=restored,
            skipped=skipped,
            warnings=warnings + [str(exc)],
        )


def rollback_to_version(version_id: str, root: Path | None = None) -> RollbackResult:
    """Rollback overlays/config sidecars to a registered version snapshot."""
    version = get_version(version_id, root)
    if version is None:
        plan = RollbackPlan(version_id=version_id, snapshot_id="", partial=True, steps=[])
        return RollbackResult(
            success=False,
            plan=plan,
            warnings=[f"version not found: {version_id}"],
        )

    snapshot_id = version.snapshot_id
    if not snapshot_id:
        plan = RollbackPlan(version_id=version_id, snapshot_id="", partial=True, steps=[])
        return RollbackResult(
            success=False,
            plan=plan,
            warnings=[f"version {version_id} has no linked snapshot_id"],
        )

    try:
        bundle = load_system_snapshot(snapshot_id, root)
    except FileNotFoundError:
        plan = RollbackPlan(version_id=version_id, snapshot_id=snapshot_id, partial=True, steps=[])
        return RollbackResult(
            success=False,
            plan=plan,
            warnings=[f"snapshot missing for version {version_id}: {snapshot_id}"],
        )

    plan = _build_plan(version_id, snapshot_id, True)
    return _execute_rollback_plan(plan, bundle, root=root)


def rollback_last_stable(root: Path | None = None) -> RollbackResult:
    """Rollback to the newest STABLE registered version."""
    version = find_last_stable_version(root)
    if version is None:
        base = root or Path(__file__).resolve().parents[4]
        stable_snapshots = [
            m
            for m in _list_stable_snapshot_metadata(root)
            if m.version_status == "STABLE"
        ]
        if not stable_snapshots:
            plan = RollbackPlan(version_id="", snapshot_id="", partial=True, steps=[])
            return RollbackResult(
                success=False,
                plan=plan,
                warnings=["no STABLE version or snapshot found"],
            )
        snap = stable_snapshots[0]
        bundle = load_system_snapshot(snap.snapshot_id, root)
        plan = _build_plan("snapshot_stable", snap.snapshot_id, True)
        return _execute_rollback_plan(plan, bundle, root=base)

    return rollback_to_version(version.version_id, root)


def archive_snapshot(snapshot_id: str, dest: Path, root: Path | None = None) -> Path:
    """Copy a snapshot directory to an external archive path (optional utility)."""
    src = snapshots_root(root) / snapshot_id
    if not src.exists():
        raise FileNotFoundError(snapshot_id)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / snapshot_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target)
    return target


def _list_stable_snapshot_metadata(root: Path | None):
    from sports_prop_edge.core.versioning.snapshot_manager import list_snapshots

    return [m for m in list_snapshots(root) if m.version_status == "STABLE"]
