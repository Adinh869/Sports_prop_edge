"""Versioning and rollback type definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

VersionStatus = Literal["STABLE", "EXPERIMENTAL", "FAILED"]

COMPONENT_CORRELATION = "correlation_model"
COMPONENT_CALIBRATION = "calibration_overlay"
COMPONENT_GOVERNANCE = "governance_config"
COMPONENT_RISK = "risk_parameters"

TRACKED_COMPONENTS = (
    COMPONENT_CORRELATION,
    COMPONENT_CALIBRATION,
    COMPONENT_GOVERNANCE,
    COMPONENT_RISK,
)


@dataclass(frozen=True)
class ModelVersion:
    """Registered version for a tracked system component."""

    version_id: str
    component: str
    name: str
    config_hash: str
    status: VersionStatus
    snapshot_id: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelVersion:
        return cls(
            version_id=str(data.get("version_id", "")),
            component=str(data.get("component", "")),
            name=str(data.get("name", "")),
            config_hash=str(data.get("config_hash", "")),
            status=str(data.get("status", "EXPERIMENTAL")),  # type: ignore[arg-type]
            snapshot_id=str(data.get("snapshot_id", "")),
            created_at=str(data.get("created_at", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class SnapshotMetadata:
    """Filesystem snapshot bundle metadata."""

    snapshot_id: str
    slate_id: str
    captured_at: str
    version_status: VersionStatus
    config_hashes: dict[str, str]
    system_health_score: float = 0.0
    version_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotMetadata:
        return cls(
            snapshot_id=str(data.get("snapshot_id", "")),
            slate_id=str(data.get("slate_id", "")),
            captured_at=str(data.get("captured_at", "")),
            version_status=str(data.get("version_status", "EXPERIMENTAL")),  # type: ignore[arg-type]
            config_hashes=dict(data.get("config_hashes", {})),
            system_health_score=float(data.get("system_health_score", 0.0)),
            version_ids=list(data.get("version_ids", [])),
        )


@dataclass
class RollbackPlan:
    """Describes what will be restored during rollback."""

    version_id: str
    snapshot_id: str
    restore_learning_overlay: bool = True
    restore_governance_state: bool = True
    restore_correlation_config: bool = True
    restore_calibration_overlay: bool = True
    restore_risk_parameters: bool = True
    partial: bool = False
    steps: list[str] = field(default_factory=list)


@dataclass
class RollbackResult:
    """Outcome of a rollback operation."""

    success: bool
    plan: RollbackPlan
    restored: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SnapshotBundle:
    """Loaded snapshot package."""

    metadata: SnapshotMetadata
    system_state: dict[str, Any]
    learning_overlay: dict[str, Any] | None = None
    governance_state: dict[str, Any] | None = None
    correlation_config: dict[str, Any] | None = None
    risk_parameters: dict[str, Any] | None = None
    governance_config: dict[str, Any] | None = None
