"""Production versioning, snapshot persistence, and rollback (opt-in)."""

from sports_prop_edge.core.versioning.model_registry import (
    find_last_stable_version,
    get_latest_version,
    get_version,
    list_versions,
    register_version,
    update_version_status,
)
from sports_prop_edge.core.versioning.rollback import (
    archive_snapshot,
    rollback_last_stable,
    rollback_to_version,
)
from sports_prop_edge.core.versioning.snapshot_manager import (
    collect_config_hashes,
    config_hash,
    list_snapshots,
    load_system_snapshot,
    save_system_snapshot,
)
from sports_prop_edge.core.versioning.versioning_types import (
    COMPONENT_CALIBRATION,
    COMPONENT_CORRELATION,
    COMPONENT_GOVERNANCE,
    COMPONENT_RISK,
    ModelVersion,
    RollbackPlan,
    RollbackResult,
    SnapshotBundle,
    SnapshotMetadata,
    TRACKED_COMPONENTS,
    VersionStatus,
)

__all__ = [
    "COMPONENT_CALIBRATION",
    "COMPONENT_CORRELATION",
    "COMPONENT_GOVERNANCE",
    "COMPONENT_RISK",
    "ModelVersion",
    "RollbackPlan",
    "RollbackResult",
    "SnapshotBundle",
    "SnapshotMetadata",
    "TRACKED_COMPONENTS",
    "VersionStatus",
    "archive_snapshot",
    "collect_config_hashes",
    "config_hash",
    "find_last_stable_version",
    "get_latest_version",
    "get_version",
    "list_snapshots",
    "list_versions",
    "load_system_snapshot",
    "register_version",
    "rollback_last_stable",
    "rollback_to_version",
    "save_system_snapshot",
    "update_version_status",
]
