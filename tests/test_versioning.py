"""Tests for versioning, model registry, snapshots, and rollback."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sports_prop_edge.core.versioning import (
    COMPONENT_CORRELATION,
    collect_config_hashes,
    config_hash,
    get_latest_version,
    list_snapshots,
    list_versions,
    load_system_snapshot,
    register_version,
    rollback_last_stable,
    rollback_to_version,
    save_system_snapshot,
    update_version_status,
)
from sports_prop_edge.strategy.learning_feedback import LearningOverlay, load_learning_overlay
from sports_prop_edge.strategy.learning_governance import GovernanceState, load_governance_state
from sports_prop_edge.strategy.portfolio_optimizer import PortfolioConfig, optimize_slate_portfolio
from sports_prop_edge.strategy.portfolio_simulation import SimulationConfig, simulate_portfolio
from sports_prop_edge.strategy.system_observability import build_slate_snapshot


def _project_root(tmp_path: Path) -> Path:
    (tmp_path / "data" / "config").mkdir(parents=True)
    return tmp_path


def _minimal_snapshot(slate_id: str = "ver-test"):
    sgp = pd.DataFrame(
        [
            {
                "card": "A O 20.5 Points + B O 8.5 Rebounds",
                "sport": "NBA",
                "matchup": "NBA|bos vs nyk|2026-06-10",
                "leg1_player": "player a",
                "leg2_player": "player b",
                "leg1_model_probability": 0.60,
                "leg2_model_probability": 0.58,
                "pair_hit_probability": 0.60 * 0.58 * 0.91,
                "pair_joint_edge": 0.04,
                "risk_adjusted_joint_edge": 0.035,
                "exposure_multiplier": 0.90,
                "correlation_factor": 0.91,
                "correlation_regime": "stable",
                "risk_confidence_score": 0.75,
                "position_sizing_tier": "REDUCED",
            }
        ]
    )
    portfolio = optimize_slate_portfolio(sgp, None, config=PortfolioConfig(bankroll=100.0))
    sim = simulate_portfolio(
        portfolio,
        sgp,
        None,
        config=SimulationConfig(n_simulations=500, random_seed=1),
        bankroll=100.0,
    )
    return build_slate_snapshot(
        slate_id=slate_id,
        sgp_df=sgp,
        portfolio=portfolio,
        simulation=sim,
    )


def test_config_hash_stable():
    h1 = config_hash({"a": 1, "b": 2})
    h2 = config_hash({"b": 2, "a": 1})
    assert h1 == h2
    assert len(h1) == 16


def test_register_and_list_versions(tmp_path):
    root = _project_root(tmp_path)
    h = config_hash({"corr": 1})
    v1 = register_version(
        "baseline",
        h,
        {"component": COMPONENT_CORRELATION},
        component=COMPONENT_CORRELATION,
        root=root,
    )
    v2 = register_version(
        "tuned",
        config_hash({"corr": 2}),
        {"component": COMPONENT_CORRELATION},
        component=COMPONENT_CORRELATION,
        root=root,
    )
    assert v1.version_id != v2.version_id
    versions = list_versions(COMPONENT_CORRELATION, root=root)
    assert len(versions) == 2
    assert versions[0].name == "tuned"
    assert get_latest_version(COMPONENT_CORRELATION, root=root).name == "tuned"


def test_save_and_load_snapshot(tmp_path):
    root = _project_root(tmp_path)
    snap = _minimal_snapshot()
    overlay = LearningOverlay()
    gov = GovernanceState(cycle=3, last_updated="2026-06-01T00:00:00Z")
    meta = save_system_snapshot(
        snap,
        learning_overlay=overlay,
        governance_state=gov,
        root=root,
    )
    assert meta.snapshot_id
    assert COMPONENT_CORRELATION in meta.config_hashes

    bundle = load_system_snapshot(meta.snapshot_id, root=root)
    assert bundle.metadata.snapshot_id == meta.snapshot_id
    assert bundle.system_state.get("slate_id") == "ver-test"
    assert bundle.learning_overlay is not None
    assert bundle.governance_state is not None

    listed = list_snapshots(root=root)
    assert len(listed) == 1


def test_rollback_restores_overlay_and_governance(tmp_path):
    root = _project_root(tmp_path)
    snap = _minimal_snapshot("rollback-test")
    overlay = LearningOverlay(
        global_ev_bias_factor=1.05,
        calibration_drift={"NBA|points|0.55-0.60": 1.04},
    )
    gov = GovernanceState(cycle=7, frozen=True, last_updated="2026-06-02T00:00:00Z")
    meta = save_system_snapshot(
        snap,
        learning_overlay=overlay,
        governance_state=gov,
        version_status="STABLE",
        root=root,
    )

    # Mutate live config
    mutated = LearningOverlay(global_ev_bias_factor=0.98)
    from sports_prop_edge.strategy.learning_feedback import save_learning_overlay

    save_learning_overlay(mutated, root)
    from sports_prop_edge.strategy.learning_governance import save_governance_state

    save_governance_state(GovernanceState(cycle=99), root)

    version = register_version(
        "stable-v1",
        meta.config_hashes[COMPONENT_CORRELATION],
        {"component": COMPONENT_CORRELATION},
        component=COMPONENT_CORRELATION,
        status="STABLE",
        snapshot_id=meta.snapshot_id,
        root=root,
    )

    result = rollback_to_version(version.version_id, root=root)
    assert result.success
    assert "learning_overlay" in result.restored
    assert "governance_state" in result.restored

    restored_overlay = load_learning_overlay(root)
    assert restored_overlay.global_ev_bias_factor == pytest.approx(1.05)
    restored_gov = load_governance_state(root)
    assert restored_gov.cycle == 7
    assert restored_gov.frozen is True


def test_rollback_last_stable(tmp_path):
    root = _project_root(tmp_path)
    snap = _minimal_snapshot("stable-rollback")
    meta = save_system_snapshot(snap, version_status="STABLE", root=root)
    register_version(
        "stable",
        meta.config_hashes[COMPONENT_CORRELATION],
        {"component": COMPONENT_CORRELATION},
        component=COMPONENT_CORRELATION,
        status="STABLE",
        snapshot_id=meta.snapshot_id,
        root=root,
    )
    result = rollback_last_stable(root=root)
    assert result.success
    assert result.plan.snapshot_id == meta.snapshot_id


def test_partial_rollback_missing_snapshot(tmp_path):
    root = _project_root(tmp_path)
    version = register_version(
        "orphan",
        "deadbeef",
        {"component": COMPONENT_CORRELATION},
        component=COMPONENT_CORRELATION,
        snapshot_id="missing-snap",
        root=root,
    )
    result = rollback_to_version(version.version_id, root=root)
    assert not result.success
    assert result.plan.partial


def test_update_version_status(tmp_path):
    root = _project_root(tmp_path)
    version = register_version(
        "exp",
        "abc",
        {"component": COMPONENT_CORRELATION},
        component=COMPONENT_CORRELATION,
        status="EXPERIMENTAL",
        root=root,
    )
    updated = update_version_status(version.version_id, "FAILED", root=root)
    assert updated is not None
    assert updated.status == "FAILED"


def test_collect_config_hashes(tmp_path):
    root = _project_root(tmp_path)
    hashes = collect_config_hashes(root=root)
    assert set(hashes.keys()) == {
        "correlation_model",
        "calibration_overlay",
        "governance_config",
        "risk_parameters",
    }
