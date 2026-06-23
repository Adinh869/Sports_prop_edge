"""Tests for production deployment layer."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from sports_prop_edge.deployment.app import create_production_app
from sports_prop_edge.deployment.config import ServiceConfig, reset_config
from sports_prop_edge.live.engine import LiveEngine


def _sgp_rows() -> list[dict]:
    return [
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


@pytest.fixture
def client(tmp_path):
    reset_config()
    (tmp_path / "data" / "config").mkdir(parents=True)
    config = ServiceConfig(
        api_key=None,
        project_root=tmp_path,
        env="test",
        bankroll=100.0,
    )
    engine = LiveEngine(root=tmp_path)
    app = create_production_app(config=config, engine=engine)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def secured_client(tmp_path):
    reset_config()
    (tmp_path / "data" / "config").mkdir(parents=True)
    config = ServiceConfig(api_key="secret-key", project_root=tmp_path, env="test")
    engine = LiveEngine(root=tmp_path)
    app = create_production_app(config=config, engine=engine)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/system/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["service"] == "sports-prop-edge"
    assert body["pipeline_status"] == "IDLE"


def test_run_slate_post(client):
    response = client.post(
        "/slate/deploy-test/run",
        json={"sgps": _sgp_rows()},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["slate_id"] == "deploy-test"
    assert body["ok"] is True


def test_run_slate_get_without_inputs_returns_404(client):
    response = client.get("/slate/missing/run")
    assert response.status_code == 404


def test_api_key_required_when_configured(secured_client):
    no_key = secured_client.post("/slate/secure/run", json={"sgps": _sgp_rows()})
    assert no_key.status_code == 401

    with_key = secured_client.post(
        "/slate/secure/run",
        json={"sgps": _sgp_rows()},
        headers={"X-API-Key": "secret-key"},
    )
    assert with_key.status_code == 200


def test_health_public_when_api_key_configured(secured_client):
    response = secured_client.get("/system/health")
    assert response.status_code == 200
    assert response.json()["api_key_required"] is True


def test_config_from_env(monkeypatch, tmp_path):
    reset_config()
    monkeypatch.setenv("SPE_PORT", "9001")
    monkeypatch.setenv("SPE_API_KEY", "from-env")
    monkeypatch.setenv("SPE_BANKROLL", "250")
    config = ServiceConfig.from_env(dotenv_path=tmp_path / "missing.env")
    assert config.port == 9001
    assert config.api_key == "from-env"
    assert config.bankroll == 250.0
