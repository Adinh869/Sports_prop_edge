"""Production FastAPI application factory."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sports_prop_edge.deployment.config import ServiceConfig, get_config
from sports_prop_edge.deployment.security import require_api_key
from sports_prop_edge.live.api import _serialize_run_result, _serialize_status
from sports_prop_edge.live.engine import LiveEngine, LiveEngineConfig


class SlateRunRequest(BaseModel):
    """Optional slate payload for register-and-run."""

    props: list[dict[str, Any]] | None = None
    sgps: list[dict[str, Any]] = Field(default_factory=list)
    power_cards: list[dict[str, Any]] | None = None


def create_production_app(
    config: ServiceConfig | None = None,
    engine: LiveEngine | None = None,
) -> FastAPI:
    """Build the deployable FastAPI service."""
    cfg = config or get_config()
    live_engine = engine or LiveEngine(
        root=cfg.project_root,
        config=LiveEngineConfig(bankroll=cfg.bankroll),
    )

    app = FastAPI(
        title="Sports Prop Edge API",
        version="1.0.0",
        description="Production live probabilistic decision engine",
    )

    origins = [o.strip() for o in cfg.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "sports-prop-edge", "status": "up", "env": cfg.env}

    @app.get("/system/health")
    def system_health() -> dict[str, Any]:
        """Container/orchestrator health probe — always 200 when process is alive."""
        last_results = list(live_engine._last_results.values())
        payload: dict[str, Any] = {
            "status": "OK",
            "service": "sports-prop-edge",
            "env": cfg.env,
            "circuit_state": live_engine.breaker.state.value.lower(),
            "slates_tracked": len(last_results),
            "version_id": (
                live_engine.version_context.version.version_id
                if live_engine.version_context.version
                else None
            ),
            "api_key_required": bool(cfg.api_key),
        }
        if last_results:
            scores = [r.snapshot.system_health_score for r in last_results]
            statuses = [r.health.status for r in last_results]
            worst = (
                "CRITICAL"
                if "CRITICAL" in statuses
                else ("DEGRADED" if "DEGRADED" in statuses else "OK")
            )
            payload["pipeline_status"] = worst
            payload["avg_health_score"] = sum(scores) / len(scores)
            payload["slates"] = {r.slate_id: _serialize_status(r) for r in last_results}
        else:
            payload["pipeline_status"] = "IDLE"
        return payload

    @app.get("/slate/{slate_id}/run", dependencies=[Depends(require_api_key)])
    def run_slate_get(slate_id: str) -> dict[str, Any]:
        """Execute live pipeline for a pre-registered slate."""
        return _run_slate(live_engine, slate_id)

    @app.post("/slate/{slate_id}/run", dependencies=[Depends(require_api_key)])
    def run_slate_post(slate_id: str, body: SlateRunRequest) -> dict[str, Any]:
        """Register slate inputs and execute live pipeline in one request."""
        live_engine.register_slate_inputs(
            slate_id,
            _rows_to_df(body.props),
            _rows_to_df(body.sgps),
            _rows_to_df(body.power_cards),
        )
        return _run_slate(live_engine, slate_id)

    @app.get("/slate/{slate_id}/status", dependencies=[Depends(require_api_key)])
    def slate_status(slate_id: str) -> dict[str, Any]:
        result = live_engine.get_last_result(slate_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"no run history for slate {slate_id}")
        return _serialize_status(result)

    app.state.config = cfg
    app.state.engine = live_engine
    return app


def _run_slate(engine: LiveEngine, slate_id: str) -> dict[str, Any]:
    inputs = engine.cache.get_inputs(slate_id)
    if inputs is None:
        raise HTTPException(
            status_code=404,
            detail=f"no registered inputs for slate {slate_id}; POST payload to /slate/{slate_id}/run",
        )
    result = engine.run_slate_live(
        slate_id,
        inputs.props,
        inputs.sgps,
        inputs.power_cards,
    )
    return _serialize_run_result(result)


def _rows_to_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame | None:
    if not rows:
        return None
    return pd.DataFrame(rows)


app = create_production_app()
