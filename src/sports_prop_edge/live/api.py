"""Minimal FastAPI-style HTTP interface over LiveEngine."""

from __future__ import annotations

import json
from typing import Any

from sports_prop_edge.live.engine import LiveEngine, LiveRunResult

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore[misc, assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    JSONResponse = None  # type: ignore[misc, assignment]
    StreamingResponse = None  # type: ignore[misc, assignment]
    _FASTAPI_AVAILABLE = False


def _serialize_run_result(result: LiveRunResult) -> dict[str, Any]:
    return {
        "slate_id": result.slate_id,
        "ok": result.ok,
        "used_fallback": result.used_fallback,
        "circuit_state": result.circuit_state,
        "warnings": result.warnings,
        "version_id": result.version.version_id if result.version else None,
        "snapshot": result.snapshot.to_dict(),
        "portfolio": {
            "optimized_objective": result.portfolio.optimized_objective,
            "slate_risk_status": result.portfolio.slate_risk_status,
            "total_allocated_weight": result.portfolio.total_allocated_weight,
            "solver_method": result.portfolio.solver_method,
        },
        "simulation": {
            "expected_return": result.simulation.expected_return,
            "simulated_mean_return": result.simulation.simulated_mean_return,
            "portfolio_std_dev": result.simulation.portfolio_std_dev,
            "var_5th_percentile": result.simulation.var_5th_percentile,
            "probability_of_loss": result.simulation.probability_of_loss,
            "ev_divergence_pct": result.simulation.ev_divergence_pct,
            "n_simulations": result.simulation.n_simulations,
        },
        "health": {
            "status": result.health.status,
            "system_health_score": result.health.system_health_score,
            "issues": result.health.issues,
        },
        "alerts": [a.to_dict() for a in result.alerts],
    }


def _serialize_status(result: LiveRunResult | None) -> dict[str, Any]:
    if result is None:
        return {"status": "unknown", "slate_id": None}
    return {
        "slate_id": result.slate_id,
        "status": result.health.status,
        "ok": result.ok,
        "stability_status": result.snapshot.stability_status,
        "system_health_score": result.snapshot.system_health_score,
        "circuit_state": result.circuit_state,
        "cached": "served from cache" in result.warnings,
    }


def create_app(engine: LiveEngine | None = None):
    """Create a FastAPI application wired to a LiveEngine instance."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError("fastapi is required for create_app(); pip install fastapi uvicorn")

    live_engine = engine or LiveEngine()
    app = FastAPI(title="Sports Prop Edge Live API", version="1.0.0")

    @app.get("/slate/{slate_id}/run")
    def run_slate(slate_id: str) -> dict[str, Any]:
        inputs = live_engine.cache.get_inputs(slate_id)
        if inputs is None:
            raise HTTPException(status_code=404, detail=f"no registered inputs for slate {slate_id}")
        result = live_engine.run_slate_live(
            slate_id,
            inputs.props,
            inputs.sgps,
            inputs.power_cards,
        )
        return _serialize_run_result(result)

    @app.get("/slate/{slate_id}/status")
    def slate_status(slate_id: str) -> dict[str, Any]:
        result = live_engine.get_last_result(slate_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"no run history for slate {slate_id}")
        return _serialize_status(result)

    @app.get("/live/stream")
    def live_stream():
        """Server-sent style stream of last known slate updates (polling snapshot)."""

        def _events():
            for slate_id, result in live_engine._last_results.items():
                payload = _serialize_status(result)
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(_events(), media_type="text/event-stream")

    @app.get("/system/health")
    def system_health_endpoint() -> dict[str, Any]:
        results = list(live_engine._last_results.values())
        if not results:
            return {
                "status": "UNKNOWN",
                "slates_tracked": 0,
                "circuit_state": live_engine.breaker.state.value,
                "version_id": (
                    live_engine.version_context.version.version_id
                    if live_engine.version_context.version
                    else None
                ),
            }

        scores = [r.snapshot.system_health_score for r in results]
        statuses = [r.health.status for r in results]
        worst = "CRITICAL" if "CRITICAL" in statuses else ("DEGRADED" if "DEGRADED" in statuses else "OK")
        return {
            "status": worst,
            "slates_tracked": len(results),
            "avg_health_score": sum(scores) / len(scores),
            "circuit_state": live_engine.breaker.state.value,
            "version_id": (
                live_engine.version_context.version.version_id
                if live_engine.version_context.version
                else None
            ),
            "slates": {r.slate_id: _serialize_status(r) for r in results},
        }

    return app


def create_router_handlers(engine: LiveEngine):
    """Lightweight dict-based handlers when FastAPI is not installed."""
    return {
        "run_slate": lambda slate_id: _run_slate_handler(engine, slate_id),
        "slate_status": lambda slate_id: _slate_status_handler(engine, slate_id),
        "system_health": lambda: _system_health_handler(engine),
    }


def _run_slate_handler(engine: LiveEngine, slate_id: str) -> dict[str, Any]:
    inputs = engine.cache.get_inputs(slate_id)
    if inputs is None:
        return {"error": f"no registered inputs for slate {slate_id}", "status_code": 404}
    result = engine.run_slate_live(slate_id, inputs.props, inputs.sgps, inputs.power_cards)
    return _serialize_run_result(result)


def _slate_status_handler(engine: LiveEngine, slate_id: str) -> dict[str, Any]:
    result = engine.get_last_result(slate_id)
    if result is None:
        return {"error": f"no run history for slate {slate_id}", "status_code": 404}
    return _serialize_status(result)


def _system_health_handler(engine: LiveEngine) -> dict[str, Any]:
    results = list(engine._last_results.values())
    if not results:
        return {
            "status": "UNKNOWN",
            "slates_tracked": 0,
            "circuit_state": engine.breaker.state.value,
        }
    statuses = [r.health.status for r in results]
    worst = "CRITICAL" if "CRITICAL" in statuses else ("DEGRADED" if "DEGRADED" in statuses else "OK")
    return {"status": worst, "slates_tracked": len(results), "circuit_state": engine.breaker.state.value}
