"""Optional API key authentication for the deployment HTTP layer."""

from __future__ import annotations

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from sports_prop_edge.deployment.config import ServiceConfig, get_config

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
) -> None:
    """Validate ``X-API-Key`` when ``SPE_API_KEY`` is configured."""
    cfg: ServiceConfig | None = getattr(request.app.state, "config", None)
    if cfg is None:
        cfg = get_config()
    if not cfg.api_key:
        return
    if not api_key or api_key != cfg.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
