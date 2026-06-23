"""Optional API key authentication for the deployment HTTP layer."""

from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from sports_prop_edge.deployment.config import ServiceConfig, get_config

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
    config: ServiceConfig | None = None,
) -> None:
    """Validate ``X-API-Key`` when ``SPE_API_KEY`` is configured."""
    cfg = config or get_config()
    if not cfg.api_key:
        return
    if not api_key or api_key != cfg.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
