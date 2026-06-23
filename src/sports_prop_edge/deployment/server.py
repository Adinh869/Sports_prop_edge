"""Uvicorn production server entrypoint."""

from __future__ import annotations

import uvicorn

from sports_prop_edge.deployment.config import get_config


def main() -> None:
    """Run the API service with uvicorn."""
    config = get_config()
    uvicorn.run(
        "sports_prop_edge.deployment.app:app",
        host=config.host,
        port=config.port,
        workers=config.workers if not config.reload else 1,
        reload=config.reload,
        log_level=config.log_level,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
