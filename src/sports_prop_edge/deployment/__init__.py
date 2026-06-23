"""Production deployment layer (HTTP service, config, container)."""

from sports_prop_edge.deployment.app import app, create_production_app
from sports_prop_edge.deployment.config import ServiceConfig, get_config, reset_config
from sports_prop_edge.deployment.security import require_api_key

__all__ = [
    "ServiceConfig",
    "app",
    "create_production_app",
    "get_config",
    "require_api_key",
    "reset_config",
]
