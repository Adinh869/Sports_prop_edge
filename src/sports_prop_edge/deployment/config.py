"""Environment-driven service configuration with optional .env loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ServiceConfig:
    """Production API service settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    env: str = "production"
    api_key: str | None = None
    project_root: Path = Path()
    bankroll: float = 100.0
    cors_origins: str = "*"

    @classmethod
    def from_env(cls, *, dotenv_path: Path | None = None) -> ServiceConfig:
        """Load settings from environment variables and optional .env file."""
        if load_dotenv is not None:
            if dotenv_path is not None:
                load_dotenv(dotenv_path)
            else:
                root = _project_root()
                load_dotenv(root / ".env")

        root_raw = os.getenv("SPE_PROJECT_ROOT")
        project_root = Path(root_raw) if root_raw else _project_root()

        api_key = os.getenv("SPE_API_KEY") or os.getenv("API_KEY")
        if api_key is not None and not api_key.strip():
            api_key = None

        return cls(
            host=os.getenv("SPE_HOST", "0.0.0.0"),
            port=_env_int("SPE_PORT", 8000),
            workers=_env_int("SPE_WORKERS", 1),
            reload=_env_bool("SPE_RELOAD", False),
            log_level=os.getenv("SPE_LOG_LEVEL", "info").lower(),
            env=os.getenv("SPE_ENV", os.getenv("ENV", "production")),
            api_key=api_key,
            project_root=project_root,
            bankroll=_env_float("SPE_BANKROLL", 100.0),
            cors_origins=os.getenv("SPE_CORS_ORIGINS", "*"),
        )


_config: ServiceConfig | None = None


def get_config() -> ServiceConfig:
    """Return cached service configuration."""
    global _config
    if _config is None:
        _config = ServiceConfig.from_env()
    return _config


def reset_config() -> None:
    """Clear cached config (testing helper)."""
    global _config
    _config = None
