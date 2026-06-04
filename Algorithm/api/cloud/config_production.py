"""Phase 22.1: Production Environment Configuration."""
from __future__ import annotations

import os
from typing import Any, Dict, List

PRODUCTION_ENVIRONMENTS: Dict[str, Dict[str, Any]] = {
    "development": {
        "debug": True,
        "db_pool_size": 5,
        "db_pool_max": 10,
        "worker_count": 1,
        "log_level": "DEBUG",
        "cache_ttl_minutes": 5,
        "rate_limit_rpm": 1000,
        "cors_origins": ["*"],
    },
    "staging": {
        "debug": False,
        "db_pool_size": 10,
        "db_pool_max": 20,
        "worker_count": 2,
        "log_level": "INFO",
        "cache_ttl_minutes": 15,
        "rate_limit_rpm": 5000,
        "cors_origins": ["https://staging.axiom.io"],
    },
    "production": {
        "debug": False,
        "db_pool_size": 20,
        "db_pool_max": 50,
        "worker_count": 4,
        "log_level": "WARNING",
        "cache_ttl_minutes": 30,
        "rate_limit_rpm": 10000,
        "cors_origins": ["https://axiom.io", "https://app.axiom.io"],
    },
}

_REQUIRED_KEYS = ["db_pool_size", "worker_count", "log_level", "cors_origins",
                  "debug", "db_pool_max", "cache_ttl_minutes", "rate_limit_rpm"]

_REQUIRED_SECRETS = ["DATABASE_URL", "FTIP_API_KEY", "POLYGON_API_KEY"]
_OPTIONAL_SECRETS = ["ANTHROPIC_API_KEY", "FRED_API_KEY"]


def get_production_config() -> Dict[str, Any]:
    env_name = os.getenv("AXIOM_ENV", "development")
    return PRODUCTION_ENVIRONMENTS.get(env_name, PRODUCTION_ENVIRONMENTS["development"]).copy()


def validate_production_secrets() -> Dict[str, Any]:
    missing_required: List[str] = [k for k in _REQUIRED_SECRETS if not os.getenv(k)]
    missing_optional: List[str] = [k for k in _OPTIONAL_SECRETS if not os.getenv(k)]

    warnings: List[str] = []
    for k in missing_optional:
        warnings.append(f"Optional secret {k} not set — some features may be degraded")

    if missing_required:
        warnings.insert(0, f"Missing {len(missing_required)} required secret(s): {', '.join(missing_required)}")

    return {
        "ready_for_production": len(missing_required) == 0,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "warnings": warnings,
    }
