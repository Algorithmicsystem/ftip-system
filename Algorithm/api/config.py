import os
from typing import Any, Optional


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def env_bool(name: str, default: bool = False) -> bool:
    value = env(name)
    if value is None:
        return default
    return value == "1" or value.lower() in {"true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = env(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    value = env(name)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def db_enabled() -> bool:
    return env_bool("FTIP_DB_ENABLED", False)


def db_required() -> bool:
    return env_bool("FTIP_DB_REQUIRED", False)


def migrations_auto() -> bool:
    return env_bool("FTIP_MIGRATIONS_AUTO", False)


def llm_enabled() -> bool:
    return env_bool("FTIP_LLM_ENABLED", False)


def openai_api_key() -> Optional[str]:
    return env("OPENAI_API_KEY")


def llm_model() -> str:
    return env("FTIP_LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini"


def llm_timeout_seconds() -> int:
    return env_int("FTIP_LLM_TIMEOUT", 30)


def llm_max_retries() -> int:
    return env_int("FTIP_LLM_MAX_RETRIES", 2)


def assistant_rate_limit() -> int:
    return env_int("FTIP_ASSISTANT_RATE_LIMIT", 30)


def safe_dict() -> dict[str, Any]:
    return {
        "llm_enabled": llm_enabled(),
        "db_enabled": db_enabled(),
    }
