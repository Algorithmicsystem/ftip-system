import os
import sys
from typing import Any, Optional

from dotenv import find_dotenv, load_dotenv

if "pytest" not in sys.modules:
    load_dotenv(find_dotenv(usecwd=True))


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


def db_write_enabled() -> bool:
    return env_bool("FTIP_DB_WRITE_ENABLED", True)


def db_read_enabled() -> bool:
    return env_bool("FTIP_DB_READ_ENABLED", True)


def db_required() -> bool:
    return env_bool("FTIP_DB_REQUIRED", False)


def migrations_auto() -> bool:
    return env_bool("FTIP_MIGRATIONS_AUTO", False)


def llm_enabled() -> bool:
    return env_bool("FTIP_LLM_ENABLED", False)


def openai_api_key() -> Optional[str]:
    return env("OPENAI_API_KEY") or env("OpenAI_ftip-system")


def llm_model() -> str:
    return (
        env("FTIP_OPENAI_MODEL")
        or env("FTIP_LLM_MODEL", "gpt-4o-mini")
        or "gpt-4o-mini"
    )


def llm_timeout_seconds() -> int:
    return env_int("FTIP_LLM_TIMEOUT", 30)


def llm_max_retries() -> int:
    return env_int("FTIP_LLM_MAX_RETRIES", 2)


def llm_max_tokens() -> int:
    return env_int("FTIP_LLM_MAX_TOKENS", 700)


def llm_temperature() -> float:
    return env_float("FTIP_LLM_TEMPERATURE", 0.2)


def assistant_rate_limit() -> int:
    return env_int("FTIP_ASSISTANT_RATE_LIMIT", 30)


def narrator_rate_limit() -> int:
    return env_int("FTIP_NARRATOR_RATE_LIMIT", 30)


def narrator_rate_window_seconds() -> int:
    return env_int("FTIP_NARRATOR_RATE_WINDOW_SECONDS", 600)


def massive_api_key() -> Optional[str]:
    return env("MASSIVE_API_KEY") or env("POLYGON_API_KEY")


def massive_base_url() -> str:
    return env("MASSIVE_BASE_URL") or env("POLYGON_BASE_URL") or "https://api.polygon.io"


def finnhub_api_key() -> Optional[str]:
    return env("FINNHUB_API_KEY")


def fred_api_key() -> Optional[str]:
    return env("FRED_API_KEY")


def secedgar_api_key() -> Optional[str]:
    return env("SECEDGAR_API_KEY")


def sec_user_agent() -> str:
    return env("SEC_USER_AGENT") or "FTIP System/1.0 (ops@ftip.local)"


def alphavantage_api_key() -> Optional[str]:
    return env("ALPHAVANTAGE_API_KEY")


def gnews_api_key() -> Optional[str]:
    return env("GNEWS_API_KEY")


def news_api_key() -> Optional[str]:
    return env("NEWS_API_KEY")


def _data_fabric_default_enabled() -> bool:
    return "pytest" not in sys.modules


def data_fabric_enabled() -> bool:
    return env_bool("FTIP_DATA_FABRIC_ENABLED", _data_fabric_default_enabled())


def data_fabric_timeout_seconds() -> int:
    return env_int("FTIP_DATA_FABRIC_TIMEOUT_SECONDS", 8)


def data_fabric_news_days() -> int:
    return env_int("FTIP_DATA_FABRIC_NEWS_DAYS", 14)


def data_fabric_news_limit() -> int:
    return env_int("FTIP_DATA_FABRIC_NEWS_LIMIT", 24)


def gdelt_enabled() -> bool:
    return env_bool("GDELT_ENABLED", data_fabric_enabled())


def world_bank_enabled() -> bool:
    return env_bool("WORLD_BANK_ENABLED", data_fabric_enabled())


def stooq_enabled() -> bool:
    return env_bool("STOOQ_ENABLED", True)


def safe_dict() -> dict[str, Any]:
    return {
        "llm_enabled": llm_enabled(),
        "db_enabled": db_enabled(),
        "data_fabric_enabled": data_fabric_enabled(),
    }
