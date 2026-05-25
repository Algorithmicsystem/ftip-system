from __future__ import annotations

import csv
import datetime as dt
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import importlib.util
import httpx

from api import config
from api.source_governance import active_source_profile, source_allowed

from .alphavantage import fetch_daily_adjusted_bars
from .errors import ProviderError, ProviderUnavailable, SymbolNoData
from .quality import (
    provider_attempt,
    provider_capability_profile,
    provider_result_metadata,
)
from .symbols import canonical_symbol, detect_country_exchange, provider_symbol

logger = logging.getLogger(__name__)

_yf_spec = importlib.util.find_spec("yfinance")
if _yf_spec:
    import yfinance as yf
else:  # pragma: no cover - optional dependency
    yf = None


_PROVIDER_ALIASES = {
    "massive": "massive_polygon",
    "polygon": "massive_polygon",
    "massive_polygon": "massive_polygon",
    "alpha_vantage": "alphavantage",
}


def _unwrap_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "iloc"):
        try:
            if len(value) == 1:
                return value.iloc[0]
        except Exception:
            pass
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _float_scalar_or_none(value: Any) -> Optional[float]:
    scalar = _unwrap_scalar(value)
    if scalar in (None, ""):
        return None
    try:
        return float(scalar)
    except (TypeError, ValueError):
        return None


def _int_scalar_or_none(value: Any) -> Optional[int]:
    scalar = _unwrap_scalar(value)
    if scalar in (None, ""):
        return None
    try:
        return int(float(scalar))
    except (TypeError, ValueError):
        return None


def _daily_provider_attempts() -> List[Tuple[str, Any]]:
    attempts: List[Tuple[str, Any]] = []
    if source_allowed("massive_polygon"):
        attempts.append(("massive_polygon", _fetch_daily_massive))
    if source_allowed("alphavantage"):
        attempts.append(("alphavantage", fetch_daily_adjusted_bars))
    if config.stooq_enabled() and source_allowed("stooq"):
        attempts.append(("stooq", _fetch_daily_stooq))
    if source_allowed("yfinance"):
        attempts.append(("yfinance", _fetch_daily_yfinance))
    return attempts


def _normalize_provider_name(provider_name: Optional[str]) -> Optional[str]:
    normalized = str(provider_name or "").strip().lower()
    if not normalized:
        return None
    return _PROVIDER_ALIASES.get(normalized, normalized)


def _suppressed_provider_attempts(
    disabled_providers: Optional[List[str]],
) -> List[Dict[str, Any]]:
    disabled = {
        _normalize_provider_name(item)
        for item in (disabled_providers or [])
        if _normalize_provider_name(item)
    }
    if not disabled:
        return []
    return [
        provider_attempt(
            provider_name,
            status="suppressed",
            source_type="market_data",
            reason_code="SUPPRESSED_BY_RUN_POLICY",
            reason_detail="suppressed by run policy",
            response_quality="degraded",
            fallback_used=index > 0,
        )
        for index, (provider_name, _fetcher) in enumerate(_daily_provider_attempts())
        if _normalize_provider_name(provider_name) in disabled
    ]


def _ordered_provider_attempts(
    *,
    preferred_provider: Optional[str] = None,
    disabled_providers: Optional[List[str]] = None,
) -> List[Tuple[str, Any]]:
    disabled = {
        _normalize_provider_name(item)
        for item in (disabled_providers or [])
        if _normalize_provider_name(item)
    }
    attempts = _daily_provider_attempts()
    ranked: List[Tuple[int, int, Tuple[str, Any]]] = []
    normalized_preferred = _normalize_provider_name(preferred_provider)
    for index, attempt in enumerate(attempts):
        provider_name = _normalize_provider_name(attempt[0]) or str(attempt[0]).strip().lower()
        profile = provider_capability_profile(provider_name, capability="daily_bars")
        priority = int(profile.get("fallback_priority") or 99)
        if normalized_preferred and provider_name == normalized_preferred:
            priority = -1
        ranked.append((priority, index, attempt))
    ordered = [attempt for _priority, _index, attempt in sorted(ranked, key=lambda item: (item[0], item[1]))]
    return [
        attempt
        for attempt in ordered
        if _normalize_provider_name(attempt[0]) not in disabled
    ]

def _date_range_filter(
    rows: List[Dict[str, object]], start: dt.date, end: dt.date
) -> List[Dict[str, object]]:
    filtered = []
    for row in rows:
        as_of = row.get("as_of_date")
        if isinstance(as_of, dt.date) and start <= as_of <= end:
            filtered.append(row)
    return filtered


def _fetch_daily_stooq(
    symbol: str, start: dt.date, end: dt.date
) -> List[Dict[str, object]]:
    stooq_symbol = provider_symbol(symbol, "stooq")
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    resp = httpx.get(url, timeout=15)
    if resp.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"stooq HTTP {resp.status_code}",
            provider_name="stooq",
            source_type="market_data",
        )
    reader = csv.DictReader(io.StringIO(resp.text))
    rows: List[Dict[str, object]] = []
    for row in reader:
        try:
            as_of = dt.date.fromisoformat(row["Date"])
        except Exception:
            continue
        rows.append(
            {
                "symbol": canonical_symbol(symbol),
                "as_of_date": as_of,
                "open": float(row.get("Open") or 0) or None,
                "high": float(row.get("High") or 0) or None,
                "low": float(row.get("Low") or 0) or None,
                "close": float(row.get("Close") or 0) or None,
                "volume": (
                    int(float(row.get("Volume") or 0)) if row.get("Volume") else None
                ),
                "source": "stooq",
            }
        )
    rows = _date_range_filter(rows, start, end)
    if not rows:
        reason_code = "PROVIDER_PARSE_FAILURE" if "Date" in (resp.text or "") else "NO_DATA"
        raise SymbolNoData(
            reason_code,
            "stooq daily bars could not be parsed" if reason_code == "PROVIDER_PARSE_FAILURE" else "no daily bars returned",
            provider_name="stooq",
            source_type="market_data",
        )
    return rows


def _fetch_daily_massive(
    symbol: str, start: dt.date, end: dt.date
) -> List[Dict[str, object]]:
    def _legacy_rows() -> List[Dict[str, object]]:
        try:
            from api import main as app_main
        except Exception:
            return []
        fetcher = getattr(app_main, "massive_fetch_daily_bars", None)
        if not callable(fetcher):
            return []
        try:
            payload = fetcher(canonical_symbol(symbol), start.isoformat(), end.isoformat())
        except Exception:
            return []
        rows: List[Dict[str, object]] = []
        for item in payload or []:
            timestamp = getattr(item, "timestamp", None)
            if timestamp is None and isinstance(item, dict):
                timestamp = item.get("timestamp") or item.get("as_of_date") or item.get("date")
            if timestamp is None:
                continue
            try:
                as_of = (
                    timestamp
                    if isinstance(timestamp, dt.date)
                    else dt.date.fromisoformat(str(timestamp)[:10])
                )
            except Exception:
                continue
            close = getattr(item, "close", None)
            volume = getattr(item, "volume", None)
            if isinstance(item, dict):
                close = item.get("close", close)
                volume = item.get("volume", volume)
            if close is None:
                continue
            rows.append(
                {
                    "symbol": canonical_symbol(symbol),
                    "as_of_date": as_of,
                    "open": getattr(item, "open", None) if not isinstance(item, dict) else item.get("open"),
                    "high": getattr(item, "high", None) if not isinstance(item, dict) else item.get("high"),
                    "low": getattr(item, "low", None) if not isinstance(item, dict) else item.get("low"),
                    "close": float(close),
                    "volume": int(float(volume)) if volume is not None else None,
                    "source": "massive_polygon",
                }
            )
        return _date_range_filter(rows, start, end)

    api_key = config.massive_api_key()
    if not api_key:
        legacy_rows = _legacy_rows()
        if legacy_rows:
            return legacy_rows
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            "MASSIVE_API_KEY not set",
            provider_name="massive_polygon",
            source_type="market_data",
        )
    base_url = config.massive_base_url().rstrip("/")
    url = f"{base_url}/v2/aggs/ticker/{canonical_symbol(symbol)}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    resp = httpx.get(
        url,
        params={
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": api_key,
        },
        timeout=config.data_fabric_timeout_seconds(),
    )
    if resp.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"massive/polygon HTTP {resp.status_code}",
            provider_name="massive_polygon",
            source_type="market_data",
        )
    payload = resp.json()
    results = payload.get("results") or []
    if not isinstance(results, list) or not results:
        raise SymbolNoData(
            "NO_DATA",
            "massive/polygon returned no daily bars",
            provider_name="massive_polygon",
            source_type="market_data",
        )
    rows: List[Dict[str, object]] = []
    for row in results:
        timestamp = row.get("t")
        close = row.get("c")
        if timestamp is None or close is None:
            continue
        as_of = dt.datetime.utcfromtimestamp(int(timestamp) / 1000.0).date()
        rows.append(
            {
                "symbol": canonical_symbol(symbol),
                "as_of_date": as_of,
                "open": float(row.get("o")) if row.get("o") is not None else None,
                "high": float(row.get("h")) if row.get("h") is not None else None,
                "low": float(row.get("l")) if row.get("l") is not None else None,
                "close": float(close),
                "volume": int(float(row.get("v"))) if row.get("v") is not None else None,
                "source": "massive_polygon",
            }
        )
    if not rows:
        raise ProviderUnavailable(
            "PROVIDER_PARSE_FAILURE",
            "massive/polygon bars could not be parsed",
            provider_name="massive_polygon",
            source_type="market_data",
        )
    return rows


def _fetch_daily_yfinance(
    symbol: str, start: dt.date, end: dt.date
) -> List[Dict[str, object]]:
    if yf is None:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            "yfinance not installed",
            provider_name="yfinance",
            source_type="market_data",
        )
    y_symbol = provider_symbol(symbol, "yfinance")
    df = yf.download(
        y_symbol,
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat(),
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise SymbolNoData(
            "NO_DATA",
            "no daily bars returned",
            provider_name="yfinance",
            source_type="market_data",
        )
    rows = []
    for idx, row in df.iterrows():
        as_of = (
            idx.date() if hasattr(idx, "date") else dt.date.fromisoformat(str(idx)[:10])
        )
        rows.append(
            {
                "symbol": canonical_symbol(symbol),
                "as_of_date": as_of,
                "open": _float_scalar_or_none(row.get("Open")),
                "high": _float_scalar_or_none(row.get("High")),
                "low": _float_scalar_or_none(row.get("Low")),
                "close": _float_scalar_or_none(row.get("Close")),
                "volume": _int_scalar_or_none(row.get("Volume")),
                "source": "yfinance",
            }
        )
    if not rows:
        raise ProviderUnavailable(
            "PROVIDER_PARSE_FAILURE",
            "yfinance daily bars could not be parsed",
            provider_name="yfinance",
            source_type="market_data",
        )
    return _date_range_filter(rows, start, end)


def fetch_daily_bars(
    symbol: str, start_date: dt.date, end_date: dt.date
) -> List[Dict[str, object]]:
    rows, _metadata = fetch_daily_bars_with_meta(symbol, start_date, end_date)
    return rows


def fetch_daily_bars_with_meta(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    preferred_provider: Optional[str] = None,
    disabled_providers: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, Any]]:
    disabled_set = {
        _normalize_provider_name(item)
        for item in (disabled_providers or [])
        if _normalize_provider_name(item)
    }
    suppressed_attempts = _suppressed_provider_attempts(disabled_providers)
    attempts = _ordered_provider_attempts(
        preferred_provider=preferred_provider,
        disabled_providers=disabled_providers,
    )
    if not attempts:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"no daily-bar providers are allowed under source profile {active_source_profile()}",
            provider_name="daily_bars",
            source_type="market_data",
            metadata={
                "provider_context": provider_result_metadata(
                    "daily_bars",
                    source_type="market_data",
                    end_date=end_date,
                    response_quality="degraded",
                    error_summary="all providers suppressed or disallowed",
                    attempts=suppressed_attempts,
                    capability="daily_bars",
                )
            },
        )

    attempt_log: List[Dict[str, Any]] = list(suppressed_attempts)
    last_error: Optional[ProviderError] = None
    for index, (provider_name, fetcher) in enumerate(attempts):
        fallback_used = index > 0 or bool(suppressed_attempts)
        try:
            rows = fetcher(symbol, start_date, end_date)
            attempt_log.append(
                provider_attempt(
                    provider_name,
                    status="success",
                    source_type="market_data",
                    response_quality="complete",
                    fallback_used=fallback_used,
                )
            )
            metadata = provider_result_metadata(
                provider_name,
                source_type="market_data",
                end_date=end_date,
                fallback_used=fallback_used,
                response_quality="complete",
                attempts=attempt_log,
                capability="daily_bars",
            )
            metadata["preferred_provider"] = _normalize_provider_name(preferred_provider)
            metadata["suppressed_providers"] = sorted(item for item in disabled_set if item)
            metadata["available_providers"] = [
                _normalize_provider_name(name) or str(name).strip().lower()
                for name, _fetcher in _ordered_provider_attempts(
                    preferred_provider=preferred_provider,
                    disabled_providers=None,
                )
            ]
            return rows, metadata
        except ProviderError as exc:
            last_error = exc
            attempt_log.append(
                provider_attempt(
                    provider_name,
                    status="failed",
                    source_type="market_data",
                    reason_code=exc.reason_code,
                    reason_detail=exc.reason_detail,
                    response_quality="degraded",
                    fallback_used=fallback_used,
                )
            )
            logger.info(
                "%s daily failed, falling back",
                provider_name,
                extra={"symbol": symbol, "reason": exc.reason_detail},
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive
            last_error = ProviderUnavailable(
                "PROVIDER_UNAVAILABLE",
                str(exc),
                provider_name=provider_name,
                source_type="market_data",
            )
            attempt_log.append(
                provider_attempt(
                    provider_name,
                    status="failed",
                    source_type="market_data",
                    reason_code="PROVIDER_UNAVAILABLE",
                    reason_detail=str(exc),
                    response_quality="degraded",
                    fallback_used=fallback_used,
                )
            )
            logger.info(
                "%s daily unexpected error",
                provider_name,
                extra={"symbol": symbol, "error": str(exc)},
            )

    if last_error is None:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            "daily bars unavailable",
            provider_name="daily_bars",
            source_type="market_data",
            metadata={"attempts": attempt_log},
        )

    last_error.metadata = {
        **dict(last_error.metadata or {}),
        "attempts": attempt_log,
        "provider_context": provider_result_metadata(
            last_error.provider_name or "unknown",
            source_type="market_data",
            end_date=end_date,
            fallback_used=bool(attempt_log[:-1]),
            response_quality="degraded",
            error_summary=last_error.reason_detail,
            attempts=attempt_log,
            capability="daily_bars",
        ),
    }
    if isinstance(last_error.metadata.get("provider_context"), dict):
        last_error.metadata["provider_context"]["preferred_provider"] = _normalize_provider_name(
            preferred_provider
        )
        last_error.metadata["provider_context"]["suppressed_providers"] = sorted(
            item for item in disabled_set if item
        )
    raise last_error


def fetch_reference_bars(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> List[Dict[str, object]]:
    errors: List[str] = []
    attempts = []
    if source_allowed("stooq"):
        attempts.append(_fetch_daily_stooq)
    if source_allowed("alphavantage"):
        attempts.append(fetch_daily_adjusted_bars)
    if source_allowed("yfinance"):
        attempts.append(_fetch_daily_yfinance)
    if not attempts:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"no reference-bar providers are allowed under source profile {active_source_profile()}",
        )
    for fetcher in attempts:
        try:
            return fetcher(symbol, start_date, end_date)
        except Exception as exc:
            errors.append(str(exc))
    raise ProviderUnavailable(
        "PROVIDER_UNAVAILABLE",
        "; ".join(errors) or "reference bars unavailable",
    )


def fetch_intraday_bars(
    symbol: str,
    start_ts: dt.datetime,
    end_ts: dt.datetime,
    timeframe: str = "5m",
) -> List[Dict[str, object]]:
    if yf is None:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            "yfinance not installed",
            provider_name="yfinance",
            source_type="market_data",
        )
    y_symbol = provider_symbol(symbol, "yfinance")
    df = yf.download(
        y_symbol,
        start=start_ts.isoformat(),
        end=end_ts.isoformat(),
        interval=timeframe,
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise SymbolNoData(
            "NO_DATA",
            "no intraday bars returned",
            provider_name="yfinance",
            source_type="market_data",
        )

    rows: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime().astimezone(dt.timezone.utc)
        rows.append(
            {
                "symbol": canonical_symbol(symbol),
                "ts": ts,
                "timeframe": timeframe,
                "open": _float_scalar_or_none(row.get("Open")),
                "high": _float_scalar_or_none(row.get("High")),
                "low": _float_scalar_or_none(row.get("Low")),
                "close": _float_scalar_or_none(row.get("Close")),
                "volume": _int_scalar_or_none(row.get("Volume")),
                "source": "yfinance",
            }
        )
    return rows
