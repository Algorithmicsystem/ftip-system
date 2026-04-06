from __future__ import annotations

import csv
import datetime as dt
import io
import logging
from typing import Dict, List

import importlib.util
import requests

from api import config

from .alphavantage import fetch_daily_adjusted_bars
from .errors import ProviderError, ProviderUnavailable, SymbolNoData
from .symbols import canonical_symbol, detect_country_exchange, provider_symbol

logger = logging.getLogger(__name__)

_yf_spec = importlib.util.find_spec("yfinance")
if _yf_spec:
    import yfinance as yf
else:  # pragma: no cover - optional dependency
    yf = None

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
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE", f"stooq HTTP {resp.status_code}"
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
        raise SymbolNoData("NO_DATA", "no daily bars returned")
    return rows


def _fetch_daily_massive(
    symbol: str, start: dt.date, end: dt.date
) -> List[Dict[str, object]]:
    api_key = config.massive_api_key()
    if not api_key:
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "MASSIVE_API_KEY not set")
    base_url = config.massive_base_url().rstrip("/")
    url = f"{base_url}/v2/aggs/ticker/{canonical_symbol(symbol)}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    resp = requests.get(
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
            "PROVIDER_UNAVAILABLE", f"massive/polygon HTTP {resp.status_code}"
        )
    payload = resp.json()
    results = payload.get("results") or []
    if not isinstance(results, list) or not results:
        raise SymbolNoData("NO_DATA", "massive/polygon returned no daily bars")
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
        raise SymbolNoData("NO_DATA", "massive/polygon bars could not be parsed")
    return rows


def _fetch_daily_yfinance(
    symbol: str, start: dt.date, end: dt.date
) -> List[Dict[str, object]]:
    if yf is None:
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "yfinance not installed")
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
        raise SymbolNoData("NO_DATA", "no daily bars returned")
    rows = []
    for idx, row in df.iterrows():
        as_of = (
            idx.date() if hasattr(idx, "date") else dt.date.fromisoformat(str(idx)[:10])
        )
        rows.append(
            {
                "symbol": canonical_symbol(symbol),
                "as_of_date": as_of,
                "open": float(row.get("Open")) if row.get("Open") is not None else None,
                "high": float(row.get("High")) if row.get("High") is not None else None,
                "low": float(row.get("Low")) if row.get("Low") is not None else None,
                "close": (
                    float(row.get("Close")) if row.get("Close") is not None else None
                ),
                "volume": (
                    int(row.get("Volume")) if row.get("Volume") is not None else None
                ),
                "source": "yfinance",
            }
        )
    return _date_range_filter(rows, start, end)


def fetch_daily_bars(
    symbol: str, start_date: dt.date, end_date: dt.date
) -> List[Dict[str, object]]:
    attempts = [
        ("massive_polygon", _fetch_daily_massive),
        ("alphavantage", fetch_daily_adjusted_bars),
    ]
    if config.stooq_enabled():
        attempts.append(("stooq", _fetch_daily_stooq))
    for provider_name, fetcher in attempts:
        try:
            return fetcher(symbol, start_date, end_date)
        except ProviderUnavailable as exc:
            logger.info(
                "%s daily failed, falling back",
                provider_name,
                extra={"symbol": symbol, "reason": exc.reason_detail},
            )
        except SymbolNoData:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.info(
                "%s daily unexpected error",
                provider_name,
                extra={"symbol": symbol, "error": str(exc)},
            )
    return _fetch_daily_yfinance(symbol, start_date, end_date)


def fetch_reference_bars(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> List[Dict[str, object]]:
    errors: List[str] = []
    for fetcher in (_fetch_daily_stooq, fetch_daily_adjusted_bars, _fetch_daily_yfinance):
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
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "yfinance not installed")
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
        raise SymbolNoData("NO_DATA", "no intraday bars returned")

    rows: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime().astimezone(dt.timezone.utc)
        rows.append(
            {
                "symbol": canonical_symbol(symbol),
                "ts": ts,
                "timeframe": timeframe,
                "open": float(row.get("Open")) if row.get("Open") is not None else None,
                "high": float(row.get("High")) if row.get("High") is not None else None,
                "low": float(row.get("Low")) if row.get("Low") is not None else None,
                "close": (
                    float(row.get("Close")) if row.get("Close") is not None else None
                ),
                "volume": (
                    int(row.get("Volume")) if row.get("Volume") is not None else None
                ),
                "source": "yfinance",
            }
        )
    return rows
