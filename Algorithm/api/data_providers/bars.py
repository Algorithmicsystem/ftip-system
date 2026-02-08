from __future__ import annotations

import csv
import datetime as dt
import io
import logging
from typing import Dict, List

import importlib.util
import requests

from .symbols import canonical_symbol, detect_country_exchange, provider_symbol

logger = logging.getLogger(__name__)

_yf_spec = importlib.util.find_spec("yfinance")
if _yf_spec:
    import yfinance as yf
else:  # pragma: no cover - optional dependency
    yf = None


class ProviderError(Exception):
    def __init__(self, reason_code: str, reason_detail: str) -> None:
        super().__init__(reason_detail)
        self.reason_code = reason_code
        self.reason_detail = reason_detail


class ProviderUnavailable(ProviderError):
    pass


class SymbolNoData(ProviderError):
    pass


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
    info = detect_country_exchange(symbol)
    if info.get("country") != "US":
        raise ProviderUnavailable(
            "PROVIDER_UNSUPPORTED", "stooq daily only supports US symbols"
        )
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
    try:
        return _fetch_daily_stooq(symbol, start_date, end_date)
    except ProviderUnavailable as exc:
        logger.info(
            "stooq daily failed, falling back",
            extra={"symbol": symbol, "reason": exc.reason_detail},
        )
    except SymbolNoData:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.info(
            "stooq daily unexpected error", extra={"symbol": symbol, "error": str(exc)}
        )

    return _fetch_daily_yfinance(symbol, start_date, end_date)


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
