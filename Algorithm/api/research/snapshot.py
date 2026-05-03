from __future__ import annotations

import datetime as dt
import json
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from api import config, db
from api.data import service as pit_service
from api.data_providers import canonical_symbol


CANONICAL_SNAPSHOT_VERSION = "phase9_canonical_snapshot_v1"

_SECTOR_PROXY_MAP = {
    "technology": "XLK",
    "financial services": "XLF",
    "financial": "XLF",
    "healthcare": "XLV",
    "energy": "XLE",
    "industrials": "XLI",
    "consumer defensive": "XLP",
    "consumer staples": "XLP",
    "utilities": "XLU",
    "communication services": "XLC",
    "consumer cyclical": "XLY",
    "materials": "XLB",
    "real estate": "XLRE",
}

_EVENT_KEYWORDS = (
    "earnings",
    "guidance",
    "outlook",
    "forecast",
    "revenue",
    "margin",
    "quarter",
    "results",
    "filing",
    "filed",
    "8-k",
    "10-q",
    "10-k",
    "sec",
)


def _db_ready() -> bool:
    try:
        return bool(
            db.db_enabled() and db.db_read_enabled() and config.env("DATABASE_URL")
        )
    except Exception:
        return False


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _iso_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value)


def _iso_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc).isoformat()
    return str(value)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _normalize_bar_dict(row: Dict[str, Any], *, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
    row_date_text = _iso_date(row.get("as_of_date") or row.get("date") or row.get("timestamp"))
    if not row_date_text:
        return None
    try:
        row_date = dt.date.fromisoformat(row_date_text)
    except ValueError:
        return None
    if row_date > as_of_date:
        return None

    close = _safe_float(row.get("close"))
    if close is None:
        return None
    open_px = _safe_float(row.get("open"))
    high_px = _safe_float(row.get("high"))
    low_px = _safe_float(row.get("low"))
    return {
        "as_of_date": row_date.isoformat(),
        "open": open_px if open_px is not None else close,
        "high": high_px if high_px is not None else close,
        "low": low_px if low_px is not None else close,
        "close": close,
        "volume": _safe_int(row.get("volume")),
        "source": row.get("source"),
        "ingested_at": _iso_datetime(row.get("ingested_at")),
    }


def _normalize_bar_rows(
    rows: Iterable[Dict[str, Any]], *, as_of_date: dt.date
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        item = _normalize_bar_dict(row, as_of_date=as_of_date)
        if item:
            normalized.append(item)
    normalized.sort(key=lambda item: item["as_of_date"])
    return normalized


def _db_rows_to_market_bars(rows: Sequence[Sequence[Any]], *, prosperity: bool = False) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if prosperity:
            normalized.append(
                {
                    "date": row[0],
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                    "source": row[6],
                }
            )
        else:
            normalized.append(
                {
                    "as_of_date": row[0],
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                    "source": row[6],
                    "ingested_at": row[7] if len(row) > 7 else None,
                }
            )
    return normalized


def _query_market_bars(
    symbol: str, from_date: dt.date, to_date: dt.date
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    if not _db_ready():
        return [], "none", []
    fallbacks: List[str] = []
    rows = db.safe_fetchall(
        """
        SELECT as_of_date, open, high, low, close, volume, source, ingested_at
        FROM market_bars_daily
        WHERE symbol = %s
          AND as_of_date BETWEEN %s AND %s
        ORDER BY as_of_date ASC
        """,
        (symbol, from_date, to_date),
    )
    if rows:
        return _db_rows_to_market_bars(rows), "market_bars_daily", fallbacks

    fallbacks.append("market_bars_daily")
    try:
        pit_rows = pit_service.query_prices_daily(
            symbol,
            from_date,
            to_date,
            dt.datetime.combine(to_date, dt.time.max, tzinfo=dt.timezone.utc),
        )
    except Exception:
        pit_rows = []
    if pit_rows:
        return (
            [
                {
                    "as_of_date": _iso_date(row.get("date")),
                    "open": _safe_float(row.get("open")),
                    "high": _safe_float(row.get("high")),
                    "low": _safe_float(row.get("low")),
                    "close": _safe_float(row.get("close")),
                    "volume": _safe_int(row.get("volume")),
                    "source": "prices_daily_versioned",
                    "ingested_at": _iso_datetime(row.get("as_of_ts")),
                }
                for row in pit_rows
            ],
            "prices_daily_versioned",
            fallbacks,
        )

    fallbacks.append("prices_daily_versioned")
    prosperity_rows = db.safe_fetchall(
        """
        SELECT date, open, high, low, close, volume, source
        FROM prosperity_daily_bars
        WHERE symbol = %s
          AND date BETWEEN %s AND %s
        ORDER BY date ASC
        """,
        (symbol, from_date, to_date),
    )
    return _db_rows_to_market_bars(prosperity_rows, prosperity=True), "prosperity_daily_bars", fallbacks


def _query_reference_bars(symbols: Sequence[str], from_date: dt.date, to_date: dt.date) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols:
        bars, source, fallbacks = _query_market_bars(symbol, from_date, to_date)
        if not bars:
            continue
        out[symbol] = {
            "bars": bars,
            "source": source,
            "fallbacks": fallbacks,
            "latest_close": bars[-1].get("close"),
        }
    return out


def _load_symbol_meta(symbol: str) -> Dict[str, Any]:
    if not _db_ready():
        return {"symbol": symbol}
    row = db.safe_fetchone(
        """
        SELECT symbol, exchange, country, currency, name, sector
        FROM market_symbols
        WHERE symbol = %s
        """,
        (symbol,),
    )
    if row:
        return {
            "symbol": row[0],
            "exchange": row[1],
            "country": row[2],
            "currency": row[3],
            "name": row[4],
            "sector": row[5],
        }
    row = db.safe_fetchone(
        """
        SELECT symbol, exchange, country, currency
        FROM symbols
        WHERE symbol = %s
        """,
        (symbol,),
    )
    if row:
        return {
            "symbol": row[0],
            "exchange": row[1],
            "country": row[2],
            "currency": row[3],
        }
    return {"symbol": symbol}


def _load_fundamentals(symbol: str, as_of_date: dt.date) -> Tuple[List[Dict[str, Any]], str]:
    if not _db_ready():
        return [], "none"
    as_of_ts = dt.datetime.combine(as_of_date, dt.time.max, tzinfo=dt.timezone.utc)
    try:
        pit_rows = pit_service.query_latest_fundamentals(symbol, as_of_ts)
    except Exception:
        pit_rows = []
    if pit_rows:
        return (
            [
                {
                    "metric_key": row.get("metric_key"),
                    "metric_value": _safe_float(row.get("metric_value")),
                    "period_end": _iso_date(row.get("period_end")),
                    "published_ts": _iso_datetime(row.get("published_ts")),
                    "as_of_ts": _iso_datetime(row.get("as_of_ts")),
                    "data_version_id": row.get("data_version_id"),
                }
                for row in pit_rows
            ],
            "fundamentals_pit",
        )

    rows = db.safe_fetchall(
        """
        SELECT fiscal_period_end, report_date, revenue, eps, gross_margin, op_margin, fcf, source, ingested_at
        FROM fundamentals_quarterly
        WHERE symbol = %s
          AND fiscal_period_end <= %s
        ORDER BY fiscal_period_end DESC
        LIMIT 8
        """,
        (symbol, as_of_date),
    )
    return (
        [
            {
                "period_end": _iso_date(row[0]),
                "report_date": _iso_date(row[1]),
                "revenue": _safe_float(row[2]),
                "eps": _safe_float(row[3]),
                "gross_margin": _safe_float(row[4]),
                "op_margin": _safe_float(row[5]),
                "fcf": _safe_float(row[6]),
                "source": row[7],
                "ingested_at": _iso_datetime(row[8]),
            }
            for row in rows
        ],
        "fundamentals_quarterly",
    )


def _load_news(symbol: str, as_of_date: dt.date) -> Tuple[List[Dict[str, Any]], str]:
    if not _db_ready():
        return [], "none"
    as_of_ts = dt.datetime.combine(as_of_date, dt.time.max, tzinfo=dt.timezone.utc)
    start_ts = as_of_ts - dt.timedelta(days=30)
    try:
        pit_rows = pit_service.query_news(symbol, as_of_ts, start_ts=start_ts, end_ts=as_of_ts, limit=200)
    except Exception:
        pit_rows = []
    if pit_rows:
        return (
            [
                {
                    "published_at": _iso_datetime(row.get("published_ts")),
                    "source": row.get("source"),
                    "title": row.get("headline"),
                    "content_snippet": row.get("full_text"),
                    "credibility": _safe_float(row.get("credibility")),
                    "as_of_ts": _iso_datetime(row.get("as_of_ts")),
                    "data_version_id": row.get("data_version_id"),
                }
                for row in pit_rows
            ],
            "news_items",
        )

    rows = db.safe_fetchall(
        """
        SELECT published_at, source, title, url, content_snippet, ingested_at
        FROM news_raw
        WHERE symbol = %s
          AND published_at <= %s
          AND published_at >= %s
        ORDER BY published_at DESC, ingested_at DESC
        LIMIT 200
        """,
        (symbol, as_of_ts, start_ts),
    )
    return (
        [
            {
                "published_at": _iso_datetime(row[0]),
                "source": row[1],
                "title": row[2],
                "url": row[3],
                "content_snippet": row[4],
                "ingested_at": _iso_datetime(row[5]),
            }
            for row in rows
        ],
        "news_raw",
    )


def _load_sentiment_history(symbol: str, as_of_date: dt.date) -> List[Dict[str, Any]]:
    if not _db_ready():
        return []
    rows = db.safe_fetchall(
        """
        SELECT as_of_date, headline_count, sentiment_mean, sentiment_pos, sentiment_neg, sentiment_neu, sentiment_score, source, computed_at
        FROM sentiment_daily
        WHERE symbol = %s
          AND as_of_date <= %s
          AND as_of_date >= %s
        ORDER BY as_of_date ASC
        """,
        (symbol, as_of_date, as_of_date - dt.timedelta(days=120)),
    )
    return [
        {
            "as_of_date": _iso_date(row[0]),
            "headline_count": _safe_int(row[1]),
            "sentiment_mean": _safe_float(row[2]),
            "sentiment_pos": _safe_int(row[3]),
            "sentiment_neg": _safe_int(row[4]),
            "sentiment_neu": _safe_int(row[5]),
            "sentiment_score": _safe_float(row[6]),
            "source": row[7],
            "computed_at": _iso_datetime(row[8]),
        }
        for row in rows
    ]


def _load_quality(symbol: str, as_of_date: dt.date) -> Dict[str, Any]:
    if not _db_ready():
        return {}
    row = db.safe_fetchone(
        """
        SELECT bars_ok, fundamentals_ok, sentiment_ok, intraday_ok, missingness, anomaly_flags, quality_score, updated_at
        FROM quality_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )
    if not row:
        return {}
    return {
        "bars_ok": row[0],
        "fundamentals_ok": row[1],
        "sentiment_ok": row[2],
        "intraday_ok": row[3],
        "missingness": _safe_float(row[4]),
        "anomaly_flags": row[5] or {},
        "quality_score": _safe_int(row[6]),
        "updated_at": _iso_datetime(row[7]),
    }


def _load_intraday(symbol: str, as_of_date: dt.date) -> List[Dict[str, Any]]:
    if not _db_ready():
        return []
    start_ts = dt.datetime.combine(as_of_date - dt.timedelta(days=2), dt.time.min).replace(
        tzinfo=dt.timezone.utc
    )
    rows = db.safe_fetchall(
        """
        SELECT ts, timeframe, open, high, low, close, volume, source, ingested_at
        FROM market_bars_intraday
        WHERE symbol = %s
          AND ts >= %s
        ORDER BY ts DESC
        LIMIT 96
        """,
        (symbol, start_ts),
    )
    return [
        {
            "ts": _iso_datetime(row[0]),
            "timeframe": row[1],
            "open": _safe_float(row[2]),
            "high": _safe_float(row[3]),
            "low": _safe_float(row[4]),
            "close": _safe_float(row[5]),
            "volume": _safe_int(row[6]),
            "source": row[7],
            "ingested_at": _iso_datetime(row[8]),
        }
        for row in rows
    ]


def _event_keyword_matches(news_items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for item in news_items:
        text = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("content_snippet") or ""),
            ]
        ).lower()
        matched_keywords = [keyword for keyword in _EVENT_KEYWORDS if keyword in text]
        if matched_keywords:
            matches.append(
                {
                    "published_at": item.get("published_at"),
                    "title": item.get("title"),
                    "matched_keywords": matched_keywords,
                }
            )
    return matches


def _latest_event_date(
    fundamentals: Sequence[Dict[str, Any]],
    event_matches: Sequence[Dict[str, Any]],
) -> Optional[dt.date]:
    candidates: List[dt.date] = []
    for row in fundamentals:
        for field in ("report_date", "published_ts", "as_of_ts", "period_end"):
            parsed = _iso_date(row.get(field))
            if not parsed:
                continue
            try:
                candidates.append(dt.date.fromisoformat(parsed[:10]))
            except ValueError:
                continue
    for item in event_matches:
        parsed = _iso_date(item.get("published_at"))
        if not parsed:
            continue
        try:
            candidates.append(dt.date.fromisoformat(parsed[:10]))
        except ValueError:
            continue
    return max(candidates) if candidates else None


def _load_event_context(
    *,
    as_of_date: dt.date,
    fundamentals: Sequence[Dict[str, Any]],
    news_items: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    matches = _event_keyword_matches(news_items)
    latest_event_date = _latest_event_date(fundamentals, matches)
    estimated_next_event_date = (
        latest_event_date + dt.timedelta(days=90)
        if latest_event_date is not None
        else None
    )
    days_to_next_event = (
        max((estimated_next_event_date - as_of_date).days, 0)
        if estimated_next_event_date is not None
        else None
    )
    days_since_last_major_event = (
        max((as_of_date - latest_event_date).days, 0)
        if latest_event_date is not None
        else None
    )
    recent_3d = 0
    recent_7d = 0
    recent_21d = 0
    for item in matches:
        published = _iso_date(item.get("published_at"))
        if not published:
            continue
        try:
            published_date = dt.date.fromisoformat(published[:10])
        except ValueError:
            continue
        if published_date > as_of_date:
            continue
        delta = (as_of_date - published_date).days
        if delta <= 3:
            recent_3d += 1
        if delta <= 7:
            recent_7d += 1
        if delta <= 21:
            recent_21d += 1

    earnings_window_flag = bool(
        (days_to_next_event is not None and days_to_next_event <= 7)
        or (days_since_last_major_event is not None and days_since_last_major_event <= 2)
    )
    post_event_instability_flag = bool(
        days_since_last_major_event is not None
        and 0 <= days_since_last_major_event <= 5
        and (recent_3d > 0 or recent_7d >= 2)
    )
    event_density_score = min((recent_7d + recent_21d * 0.35) / 8.0, 1.0)
    catalyst_burst_score = min(recent_3d / max((recent_21d - recent_7d) / 2.0, 1.0), 3.0) / 3.0
    return {
        "latest_event_date": latest_event_date.isoformat() if latest_event_date else None,
        "estimated_next_event_date": (
            estimated_next_event_date.isoformat()
            if estimated_next_event_date
            else None
        ),
        "days_to_next_event": days_to_next_event,
        "days_since_last_major_event": days_since_last_major_event,
        "earnings_window_flag": earnings_window_flag,
        "post_event_instability_flag": post_event_instability_flag,
        "event_match_count_3d": recent_3d,
        "event_match_count_7d": recent_7d,
        "event_match_count_21d": recent_21d,
        "event_density_score": round(event_density_score * 100.0, 2),
        "catalyst_burst_score": round(catalyst_burst_score * 100.0, 2),
        "major_event_matches": matches[:8],
    }


def _load_breadth_context(
    symbol_meta: Dict[str, Any],
    as_of_date: dt.date,
) -> Dict[str, Any]:
    if not _db_ready():
        return {}

    rows = db.safe_fetchall(
        """
        SELECT b.symbol,
               COALESCE(m.sector, ''),
               b.as_of_date,
               b.close
        FROM market_bars_daily b
        LEFT JOIN market_symbols m ON m.symbol = b.symbol
        WHERE b.as_of_date BETWEEN %s AND %s
          AND EXISTS (
              SELECT 1
              FROM market_symbols ms
              WHERE ms.symbol = b.symbol
                AND ms.is_active = TRUE
          )
        ORDER BY b.symbol ASC, b.as_of_date ASC
        """,
        (as_of_date - dt.timedelta(days=70), as_of_date),
    )
    if not rows:
        return {}

    sector = str(symbol_meta.get("sector") or "").strip().lower()
    symbol_rows: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sym = str(row[0] or "").strip().upper()
        if not sym:
            continue
        bucket = symbol_rows.setdefault(
            sym,
            {"sector": str(row[1] or "").strip().lower(), "closes": []},
        )
        close = _safe_float(row[3])
        if close is not None:
            bucket["closes"].append(close)

    derived_rows: List[Dict[str, Any]] = []
    for sym, payload in symbol_rows.items():
        closes = payload.get("closes") or []
        if len(closes) < 22:
            continue
        ret_1d = closes[-1] / closes[-2] - 1.0 if closes[-2] != 0 else None
        ret_21d = closes[-1] / closes[-22] - 1.0 if closes[-22] != 0 else None
        trend_sma20_50 = None
        if len(closes) >= 50:
            ma20 = sum(closes[-20:]) / 20.0
            ma50 = sum(closes[-50:]) / 50.0
            if ma50 != 0:
                trend_sma20_50 = ma20 / ma50 - 1.0
        derived_rows.append(
            {
                "symbol": sym,
                "sector": payload.get("sector"),
                "ret_1d": ret_1d,
                "ret_21d": ret_21d,
                "trend_sma20_50": trend_sma20_50,
            }
        )

    if not derived_rows:
        return {}

    universe_count = len(derived_rows)
    ret_1d_values = [row.get("ret_1d") for row in derived_rows]
    ret_21d_values = [row.get("ret_21d") for row in derived_rows]
    trend_values = [row.get("trend_sma20_50") for row in derived_rows]
    non_null_ret_1d = [value for value in ret_1d_values if value is not None]
    non_null_ret_21d = [value for value in ret_21d_values if value is not None]
    non_null_trend = [value for value in trend_values if value is not None]
    advancing_1d_ratio = (
        sum(1 for value in non_null_ret_1d if value > 0) / len(non_null_ret_1d)
        if non_null_ret_1d
        else None
    )
    advancing_21d_ratio = (
        sum(1 for value in non_null_ret_21d if value > 0) / len(non_null_ret_21d)
        if non_null_ret_21d
        else None
    )
    above_trend_ratio = (
        sum(1 for value in non_null_trend if value > 0) / len(non_null_trend)
        if non_null_trend
        else None
    )

    sector_rows = [row for row in derived_rows if str(row.get("sector") or "").strip().lower() == sector] if sector else []
    sector_ret_21d = [row.get("ret_21d") for row in sector_rows if row.get("ret_21d") is not None]
    sector_trend = [row.get("trend_sma20_50") for row in sector_rows if row.get("trend_sma20_50") is not None]
    sector_participation_ratio = (
        sum(1 for value in sector_trend if value > 0) / len(sector_trend)
        if sector_trend
        else None
    )

    dispersion = None
    if len(non_null_ret_21d) >= 2:
        try:
            dispersion = float(statistics.pstdev(non_null_ret_21d))
        except statistics.StatisticsError:
            dispersion = None
    sector_dispersion = None
    if len(sector_ret_21d) >= 2:
        try:
            sector_dispersion = float(statistics.pstdev(sector_ret_21d))
        except statistics.StatisticsError:
            sector_dispersion = None

    sorted_rets = sorted(non_null_ret_21d)
    if sorted_rets:
        bucket_size = max(1, len(sorted_rets) // 10)
        leader_strength = sum(sorted_rets[-bucket_size:]) / bucket_size
        laggard_pressure = sum(sorted_rets[:bucket_size]) / bucket_size
        positive_sum = sum(value for value in sorted_rets if value > 0) or 1e-9
        leadership_concentration = min(
            max(sum(value for value in sorted_rets[-bucket_size:] if value > 0) / positive_sum, 0.0),
            1.0,
        )
    else:
        leader_strength = None
        laggard_pressure = None
        leadership_concentration = None

    top_symbols = [
        {
            "symbol": row["symbol"],
            "ret_21d": _safe_float(row["ret_21d"]),
        }
        for row in sorted(derived_rows, key=lambda item: _safe_float(item.get("ret_21d")) or -999.0, reverse=True)[:5]
    ]

    leader_momentum = [row.get("ret_21d") for row in derived_rows if row.get("ret_21d") is not None]
    leadership_rotation = None
    if leader_momentum and non_null_ret_21d:
        try:
            leadership_rotation = min(
                abs(statistics.fmean(leader_momentum[-10:]) - statistics.fmean(non_null_ret_21d[-10:])),
                1.0,
            )
        except statistics.StatisticsError:
            leadership_rotation = None

    leadership_instability = None
    if leadership_concentration is not None or dispersion is not None or leadership_rotation is not None:
        pieces = [
            leadership_concentration,
            min(dispersion / 0.06, 1.0) if dispersion is not None else None,
            leadership_rotation,
        ]
        clean = [float(value) for value in pieces if value is not None]
        leadership_instability = sum(clean) / len(clean) if clean else None

    return {
        "as_of_date": as_of_date.isoformat(),
        "universe_count": universe_count,
        "sector_universe_count": len(sector_rows),
        "advancing_1d_ratio": advancing_1d_ratio,
        "advancing_21d_ratio": advancing_21d_ratio,
        "above_trend_ratio": above_trend_ratio,
        "sector_participation_ratio": sector_participation_ratio,
        "cross_sectional_dispersion": dispersion,
        "sector_dispersion": sector_dispersion,
        "leader_strength": leader_strength,
        "laggard_pressure": laggard_pressure,
        "leadership_concentration": leadership_concentration,
        "leadership_rotation": leadership_rotation,
        "leadership_instability": leadership_instability,
        "top_leaders": top_symbols,
        "source": "market_bars_daily",
    }


def _reference_symbols(symbol_meta: Dict[str, Any]) -> List[str]:
    symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "UUP"]
    sector = str(symbol_meta.get("sector") or "").strip().lower()
    sector_proxy = _SECTOR_PROXY_MAP.get(sector)
    if sector_proxy and sector_proxy not in symbols:
        symbols.append(sector_proxy)
    return symbols


def build_research_snapshot(
    symbol: str,
    as_of_date: dt.date,
    lookback: int = 252,
    *,
    lookback_days: int = 420,
    market_bars: Optional[Sequence[Dict[str, Any]]] = None,
    source_hint: Optional[str] = None,
    include_intraday: bool = True,
    include_reference_context: bool = True,
) -> Dict[str, Any]:
    sym = canonical_symbol(symbol)
    from_date = as_of_date - dt.timedelta(days=max(int(lookback_days), 30))
    symbol_meta = _load_symbol_meta(sym)

    fallbacks_used: List[str] = []
    if market_bars is None:
        raw_bars, bars_source, fallbacks = _query_market_bars(sym, from_date, as_of_date)
        fallbacks_used.extend(fallbacks)
    else:
        raw_bars = list(market_bars)
        bars_source = source_hint or "provided_bars"
    price_bars = _normalize_bar_rows(raw_bars, as_of_date=as_of_date)

    fundamentals, fundamentals_source = _load_fundamentals(sym, as_of_date)
    news_items, news_source = _load_news(sym, as_of_date)
    sentiment_history = _load_sentiment_history(sym, as_of_date)
    quality = _load_quality(sym, as_of_date)
    intraday_bars = _load_intraday(sym, as_of_date) if include_intraday else []
    event_context = _load_event_context(
        as_of_date=as_of_date,
        fundamentals=fundamentals,
        news_items=news_items,
    )
    breadth_context = _load_breadth_context(symbol_meta, as_of_date)
    reference_context: Dict[str, Any] = {}
    if include_reference_context:
        reference_context = _query_reference_bars(
            _reference_symbols(symbol_meta),
            from_date,
            as_of_date,
        )

    snapshot = {
        "snapshot_version": CANONICAL_SNAPSHOT_VERSION,
        "symbol": sym,
        "as_of_date": as_of_date.isoformat(),
        "as_of_ts": dt.datetime.combine(
            as_of_date, dt.time.max, tzinfo=dt.timezone.utc
        ).isoformat(),
        "requested_lookback": int(lookback),
        "available_history_bars": len(price_bars),
        "symbol_meta": symbol_meta,
        "price_bars": price_bars,
        "intraday_bars": intraday_bars,
        "fundamentals": fundamentals,
        "news": news_items,
        "sentiment_history": sentiment_history,
        "quality": quality,
        "event_context": event_context,
        "breadth_context": breadth_context,
        "reference_context": reference_context,
        "coverage": {
            "bars": len(price_bars),
            "intraday_bars": len(intraday_bars),
            "fundamentals": len(fundamentals),
            "news": len(news_items),
            "sentiment_points": len(sentiment_history),
            "event_matches": len(event_context.get("major_event_matches") or []),
            "breadth_universe_count": breadth_context.get("universe_count") or 0,
            "reference_symbols": sorted(reference_context.keys()),
        },
        "provenance": {
            "market_bars_source": bars_source,
            "fundamentals_source": fundamentals_source,
            "news_source": news_source,
            "sentiment_source": "sentiment_daily",
            "quality_source": "quality_daily" if quality else "none",
            "event_source": "fundamentals+news_heuristic",
            "breadth_source": breadth_context.get("source") or "none",
            "fallbacks_used": fallbacks_used,
            "point_in_time_precision": {
                "prices": "versioned_pit" if bars_source == "prices_daily_versioned" else "operational",
                "fundamentals": "versioned_pit" if fundamentals_source == "fundamentals_pit" else "coarse_operational",
                "news": "versioned_pit" if news_source == "news_items" else "operational",
            },
        },
    }
    snapshot["snapshot_id"] = _hash_payload(snapshot)
    snapshot["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return snapshot


def build_research_snapshot_from_bars(
    symbol: str,
    as_of_date: dt.date,
    lookback: int,
    bars: Sequence[Dict[str, Any]],
    *,
    source_hint: str = "provided_market_bars",
    include_reference_context: bool = False,
) -> Dict[str, Any]:
    max_window = 420
    if bars:
        try:
            first_date = min(
                dt.date.fromisoformat(
                    _iso_date(bar.get("as_of_date") or bar.get("date") or bar.get("timestamp")) or as_of_date.isoformat()
                )
                for bar in bars
                if _iso_date(bar.get("as_of_date") or bar.get("date") or bar.get("timestamp"))
            )
            max_window = max((as_of_date - first_date).days + 5, 30)
        except Exception:
            max_window = 420
    return build_research_snapshot(
        symbol,
        as_of_date,
        lookback,
        lookback_days=max_window,
        market_bars=bars,
        source_hint=source_hint,
        include_reference_context=include_reference_context,
    )


def build_research_snapshot_from_candles(
    symbol: str,
    as_of_date: dt.date,
    lookback: int,
    candles: Sequence[Any],
    *,
    source_hint: str = "provided_market_bars",
    include_reference_context: bool = False,
) -> Dict[str, Any]:
    bars: List[Dict[str, Any]] = []
    for candle in candles:
        bars.append(
            {
                "date": getattr(candle, "timestamp", None),
                "open": getattr(candle, "open", None),
                "high": getattr(candle, "high", None),
                "low": getattr(candle, "low", None),
                "close": getattr(candle, "close", None),
                "volume": getattr(candle, "volume", None),
                "source": getattr(candle, "source", None),
            }
        )
    return build_research_snapshot_from_bars(
        symbol,
        as_of_date,
        lookback,
        bars,
        source_hint=source_hint,
        include_reference_context=include_reference_context,
    )
