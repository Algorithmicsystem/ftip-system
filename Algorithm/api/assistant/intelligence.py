from __future__ import annotations

import datetime as dt
import math
import re
import statistics
import uuid
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from api import db
from api.assistant import data_fabric


ANALYSIS_JOB_KIND = "analysis_job_context"
DATA_BUNDLE_KIND = "normalized_data_bundle"
FEATURE_FACTOR_BUNDLE_KIND = "feature_factor_bundle"

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "after",
    "amid",
    "over",
    "under",
    "stock",
    "shares",
    "company",
    "corp",
    "quarter",
    "market",
    "markets",
    "earnings",
    "analyst",
    "price",
    "today",
    "says",
}
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
_GEOPOLITICAL_KEYWORDS = {
    "rates_policy": {"fed", "inflation", "rate", "rates", "yield", "treasury", "cpi", "pce"},
    "regulation_policy": {"regulation", "regulatory", "antitrust", "export", "license", "sec", "doj", "ftc", "fda"},
    "trade_geopolitics": {"tariff", "sanction", "war", "conflict", "ukraine", "russia", "china", "taiwan", "middle", "east"},
    "election_policy": {"election", "congress", "senate", "white", "house", "administration", "president"},
    "commodity_supply": {"oil", "gas", "copper", "lithium", "rare", "earth", "supply", "chain"},
}


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_int(value: Any) -> Optional[int]:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(statistics.fmean(clean))


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(statistics.median(clean))


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(clean) < 2:
        return None
    return float(statistics.pstdev(clean))


def _clamp(value: Optional[float], low: float, high: float) -> Optional[float]:
    if value is None:
        return None
    return max(low, min(high, float(value)))


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _pct_change(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    if current is None or prior in (None, 0):
        return None
    return float(current) / float(prior) - 1.0


def _score_100(value: Optional[float], *, low: float = -1.0, high: float = 1.0) -> Optional[float]:
    if value is None:
        return None
    if math.isclose(high, low):
        return 50.0
    clipped = max(low, min(high, float(value)))
    return 100.0 * ((clipped - low) / (high - low))


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


def _first_available(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _days_stale(as_of_date: Optional[str], latest_ts: Optional[str]) -> Optional[int]:
    if not as_of_date or not latest_ts:
        return None
    try:
        as_of = dt.date.fromisoformat(as_of_date)
        latest = dt.datetime.fromisoformat(latest_ts.replace("Z", "+00:00")).date()
    except Exception:
        return None
    return max((as_of - latest).days, 0)


def _domain_status(days_stale: Optional[int], *, fresh_days: int, usable_days: int) -> str:
    if days_stale is None:
        return "unknown"
    if days_stale <= fresh_days:
        return "fresh"
    if days_stale <= usable_days:
        return "stale_but_usable"
    return "stale"


def _to_returns(close_values: Sequence[Optional[float]]) -> List[float]:
    returns: List[float] = []
    previous: Optional[float] = None
    for current in close_values:
        if current is None:
            previous = None
            continue
        if previous not in (None, 0):
            returns.append(float(current) / float(previous) - 1.0)
        previous = current
    return returns


def _realized_vol(close_values: Sequence[Optional[float]], window: int) -> Optional[float]:
    returns = _to_returns(close_values)
    if len(returns) < window:
        return None
    windowed = returns[-window:]
    sigma = _std(windowed)
    if sigma is None:
        return None
    return float(sigma * math.sqrt(252.0))


def _linear_slope(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(clean) < 3:
        return None
    x_values = list(range(len(clean)))
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(clean)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, clean))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _moving_average(values: Sequence[Optional[float]], window: int) -> Optional[float]:
    if len(values) < window:
        return None
    return _mean(values[-window:])


def _normalize_title(title: str) -> str:
    title = re.sub(r"\s+", " ", (title or "").strip().lower())
    return re.sub(r"[^a-z0-9 ]+", "", title)


def _extract_topics(titles: Sequence[str], symbol: str) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    ignored = {symbol.lower(), symbol.lower().replace(".", ""), *(_STOPWORDS)}
    for title in titles:
        for token in _TOKEN_RE.findall(title or ""):
            lowered = token.lower()
            if lowered in ignored:
                continue
            counter[lowered] += 1
    topics: List[Dict[str, Any]] = []
    for keyword, count in counter.most_common(5):
        topics.append({"topic": keyword, "count": count})
    return topics


def _db_ready() -> bool:
    return db.db_enabled() and db.db_read_enabled()


def _safe_fetchall(sql: str, params: Sequence[Any]) -> List[Sequence[Any]]:
    if not _db_ready():
        return []
    try:
        return list(db.safe_fetchall(sql, params))
    except Exception:
        return []


def _safe_fetchone(sql: str, params: Sequence[Any]) -> Optional[Sequence[Any]]:
    if not _db_ready():
        return None
    try:
        return db.safe_fetchone(sql, params)
    except Exception:
        return None


def build_analysis_job_context(
    request: Dict[str, Any],
    *,
    session_id: str,
    trace_id: Optional[str] = None,
    as_of_date: Any = None,
    freshness: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scenario = str(
        request.get("scenario_mode") or request.get("scenario") or "base"
    ).strip() or "base"
    analysis_depth = str(request.get("analysis_depth") or "standard").strip() or "standard"
    refresh_mode = str(request.get("refresh_mode") or "refresh_stale").strip() or "refresh_stale"
    market_regime = str(
        request.get("market_regime") or request.get("macro_sensitivity") or "auto"
    ).strip() or "auto"
    required_domains = [
        "market_price_volume",
        "technical_market_structure",
        "sentiment_narrative_flow",
        "quality_provenance",
    ]
    if analysis_depth in {"standard", "deep"}:
        required_domains.extend(
            [
                "fundamental_filing",
                "macro_cross_asset",
                "geopolitical_policy",
                "relative_context",
            ]
        )
    if analysis_depth == "deep":
        required_domains.append("intraday_microstructure")

    return {
        "job_id": request.get("job_id") or str(uuid.uuid4()),
        "trace_id": trace_id or str(uuid.uuid4()),
        "session_id": session_id,
        "symbol": str(request.get("symbol") or "").strip().upper(),
        "as_of_date": _iso_date(as_of_date),
        "horizon": str(request.get("horizon") or "").strip(),
        "risk_mode": str(request.get("risk_mode") or "").strip(),
        "scenario": scenario,
        "analysis_depth": analysis_depth,
        "refresh_mode": refresh_mode,
        "market_regime": market_regime,
        "requested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "required_domains": required_domains,
        "freshness": freshness or {},
    }


def _load_symbol_meta(symbol: str) -> Dict[str, Any]:
    row = _safe_fetchone(
        """
        SELECT symbol, exchange, country, currency, name, sector
        FROM market_symbols
        WHERE symbol = %s
        """,
        (symbol,),
    )
    if not row:
        return {"symbol": symbol}
    return {
        "symbol": row[0],
        "exchange": row[1],
        "country": row[2],
        "currency": row[3],
        "name": row[4],
        "sector": row[5],
    }


def _load_daily_bars(symbol: str, as_of_date: dt.date, lookback_days: int = 420) -> List[Dict[str, Any]]:
    start_date = as_of_date - dt.timedelta(days=lookback_days)
    rows = _safe_fetchall(
        """
        SELECT as_of_date, open, high, low, close, volume, source, ingested_at
        FROM market_bars_daily
        WHERE symbol = %s
          AND as_of_date >= %s
          AND as_of_date <= %s
        ORDER BY as_of_date ASC
        """,
        (symbol, start_date, as_of_date),
    )
    return [
        {
            "as_of_date": _iso_date(row[0]),
            "open": _safe_float(row[1]),
            "high": _safe_float(row[2]),
            "low": _safe_float(row[3]),
            "close": _safe_float(row[4]),
            "volume": _safe_int(row[5]),
            "source": row[6],
            "ingested_at": _iso_datetime(row[7]),
        }
        for row in rows
    ]


def _load_intraday_bars(symbol: str, as_of_date: dt.date) -> List[Dict[str, Any]]:
    start_ts = dt.datetime.combine(as_of_date - dt.timedelta(days=2), dt.time.min).replace(
        tzinfo=dt.timezone.utc
    )
    rows = _safe_fetchall(
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
    output = [
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
    output.reverse()
    return output


def _load_fundamentals(symbol: str, as_of_date: dt.date) -> List[Dict[str, Any]]:
    rows = _safe_fetchall(
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
    return [
        {
            "fiscal_period_end": _iso_date(row[0]),
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
    ]


def _load_sentiment_history(symbol: str, as_of_date: dt.date) -> List[Dict[str, Any]]:
    start_date = as_of_date - dt.timedelta(days=14)
    rows = _safe_fetchall(
        """
        SELECT as_of_date, headline_count, sentiment_mean, sentiment_pos, sentiment_neg,
               sentiment_neu, sentiment_score, source, computed_at
        FROM sentiment_daily
        WHERE symbol = %s
          AND as_of_date >= %s
          AND as_of_date <= %s
        ORDER BY as_of_date ASC
        """,
        (symbol, start_date, as_of_date),
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


def _load_recent_news(symbol: str, as_of_date: dt.date) -> List[Dict[str, Any]]:
    start_ts = dt.datetime.combine(as_of_date - dt.timedelta(days=14), dt.time.min).replace(
        tzinfo=dt.timezone.utc
    )
    end_ts = dt.datetime.combine(as_of_date, dt.time.max).replace(tzinfo=dt.timezone.utc)
    rows = _safe_fetchall(
        """
        SELECT published_at, source, title, url, content_snippet, ingested_at
        FROM news_raw
        WHERE symbol = %s
          AND published_at >= %s
          AND published_at <= %s
        ORDER BY published_at DESC
        LIMIT 25
        """,
        (symbol, start_ts, end_ts),
    )
    return [
        {
            "published_at": _iso_datetime(row[0]),
            "source": row[1],
            "title": row[2],
            "url": row[3],
            "content_snippet": row[4],
            "ingested_at": _iso_datetime(row[5]),
        }
        for row in rows
    ]


def _load_peer_rows(symbol: str, sector: Optional[str], as_of_date: dt.date) -> List[Dict[str, Any]]:
    if not sector:
        return []
    rows = _safe_fetchall(
        """
        SELECT ms.symbol, fd.ret_21d, fd.mom_vol_adj_21d, fd.regime_label, sd.action, sd.score
        FROM market_symbols ms
        JOIN features_daily fd
          ON ms.symbol = fd.symbol
        LEFT JOIN signals_daily sd
          ON fd.symbol = sd.symbol
         AND fd.as_of_date = sd.as_of_date
        WHERE ms.sector = %s
          AND ms.symbol <> %s
          AND fd.as_of_date = %s
        ORDER BY ABS(COALESCE(sd.score, 0)) DESC, ms.symbol ASC
        LIMIT 20
        """,
        (sector, symbol, as_of_date),
    )
    return [
        {
            "symbol": row[0],
            "ret_21d": _safe_float(row[1]),
            "mom_vol_adj_21d": _safe_float(row[2]),
            "regime_label": row[3],
            "action": row[4],
            "score": _safe_float(row[5]),
        }
        for row in rows
    ]


def _load_proxy_bars(proxies: Sequence[str], as_of_date: dt.date) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for proxy in proxies:
        bars = _load_daily_bars(proxy, as_of_date, lookback_days=90)
        if not bars:
            continue
        close_values = [row.get("close") for row in bars]
        output[proxy] = {
            "symbol": proxy,
            "ret_21d": _pct_change(close_values[-1], close_values[-22]) if len(close_values) >= 22 else None,
            "ret_63d": _pct_change(close_values[-1], close_values[-64]) if len(close_values) >= 64 else None,
            "vol_21d": _realized_vol(close_values, 21),
            "latest_close": close_values[-1],
        }
    return output


def _market_domain(
    symbol: str,
    as_of_date: dt.date,
    freshness: Dict[str, Any],
    key_features: Dict[str, Any],
    quality: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    daily_bars = _load_daily_bars(symbol, as_of_date)
    intraday_bars = _load_intraday_bars(symbol, as_of_date)

    close_values = [row.get("close") for row in daily_bars]
    open_values = [row.get("open") for row in daily_bars]
    high_values = [row.get("high") for row in daily_bars]
    low_values = [row.get("low") for row in daily_bars]
    volume_values = [row.get("volume") for row in daily_bars]

    latest_close = close_values[-1] if close_values else None
    prev_close = close_values[-2] if len(close_values) >= 2 else None
    latest_open = open_values[-1] if open_values else None
    latest_volume = volume_values[-1] if volume_values else None
    avg_volume_20 = _mean(volume_values[-20:])

    gap_pct = _pct_change(latest_open, prev_close)
    volume_anomaly = _ratio(latest_volume, avg_volume_20)
    realized_vol_5d = _realized_vol(close_values, 5)
    realized_vol_21d = _realized_vol(close_values, 21) or _safe_float(key_features.get("vol_21d"))
    realized_vol_63d = _realized_vol(close_values, 63) or _safe_float(key_features.get("vol_63d"))
    atr_pct = _safe_float(key_features.get("atr_pct"))
    ret_63d = _pct_change(latest_close, close_values[-64]) if len(close_values) >= 64 else None
    ret_126d = _pct_change(latest_close, close_values[-127]) if len(close_values) >= 127 else None
    ret_252d = _pct_change(latest_close, close_values[-253]) if len(close_values) >= 253 else None
    support_21d = min((value for value in low_values[-21:] if value is not None), default=None)
    resistance_21d = max((value for value in high_values[-21:] if value is not None), default=None)
    range_21d = None
    if support_21d is not None and resistance_21d is not None:
        range_21d = resistance_21d - support_21d
    compression_ratio = _ratio(range_21d, latest_close) if latest_close else None
    breakout_distance_63d = None
    if latest_close is not None and len(high_values) >= 63:
        trailing_high = max((value for value in high_values[-63:] if value is not None), default=None)
        if trailing_high is not None and trailing_high != 0:
            breakout_distance_63d = latest_close / trailing_high - 1.0

    latest_bar_ingested = daily_bars[-1].get("ingested_at") if daily_bars else freshness.get("bars_updated_at")
    intraday_ingested = intraday_bars[-1].get("ingested_at") if intraday_bars else None
    bars_staleness = _days_stale(_iso_date(as_of_date), latest_bar_ingested)
    intraday_staleness = _days_stale(_iso_date(as_of_date), intraday_ingested)

    domain = {
        "latest_close": latest_close,
        "previous_close": prev_close,
        "latest_open": latest_open,
        "day_return": _safe_float(key_features.get("ret_1d")),
        "ret_5d": _safe_float(key_features.get("ret_5d")),
        "ret_10d": _pct_change(latest_close, close_values[-11]) if len(close_values) >= 11 else None,
        "ret_21d": _safe_float(key_features.get("ret_21d")),
        "ret_63d": ret_63d,
        "ret_126d": ret_126d,
        "ret_252d": ret_252d,
        "realized_vol_5d": realized_vol_5d,
        "realized_vol_21d": realized_vol_21d,
        "realized_vol_63d": realized_vol_63d,
        "atr_pct": atr_pct,
        "gap_pct": gap_pct,
        "volume_anomaly": volume_anomaly,
        "support_21d": support_21d,
        "resistance_21d": resistance_21d,
        "compression_ratio": compression_ratio,
        "breakout_distance_63d": breakout_distance_63d,
        "recent_bars": daily_bars[-5:],
        "recent_intraday_bars": intraday_bars[-12:],
        "meta": {
            "sources": sorted(
                [
                    source
                    for source in {
                        *(row.get("source") for row in daily_bars if row.get("source")),
                        "signals_daily" if quality.get("bars_ok") is not None else None,
                    }
                    if source
                ]
            ),
            "row_count": len(daily_bars),
            "intraday_row_count": len(intraday_bars),
            "latest_ingested_at": latest_bar_ingested,
            "intraday_latest_ingested_at": intraday_ingested,
            "bars_status": _domain_status(bars_staleness, fresh_days=1, usable_days=5),
            "intraday_status": _domain_status(intraday_staleness, fresh_days=0, usable_days=2),
            "coverage_score": _clamp(_ratio(len(daily_bars), 252), 0.0, 1.0),
        },
    }
    return domain, daily_bars, intraday_bars


def _technical_domain(
    daily_bars: List[Dict[str, Any]],
    key_features: Dict[str, Any],
) -> Dict[str, Any]:
    close_values = [row.get("close") for row in daily_bars]
    volume_values = [row.get("volume") for row in daily_bars]
    latest_close = close_values[-1] if close_values else None
    ma_10 = _moving_average(close_values, 10)
    ma_21 = _moving_average(close_values, 21)
    ma_63 = _moving_average(close_values, 63)
    ma_126 = _moving_average(close_values, 126)
    slope_21 = _linear_slope(close_values[-21:]) or _safe_float(key_features.get("trend_slope_21d"))
    slope_63 = _linear_slope(close_values[-63:]) or _safe_float(key_features.get("trend_slope_63d"))
    trend_curvature = None
    if slope_21 is not None and slope_63 is not None:
        trend_curvature = slope_21 - slope_63
    mean_reversion_gap = None
    if latest_close is not None and ma_21 not in (None, 0):
        mean_reversion_gap = latest_close / ma_21 - 1.0
    breakout_state = "neutral"
    if latest_close is not None and ma_21 is not None and ma_63 is not None:
        if latest_close > ma_21 > ma_63:
            breakout_state = "trend_extension"
        elif latest_close < ma_21 < ma_63:
            breakout_state = "downtrend_extension"
        elif latest_close > ma_21 and ma_21 < ma_63:
            breakout_state = "transition_higher"
        elif latest_close < ma_21 and ma_21 > ma_63:
            breakout_state = "transition_lower"
    volume_price_alignment = None
    if len(close_values) >= 6 and len(volume_values) >= 6:
        short_return = _pct_change(close_values[-1], close_values[-6])
        volume_change = _ratio(volume_values[-1], _mean(volume_values[-6:-1]))
        if short_return is not None and volume_change is not None:
            volume_price_alignment = short_return * volume_change
    return {
        "moving_averages": {
            "ma_10": ma_10,
            "ma_21": ma_21,
            "ma_63": ma_63,
            "ma_126": ma_126,
        },
        "trend_slope_21d": slope_21,
        "trend_slope_63d": slope_63,
        "trend_curvature": trend_curvature,
        "mean_reversion_gap": mean_reversion_gap,
        "breakout_state": breakout_state,
        "volume_price_alignment": volume_price_alignment,
        "regime_label": key_features.get("regime_label"),
        "regime_strength": _safe_float(key_features.get("regime_strength")),
        "meta": {
            "coverage_score": _clamp(_ratio(len(close_values), 126), 0.0, 1.0),
        },
    }


def _fundamental_domain(
    fundamentals: List[Dict[str, Any]],
    as_of_date: dt.date,
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    latest = fundamentals[0] if fundamentals else {}
    prior_year = fundamentals[4] if len(fundamentals) >= 5 else {}
    revenue_growth_yoy = _pct_change(latest.get("revenue"), prior_year.get("revenue"))
    gross_margin_trend = None
    if latest.get("gross_margin") is not None and prior_year.get("gross_margin") is not None:
        gross_margin_trend = latest["gross_margin"] - prior_year["gross_margin"]
    op_margin_trend = None
    if latest.get("op_margin") is not None and prior_year.get("op_margin") is not None:
        op_margin_trend = latest["op_margin"] - prior_year["op_margin"]
    margin_stability = None
    gross_margin_series = [row.get("gross_margin") for row in fundamentals]
    if len([value for value in gross_margin_series if value is not None]) >= 3:
        sigma = _std(gross_margin_series)
        if sigma is not None:
            margin_stability = max(0.0, 1.0 - min(sigma, 0.25) / 0.25)
    positive_fcf_ratio = None
    fcf_values = [row.get("fcf") for row in fundamentals if row.get("fcf") is not None]
    if fcf_values:
        positive_fcf_ratio = sum(1 for value in fcf_values if value > 0) / len(fcf_values)
    filing_recency_days = None
    latest_report_date = latest.get("report_date") or latest.get("fiscal_period_end")
    if latest_report_date:
        try:
            filing_recency_days = (as_of_date - dt.date.fromisoformat(str(latest_report_date))).days
        except Exception:
            filing_recency_days = None

    coverage_score = len(fundamentals) / 4.0 if fundamentals else 0.0
    return {
        "latest_quarter": latest,
        "quarterly_series": fundamentals[:4],
        "revenue_growth_yoy": revenue_growth_yoy,
        "gross_margin_trend": gross_margin_trend,
        "op_margin_trend": op_margin_trend,
        "margin_stability": margin_stability,
        "positive_fcf_ratio": positive_fcf_ratio,
        "filing_recency_days": filing_recency_days,
        "fundamentals_ok": quality.get("fundamentals_ok"),
        "meta": {
            "coverage_score": _clamp(coverage_score, 0.0, 1.0),
            "latest_report_date": latest_report_date,
            "source": latest.get("source"),
            "latest_ingested_at": latest.get("ingested_at"),
            "status": "fresh" if filing_recency_days is not None and filing_recency_days <= 120 else "stale_but_usable" if filing_recency_days is not None and filing_recency_days <= 180 else "limited",
        },
    }


def _sentiment_domain(
    symbol: str,
    sentiment_history: List[Dict[str, Any]],
    recent_news: List[Dict[str, Any]],
    key_features: Dict[str, Any],
) -> Dict[str, Any]:
    latest_sentiment = sentiment_history[-1] if sentiment_history else {}
    titles = [row.get("title") or "" for row in recent_news]
    normalized_titles = [_normalize_title(title) for title in titles if title]
    unique_ratio = _ratio(len(set(normalized_titles)), len(normalized_titles))
    topics = _extract_topics(titles, symbol)
    top_topic_share = None
    if topics and titles:
        top_topic_share = topics[0]["count"] / len(titles)
    disagreement_score = None
    pos = latest_sentiment.get("sentiment_pos")
    neg = latest_sentiment.get("sentiment_neg")
    headline_count = latest_sentiment.get("headline_count") or len(titles)
    if headline_count:
        disagreement_score = (2 * min(pos or 0, neg or 0)) / headline_count

    headline_series = [row.get("headline_count") for row in sentiment_history]
    attention_crowding = None
    if headline_series:
        recent = headline_series[-1]
        baseline = _mean(headline_series[:-1] or headline_series)
        attention_crowding = _ratio(recent, baseline)

    sentiment_trend = None
    sentiment_series = [row.get("sentiment_score") for row in sentiment_history]
    if len([value for value in sentiment_series if value is not None]) >= 3:
        sentiment_trend = _linear_slope(sentiment_series[-5:])

    hype_price_divergence = None
    sentiment_score = _safe_float(key_features.get("sentiment_score"))
    ret_5d = _safe_float(key_features.get("ret_5d"))
    if sentiment_score is not None and ret_5d is not None:
        hype_price_divergence = sentiment_score - ret_5d

    latest_news_ingested = recent_news[0].get("ingested_at") if recent_news else None
    latest_sentiment_at = latest_sentiment.get("computed_at")
    return {
        "sentiment_score": sentiment_score,
        "sentiment_surprise": _safe_float(key_features.get("sentiment_surprise")),
        "sentiment_trend": sentiment_trend,
        "headline_count": headline_count,
        "attention_crowding": attention_crowding,
        "novelty_ratio": unique_ratio,
        "narrative_concentration": top_topic_share,
        "disagreement_score": disagreement_score,
        "hype_price_divergence": hype_price_divergence,
        "top_narratives": topics,
        "recent_headlines": recent_news[:8],
        "meta": {
            "news_status": _domain_status(_days_stale(sentiment_history[-1]["as_of_date"] if sentiment_history else None, latest_news_ingested), fresh_days=1, usable_days=5),
            "sentiment_status": _domain_status(_days_stale(sentiment_history[-1]["as_of_date"] if sentiment_history else None, latest_sentiment_at), fresh_days=1, usable_days=5),
            "latest_news_ingested_at": latest_news_ingested,
            "latest_sentiment_at": latest_sentiment_at,
            "coverage_score": _clamp(_ratio(len(recent_news), 12), 0.0, 1.0),
        },
    }


def _macro_cross_asset_domain(
    symbol_meta: Dict[str, Any],
    as_of_date: dt.date,
    job_context: Dict[str, Any],
    market_domain: Dict[str, Any],
) -> Dict[str, Any]:
    sector = (symbol_meta.get("sector") or "").strip().lower()
    preferred_proxy = _SECTOR_PROXY_MAP.get(sector)
    proxy_universe = [item for item in [preferred_proxy, "SPY", "QQQ", "IWM", "TLT", "GLD"] if item]
    proxies = _load_proxy_bars(proxy_universe, as_of_date)
    market_regime = job_context.get("market_regime") or "auto"
    benchmark = proxies.get(preferred_proxy) if preferred_proxy else None
    if benchmark is None:
        benchmark = proxies.get("SPY") or proxies.get("QQQ") or next(iter(proxies.values()), None)

    alignment = None
    symbol_ret_21d = market_domain.get("ret_21d")
    benchmark_ret_21d = benchmark.get("ret_21d") if benchmark else None
    if symbol_ret_21d is not None and benchmark_ret_21d is not None:
        alignment = 1.0 - min(abs(symbol_ret_21d - benchmark_ret_21d), 0.4) / 0.4

    risk_on_score = None
    if proxies:
        equity_returns = [proxies[key].get("ret_21d") for key in ("SPY", "QQQ", "IWM") if key in proxies]
        risk_on_score = _mean(equity_returns)

    stress_overlay = None
    if "TLT" in proxies and "SPY" in proxies:
        stress_overlay = (proxies["TLT"].get("ret_21d") or 0.0) - (proxies["SPY"].get("ret_21d") or 0.0)

    inferred_regime = market_regime
    if market_regime == "auto":
        if risk_on_score is not None and risk_on_score > 0.04:
            inferred_regime = "risk_on"
        elif risk_on_score is not None and risk_on_score < -0.04:
            inferred_regime = "risk_off"
        else:
            inferred_regime = "neutral"

    return {
        "requested_market_regime": market_regime,
        "inferred_market_regime": inferred_regime,
        "benchmark_proxy": benchmark.get("symbol") if benchmark else None,
        "benchmark_ret_21d": benchmark_ret_21d,
        "benchmark_vol_21d": benchmark.get("vol_21d") if benchmark else None,
        "risk_on_score": risk_on_score,
        "stress_overlay": stress_overlay,
        "macro_alignment_score": _score_100(alignment, low=0.0, high=1.0),
        "proxy_snapshot": proxies,
        "meta": {
            "coverage_score": _clamp(_ratio(len(proxies), len(proxy_universe)), 0.0, 1.0),
            "status": "fresh" if proxies else "limited",
            "relevance_note": (
                f"Cross-asset context is anchored to {benchmark.get('symbol')}." if benchmark else "Cross-asset benchmark coverage is limited."
            ),
        },
    }


def _geopolitical_domain(recent_news: List[Dict[str, Any]]) -> Dict[str, Any]:
    category_counts = {key: 0 for key in _GEOPOLITICAL_KEYWORDS}
    matched_titles: List[Dict[str, Any]] = []
    for row in recent_news:
        title = (row.get("title") or "").lower()
        title_matches: List[str] = []
        for category, keywords in _GEOPOLITICAL_KEYWORDS.items():
            if any(keyword in title for keyword in keywords):
                category_counts[category] += 1
                title_matches.append(category)
        if title_matches:
            matched_titles.append(
                {
                    "title": row.get("title"),
                    "published_at": row.get("published_at"),
                    "matches": title_matches,
                }
            )
    headline_total = len(recent_news) or 1
    weighted_hits = (
        1.0 * category_counts["rates_policy"]
        + 1.2 * category_counts["regulation_policy"]
        + 1.5 * category_counts["trade_geopolitics"]
        + 1.0 * category_counts["election_policy"]
        + 0.8 * category_counts["commodity_supply"]
    )
    exogenous_event_score = min(weighted_hits / headline_total, 1.0)
    return {
        "category_counts": category_counts,
        "exogenous_event_score": exogenous_event_score,
        "relevant_headlines": matched_titles[:6],
        "policy_sensitive": weighted_hits > 0,
        "meta": {
            "coverage_score": _clamp(_ratio(len(recent_news), 10), 0.0, 1.0),
            "status": "fresh" if recent_news else "limited",
        },
    }


def _relative_context_domain(
    symbol: str,
    as_of_date: dt.date,
    symbol_meta: Dict[str, Any],
    market_domain: Dict[str, Any],
    key_features: Dict[str, Any],
) -> Dict[str, Any]:
    sector = symbol_meta.get("sector")
    peers = _load_peer_rows(symbol, sector, as_of_date)
    ret_21d = _safe_float(key_features.get("ret_21d"))
    mom_21d = _safe_float(key_features.get("mom_vol_adj_21d"))
    sector_ret_median = _median(row.get("ret_21d") for row in peers)
    sector_mom_median = _median(row.get("mom_vol_adj_21d") for row in peers)
    rel_ret = ret_21d - sector_ret_median if ret_21d is not None and sector_ret_median is not None else None
    rel_mom = mom_21d - sector_mom_median if mom_21d is not None and sector_mom_median is not None else None
    percentile = None
    peer_returns = [row.get("ret_21d") for row in peers if row.get("ret_21d") is not None]
    if ret_21d is not None and peer_returns:
        wins = sum(1 for value in peer_returns if value <= ret_21d)
        percentile = wins / len(peer_returns)
    peer_dispersion = _std(peer_returns)
    return {
        "sector": sector,
        "peer_count": len(peers),
        "sector_median_ret_21d": sector_ret_median,
        "sector_median_mom_vol_adj_21d": sector_mom_median,
        "relative_ret_21d": rel_ret,
        "relative_momentum": rel_mom,
        "relative_strength_percentile": percentile,
        "peer_dispersion_score": peer_dispersion,
        "peer_snapshot": peers[:8],
        "meta": {
            "coverage_score": _clamp(_ratio(len(peers), 8), 0.0, 1.0),
            "status": "fresh" if peers else "limited",
        },
    }


def _quality_provenance_domain(
    freshness: Dict[str, Any],
    quality: Dict[str, Any],
    data_domains: Dict[str, Any],
    as_of_date: dt.date,
) -> Dict[str, Any]:
    freshness_summary = {
        "bars": {
            "updated_at": freshness.get("bars_updated_at"),
            "status": _domain_status(_days_stale(_iso_date(as_of_date), freshness.get("bars_updated_at")), fresh_days=1, usable_days=5),
        },
        "news": {
            "updated_at": freshness.get("news_updated_at"),
            "status": _domain_status(_days_stale(_iso_date(as_of_date), freshness.get("news_updated_at")), fresh_days=1, usable_days=7),
        },
        "sentiment": {
            "updated_at": freshness.get("sentiment_updated_at"),
            "status": _domain_status(_days_stale(_iso_date(as_of_date), freshness.get("sentiment_updated_at")), fresh_days=1, usable_days=5),
        },
    }
    domain_confidence = {
        "market": 0.95 if quality.get("bars_ok") else 0.55,
        "fundamentals": 0.9 if quality.get("fundamentals_ok") else 0.35,
        "sentiment": 0.85 if quality.get("sentiment_ok") else 0.4,
        "intraday": 0.8 if quality.get("intraday_ok") else 0.25,
        "macro": data_domains.get("macro_cross_asset", {}).get("meta", {}).get("coverage_score"),
        "relative": data_domains.get("relative_context", {}).get("meta", {}).get("coverage_score"),
    }
    warnings = list(quality.get("warnings") or [])
    if quality.get("missingness") not in (None, ""):
        missingness = _safe_float(quality.get("missingness"))
        if missingness is not None and missingness > 0.15:
            warnings.append("High data missingness is degrading conviction.")
    return {
        "quality_score": _safe_float(quality.get("quality_score")),
        "missingness": _safe_float(quality.get("missingness")),
        "anomaly_flags": quality.get("anomaly_flags") or [],
        "coverage_flags": {
            "bars_ok": quality.get("bars_ok"),
            "fundamentals_ok": quality.get("fundamentals_ok"),
            "sentiment_ok": quality.get("sentiment_ok"),
            "intraday_ok": quality.get("intraday_ok"),
            "news_ok": quality.get("news_ok"),
        },
        "freshness_summary": freshness_summary,
        "domain_confidence": domain_confidence,
        "warnings": warnings,
        "meta": {
            "status": "fresh" if not warnings else "mixed",
        },
    }


def build_normalized_data_bundle(
    *,
    job_context: Dict[str, Any],
    freshness: Dict[str, Any],
    signal: Dict[str, Any],
    key_features: Dict[str, Any],
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    symbol = str(job_context.get("symbol") or "").upper()
    as_of_date = dt.date.fromisoformat(str(job_context.get("as_of_date")))
    symbol_meta = _load_symbol_meta(symbol)
    market_domain, daily_bars, intraday_bars = _market_domain(
        symbol, as_of_date, freshness, key_features, quality
    )
    technical_domain = _technical_domain(daily_bars, key_features)
    fundamentals = _load_fundamentals(symbol, as_of_date)
    fundamental_domain = _fundamental_domain(fundamentals, as_of_date, quality)
    sentiment_history = _load_sentiment_history(symbol, as_of_date)
    recent_news = _load_recent_news(symbol, as_of_date)
    sentiment_domain = _sentiment_domain(symbol, sentiment_history, recent_news, key_features)
    macro_domain = _macro_cross_asset_domain(symbol_meta, as_of_date, job_context, market_domain)
    geopolitical_domain = _geopolitical_domain(recent_news)
    relative_domain = _relative_context_domain(
        symbol, as_of_date, symbol_meta, market_domain, key_features
    )
    quality_domain = _quality_provenance_domain(
        freshness,
        quality,
        {
            "macro_cross_asset": macro_domain,
            "relative_context": relative_domain,
        },
        as_of_date,
    )

    data_bundle = {
        "symbol_meta": symbol_meta,
        "market_price_volume": market_domain,
        "technical_market_structure": technical_domain,
        "fundamental_filing": fundamental_domain,
        "sentiment_narrative_flow": sentiment_domain,
        "macro_cross_asset": macro_domain,
        "geopolitical_policy": geopolitical_domain,
        "relative_context": relative_domain,
        "quality_provenance": quality_domain,
        "raw_supporting_fields": {
            "signal": signal,
            "key_features": key_features,
            "quality": quality,
            "recent_news_headlines": [row.get("title") for row in recent_news[:8]],
            "sentiment_history": sentiment_history[-5:],
            "recent_daily_bars": daily_bars[-10:],
            "recent_intraday_bars": intraday_bars[-12:],
            "fundamental_quarters": fundamentals[:4],
        },
    }
    overlay = data_fabric.enrich_data_bundle(
        job_context=job_context,
        symbol_meta=symbol_meta,
        data_bundle=data_bundle,
    )
    merged_bundle = data_fabric.merge_into_data_bundle(
        data_bundle=data_bundle,
        overlay=overlay,
    )
    merged_bundle["raw_supporting_fields"]["external_data_fabric"] = overlay
    return merged_bundle


def build_feature_factor_bundle(
    *,
    data_bundle: Dict[str, Any],
    signal: Dict[str, Any],
    key_features: Dict[str, Any],
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    market = data_bundle.get("market_price_volume") or {}
    technical = data_bundle.get("technical_market_structure") or {}
    fundamentals = data_bundle.get("fundamental_filing") or {}
    sentiment = data_bundle.get("sentiment_narrative_flow") or {}
    macro = data_bundle.get("macro_cross_asset") or {}
    geopolitical = data_bundle.get("geopolitical_policy") or {}
    relative = data_bundle.get("relative_context") or {}
    quality_domain = data_bundle.get("quality_provenance") or {}
    fundamental_metrics = fundamentals.get("normalized_metrics") or {}
    fundamental_quality = fundamentals.get("quality_proxies") or {}
    latest_quarter = fundamentals.get("latest_quarter") or {}

    price_momentum = {
        "ret_1d": _safe_float(key_features.get("ret_1d")),
        "ret_5d": _safe_float(key_features.get("ret_5d")),
        "ret_10d": market.get("ret_10d"),
        "ret_21d": _safe_float(key_features.get("ret_21d")),
        "ret_63d": market.get("ret_63d"),
        "ret_126d": market.get("ret_126d"),
        "ret_252d": market.get("ret_252d"),
        "trend_slope_21d": technical.get("trend_slope_21d"),
        "trend_slope_63d": technical.get("trend_slope_63d"),
        "trend_curvature": technical.get("trend_curvature"),
        "momentum_consistency_score": _score_100(
            _mean(
                [
                    1.0 if (value or 0.0) > 0 else 0.0
                    for value in [
                        _safe_float(key_features.get("ret_1d")),
                        _safe_float(key_features.get("ret_5d")),
                        market.get("ret_10d"),
                        _safe_float(key_features.get("ret_21d")),
                        market.get("ret_63d"),
                    ]
                    if value is not None
                ]
            ),
            low=0.0,
            high=1.0,
        ),
        "breakout_follow_through_score": _score_100(
            _ratio(
                (market.get("breakout_distance_63d") or 0.0) + max(technical.get("mean_reversion_gap") or 0.0, 0.0),
                max(market.get("atr_pct") or 0.05, 0.01),
            ),
            low=-2.0,
            high=2.0,
        ),
        "exhaustion_score": _score_100(
            (market.get("ret_5d") or 0.0) - (market.get("ret_21d") or 0.0) - (market.get("atr_pct") or 0.0),
            low=-0.2,
            high=0.2,
        ),
    }

    vol_risk = {
        "realized_vol_5d": market.get("realized_vol_5d"),
        "realized_vol_21d": market.get("realized_vol_21d"),
        "realized_vol_63d": market.get("realized_vol_63d"),
        "vol_of_vol": _ratio(
            (market.get("realized_vol_5d") or 0.0) - (market.get("realized_vol_21d") or 0.0),
            max(market.get("realized_vol_21d") or 0.01, 0.01),
        ),
        "atr_normalization": _ratio(market.get("atr_pct"), market.get("realized_vol_21d")),
        "gap_instability": _score_100(abs(market.get("gap_pct") or 0.0), low=0.0, high=0.08),
        "downside_asymmetry": _score_100(
            abs(min(market.get("ret_21d") or 0.0, 0.0)) - max(market.get("ret_21d") or 0.0, 0.0),
            low=-0.2,
            high=0.2,
        ),
        "stress_reactivity": _score_100(
            (macro.get("stress_overlay") or 0.0) + (geopolitical.get("exogenous_event_score") or 0.0),
            low=-0.1,
            high=0.25,
        ),
    }

    regime_label = technical.get("regime_label") or "unknown"
    if (market.get("realized_vol_21d") or 0.0) >= 0.45:
        regime_label = "high_vol"
    elif (market.get("compression_ratio") or 1.0) <= 0.05:
        regime_label = "squeeze"
    elif technical.get("breakout_state") in {"transition_higher", "transition_lower"}:
        regime_label = "transition"
    elif technical.get("breakout_state") in {"trend_extension", "downtrend_extension"}:
        regime_label = "trend"
    else:
        regime_label = "chop"

    regime_engine = {
        "regime_label": regime_label,
        "regime_confidence": _score_100(
            abs(technical.get("trend_slope_63d") or 0.0) + abs(technical.get("trend_curvature") or 0.0),
            low=0.0,
            high=0.5,
        ),
        "regime_instability": _score_100(
            (abs(technical.get("trend_curvature") or 0.0) * 10.0)
            + ((vol_risk.get("gap_instability") or 50.0) / 100.0),
            low=0.0,
            high=2.0,
        ),
        "regime_transition_probability": _score_100(
            abs(technical.get("trend_curvature") or 0.0) + abs((market.get("gap_pct") or 0.0) * 3.0),
            low=0.0,
            high=0.5,
        ),
    }

    sentiment_intelligence = {
        "sentiment_level": sentiment.get("sentiment_score"),
        "sentiment_trend": sentiment.get("sentiment_trend"),
        "novelty_ratio": sentiment.get("novelty_ratio"),
        "narrative_concentration": sentiment.get("narrative_concentration"),
        "attention_crowding": sentiment.get("attention_crowding"),
        "disagreement_score": sentiment.get("disagreement_score"),
        "hype_to_price_divergence": sentiment.get("hype_price_divergence"),
    }

    fundamental_intelligence = {
        "growth_quality": _first_available(
            _score_100(
                _first_available(
                    fundamental_metrics.get("revenue_growth_yoy"),
                    fundamentals.get("revenue_growth_yoy"),
                ),
                low=-0.3,
                high=0.4,
            ),
            fundamental_quality.get("business_quality_durability"),
        ),
        "profitability_quality": _first_available(
            fundamental_quality.get("profitability_strength"),
            _score_100(
                _mean(
                    [
                        fundamental_metrics.get("operating_margin"),
                        fundamental_metrics.get("net_margin"),
                        latest_quarter.get("op_margin"),
                    ]
                ),
                low=-0.1,
                high=0.4,
            ),
        ),
        "balance_sheet_resilience": _first_available(
            fundamental_quality.get("balance_sheet_resilience"),
            _score_100(
                _mean(
                    [
                        fundamental_metrics.get("current_ratio"),
                        1.0 - min((fundamental_metrics.get("debt_to_equity") or 3.0), 3.0) / 3.0,
                    ]
                ),
                low=0.0,
                high=2.0,
            ),
            quality.get("fundamentals_ok") is True and 70.0 or 35.0,
        ),
        "cash_flow_durability": _first_available(
            fundamental_quality.get("cash_flow_durability"),
            _score_100(
                _first_available(
                    fundamental_metrics.get("positive_fcf_ratio"),
                    fundamentals.get("positive_fcf_ratio"),
                ),
                low=0.0,
                high=1.0,
            ),
        ),
        "accounting_quality": _first_available(
            fundamental_quality.get("reporting_quality_proxy"),
            _score_100(
                _first_available(
                    fundamentals.get("margin_stability"),
                    fundamental_metrics.get("operating_margin_stability"),
                    fundamental_metrics.get("gross_margin_stability"),
                ),
                low=0.0,
                high=100.0,
            ),
        ),
        "capital_efficiency_proxy": _score_100(
            _mean(
                [
                    fundamental_metrics.get("gross_margin"),
                    fundamental_metrics.get("return_on_equity"),
                ]
            ),
            low=0.0,
            high=0.8,
        ),
        "filing_recency_drift": _first_available(
            fundamental_quality.get("filing_recency_score"),
            _score_100(
                1.0 - min((fundamentals.get("filing_recency_days") or 365), 365) / 365.0,
                low=0.0,
                high=1.0,
            ),
        ),
    }

    macro_sensitivity = {
        "macro_alignment_score": macro.get("macro_alignment_score"),
        "macro_stress_fragility": _score_100(
            (macro.get("stress_overlay") or 0.0) + (geopolitical.get("exogenous_event_score") or 0.0),
            low=-0.05,
            high=0.25,
        ),
        "risk_on_score": _score_100(macro.get("risk_on_score"), low=-0.08, high=0.08),
        "geopolitical_stress_score": _score_100(geopolitical.get("exogenous_event_score"), low=0.0, high=1.0),
    }

    relative_peer = {
        "sector_relative_momentum": _score_100(relative.get("relative_momentum"), low=-0.4, high=0.4),
        "market_relative_behavior": _score_100(relative.get("relative_ret_21d"), low=-0.3, high=0.3),
        "relative_strength_percentile": _score_100(relative.get("relative_strength_percentile"), low=0.0, high=1.0),
        "dispersion_score": _score_100(relative.get("peer_dispersion_score"), low=0.0, high=0.25),
        "peer_divergence_score": _score_100(abs(relative.get("relative_ret_21d") or 0.0), low=0.0, high=0.25),
    }

    market_structure_integrity = _mean(
        [
            price_momentum.get("momentum_consistency_score"),
            _score_100(technical.get("volume_price_alignment"), low=-0.2, high=0.2),
            100.0 - (vol_risk.get("gap_instability") or 50.0),
        ]
    )
    narrative_crowding_index = _mean(
        [
            _score_100(sentiment.get("attention_crowding"), low=0.5, high=3.0),
            _score_100(sentiment.get("narrative_concentration"), low=0.1, high=0.8),
            _score_100(abs(sentiment.get("hype_price_divergence") or 0.0), low=0.0, high=0.4),
        ]
    )
    regime_stability_score = _mean(
        [
            regime_engine.get("regime_confidence"),
            100.0 - (regime_engine.get("regime_instability") or 50.0),
        ]
    )
    fundamental_durability_score = _mean(
        [
            fundamental_intelligence.get("growth_quality"),
            fundamental_intelligence.get("profitability_quality"),
            fundamental_intelligence.get("cash_flow_durability"),
            fundamental_intelligence.get("accounting_quality"),
            fundamental_intelligence.get("filing_recency_drift"),
        ]
    )
    cross_domain_conviction = _mean(
        [
            _score_100(_safe_float(signal.get("confidence")), low=0.0, high=1.0),
            market_structure_integrity,
            regime_stability_score,
            macro_sensitivity.get("macro_alignment_score"),
            relative_peer.get("relative_strength_percentile"),
        ]
    )
    signal_fragility_index = _mean(
        [
            vol_risk.get("gap_instability"),
            macro_sensitivity.get("macro_stress_fragility"),
            narrative_crowding_index,
            _score_100(_safe_float(quality_domain.get("missingness")), low=0.0, high=0.25),
        ]
    )
    opportunity_quality = _mean(
        [
            price_momentum.get("momentum_consistency_score"),
            fundamental_durability_score,
            cross_domain_conviction,
            100.0 - (signal_fragility_index or 50.0),
        ]
    )

    composite_intelligence = {
        "Market Structure Integrity Score": market_structure_integrity,
        "Narrative Crowding Index": narrative_crowding_index,
        "Regime Stability Score": regime_stability_score,
        "Macro Alignment Score": macro_sensitivity.get("macro_alignment_score"),
        "Fundamental Durability Score": fundamental_durability_score,
        "Cross-Domain Conviction Score": cross_domain_conviction,
        "Signal Fragility Index": signal_fragility_index,
        "Opportunity Quality Score": opportunity_quality,
    }

    return {
        "multi_horizon_price_momentum": price_momentum,
        "volatility_risk_microstructure": vol_risk,
        "regime_engine": regime_engine,
        "sentiment_intelligence": sentiment_intelligence,
        "fundamental_intelligence": fundamental_intelligence,
        "macro_sensitivity": macro_sensitivity,
        "relative_peer": relative_peer,
        "composite_intelligence": composite_intelligence,
    }
