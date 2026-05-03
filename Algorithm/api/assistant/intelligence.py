from __future__ import annotations

import datetime as dt
import math
import re
import statistics
import uuid
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from api import db
from api.assistant.coverage import (
    RETURN_HORIZON_REQUIREMENTS,
    TECHNICAL_HORIZON_REQUIREMENTS,
    availability_payload,
    classify_horizon_coverage,
)
from api.assistant import data_fabric
from api.assistant.phase3 import build_feature_factor_bundle as build_phase3_feature_factor_bundle


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


def _max_drawdown(values: Sequence[Optional[float]]) -> Optional[float]:
    clean: List[float] = []
    for value in values:
        number = _safe_float(value)
        if number is not None:
            clean.append(number)
    if len(clean) < 2:
        return None
    peak = clean[0]
    worst_drawdown = 0.0
    for value in clean:
        peak = max(peak, value)
        if peak == 0:
            continue
        worst_drawdown = min(worst_drawdown, (value - peak) / peak)
    return float(worst_drawdown)


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


def _atr_pct_from_bars(rows: Sequence[Dict[str, Any]], window: int = 14) -> Optional[float]:
    if len(rows) < max(window, 2):
        return None
    true_ranges: List[float] = []
    for index in range(max(len(rows) - window, 1), len(rows)):
        row = rows[index]
        prev_close = rows[index - 1].get("close") if index > 0 else None
        high = row.get("high")
        low = row.get("low")
        if high is None or low is None:
            continue
        candidates = [high - low]
        if prev_close is not None:
            candidates.extend([abs(high - prev_close), abs(low - prev_close)])
        true_ranges.append(max(candidates))
    atr = _mean(true_ranges)
    latest_close = rows[-1].get("close")
    if atr is None or latest_close in (None, 0):
        return None
    return atr / latest_close


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

    day_return = _first_available(_pct_change(latest_close, prev_close), _safe_float(key_features.get("ret_1d")))
    ret_5d = _first_available(
        _pct_change(latest_close, close_values[-6]) if len(close_values) >= 6 else None,
        _safe_float(key_features.get("ret_5d")),
    )
    ret_21d = _first_available(
        _pct_change(latest_close, close_values[-22]) if len(close_values) >= 22 else None,
        _safe_float(key_features.get("ret_21d")),
    )
    gap_pct = _pct_change(latest_open, prev_close)
    volume_anomaly = _ratio(latest_volume, avg_volume_20)
    realized_vol_5d = _realized_vol(close_values, 5)
    realized_vol_10d = _realized_vol(close_values, 10)
    realized_vol_21d = _realized_vol(close_values, 21) or _safe_float(key_features.get("vol_21d"))
    realized_vol_63d = _realized_vol(close_values, 63) or _safe_float(key_features.get("vol_63d"))
    realized_vol_126d = _realized_vol(close_values, 126)
    realized_vol_252d = _realized_vol(close_values, 252)
    atr_pct = _first_available(_safe_float(key_features.get("atr_pct")), _atr_pct_from_bars(daily_bars))
    ret_3d = _pct_change(latest_close, close_values[-4]) if len(close_values) >= 4 else None
    ret_63d = _pct_change(latest_close, close_values[-64]) if len(close_values) >= 64 else None
    ret_126d = _pct_change(latest_close, close_values[-127]) if len(close_values) >= 127 else None
    ret_252d = _pct_change(latest_close, close_values[-253]) if len(close_values) >= 253 else None
    returns = _to_returns(close_values)
    positive_day_ratio_10d = (
        sum(1 for value in returns[-10:] if value > 0) / 10.0 if len(returns) >= 10 else None
    )
    positive_day_ratio_21d = (
        sum(1 for value in returns[-21:] if value > 0) / 21.0 if len(returns) >= 21 else None
    )
    positive_day_ratio_63d = (
        sum(1 for value in returns[-63:] if value > 0) / 63.0 if len(returns) >= 63 else None
    )
    return_dispersion_10d = _std(returns[-10:]) if len(returns) >= 10 else None
    return_dispersion_21d = _std(returns[-21:]) if len(returns) >= 21 else None
    return_dispersion_63d = _std(returns[-63:]) if len(returns) >= 63 else None
    downside_21d = [value for value in returns[-21:] if value < 0] if len(returns) >= 21 else []
    upside_21d = [value for value in returns[-21:] if value > 0] if len(returns) >= 21 else []
    downside_63d = [value for value in returns[-63:] if value < 0] if len(returns) >= 63 else []
    upside_63d = [value for value in returns[-63:] if value > 0] if len(returns) >= 63 else []
    downside_vol_21d = _std(downside_21d) if len(downside_21d) >= 2 else None
    upside_vol_21d = _std(upside_21d) if len(upside_21d) >= 2 else None
    downside_vol_63d = _std(downside_63d) if len(downside_63d) >= 2 else None
    upside_vol_63d = _std(upside_63d) if len(upside_63d) >= 2 else None
    downside_asymmetry_21d = _ratio(downside_vol_21d, upside_vol_21d)
    downside_asymmetry_63d = _ratio(downside_vol_63d, upside_vol_63d)
    maxdd_21d = _max_drawdown(close_values[-21:]) if len(close_values) >= 21 else None
    maxdd_126d = _max_drawdown(close_values[-126:]) if len(close_values) >= 126 else None
    support_window_days = min(len(low_values), 21) if len(low_values) >= 5 else 0
    breakout_window_days = min(len(high_values), 63) if len(high_values) >= 21 else 0
    support_21d = (
        min((value for value in low_values[-support_window_days:] if value is not None), default=None)
        if support_window_days
        else None
    )
    resistance_21d = (
        max((value for value in high_values[-support_window_days:] if value is not None), default=None)
        if support_window_days
        else None
    )
    range_21d = None
    if support_21d is not None and resistance_21d is not None:
        range_21d = resistance_21d - support_21d
    compression_ratio = _ratio(range_21d, latest_close) if latest_close else None
    range_position_21d = None
    if range_21d not in (None, 0) and latest_close is not None and support_21d is not None:
        range_position_21d = (latest_close - support_21d) / range_21d
    range_expansion_ratio = None
    if realized_vol_5d is not None and realized_vol_21d not in (None, 0):
        range_expansion_ratio = realized_vol_5d / realized_vol_21d
    breakout_distance_63d = None
    if latest_close is not None and breakout_window_days:
        trailing_high = max(
            (value for value in high_values[-breakout_window_days:] if value is not None),
            default=None,
        )
        if trailing_high is not None and trailing_high != 0:
            breakout_distance_63d = latest_close / trailing_high - 1.0
    abs_gap_mean_10d = None
    gap_instability_10d = None
    if len(open_values) >= 11 and len(close_values) >= 11:
        gaps = []
        for idx in range(len(open_values) - 10, len(open_values)):
            open_value = open_values[idx]
            prior_close = close_values[idx - 1] if idx > 0 else None
            gap_value = _pct_change(open_value, prior_close)
            if gap_value is not None:
                gaps.append(abs(gap_value))
        abs_gap_mean_10d = _mean(gaps)
        gap_instability_10d = _ratio(abs_gap_mean_10d, atr_pct)
    up_down_volume_ratio_21d = None
    if len(returns) >= 21 and len(volume_values) >= 22:
        up_volume = 0.0
        down_volume = 0.0
        for offset, ret_value in enumerate(returns[-21:], start=len(volume_values) - 21):
            volume_value = volume_values[offset]
            if volume_value is None:
                continue
            if ret_value > 0:
                up_volume += float(volume_value)
            elif ret_value < 0:
                down_volume += float(volume_value)
        up_down_volume_ratio_21d = _ratio(up_volume, down_volume or 1.0)
    vol_of_vol_proxy = None
    if realized_vol_5d is not None and realized_vol_21d not in (None, 0):
        vol_of_vol_proxy = abs(realized_vol_5d - realized_vol_21d) / realized_vol_21d

    latest_bar_ingested = daily_bars[-1].get("ingested_at") if daily_bars else freshness.get("bars_updated_at")
    intraday_ingested = intraday_bars[-1].get("ingested_at") if intraday_bars else None
    bars_staleness = _days_stale(_iso_date(as_of_date), latest_bar_ingested)
    intraday_staleness = _days_stale(_iso_date(as_of_date), intraday_ingested)
    horizon_info = classify_horizon_coverage(len(close_values), requirements=RETURN_HORIZON_REQUIREMENTS)
    fallback_sources = []
    if day_return is not None and _pct_change(latest_close, prev_close) is None and key_features.get("ret_1d") is not None:
        fallback_sources.append("features_daily")
    if ret_5d is not None and not (len(close_values) >= 6):
        fallback_sources.append("features_daily")
    if ret_21d is not None and not (len(close_values) >= 22):
        fallback_sources.append("features_daily")
    if realized_vol_21d is not None and _realized_vol(close_values, 21) is None and key_features.get("vol_21d") is not None:
        fallback_sources.append("features_daily")
    if realized_vol_63d is not None and _realized_vol(close_values, 63) is None and key_features.get("vol_63d") is not None:
        fallback_sources.append("features_daily")
    if atr_pct is not None and _safe_float(key_features.get("atr_pct")) is not None:
        fallback_sources.append("features_daily")
    fallback_sources = sorted(set(fallback_sources))
    market_note = (
        f"Return context is populated through {horizon_info['available_through']}; longer-horizon metrics remain constrained by usable history."
        if horizon_info["available_horizons"] and horizon_info["missing_horizons"]
        else "Daily bar coverage is currently too thin for stable return-horizon analysis."
        if not daily_bars
        else "Daily bar coverage supports the standard return stack."
    )

    domain = {
        "latest_close": latest_close,
        "previous_close": prev_close,
        "latest_open": latest_open,
        "day_return": day_return,
        "ret_3d": ret_3d,
        "ret_5d": ret_5d,
        "ret_10d": _pct_change(latest_close, close_values[-11]) if len(close_values) >= 11 else None,
        "ret_21d": ret_21d,
        "ret_63d": ret_63d,
        "ret_126d": ret_126d,
        "ret_252d": ret_252d,
        "realized_vol_5d": realized_vol_5d,
        "realized_vol_10d": realized_vol_10d,
        "realized_vol_21d": realized_vol_21d,
        "realized_vol_63d": realized_vol_63d,
        "realized_vol_126d": realized_vol_126d,
        "realized_vol_252d": realized_vol_252d,
        "atr_pct": atr_pct,
        "gap_pct": gap_pct,
        "volume_anomaly": volume_anomaly,
        "positive_day_ratio_10d": positive_day_ratio_10d,
        "positive_day_ratio_21d": positive_day_ratio_21d,
        "positive_day_ratio_63d": positive_day_ratio_63d,
        "return_dispersion_10d": return_dispersion_10d,
        "return_dispersion_21d": return_dispersion_21d,
        "return_dispersion_63d": return_dispersion_63d,
        "downside_vol_21d": downside_vol_21d,
        "upside_vol_21d": upside_vol_21d,
        "downside_vol_63d": downside_vol_63d,
        "upside_vol_63d": upside_vol_63d,
        "downside_asymmetry_21d": downside_asymmetry_21d,
        "downside_asymmetry_63d": downside_asymmetry_63d,
        "maxdd_21d": maxdd_21d,
        "maxdd_63d": _max_drawdown(close_values[-63:]) if len(close_values) >= 63 else None,
        "maxdd_126d": maxdd_126d,
        "gap_instability_10d": gap_instability_10d,
        "abs_gap_mean_10d": abs_gap_mean_10d,
        "up_down_volume_ratio_21d": up_down_volume_ratio_21d,
        "vol_of_vol_proxy": vol_of_vol_proxy,
        "support_21d": support_21d,
        "resistance_21d": resistance_21d,
        "support_window_days": support_window_days or None,
        "compression_ratio": compression_ratio,
        "range_position_21d": range_position_21d,
        "range_expansion_ratio": range_expansion_ratio,
        "breakout_distance_63d": breakout_distance_63d,
        "breakout_window_days": breakout_window_days or None,
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
            **availability_payload(
                has_data=bool(daily_bars),
                coverage_score=_clamp(_ratio(len(daily_bars), 252), 0.0, 1.0),
                freshness_status=_domain_status(bars_staleness, fresh_days=1, usable_days=5),
                available_horizons=horizon_info["available_horizons"],
                missing_horizons=horizon_info["missing_horizons"],
                missing_reason="unavailable" if not daily_bars else horizon_info["missing_reason"],
                fallback_used=bool(fallback_sources),
                fallback_source=fallback_sources,
                data_quality_note=market_note,
            ),
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
    ma_stack_alignment = None
    if latest_close is not None and ma_10 is not None and ma_21 is not None and ma_63 is not None:
        if latest_close > ma_10 > ma_21 > ma_63:
            ma_stack_alignment = 1.0
        elif latest_close < ma_10 < ma_21 < ma_63:
            ma_stack_alignment = -1.0
        else:
            ma_stack_alignment = 0.0
    slope_21_direct = _linear_slope(close_values[-21:])
    slope_63_direct = _linear_slope(close_values[-63:])
    slope_21 = slope_21_direct or _safe_float(key_features.get("trend_slope_21d"))
    slope_63 = slope_63_direct or _safe_float(key_features.get("trend_slope_63d"))
    trend_curvature = None
    if slope_21 is not None and slope_63 is not None:
        trend_curvature = slope_21 - slope_63
    mean_reversion_gap = None
    if latest_close is not None and ma_21 not in (None, 0):
        mean_reversion_gap = latest_close / ma_21 - 1.0
    breakout_state = None
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
    horizon_info = classify_horizon_coverage(
        len(close_values),
        requirements=TECHNICAL_HORIZON_REQUIREMENTS,
    )
    technical_fallback_sources = []
    if slope_21 is not None and slope_21_direct is None and key_features.get("trend_slope_21d") is not None:
        technical_fallback_sources.append("features_daily")
    if slope_63 is not None and slope_63_direct is None and key_features.get("trend_slope_63d") is not None:
        technical_fallback_sources.append("features_daily")
    technical_fallback_sources = sorted(set(technical_fallback_sources))
    technical_note = (
        f"Technical structure is populated through {horizon_info['available_through']}; longer moving-average and trend layers remain history-constrained."
        if horizon_info["available_horizons"] and horizon_info["missing_horizons"]
        else "Technical structure is currently too thin for a stable multi-horizon read."
        if not close_values
        else "Technical structure supports the standard moving-average stack."
    )
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
        "ma_stack_alignment": ma_stack_alignment,
        "mean_reversion_gap": mean_reversion_gap,
        "breakout_state": breakout_state,
        "volume_price_alignment": volume_price_alignment,
        "regime_label": key_features.get("regime_label"),
        "regime_strength": _safe_float(key_features.get("regime_strength")),
        "meta": {
            "coverage_score": _clamp(_ratio(len(close_values), 126), 0.0, 1.0),
            **availability_payload(
                has_data=bool(close_values),
                coverage_score=_clamp(_ratio(len(close_values), 126), 0.0, 1.0),
                available_horizons=horizon_info["available_horizons"],
                missing_horizons=horizon_info["missing_horizons"],
                missing_reason="unavailable" if not close_values else horizon_info["missing_reason"],
                fallback_used=bool(technical_fallback_sources),
                fallback_source=technical_fallback_sources,
                data_quality_note=technical_note,
            ),
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

    statement_coverage = {
        "revenue": any(row.get("revenue") is not None for row in fundamentals),
        "gross_margin": any(row.get("gross_margin") is not None for row in fundamentals),
        "op_margin": any(row.get("op_margin") is not None for row in fundamentals),
        "fcf": any(row.get("fcf") is not None for row in fundamentals),
    }
    coverage_score = len(fundamentals) / 4.0 if fundamentals else 0.0
    fundamental_note = (
        "Quarterly filing coverage is partial, so durability and cash-flow reads should be treated with reduced confidence."
        if fundamentals and coverage_score < 1.0
        else "Quarterly filing coverage is currently unavailable in the local store."
        if not fundamentals
        else "Quarterly filing coverage is present across the recent filing window."
    )
    return {
        "latest_quarter": latest,
        "quarterly_series": fundamentals[:4],
        "statement_coverage": statement_coverage,
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
            **availability_payload(
                has_data=bool(fundamentals),
                coverage_score=_clamp(coverage_score, 0.0, 1.0),
                freshness_status=(
                    "fresh"
                    if filing_recency_days is not None and filing_recency_days <= 120
                    else "stale_but_usable"
                    if filing_recency_days is not None and filing_recency_days <= 180
                    else "limited"
                ),
                missing_reason="unavailable" if not fundamentals else None,
                data_quality_note=fundamental_note,
            ),
        },
    }


def _sentiment_domain(
    symbol: str,
    as_of_date: dt.date,
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
    sentiment_score = _first_available(
        _safe_float(key_features.get("sentiment_score")),
        latest_sentiment.get("sentiment_score"),
        latest_sentiment.get("sentiment_mean"),
    )
    sentiment_surprise = _safe_float(key_features.get("sentiment_surprise"))
    if sentiment_surprise is None and sentiment_series:
        baseline = _mean(sentiment_series[:-1])
        if baseline is not None and sentiment_score is not None:
            sentiment_surprise = sentiment_score - baseline
    ret_5d = _safe_float(key_features.get("ret_5d"))
    if sentiment_score is not None and ret_5d is not None:
        hype_price_divergence = sentiment_score - ret_5d

    latest_news_ingested = recent_news[0].get("ingested_at") if recent_news else None
    latest_sentiment_at = latest_sentiment.get("computed_at")
    sentiment_fallback_sources = []
    if key_features.get("sentiment_score") in (None, "") and sentiment_score is not None:
        sentiment_fallback_sources.append("sentiment_daily")
    sentiment_note = (
        "Headline flow is available, but the density is still too thin for a stable sentiment-level inference."
        if recent_news and sentiment_score is None
        else "Sentiment history is available but still partial across the recent lookback window."
        if sentiment_history and len(sentiment_history) < 5
        else "Sentiment coverage is currently too thin to support a directional read."
        if not recent_news and not sentiment_history
        else "Sentiment coverage is available across recent headlines and stored sentiment history."
    )
    return {
        "sentiment_score": sentiment_score,
        "sentiment_surprise": sentiment_surprise,
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
            "news_status": _domain_status(_days_stale(_iso_date(as_of_date), latest_news_ingested), fresh_days=1, usable_days=5),
            "sentiment_status": _domain_status(_days_stale(_iso_date(as_of_date), latest_sentiment_at), fresh_days=1, usable_days=5),
            "latest_news_ingested_at": latest_news_ingested,
            "latest_sentiment_at": latest_sentiment_at,
            "coverage_score": _clamp(_ratio(len(recent_news), 12), 0.0, 1.0),
            **availability_payload(
                has_data=bool(recent_news or sentiment_history),
                coverage_score=_clamp(
                    _mean(
                        [
                            _ratio(len(recent_news), 12),
                            _ratio(len(sentiment_history), 5),
                        ]
                    ),
                    0.0,
                    1.0,
                ),
                freshness_status=_domain_status(
                    _days_stale(_iso_date(as_of_date), latest_news_ingested or latest_sentiment_at),
                    fresh_days=1,
                    usable_days=5,
                ),
                missing_reason="unavailable" if not recent_news and not sentiment_history else None,
                fallback_used=bool(sentiment_fallback_sources),
                fallback_source=sentiment_fallback_sources,
                data_quality_note=sentiment_note,
            ),
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

    macro_fallback_sources = []
    if preferred_proxy and benchmark and benchmark.get("symbol") != preferred_proxy:
        macro_fallback_sources.append(benchmark.get("symbol"))
    macro_note = (
        f"Sector-specific benchmark coverage is thin, so the broader-market fallback {benchmark.get('symbol')} is being used."
        if preferred_proxy and benchmark and benchmark.get("symbol") != preferred_proxy
        else "Cross-asset proxy coverage is currently thin."
        if not proxies
        else f"Cross-asset context is anchored to {benchmark.get('symbol')}."
    )
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
            **availability_payload(
                has_data=bool(proxies),
                coverage_score=_clamp(_ratio(len(proxies), len(proxy_universe)), 0.0, 1.0),
                freshness_status="fresh" if proxies else "limited",
                missing_reason="unavailable" if not proxies else None,
                fallback_used=bool(macro_fallback_sources),
                fallback_source=macro_fallback_sources,
                data_quality_note=macro_note,
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
    relevant = bool(matched_titles)
    note = (
        "Recent headline flow does not currently point to a material policy, conflict, or macro shock cluster."
        if recent_news and not matched_titles
        else "Geopolitical and policy coverage is currently unavailable because recent headline coverage is thin."
        if not recent_news
        else "Event tagging is heuristic and should be read as directional context rather than a precision event model."
    )
    return {
        "category_counts": category_counts,
        "exogenous_event_score": exogenous_event_score,
        "relevant_headlines": matched_titles[:6],
        "policy_sensitive": weighted_hits > 0,
        "meta": {
            "coverage_score": _clamp(_ratio(len(recent_news), 10), 0.0, 1.0),
            "status": "fresh" if recent_news else "limited",
            **availability_payload(
                has_data=bool(recent_news),
                coverage_score=_clamp(_ratio(len(recent_news), 10), 0.0, 1.0),
                freshness_status="fresh" if recent_news else "limited",
                missing_reason="unavailable" if not recent_news else "not_relevant" if not relevant else None,
                relevant=relevant or not recent_news,
                data_quality_note=note,
            ),
        },
    }


def _relative_context_domain(
    symbol: str,
    as_of_date: dt.date,
    symbol_meta: Dict[str, Any],
    market_domain: Dict[str, Any],
    macro_domain: Dict[str, Any],
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
    benchmark_ret_21d = _safe_float(macro_domain.get("benchmark_ret_21d"))
    benchmark_proxy = macro_domain.get("benchmark_proxy")
    benchmark_relative = (
        ret_21d - benchmark_ret_21d
        if ret_21d is not None and benchmark_ret_21d is not None
        else None
    )
    relative_move_summary = {
        "vs_benchmark_ret_21d": benchmark_relative,
        "vs_sector_ret_21d": rel_ret,
        "market_relative_note": (
            f"The stock is outperforming {benchmark_proxy} on a 21-day basis."
            if benchmark_relative is not None and benchmark_proxy and benchmark_relative > 0
            else f"The stock is lagging {benchmark_proxy} on a 21-day basis."
            if benchmark_relative is not None and benchmark_proxy and benchmark_relative < 0
            else None
        ),
        "sector_relative_note": (
            "The stock is outperforming the local sector comparison set."
            if rel_ret is not None and rel_ret > 0
            else "The stock is lagging the local sector comparison set."
            if rel_ret is not None and rel_ret < 0
            else None
        ),
    }
    relative_note = (
        "Peer-relative context is currently unavailable because no same-sector comparison set was found."
        if not peers
        else "Peer-relative context is partial because the comparison set is still shallow."
        if len(peers) < 5
        else "Peer-relative context is populated against the local sector comparison set."
    )
    return {
        "sector": sector,
        "peer_count": len(peers),
        "sector_median_ret_21d": sector_ret_median,
        "sector_median_mom_vol_adj_21d": sector_mom_median,
        "relative_ret_21d": rel_ret,
        "relative_momentum": rel_mom,
        "relative_strength_percentile": percentile,
        "benchmark_proxy": benchmark_proxy,
        "benchmark_relative_strength": benchmark_relative,
        "peer_dispersion_score": peer_dispersion,
        "relative_move_summary": relative_move_summary,
        "peer_snapshot": peers[:8],
        "meta": {
            "coverage_score": _clamp(_ratio(len(peers), 8), 0.0, 1.0),
            "status": "fresh" if peers else "limited",
            **availability_payload(
                has_data=bool(peers),
                coverage_score=_clamp(_ratio(len(peers), 8), 0.0, 1.0),
                freshness_status="fresh" if peers else "limited",
                missing_reason="unavailable" if not peers else None,
                data_quality_note=relative_note,
            ),
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
            **availability_payload(
                has_data=bool(freshness_summary),
                coverage_score=_mean(
                    [
                        1.0 if item.get("status") == "fresh" else 0.6 if item.get("status") == "stale_but_usable" else 0.35
                        for item in freshness_summary.values()
                    ]
                ),
                freshness_status="fresh" if not warnings else "stale_but_usable",
                missing_reason="unavailable" if not freshness_summary else None,
                data_quality_note=(
                    "Coverage and freshness checks are current across the tracked operational domains."
                    if not warnings
                    else "Coverage and freshness checks are mixed, and the report is carrying explicit quality headwinds."
                ),
            ),
        },
    }


def _build_domain_availability_map(data_bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    domain_map = {
        "market": data_bundle.get("market_price_volume") or {},
        "technical": data_bundle.get("technical_market_structure") or {},
        "fundamentals": data_bundle.get("fundamental_filing") or {},
        "sentiment": data_bundle.get("sentiment_narrative_flow") or {},
        "macro": data_bundle.get("macro_cross_asset") or {},
        "geopolitical": data_bundle.get("geopolitical_policy") or {},
        "cross_asset": data_bundle.get("relative_context") or {},
        "event": data_bundle.get("event_catalyst_risk") or {},
        "liquidity": data_bundle.get("liquidity_execution_fragility") or {},
        "breadth": data_bundle.get("market_breadth_internals") or {},
        "cross_asset_depth": data_bundle.get("cross_asset_confirmation") or {},
        "stress": data_bundle.get("stress_spillover_conditions") or {},
        "quality": data_bundle.get("quality_provenance") or {},
    }
    availability: Dict[str, Dict[str, Any]] = {}
    for label, payload in domain_map.items():
        meta = dict(payload.get("meta") or {})
        availability[label] = {
            "coverage_status": meta.get("coverage_status") or meta.get("status") or "unknown",
            "available_horizons": list(meta.get("available_horizons") or []),
            "missing_horizons": list(meta.get("missing_horizons") or []),
            "missing_reason": meta.get("missing_reason"),
            "fallback_used": bool(meta.get("fallback_used")),
            "fallback_source": list(meta.get("fallback_source") or []),
            "data_quality_note": meta.get("data_quality_note") or meta.get("relevance_note"),
        }
    return availability


def _canonical_depth_domains(job_context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    canonical = {
        "lineage": job_context.get("canonical_lineage") or {},
        "features": job_context.get("canonical_feature_vector") or {},
        "signal": job_context.get("canonical_signal_payload") or {},
    }
    features = canonical["features"]
    signal_payload = canonical["signal"]

    def _domain_meta(has_data: bool, note: str) -> Dict[str, Any]:
        return {
            **availability_payload(
                has_data=has_data,
                coverage_score=1.0 if has_data else 0.0,
                freshness_status="fresh" if has_data else "limited",
                missing_reason=None if has_data else "unavailable",
                data_quality_note=note,
            ),
            "status": "fresh" if has_data else "limited",
        }

    event_present = any(
        features.get(key) is not None
        for key in (
            "event_overhang_score",
            "days_to_next_event",
            "event_density_score",
        )
    )
    event_domain = {
        "days_to_next_event": features.get("days_to_next_event"),
        "days_since_last_major_event": features.get("days_since_last_major_event"),
        "earnings_window_flag": features.get("earnings_window_flag"),
        "post_event_instability_flag": features.get("post_event_instability_flag"),
        "event_density_score": features.get("event_density_score"),
        "event_overhang_score": features.get("event_overhang_score"),
        "event_uncertainty_score": features.get("event_uncertainty_score"),
        "catalyst_burst_score": features.get("catalyst_burst_score"),
        "event_risk_classification": features.get("event_risk_classification"),
        "major_event_titles": features.get("major_event_titles") or [],
        "meta": _domain_meta(
            event_present,
            "Event context is being inferred from filing recency and catalyst-tagged headline bursts."
            if event_present
            else "Event and catalyst context is currently unavailable.",
        ),
    }

    liquidity_present = any(
        features.get(key) is not None
        for key in (
            "implementation_fragility_score",
            "liquidity_quality_score",
            "friction_proxy_score",
        )
    )
    liquidity_domain = {
        "liquidity_quality_score": features.get("liquidity_quality_score"),
        "gap_instability_score": features.get("gap_instability_score"),
        "range_instability_score": features.get("range_instability_score"),
        "turnover_stability_score": features.get("turnover_stability_score"),
        "volume_instability_score": features.get("volume_instability_score"),
        "tradability_caution_score": features.get("tradability_caution_score"),
        "implementation_fragility_score": features.get("implementation_fragility_score"),
        "overnight_gap_risk_score": features.get("overnight_gap_risk_score"),
        "friction_proxy_score": features.get("friction_proxy_score"),
        "execution_cleanliness_score": features.get("execution_cleanliness_score"),
        "tradability_state": features.get("tradability_state"),
        "meta": _domain_meta(
            liquidity_present,
            "Liquidity and implementation context is coming from gap behavior, range instability, volume stability, and friction proxies."
            if liquidity_present
            else "Liquidity and implementation context is currently unavailable.",
        ),
    }

    breadth_present = any(
        features.get(key) is not None
        for key in (
            "breadth_confirmation_score",
            "participation_breadth_score",
            "breadth_thrust_proxy",
        )
    )
    breadth_domain = {
        "breadth_thrust_proxy": features.get("breadth_thrust_proxy"),
        "participation_breadth_score": features.get("participation_breadth_score"),
        "breadth_confirmation_score": features.get("breadth_confirmation_score"),
        "cross_sectional_dispersion_proxy": features.get("cross_sectional_dispersion_proxy"),
        "sector_dispersion_proxy": features.get("sector_dispersion_proxy"),
        "leadership_concentration_score": features.get("leadership_concentration_score"),
        "narrow_leadership_warning": features.get("narrow_leadership_warning"),
        "broad_participation_confirmation": features.get("broad_participation_confirmation"),
        "internal_market_divergence_score": features.get("internal_market_divergence_score"),
        "leader_strength_score": features.get("leader_strength_score"),
        "laggard_pressure_score": features.get("laggard_pressure_score"),
        "leadership_rotation_score": features.get("leadership_rotation_score"),
        "leadership_instability_score": features.get("leadership_instability_score"),
        "breadth_state": features.get("breadth_state"),
        "meta": _domain_meta(
            breadth_present,
            "Market-internals context is built from participation breadth, leadership concentration, and dispersion proxies."
            if breadth_present
            else "Breadth and market-internals context is currently unavailable.",
        ),
    }

    cross_asset_present = any(
        features.get(key) is not None
        for key in (
            "benchmark_confirmation_score",
            "sector_confirmation_score",
            "cross_asset_conflict_score",
        )
    )
    cross_asset_domain = {
        "benchmark_proxy": features.get("benchmark_proxy"),
        "sector_proxy": features.get("sector_proxy"),
        "benchmark_confirmation_score": features.get("benchmark_confirmation_score"),
        "sector_confirmation_score": features.get("sector_confirmation_score"),
        "macro_asset_alignment_score": features.get("macro_asset_alignment_score"),
        "cross_asset_conflict_score": features.get("cross_asset_conflict_score"),
        "cross_asset_divergence_score": features.get("cross_asset_divergence_score"),
        "beta_context_score": features.get("beta_context_score"),
        "idiosyncratic_strength_score": features.get("idiosyncratic_strength_score"),
        "idiosyncratic_weakness_score": features.get("idiosyncratic_weakness_score"),
        "meta": _domain_meta(
            cross_asset_present,
            "Cross-asset confirmation uses benchmark, sector, and defensive-asset context."
            if cross_asset_present
            else "Cross-asset confirmation context is currently unavailable.",
        ),
    }

    stress_present = any(
        features.get(key) is not None
        for key in (
            "market_stress_score",
            "spillover_risk_score",
            "correlation_breakdown_proxy",
        )
    )
    depth_adjustments = ((signal_payload.get("signal_meta") or {}).get("depth_adjustments") or {})
    stress_domain = {
        "market_stress_score": features.get("market_stress_score"),
        "spillover_risk_score": features.get("spillover_risk_score"),
        "correlation_breakdown_proxy": features.get("correlation_breakdown_proxy"),
        "volatility_shock_score": features.get("volatility_shock_score"),
        "stress_transition_score": features.get("stress_transition_score"),
        "contagion_risk_proxy": features.get("contagion_risk_proxy"),
        "defensive_regime_flag": features.get("defensive_regime_flag"),
        "unstable_environment_flag": features.get("unstable_environment_flag"),
        "suppression_flags": signal_payload.get("suppression_flags")
        or depth_adjustments.get("suppression_flags")
        or [],
        "adjusted_confidence_notes": signal_payload.get("adjusted_confidence_notes")
        or depth_adjustments.get("adjusted_confidence_notes")
        or [],
        "meta": _domain_meta(
            stress_present,
            "Stress context is built from volatility shocks, breadth deterioration, spillover risk, and cross-asset contradictions."
            if stress_present
            else "Stress and spillover context is currently unavailable.",
        ),
    }

    return {
        "event_catalyst_risk": event_domain,
        "liquidity_execution_fragility": liquidity_domain,
        "market_breadth_internals": breadth_domain,
        "cross_asset_confirmation": cross_asset_domain,
        "stress_spillover_conditions": stress_domain,
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
    sentiment_domain = _sentiment_domain(
        symbol,
        as_of_date,
        sentiment_history,
        recent_news,
        key_features,
    )
    macro_domain = _macro_cross_asset_domain(symbol_meta, as_of_date, job_context, market_domain)
    geopolitical_domain = _geopolitical_domain(recent_news)
    relative_domain = _relative_context_domain(
        symbol, as_of_date, symbol_meta, market_domain, macro_domain, key_features
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
    canonical_depth = _canonical_depth_domains(job_context)

    data_bundle = {
        "symbol_meta": symbol_meta,
        "canonical_alpha_core": {
            "lineage": job_context.get("canonical_lineage") or {},
            "feature_vector": job_context.get("canonical_feature_vector") or {},
            "signal_payload": job_context.get("canonical_signal_payload") or {},
        },
        "market_price_volume": market_domain,
        "technical_market_structure": technical_domain,
        "fundamental_filing": fundamental_domain,
        "sentiment_narrative_flow": sentiment_domain,
        "macro_cross_asset": macro_domain,
        "geopolitical_policy": geopolitical_domain,
        "relative_context": relative_domain,
        "event_catalyst_risk": canonical_depth.get("event_catalyst_risk") or {},
        "liquidity_execution_fragility": canonical_depth.get("liquidity_execution_fragility") or {},
        "market_breadth_internals": canonical_depth.get("market_breadth_internals") or {},
        "cross_asset_confirmation": canonical_depth.get("cross_asset_confirmation") or {},
        "stress_spillover_conditions": canonical_depth.get("stress_spillover_conditions") or {},
        "quality_provenance": quality_domain,
        "raw_supporting_fields": {
            "signal": signal,
            "key_features": key_features,
            "quality": quality,
            "canonical_lineage": job_context.get("canonical_lineage") or {},
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
    domain_availability = _build_domain_availability_map(merged_bundle)
    merged_bundle["domain_availability"] = domain_availability
    merged_bundle["quality_provenance"] = {
        **(merged_bundle.get("quality_provenance") or {}),
        "domain_availability": domain_availability,
    }
    merged_bundle["normalized_domains"] = {
        "market": merged_bundle.get("market_price_volume") or {},
        "technical": merged_bundle.get("technical_market_structure") or {},
        "fundamentals": merged_bundle.get("fundamental_filing") or {},
        "news_sentiment_narrative": merged_bundle.get("sentiment_narrative_flow") or {},
        "macro": merged_bundle.get("macro_cross_asset") or {},
        "geopolitical": merged_bundle.get("geopolitical_policy") or {},
        "cross_asset": merged_bundle.get("relative_context") or {},
        "quality_provenance": merged_bundle.get("quality_provenance") or {},
    }
    merged_bundle["raw_supporting_fields"]["external_data_fabric"] = overlay
    return merged_bundle


def build_feature_factor_bundle(
    *,
    data_bundle: Dict[str, Any],
    signal: Dict[str, Any],
    key_features: Dict[str, Any],
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    return build_phase3_feature_factor_bundle(
        data_bundle=data_bundle,
        signal=signal,
        key_features=key_features,
        quality=quality,
    )
