from __future__ import annotations

import datetime as dt
import math
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence


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
    "filed",
    "filing",
    "sec",
    "10-q",
    "10-k",
    "8-k",
)


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _clamp(value: Optional[float], low: float, high: float) -> Optional[float]:
    if value is None:
        return None
    return max(low, min(high, float(value)))


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.fmean(clean))


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return None
    return float(statistics.pstdev(clean))


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.median(clean))


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _pct_change(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    if current is None or prior in (None, 0):
        return None
    return float(current) / float(prior) - 1.0


def _score_100(value: Optional[float], *, low: float = 0.0, high: float = 1.0) -> Optional[float]:
    if value is None:
        return None
    if math.isclose(low, high):
        return 50.0
    clipped = max(low, min(high, float(value)))
    return 100.0 * ((clipped - low) / (high - low))


def _parse_date(value: Any) -> Optional[dt.date]:
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value)
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _to_returns(closes: Sequence[float]) -> List[float]:
    returns: List[float] = []
    for idx in range(1, len(closes)):
        prev = closes[idx - 1]
        cur = closes[idx]
        if prev == 0:
            continue
        returns.append(float(cur / prev - 1.0))
    return returns


def _ret(closes: Sequence[float], periods: int) -> Optional[float]:
    if len(closes) <= periods:
        return None
    base = closes[-(periods + 1)]
    if base == 0:
        return None
    return float(closes[-1] / base - 1.0)


def _realized_vol(closes: Sequence[float], window: int) -> Optional[float]:
    returns = _to_returns(closes)
    if len(returns) < window:
        return None
    sigma = _std(returns[-window:])
    if sigma is None:
        return None
    return float(sigma * math.sqrt(252.0))


def _event_keyword_hits(news_items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for item in news_items:
        text = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("content_snippet") or ""),
            ]
        ).lower()
        matched = [keyword for keyword in _EVENT_KEYWORDS if keyword in text]
        if matched:
            hits.append(
                {
                    "published_at": item.get("published_at"),
                    "title": item.get("title"),
                    "matched_keywords": matched,
                }
            )
    return hits


def _reference_proxy_returns(reference_context: Dict[str, Any], periods: int = 21) -> Dict[str, Optional[float]]:
    payload: Dict[str, Optional[float]] = {}
    for symbol, context in (reference_context or {}).items():
        bars = list((context or {}).get("bars") or [])
        closes = [
            _safe_float(row.get("close"))
            for row in bars
            if _safe_float(row.get("close")) is not None
        ]
        payload[str(symbol)] = _ret(closes, periods) if closes else None
    return payload


def build_event_depth(snapshot: Dict[str, Any], base_features: Dict[str, Any]) -> Dict[str, Any]:
    as_of_date = _parse_date(snapshot.get("as_of_date")) or dt.date.today()
    event_context = dict(snapshot.get("event_context") or {})
    news_items = list(snapshot.get("news") or [])
    event_hits = list(event_context.get("major_event_matches") or [])
    if not event_hits:
        event_hits = _event_keyword_hits(news_items)

    next_event_date = _parse_date(
        event_context.get("estimated_next_event_date")
        or event_context.get("next_event_date")
    )
    latest_event_date = _parse_date(event_context.get("latest_event_date"))
    if latest_event_date is None:
        event_dates = [
            _parse_date(item.get("published_at"))
            for item in event_hits
            if _parse_date(item.get("published_at")) is not None
        ]
        latest_event_date = max(event_dates) if event_dates else None

    explicit_days_to_next = _safe_float(event_context.get("days_to_next_event"))
    explicit_days_since = _safe_float(event_context.get("days_since_last_major_event"))
    days_to_next_event = (
        int(explicit_days_to_next)
        if explicit_days_to_next is not None
        else max((next_event_date - as_of_date).days, 0)
        if next_event_date is not None
        else None
    )
    days_since_last_major_event = (
        int(explicit_days_since)
        if explicit_days_since is not None
        else max((as_of_date - latest_event_date).days, 0)
        if latest_event_date is not None
        else None
    )
    recent_3d = 0
    recent_7d = 0
    recent_21d = 0
    for item in event_hits:
        published = _parse_date(item.get("published_at"))
        if published is None or published > as_of_date:
            continue
        delta = (as_of_date - published).days
        if delta <= 3:
            recent_3d += 1
        if delta <= 7:
            recent_7d += 1
        if delta <= 21:
            recent_21d += 1

    baseline_recent = max(recent_21d - recent_7d, 0)
    catalyst_burst_ratio = _ratio(recent_3d, max(baseline_recent / 2.0, 1.0))
    explicit_density = _safe_float(event_context.get("event_density_score"))
    explicit_burst = _safe_float(event_context.get("catalyst_burst_score"))
    event_density_score = (
        explicit_density
        if explicit_density is not None
        else _score_100(_clamp(_ratio(recent_7d + recent_21d * 0.35, 8.0), 0.0, 1.0))
    )
    catalyst_burst_score = (
        explicit_burst
        if explicit_burst is not None
        else _score_100(_clamp(_ratio(catalyst_burst_ratio, 3.0), 0.0, 1.0))
    )
    proximity_score = _score_100(
        _clamp(1.0 - min(days_to_next_event or 31, 31) / 31.0, 0.0, 1.0)
    )
    recency_score = _score_100(
        _clamp(1.0 - min(days_since_last_major_event or 31, 31) / 31.0, 0.0, 1.0)
    )

    earnings_window_flag = bool(
        (days_to_next_event is not None and days_to_next_event <= 7)
        or (days_since_last_major_event is not None and days_since_last_major_event <= 2)
        or event_context.get("earnings_window_flag")
    )
    post_event_instability_flag = bool(
        event_context.get("post_event_instability_flag")
        or (
            days_since_last_major_event is not None
            and 0 <= days_since_last_major_event <= 5
            and (recent_3d > 0 or recent_7d >= 2)
        )
    )
    event_overhang_score = _mean(
        [
            proximity_score,
            recency_score,
            event_density_score,
            catalyst_burst_score,
            78.0 if earnings_window_flag else None,
            72.0 if post_event_instability_flag else None,
        ]
    )
    event_uncertainty_score = _mean(
        [
            event_overhang_score,
            _score_100(
                _clamp(abs(_safe_float(base_features.get("sentiment_surprise")) or 0.0) / 0.20, 0.0, 1.0)
            ),
            80.0 if recent_3d >= 2 else None,
            68.0 if post_event_instability_flag else None,
        ]
    )
    event_regime_adjustment = -((_clamp(_ratio(event_overhang_score or 0.0, 100.0), 0.0, 1.0) or 0.0) * 0.35)

    if (event_overhang_score or 0.0) >= 78 or (earnings_window_flag and post_event_instability_flag):
        event_risk_classification = "event_distorted"
    elif (event_overhang_score or 0.0) >= 64:
        event_risk_classification = "high_event_risk"
    elif (event_overhang_score or 0.0) >= 48:
        event_risk_classification = "moderate_event_risk"
    elif recent_3d > 0 or recent_7d >= 2:
        event_risk_classification = "catalyst_watch"
    elif (days_since_last_major_event or 999) <= 5:
        event_risk_classification = "post_event_repricing_state"
    else:
        event_risk_classification = "low_event_risk"

    return {
        "days_to_next_event": days_to_next_event,
        "days_since_last_major_event": days_since_last_major_event,
        "earnings_window_flag": earnings_window_flag,
        "post_event_instability_flag": post_event_instability_flag,
        "event_density_score": event_density_score,
        "event_overhang_score": event_overhang_score,
        "event_uncertainty_score": event_uncertainty_score,
        "catalyst_burst_score": catalyst_burst_score,
        "event_regime_adjustment": event_regime_adjustment,
        "event_risk_classification": event_risk_classification,
        "event_match_count_7d": recent_7d,
        "event_match_count_21d": recent_21d,
        "major_event_titles": [item.get("title") for item in event_hits[:5] if item.get("title")],
    }


def build_liquidity_depth(
    snapshot: Dict[str, Any],
    base_features: Dict[str, Any],
    event_depth: Dict[str, Any],
) -> Dict[str, Any]:
    price_rows = list(snapshot.get("price_bars") or [])
    gap_values: List[float] = []
    range_pct_values: List[float] = []
    dollar_volumes: List[float] = []
    volumes: List[float] = []
    closes: List[float] = []
    for idx, row in enumerate(price_rows):
        close = _safe_float(row.get("close"))
        open_px = _safe_float(row.get("open"))
        high_px = _safe_float(row.get("high"))
        low_px = _safe_float(row.get("low"))
        volume = _safe_float(row.get("volume"))
        if close is not None:
            closes.append(close)
        if close is not None and high_px is not None and low_px is not None and close != 0:
            range_pct_values.append(abs(high_px - low_px) / close)
        if close is not None and volume is not None:
            dollar_volumes.append(close * volume)
            volumes.append(volume)
        if idx > 0 and open_px is not None:
            prior_close = _safe_float(price_rows[idx - 1].get("close"))
            gap = _pct_change(open_px, prior_close)
            if gap is not None:
                gap_values.append(abs(gap))

    atr_pct = _safe_float(base_features.get("atr_pct")) or 0.0
    volatility_ann = _safe_float(base_features.get("volatility_ann")) or _safe_float(base_features.get("vol_63d")) or 0.0
    avg_gap = _mean(gap_values[-10:]) or 0.0
    gap_p95 = max(gap_values[-10:], default=avg_gap)
    range_mean = _mean(range_pct_values[-21:]) or 0.0
    range_std = _std(range_pct_values[-21:]) or 0.0
    range_instability_ratio = _ratio(range_std, range_mean or None) or 0.0
    volume_mean = _mean(volumes[-21:]) or 0.0
    volume_std = _std(volumes[-21:]) or 0.0
    volume_cv = _ratio(volume_std, volume_mean or None) or 0.0
    turnover_mean = _mean(dollar_volumes[-21:]) or 0.0
    turnover_std = _std(dollar_volumes[-21:]) or 0.0
    turnover_cv = _ratio(turnover_std, turnover_mean or None) or 0.0

    dollar_vol_21d = _safe_float(base_features.get("dollar_vol_21d")) or turnover_mean or None
    dollar_liquidity_score = None
    if dollar_vol_21d is not None and dollar_vol_21d > 0:
        dollar_liquidity_score = _score_100(
            _clamp((math.log10(dollar_vol_21d) - 5.5) / 2.5, 0.0, 1.0)
        )

    gap_instability_score = _score_100(_clamp(_ratio(avg_gap, 0.035), 0.0, 1.0))
    overnight_gap_risk_score = _score_100(_clamp(_ratio(gap_p95, 0.055), 0.0, 1.0))
    range_instability_score = _score_100(_clamp(_ratio(range_instability_ratio, 1.1), 0.0, 1.0))
    turnover_stability_score = _score_100(1.0 - (_clamp(_ratio(turnover_cv, 1.4), 0.0, 1.0) or 0.0))
    volume_instability_score = _score_100(_clamp(_ratio(volume_cv, 1.4), 0.0, 1.0))

    liquidity_quality_score = _mean(
        [
            dollar_liquidity_score,
            turnover_stability_score,
            100.0 - (gap_instability_score or 50.0),
            100.0 - (range_instability_score or 50.0),
            _score_100(1.0 - (_clamp(_ratio(atr_pct, 0.08), 0.0, 1.0) or 0.0)),
        ]
    )
    friction_proxy_score = _mean(
        [
            100.0 - (liquidity_quality_score or 50.0),
            gap_instability_score,
            range_instability_score,
            volume_instability_score,
            _score_100(_clamp(_ratio(volatility_ann, 0.55), 0.0, 1.0)),
        ]
    )
    tradability_caution_score = _mean(
        [
            friction_proxy_score,
            overnight_gap_risk_score,
            100.0 - (turnover_stability_score or 50.0),
            event_depth.get("catalyst_burst_score"),
        ]
    )
    baseline_implementation_fragility = _mean(
        [
            tradability_caution_score,
            gap_instability_score,
            range_instability_score,
            friction_proxy_score,
            event_depth.get("event_uncertainty_score"),
        ]
    )
    event_implementation_pressure = _mean(
        [
            overnight_gap_risk_score,
            event_depth.get("event_uncertainty_score"),
            event_depth.get("event_overhang_score"),
            event_depth.get("catalyst_burst_score"),
        ]
    )
    implementation_fragility_score = _mean(
        [
            baseline_implementation_fragility,
            tradability_caution_score,
            event_implementation_pressure,
            70.0 if event_depth.get("earnings_window_flag") else None,
            64.0 if event_depth.get("post_event_instability_flag") else None,
        ]
    )
    execution_cleanliness_score = (
        100.0 - implementation_fragility_score
        if implementation_fragility_score is not None
        else None
    )

    if (implementation_fragility_score or 0.0) >= 74:
        tradability_state = "implementation_fragile"
    elif (liquidity_quality_score or 0.0) >= 68 and (implementation_fragility_score or 0.0) <= 42:
        tradability_state = "clean_liquid_setup"
    elif (tradability_caution_score or 0.0) >= 55:
        tradability_state = "patient_execution_only"
    else:
        tradability_state = "tradable_with_caution"

    return {
        "liquidity_quality_score": liquidity_quality_score,
        "gap_instability_score": gap_instability_score,
        "range_instability_score": range_instability_score,
        "turnover_stability_score": turnover_stability_score,
        "volume_instability_score": volume_instability_score,
        "tradability_caution_score": tradability_caution_score,
        "implementation_fragility_score": implementation_fragility_score,
        "overnight_gap_risk_score": overnight_gap_risk_score,
        "friction_proxy_score": friction_proxy_score,
        "execution_cleanliness_score": execution_cleanliness_score,
        "avg_gap_10d": avg_gap,
        "range_instability_ratio": range_instability_ratio,
        "tradability_state": tradability_state,
    }


def build_breadth_depth(snapshot: Dict[str, Any], base_features: Dict[str, Any]) -> Dict[str, Any]:
    breadth_context = dict(snapshot.get("breadth_context") or {})
    reference_context = dict(snapshot.get("reference_context") or {})
    proxy_returns = _reference_proxy_returns(reference_context, 21)
    broad_equity_returns = [
        proxy_returns.get(symbol)
        for symbol in ("SPY", "QQQ", "IWM")
        if proxy_returns.get(symbol) is not None
    ]
    positive_proxy_ratio = _ratio(
        sum(1 for value in broad_equity_returns if (value or 0.0) > 0.0),
        len(broad_equity_returns) or None,
    )

    advancing_1d_ratio = _safe_float(breadth_context.get("advancing_1d_ratio"))
    advancing_21d_ratio = _safe_float(breadth_context.get("advancing_21d_ratio"))
    above_trend_ratio = _safe_float(breadth_context.get("above_trend_ratio"))
    sector_participation_ratio = _safe_float(breadth_context.get("sector_participation_ratio"))
    if advancing_21d_ratio is None:
        advancing_21d_ratio = positive_proxy_ratio
    if advancing_1d_ratio is None:
        advancing_1d_ratio = positive_proxy_ratio

    cross_sectional_dispersion = _safe_float(breadth_context.get("cross_sectional_dispersion"))
    sector_dispersion = _safe_float(breadth_context.get("sector_dispersion"))
    if cross_sectional_dispersion is None and broad_equity_returns:
        cross_sectional_dispersion = _std(broad_equity_returns)
    if sector_participation_ratio is None:
        sector = str((snapshot.get("symbol_meta") or {}).get("sector") or "").strip().lower()
        sector_proxy = _SECTOR_PROXY_MAP.get(sector)
        sector_return = proxy_returns.get(sector_proxy) if sector_proxy else None
        sector_participation_ratio = 1.0 if sector_return is not None and sector_return > 0 else 0.0 if sector_return is not None else None
    if above_trend_ratio is None:
        above_trend_ratio = sector_participation_ratio

    leader_strength = _safe_float(breadth_context.get("leader_strength"))
    laggard_pressure = _safe_float(breadth_context.get("laggard_pressure"))
    leadership_concentration = _safe_float(breadth_context.get("leadership_concentration"))
    leadership_rotation = _safe_float(breadth_context.get("leadership_rotation"))
    leadership_instability = _safe_float(breadth_context.get("leadership_instability"))

    breadth_thrust_proxy = _mean(
        [
            _score_100(advancing_1d_ratio),
            _score_100(above_trend_ratio),
        ]
    )
    participation_breadth_score = _mean(
        [
            _score_100(advancing_21d_ratio),
            _score_100(sector_participation_ratio),
            _score_100(above_trend_ratio),
        ]
    )
    cross_sectional_dispersion_proxy = _score_100(
        _clamp(_ratio(cross_sectional_dispersion, 0.05), 0.0, 1.0)
    )
    sector_dispersion_proxy = _score_100(
        _clamp(_ratio(sector_dispersion, 0.04), 0.0, 1.0)
    )
    leadership_concentration_score = _score_100(
        _clamp(leadership_concentration, 0.0, 1.0)
    )
    leader_strength_score = _score_100(
        _clamp(_ratio(leader_strength, 0.18), 0.0, 1.0)
    )
    laggard_pressure_score = _score_100(
        _clamp(_ratio(abs(min(laggard_pressure or 0.0, 0.0)), 0.12), 0.0, 1.0)
    )
    leadership_rotation_score = _score_100(
        _clamp(leadership_rotation, 0.0, 1.0)
    )
    leadership_instability_score = _mean(
        [
            _score_100(_clamp(leadership_instability, 0.0, 1.0)),
            leadership_concentration_score,
            cross_sectional_dispersion_proxy,
        ]
    )
    internal_market_divergence_score = _mean(
        [
            leadership_concentration_score,
            cross_sectional_dispersion_proxy,
            100.0 - (participation_breadth_score or 50.0),
        ]
    )
    breadth_confirmation_score = _mean(
        [
            breadth_thrust_proxy,
            participation_breadth_score,
            100.0 - (leadership_concentration_score or 50.0),
            100.0 - (internal_market_divergence_score or 50.0),
        ]
    )
    narrow_leadership_warning = bool(
        (leadership_concentration_score or 0.0) >= 68
        and (participation_breadth_score or 0.0) <= 55
    )
    broad_participation_confirmation = bool(
        (breadth_thrust_proxy or 0.0) >= 58
        and (participation_breadth_score or 0.0) >= 60
    )

    if broad_participation_confirmation:
        breadth_state = "broad_healthy_participation"
    elif narrow_leadership_warning:
        breadth_state = "narrow_leadership"
    elif (internal_market_divergence_score or 0.0) >= 65:
        breadth_state = "high_dispersion_unstable_internals"
    elif (breadth_confirmation_score or 0.0) <= 42:
        breadth_state = "weak_breadth_under_index_strength"
    else:
        breadth_state = "improving_internals"

    return {
        "breadth_thrust_proxy": breadth_thrust_proxy,
        "participation_breadth_score": participation_breadth_score,
        "breadth_confirmation_score": breadth_confirmation_score,
        "cross_sectional_dispersion_proxy": cross_sectional_dispersion_proxy,
        "sector_dispersion_proxy": sector_dispersion_proxy,
        "leadership_concentration_score": leadership_concentration_score,
        "narrow_leadership_warning": narrow_leadership_warning,
        "broad_participation_confirmation": broad_participation_confirmation,
        "internal_market_divergence_score": internal_market_divergence_score,
        "leader_strength_score": leader_strength_score,
        "laggard_pressure_score": laggard_pressure_score,
        "leadership_rotation_score": leadership_rotation_score,
        "leadership_instability_score": leadership_instability_score,
        "breadth_state": breadth_state,
        "breadth_universe_count": breadth_context.get("universe_count"),
    }


def build_cross_asset_depth(
    snapshot: Dict[str, Any],
    base_features: Dict[str, Any],
    breadth_depth: Dict[str, Any],
) -> Dict[str, Any]:
    reference_context = dict(snapshot.get("reference_context") or {})
    proxy_returns = _reference_proxy_returns(reference_context, 21)
    symbol_ret = _safe_float(base_features.get("ret_21d"))
    sector = str((snapshot.get("symbol_meta") or {}).get("sector") or "").strip().lower()
    sector_proxy = _SECTOR_PROXY_MAP.get(sector)
    benchmark_symbol = "SPY" if "SPY" in reference_context else "QQQ" if "QQQ" in reference_context else next(iter(reference_context.keys()), None)
    benchmark_ret = proxy_returns.get(benchmark_symbol) if benchmark_symbol else None
    sector_ret = proxy_returns.get(sector_proxy) if sector_proxy else None
    tlt_ret = proxy_returns.get("TLT")
    gld_ret = proxy_returns.get("GLD")
    uso_ret = proxy_returns.get("USO")
    uup_ret = proxy_returns.get("UUP")

    benchmark_alignment = None
    if symbol_ret is not None and benchmark_ret is not None:
        diff = abs(symbol_ret - benchmark_ret)
        sign_bonus = 1.0 if symbol_ret == 0 or benchmark_ret == 0 or math.copysign(1.0, symbol_ret) == math.copysign(1.0, benchmark_ret) else 0.35
        benchmark_alignment = _clamp((1.0 - min(diff, 0.35) / 0.35) * sign_bonus, 0.0, 1.0)
    sector_alignment = None
    if symbol_ret is not None and sector_ret is not None:
        diff = abs(symbol_ret - sector_ret)
        sign_bonus = 1.0 if symbol_ret == 0 or sector_ret == 0 or math.copysign(1.0, symbol_ret) == math.copysign(1.0, sector_ret) else 0.35
        sector_alignment = _clamp((1.0 - min(diff, 0.28) / 0.28) * sign_bonus, 0.0, 1.0)

    defensive_pressure = _mean(
        [
            _score_100(_clamp(_ratio((tlt_ret or 0.0) - (benchmark_ret or 0.0), 0.12), 0.0, 1.0)),
            _score_100(_clamp(_ratio((uup_ret or 0.0), 0.08), 0.0, 1.0)),
            _score_100(_clamp(_ratio((gld_ret or 0.0), 0.10), 0.0, 1.0)),
        ]
    )
    cyclical_support = _mean(
        [
            _score_100(_clamp(_ratio((benchmark_ret or 0.0), 0.12), 0.0, 1.0)),
            _score_100(_clamp(_ratio((sector_ret or benchmark_ret or 0.0), 0.12), 0.0, 1.0)),
            _score_100(_clamp(_ratio((uso_ret or 0.0), 0.15), 0.0, 1.0)),
        ]
    )
    macro_asset_alignment_score = _mean(
        [
            _score_100(benchmark_alignment),
            _score_100(sector_alignment),
            cyclical_support,
            100.0 - (defensive_pressure or 50.0),
        ]
    )
    divergence_components = [
        _score_100(_clamp(_ratio(abs((symbol_ret or 0.0) - (benchmark_ret or 0.0)), 0.20), 0.0, 1.0)),
        _score_100(_clamp(_ratio(abs((symbol_ret or 0.0) - (sector_ret or 0.0)), 0.18), 0.0, 1.0)),
    ]
    cross_asset_divergence_score = _mean(divergence_components)
    cross_asset_conflict_score = _mean(
        [
            100.0 - (_score_100(benchmark_alignment) or 50.0),
            100.0 - (_score_100(sector_alignment) or 50.0),
            defensive_pressure,
            breadth_depth.get("internal_market_divergence_score"),
        ]
    )
    beta_context_score = _mean(
        [
            _score_100(benchmark_alignment),
            _score_100(sector_alignment),
            breadth_depth.get("breadth_confirmation_score"),
        ]
    )
    idiosyncratic_strength_score = _score_100(
        _clamp(
            _ratio(
                max((symbol_ret or 0.0) - max(benchmark_ret or -1.0, sector_ret or -1.0), 0.0),
                0.18,
            ),
            0.0,
            1.0,
        )
    )
    idiosyncratic_weakness_score = _score_100(
        _clamp(
            _ratio(
                abs(min((symbol_ret or 0.0) - min(benchmark_ret or 1.0, sector_ret or 1.0), 0.0)),
                0.18,
            ),
            0.0,
            1.0,
        )
    )

    return {
        "benchmark_proxy": benchmark_symbol,
        "sector_proxy": sector_proxy,
        "benchmark_confirmation_score": _score_100(benchmark_alignment),
        "sector_confirmation_score": _score_100(sector_alignment),
        "macro_asset_alignment_score": macro_asset_alignment_score,
        "cross_asset_conflict_score": cross_asset_conflict_score,
        "cross_asset_divergence_score": cross_asset_divergence_score,
        "beta_context_score": beta_context_score,
        "idiosyncratic_strength_score": idiosyncratic_strength_score,
        "idiosyncratic_weakness_score": idiosyncratic_weakness_score,
        "broad_market_return_21d": benchmark_ret,
        "sector_return_21d": sector_ret,
        "defensive_pressure_score": defensive_pressure,
        "cyclical_support_score": cyclical_support,
    }


def build_stress_depth(
    snapshot: Dict[str, Any],
    base_features: Dict[str, Any],
    event_depth: Dict[str, Any],
    liquidity_depth: Dict[str, Any],
    breadth_depth: Dict[str, Any],
    cross_asset_depth: Dict[str, Any],
) -> Dict[str, Any]:
    reference_context = dict(snapshot.get("reference_context") or {})
    benchmark_bars = None
    for symbol in ("SPY", "QQQ", "IWM"):
        context = reference_context.get(symbol)
        if context and context.get("bars"):
            benchmark_bars = list(context.get("bars") or [])
            break
    benchmark_closes = [
        _safe_float(row.get("close"))
        for row in (benchmark_bars or [])
        if _safe_float(row.get("close")) is not None
    ]
    benchmark_vol_5 = _realized_vol(benchmark_closes, 5) if benchmark_closes else None
    benchmark_vol_21 = _realized_vol(benchmark_closes, 21) if benchmark_closes else None
    vol_shock_ratio = _ratio(benchmark_vol_5, benchmark_vol_21 or None)
    volatility_shock_score = _score_100(
        _clamp(_ratio((vol_shock_ratio or 1.0) - 1.0, 0.8), 0.0, 1.0)
    )

    correlation_breakdown_proxy = _mean(
        [
            breadth_depth.get("internal_market_divergence_score"),
            breadth_depth.get("leadership_instability_score"),
            cross_asset_depth.get("cross_asset_conflict_score"),
        ]
    )
    stress_transition_score = _mean(
        [
            volatility_shock_score,
            correlation_breakdown_proxy,
            liquidity_depth.get("implementation_fragility_score"),
            event_depth.get("event_uncertainty_score"),
            event_depth.get("event_overhang_score"),
            68.0 if event_depth.get("event_risk_classification") == "event_distorted" else None,
        ]
    )
    market_stress_score = _mean(
        [
            volatility_shock_score,
            100.0 - (breadth_depth.get("breadth_confirmation_score") or 50.0),
            cross_asset_depth.get("cross_asset_conflict_score"),
            stress_transition_score,
            breadth_depth.get("internal_market_divergence_score"),
            breadth_depth.get("leadership_concentration_score"),
        ]
    )
    spillover_risk_score = _mean(
        [
            market_stress_score,
            cross_asset_depth.get("cross_asset_divergence_score"),
            liquidity_depth.get("overnight_gap_risk_score"),
            event_depth.get("event_density_score"),
        ]
    )
    contagion_risk_proxy = _mean(
        [
            spillover_risk_score,
            correlation_breakdown_proxy,
            breadth_depth.get("leadership_concentration_score"),
        ]
    )
    defensive_regime_flag = bool(
        (market_stress_score or 0.0) >= 68
        or (cross_asset_depth.get("defensive_pressure_score") or 0.0) >= 62
    )
    unstable_environment_flag = bool(
        (stress_transition_score or 0.0) >= 62
        or (correlation_breakdown_proxy or 0.0) >= 66
    )

    return {
        "market_stress_score": market_stress_score,
        "spillover_risk_score": spillover_risk_score,
        "correlation_breakdown_proxy": correlation_breakdown_proxy,
        "volatility_shock_score": volatility_shock_score,
        "stress_transition_score": stress_transition_score,
        "contagion_risk_proxy": contagion_risk_proxy,
        "defensive_regime_flag": defensive_regime_flag,
        "unstable_environment_flag": unstable_environment_flag,
        "benchmark_vol_5": benchmark_vol_5,
        "benchmark_vol_21": benchmark_vol_21,
    }
