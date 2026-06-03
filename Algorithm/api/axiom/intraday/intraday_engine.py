"""Phase 10.1: Intraday Score Update Engine.

Refreshes only the two fastest-moving engines (flow + behavioral) using intraday
bar data. Daily engines anchor the signal; intraday updates nudge it.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp

INTRADAY_UPDATE_TIMES = ["10:00", "12:00", "14:00", "16:00"]
INTRADAY_FLOW_WEIGHT = 0.30
INTRADAY_BEHAVIORAL_WEIGHT = 0.20
_INTRADAY_BASE_WEIGHT = 1.0 - INTRADAY_FLOW_WEIGHT - INTRADAY_BEHAVIORAL_WEIGHT  # 0.50

_ALERT_COMPOSITE_THRESHOLD = 65.0  # composite must exceed this to be alert-eligible


@dataclass
class IntradaySnapshot:
    symbol: str
    timestamp: dt.datetime
    intraday_flow_score: Optional[float] = None
    intraday_behavioral_score: Optional[float] = None
    intraday_composite: Optional[float] = None
    vwap_deviation: Optional[float] = None
    intraday_momentum_score: Optional[float] = None
    volume_surge_score: Optional[float] = None
    session_return_pct: Optional[float] = None
    alert_eligible: bool = False


def compute_intraday_composite(
    daily_axiom_dau: float,
    intraday_flow_score: float,
    intraday_behavioral_score: float,
) -> float:
    """Blend daily AXIOM anchor with intraday engine updates."""
    composite = (
        daily_axiom_dau * _INTRADAY_BASE_WEIGHT
        + intraday_flow_score * INTRADAY_FLOW_WEIGHT
        + intraday_behavioral_score * INTRADAY_BEHAVIORAL_WEIGHT
    )
    return round(clamp(composite, 0.0, 100.0), 2)


def compute_vwap_deviation(bars: List[Dict[str, Any]]) -> float:
    """Compute VWAP and return deviation of current price vs VWAP as percentage.

    bars: list of dicts with keys: open, high, low, close, volume
    Returns 0.0 if bars is empty or total volume is zero.
    """
    if not bars:
        return 0.0

    total_volume = 0.0
    total_tpv = 0.0  # typical_price × volume

    for bar in bars:
        try:
            high = float(bar.get("high") or 0.0)
            low = float(bar.get("low") or 0.0)
            close = float(bar.get("close") or 0.0)
            volume = float(bar.get("volume") or 0.0)
        except (TypeError, ValueError):
            continue

        if volume <= 0:
            continue

        typical_price = (high + low + close) / 3.0
        total_tpv += typical_price * volume
        total_volume += volume

    if total_volume == 0.0:
        return 0.0

    vwap = total_tpv / total_volume

    # Current close is last bar's close
    try:
        current_close = float(bars[-1].get("close") or 0.0)
    except (TypeError, ValueError, IndexError):
        return 0.0

    if vwap == 0.0:
        return 0.0

    return round((current_close - vwap) / vwap * 100.0, 4)


def compute_volume_surge_score(
    current_volume_rate: float,
    avg_daily_volume: float,
) -> float:
    """Score how much current volume rate exceeds the expected intraday rate.

    current_volume_rate: shares per minute in current session
    avg_daily_volume:    30-day average daily volume
    High score = institutional activity signal (surging volume).
    Returns 50.0 if avg_daily_volume is zero.
    """
    _TRADING_MINUTES = 6.5 * 60  # 390 minutes

    expected_rate = avg_daily_volume / _TRADING_MINUTES
    if expected_rate <= 0:
        return 50.0

    surge_ratio = current_volume_rate / expected_rate
    score = (surge_ratio - 1.0) / 3.0 * 100.0
    return round(clamp(score, 0.0, 100.0), 2)


def _compute_momentum_score(bars: List[Dict[str, Any]], window: int = 6) -> Optional[float]:
    """30-min momentum: return from `window` bars ago to now, mapped to [0,100]."""
    if len(bars) < 2:
        return None
    idx_start = max(0, len(bars) - window)
    try:
        open_price = float(bars[idx_start].get("open") or bars[idx_start].get("close") or 0.0)
        close_price = float(bars[-1].get("close") or 0.0)
    except (TypeError, ValueError):
        return None

    if open_price <= 0:
        return None

    ret_pct = (close_price - open_price) / open_price * 100.0
    # Map [-3%, +3%] range to [0, 100]
    score = clamp(50.0 + ret_pct / 3.0 * 50.0, 0.0, 100.0)
    return round(score, 2)


def _compute_session_return(bars: List[Dict[str, Any]]) -> Optional[float]:
    """Return from first bar open to last bar close."""
    if len(bars) < 1:
        return None
    try:
        first_open = float(bars[0].get("open") or 0.0)
        last_close = float(bars[-1].get("close") or 0.0)
    except (TypeError, ValueError):
        return None

    if first_open <= 0:
        return None
    return round((last_close - first_open) / first_open * 100.0, 4)


def _intraday_flow_proxy(bars: List[Dict[str, Any]], avg_daily_volume: float) -> float:
    """Derive an intraday flow score from price-volume action."""
    if not bars:
        return 50.0

    vwap_dev = compute_vwap_deviation(bars)
    # Current volume rate (shares per minute, using last bar as proxy)
    try:
        session_minutes = max(len(bars), 1)
        total_vol = sum(float(b.get("volume") or 0.0) for b in bars)
        current_rate = total_vol / session_minutes
    except (TypeError, ValueError):
        current_rate = 0.0

    surge = compute_volume_surge_score(current_rate, avg_daily_volume)
    momentum = _compute_momentum_score(bars) or 50.0

    # Blend: positive VWAP deviation + high volume surge + upward momentum = high flow
    vwap_score = clamp(50.0 + vwap_dev * 5.0, 0.0, 100.0)
    return round(clamp(vwap_score * 0.40 + surge * 0.30 + momentum * 0.30, 0.0, 100.0), 2)


def run_intraday_update(
    symbol: str,
    intraday_bars: List[Dict[str, Any]],
    daily_axiom_dau: float,
    avg_daily_volume: float,
    intraday_sentiment_score: float = 50.0,
) -> IntradaySnapshot:
    """Compute all intraday components and return an IntradaySnapshot.

    Graceful: returns snapshot with all-None scores if intraday_bars is empty.
    """
    now = dt.datetime.now(dt.timezone.utc)

    if not intraday_bars:
        return IntradaySnapshot(symbol=symbol, timestamp=now, alert_eligible=False)

    vwap_dev = compute_vwap_deviation(intraday_bars)

    try:
        session_minutes = max(len(intraday_bars), 1)
        total_vol = sum(float(b.get("volume") or 0.0) for b in intraday_bars)
        current_rate = total_vol / session_minutes
    except (TypeError, ValueError):
        current_rate = 0.0

    volume_surge = compute_volume_surge_score(current_rate, avg_daily_volume)
    momentum = _compute_momentum_score(intraday_bars)
    session_ret = _compute_session_return(intraday_bars)
    flow_score = _intraday_flow_proxy(intraday_bars, avg_daily_volume)
    behavioral_score = round(clamp(intraday_sentiment_score, 0.0, 100.0), 2)
    composite = compute_intraday_composite(daily_axiom_dau, flow_score, behavioral_score)
    alert_eligible = composite >= _ALERT_COMPOSITE_THRESHOLD

    return IntradaySnapshot(
        symbol=symbol,
        timestamp=now,
        intraday_flow_score=flow_score,
        intraday_behavioral_score=behavioral_score,
        intraday_composite=composite,
        vwap_deviation=vwap_dev,
        intraday_momentum_score=momentum,
        volume_surge_score=volume_surge,
        session_return_pct=session_ret,
        alert_eligible=alert_eligible,
    )
