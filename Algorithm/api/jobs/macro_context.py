"""Macro context helpers — fetch real FRED data for CARDI and BFS inputs.

Caches results in module-level dicts with a 4-hour TTL.
All public functions wrap failures in try/except and return sensible defaults.
"""
from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TTL_SECONDS = 4 * 3600  # 4 hours

# Module-level cache: {series_id: (fetched_at_ts, value)}
_cache: Dict[str, tuple[float, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.time() - ts > _TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = (time.time(), value)


def _fetch_latest(series_id: str) -> Optional[float]:
    """Fetch the most recent non-null observation for a FRED series."""
    cached = _cache_get(series_id)
    if cached is not None:
        return cached
    try:
        from api.data_providers.fred import fetch_series
        result = fetch_series(series_id, limit=5)
        observations = result.get("observations") or []
        for obs in observations:
            v = obs.get("value")
            if v is not None:
                val = float(v)
                _cache_set(series_id, val)
                return val
    except Exception as exc:
        logger.debug("macro_context.fetch_failed series=%s error=%s", series_id, exc)
    return None


def get_vix_level(as_of_date: Optional[Any] = None) -> Optional[float]:
    """Return the most recent VIX (VIXCLS) value, or None if unavailable."""
    try:
        return _fetch_latest("VIXCLS")
    except Exception:
        return None


def get_term_spread(as_of_date: Optional[Any] = None) -> Optional[float]:
    """Return the 10Y-3M Treasury term spread (DGS10 - TB3MS), or None."""
    try:
        dgs10 = _fetch_latest("DGS10")
        tb3ms = _fetch_latest("TB3MS")
        if dgs10 is not None and tb3ms is not None:
            return round(dgs10 - tb3ms, 4)
        return None
    except Exception:
        return None


def get_cardi_inputs(as_of_date: Optional[Any] = None) -> Dict[str, Any]:
    """Return CARDI factor inputs dict.

    carry_score is derived from the term spread:
      - positive spread  → score 50-100
      - inverted yield curve → score 0-50
    """
    try:
        spread = get_term_spread()
        carry_score: Optional[float] = None
        if spread is not None:
            # spread typically ranges roughly -1.5 to +3.5; map to 0-100
            # 0 spread → 50, positive → above 50, negative (inverted) → below 50
            raw = 50.0 + spread * 14.0  # ~14 pts per 1% spread
            carry_score = round(max(0.0, min(100.0, raw)), 2)
        return {
            "carry_score": carry_score,
            "value_score": None,
            "momentum_score": None,
            "defensive_score": None,
        }
    except Exception:
        return {
            "carry_score": None,
            "value_score": None,
            "momentum_score": None,
            "defensive_score": None,
        }


def get_market_bubble_context(
    regime_label: str = "auto",
    vix: Optional[float] = None,
) -> Dict[str, Any]:
    """Return market bubble context dict.

    kindleberger_stage and narrative_intensity are driven by VIX:
      vix > 30  → distress  / intensity 75
      vix > 22  → boom      / intensity 60
      vix > 15  → normal    / intensity 50
      vix <= 15 → euphoria  / intensity 35  (low vol = complacency)
    """
    try:
        if vix is None:
            vix = get_vix_level()

        if vix is not None:
            if vix > 30:
                stage = "distress"
                intensity = 75.0
            elif vix > 22:
                stage = "boom"
                intensity = 60.0
            elif vix > 15:
                stage = "normal"
                intensity = 50.0
            else:
                stage = "euphoria"
                intensity = 35.0
        else:
            stage = "normal"
            intensity = 50.0

        return {
            "cape_z_score": None,
            "kindleberger_stage": stage,
            "narrative_intensity": intensity,
        }
    except Exception:
        return {
            "cape_z_score": None,
            "kindleberger_stage": "normal",
            "narrative_intensity": 50.0,
        }
