"""
Shared signal computation primitives extracted from api.main to break the
circular import between api.main and api.prosperity.ingest.

Callers that previously did `from api.main import Candle, ...` should now
import from here.  api.main re-exports these names for backward compatibility.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Candle(BaseModel):
    timestamp: str  # ISO date "YYYY-MM-DD" (daily)
    close: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None

    fundamental: Optional[float] = None
    sentiment: Optional[float] = None
    crowd: Optional[float] = None


class SignalResponse(BaseModel):
    symbol: str
    as_of: str
    lookback: int
    effective_lookback: int
    regime: str
    thresholds: Dict[str, float]
    score: float
    signal: str
    confidence: float
    features: Dict[str, float]
    notes: List[str] = Field(default_factory=list)

    score_mode: str = "stacked"
    base_score: Optional[float] = None
    stacked_score: Optional[float] = None
    stacked_meta: Optional[Dict[str, Any]] = None

    calibration_loaded: bool = False
    calibration_meta: Optional[Dict[str, Any]] = None
    reason_codes: List[str] = Field(default_factory=list)
    reason_details: Dict[str, Any] = Field(default_factory=dict)
    suppression_flags: List[str] = Field(default_factory=list)
    environment_penalties: Dict[str, Any] = Field(default_factory=dict)
    event_penalties: Dict[str, Any] = Field(default_factory=dict)
    liquidity_penalties: Dict[str, Any] = Field(default_factory=dict)
    breadth_penalties: Dict[str, Any] = Field(default_factory=dict)
    cross_asset_penalties: Dict[str, Any] = Field(default_factory=dict)
    stress_penalties: Dict[str, Any] = Field(default_factory=dict)
    adjusted_confidence_notes: List[str] = Field(default_factory=list)
    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None
    snapshot_id: Optional[str] = None
    snapshot_version: Optional[str] = None
    feature_version: Optional[str] = None
    signal_version: Optional[str] = None

    _EXTERNAL_EXCLUDE = {
        "thresholds", "base_score", "stacked_score", "stacked_meta",
        "environment_penalties", "event_penalties", "liquidity_penalties",
        "breadth_penalties", "cross_asset_penalties", "stress_penalties",
    }

    def external_payload(self) -> Dict[str, Any]:
        d = self.model_dump(exclude_none=True)
        for key in self._EXTERNAL_EXCLUDE:
            d.pop(key, None)
        if isinstance(d.get("calibration_meta"), dict):
            d["calibration_meta"].pop("base_score", None)
            d["calibration_meta"].pop("stacked_score", None)
        depth = (d.get("meta") or {}).get("depth_adjustments")
        if isinstance(depth, dict):
            for k in ("environment_penalties", "event_penalties", "liquidity_penalties",
                      "breadth_penalties", "cross_asset_penalties", "stress_penalties"):
                depth.pop(k, None)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _score_mode() -> str:
    v = os.getenv("FTIP_SCORE_MODE", "stacked") or "stacked"
    mode = v.strip().lower()
    return mode if mode in ("base", "stacked") else "stacked"


def _filter_upto(candles: List[Candle], as_of: str) -> List[Candle]:
    cutoff = _parse_date(as_of)
    out: List[Candle] = []
    for c in candles:
        try:
            d = _parse_date(c.timestamp)
        except Exception:
            continue
        if d <= cutoff:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Core signal computation
# ---------------------------------------------------------------------------


def compute_signal_for_symbol_from_candles(
    symbol: str,
    as_of: str,
    lookback: int,
    candles_all: List[Candle],
) -> SignalResponse:
    from api.alpha import build_canonical_features, build_canonical_signal
    from api.research import build_research_snapshot_from_candles

    candles_upto = _filter_upto(candles_all, as_of)

    if len(candles_upto) < 30:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough data to compute signal. "
                f"Need at least 30 bars <= {as_of}, got {len(candles_upto)}."
            ),
        )

    snapshot = build_research_snapshot_from_candles(
        symbol,
        _parse_date(as_of),
        lookback,
        candles_upto,
        source_hint="provided_market_bars",
        include_reference_context=True,
    )
    feature_payload = build_canonical_features(snapshot)
    signal_payload = build_canonical_signal(snapshot, feature_payload)

    return SignalResponse(
        symbol=(symbol or "").strip().upper(),
        as_of=as_of,
        lookback=int(lookback),
        effective_lookback=int(signal_payload.get("effective_lookback") or len(candles_upto)),
        regime=str(signal_payload.get("regime") or "CHOPPY"),
        thresholds=dict(signal_payload.get("thresholds") or {}),
        score=float(signal_payload.get("score") or 0.0),
        signal=str(signal_payload.get("signal") or "HOLD"),
        confidence=float(signal_payload.get("confidence") or 0.0),
        features={k: float(v) for k, v in (signal_payload.get("features") or {}).items()},
        notes=list(signal_payload.get("notes") or []),
        score_mode=str(signal_payload.get("score_mode") or "stacked"),
        base_score=_safe_float(signal_payload.get("base_score")),
        stacked_score=_safe_float(signal_payload.get("stacked_score")),
        stacked_meta=dict(signal_payload.get("stacked_meta") or {}),
        calibration_loaded=bool(signal_payload.get("calibration_loaded")),
        calibration_meta=signal_payload.get("calibration_meta"),
        reason_codes=list(signal_payload.get("reason_codes") or []),
        reason_details=dict(signal_payload.get("reason_details") or {}),
        suppression_flags=list(signal_payload.get("suppression_flags") or []),
        environment_penalties=dict(signal_payload.get("environment_penalties") or {}),
        event_penalties=dict(signal_payload.get("event_penalties") or {}),
        liquidity_penalties=dict(signal_payload.get("liquidity_penalties") or {}),
        breadth_penalties=dict(signal_payload.get("breadth_penalties") or {}),
        cross_asset_penalties=dict(signal_payload.get("cross_asset_penalties") or {}),
        stress_penalties=dict(signal_payload.get("stress_penalties") or {}),
        adjusted_confidence_notes=list(signal_payload.get("adjusted_confidence_notes") or []),
        entry_low=_safe_float(signal_payload.get("entry_low")),
        entry_high=_safe_float(signal_payload.get("entry_high")),
        stop_loss=_safe_float(signal_payload.get("stop_loss")),
        take_profit_1=_safe_float(signal_payload.get("take_profit_1")),
        take_profit_2=_safe_float(signal_payload.get("take_profit_2")),
        meta=dict(signal_payload.get("meta") or {}),
        snapshot_id=(signal_payload.get("meta") or {}).get("snapshot_id"),
        snapshot_version=(signal_payload.get("meta") or {}).get("snapshot_version"),
        feature_version=(signal_payload.get("meta") or {}).get("feature_version"),
        signal_version=(signal_payload.get("meta") or {}).get("signal_version"),
    )
