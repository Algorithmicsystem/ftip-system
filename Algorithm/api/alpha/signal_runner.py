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
        # Deterministic intelligence fields — computed here, never from LLM
        d["system_confidence"] = _compute_system_confidence(self)
        ev_for, ev_against = _build_evidence(self)
        d["evidence_for"] = ev_for
        d["evidence_against"] = ev_against
        return d


# ---------------------------------------------------------------------------
# Intelligence enrichment — deterministic, no LLM
# ---------------------------------------------------------------------------

_STRENGTH_THRESHOLDS = (0.65, 0.35)  # (strong, moderate) — below moderate is weak


def _strength_label(magnitude: float) -> str:
    if magnitude >= _STRENGTH_THRESHOLDS[0]:
        return "strong"
    if magnitude >= _STRENGTH_THRESHOLDS[1]:
        return "moderate"
    return "weak"


# Feature → (long_positive, hint) mapping used by evidence builder.
# long_positive=True means a positive value supports a BUY signal.
_FEATURE_EVIDENCE_MAP: List[tuple] = [
    ("mom_5",           True,  "very short-term momentum"),
    ("mom_21",          True,  "1-month momentum"),
    ("mom_63",          True,  "quarterly momentum"),
    ("mom_126",         True,  "6-month momentum"),
    ("mom_252",         True,  "12-month momentum"),
    ("rsi14",           None,  "RSI 14"),          # non-linear — handled specially
    ("trend_sma20_50",  True,  "SMA 20/50 trend"),
    ("trend_r2_63d",    True,  "trend quality (63d R²)"),
    ("volume_z20",      True,  "volume surge"),
    ("sentiment_score", True,  "news sentiment"),
    ("sentiment_surprise", True, "sentiment surprise"),
    ("mom_vol_adj_21d", True,  "risk-adjusted momentum"),
    ("trend_slope_63d", True,  "medium-term trend slope"),
]

_PENALTY_FEATURE_MAP: List[tuple] = [
    ("event_overhang_score",          "event risk overhang"),
    ("market_stress_score",           "market stress"),
    ("implementation_fragility_score","implementation fragility"),
    ("cross_asset_conflict_score",    "cross-asset conflict"),
    ("breadth_confirmation_score",    None),  # inverted: low = bearish breadth
]


def _compute_system_confidence(sig: "SignalResponse") -> float:
    """Derive a 0–100 confidence score purely from model outputs, no LLM."""
    base = min(abs(sig.score), 1.0) * 100.0
    # Regime quality bonus
    if sig.regime in ("TRENDING", "trend"):
        base = min(base + 5.0, 100.0)
    # Each suppression/adjustment note erodes confidence
    note_drag = min(len(sig.adjusted_confidence_notes) * 4.0, 25.0)
    # Total penalty weight across all penalty dicts
    all_penalties = {
        **sig.environment_penalties,
        **sig.event_penalties,
        **sig.liquidity_penalties,
        **sig.breadth_penalties,
        **sig.cross_asset_penalties,
        **sig.stress_penalties,
    }
    penalty_sum = sum(
        float(v) for v in all_penalties.values()
        if isinstance(v, (int, float)) and v > 0
    )
    penalty_drag = min(penalty_sum * 15.0, 30.0)
    result = max(base - note_drag - penalty_drag, 0.0)
    return round(result, 1)


def _build_evidence(
    sig: "SignalResponse",
) -> tuple:
    """Return (evidence_for, evidence_against) lists with strength labels."""
    features = sig.features or {}
    is_buy = sig.signal == "BUY"
    is_sell = sig.signal == "SELL"

    for_items: List[Dict[str, Any]] = []
    against_items: List[Dict[str, Any]] = []

    for feat_name, long_positive, hint in _FEATURE_EVIDENCE_MAP:
        raw = features.get(feat_name)
        if raw is None:
            continue
        val = float(raw)

        if feat_name == "rsi14":
            # RSI 14: >60 supports buy, <40 supports sell
            if val >= 60.0:
                magnitude = min((val - 50.0) / 50.0, 1.0)
                item = {"feature": feat_name, "description": f"RSI={val:.1f} (bullish momentum)", "value": round(val, 2), "strength_label": _strength_label(magnitude)}
                (for_items if is_buy else against_items).append(item)
            elif val <= 40.0:
                magnitude = min((50.0 - val) / 50.0, 1.0)
                item = {"feature": feat_name, "description": f"RSI={val:.1f} (bearish momentum)", "value": round(val, 2), "strength_label": _strength_label(magnitude)}
                (for_items if is_sell else against_items).append(item)
            continue

        if long_positive is None:
            continue

        supports_signal = (long_positive and is_buy and val > 0) or \
                          (long_positive and is_sell and val < 0) or \
                          (not long_positive and is_buy and val < 0) or \
                          (not long_positive and is_sell and val > 0)

        # Normalize magnitude to 0-1 using typical max ranges
        _ranges = {
            "mom_5": 0.05, "mom_21": 0.15, "mom_63": 0.25,
            "mom_126": 0.35, "mom_252": 0.50,
            "trend_sma20_50": 0.10, "trend_r2_63d": 1.0,
            "volume_z20": 3.0, "sentiment_score": 1.0,
            "sentiment_surprise": 0.5, "mom_vol_adj_21d": 1.5,
            "trend_slope_63d": 0.005,
        }
        magnitude = min(abs(val) / max(_ranges.get(feat_name, 1.0), 1e-9), 1.0)
        if magnitude < 0.05:
            continue  # noise

        item = {
            "feature": feat_name,
            "description": f"{hint}: {round(val, 4)}",
            "value": round(val, 4),
            "strength_label": _strength_label(magnitude),
        }
        (for_items if supports_signal else against_items).append(item)

    # Sort by strength: strong → moderate → weak
    _order = {"strong": 0, "moderate": 1, "weak": 2}
    for_items.sort(key=lambda x: _order.get(x["strength_label"], 3))
    against_items.sort(key=lambda x: _order.get(x["strength_label"], 3))

    return for_items[:6], against_items[:6]


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
