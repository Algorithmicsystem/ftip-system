from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from api import config
from .canonical_features import (
    CANONICAL_FEATURE_VERSION,
    classify_signal_regime,
)


CANONICAL_SIGNAL_VERSION = "phase9_canonical_signal_v1"
SIGNAL_SCHEMA_VERSION = 3


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _resolve_score_mode(mode_hint: Optional[str] = None) -> str:
    mode = (mode_hint or config.env("FTIP_SCORE_MODE", "stacked") or "stacked").strip().lower()
    return mode if mode in {"base", "stacked"} else "stacked"


def _stack_weights_for_regime(regime: str) -> Dict[str, float]:
    if regime == "TRENDING":
        return {"short": 0.15, "mid": 0.35, "long": 0.50}
    if regime == "HIGH_VOL":
        return {"short": 0.20, "mid": 0.40, "long": 0.40}
    return {"short": 0.45, "mid": 0.35, "long": 0.20}


def _load_calibration_for_symbol(symbol: Optional[str]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    sym = (symbol or "").strip().upper()
    raw_map = config.env("FTIP_CALIBRATION_JSON_MAP")
    if raw_map:
        try:
            mapping = json.loads(raw_map)
            if isinstance(mapping, dict):
                if sym and isinstance(mapping.get(sym), dict):
                    return True, mapping[sym]
                if isinstance(mapping.get("DEFAULT"), dict):
                    return True, mapping["DEFAULT"]
        except Exception:
            pass

    raw_single = config.env("FTIP_CALIBRATION_JSON")
    if raw_single:
        try:
            calibration = json.loads(raw_single)
            if isinstance(calibration, dict):
                return True, calibration
        except Exception:
            pass
    return False, None


def _thresholds_for_regime(regime: str, calibration: Optional[Dict[str, Any]]) -> Dict[str, float]:
    defaults = {
        "TRENDING": {"buy": 0.20, "sell": -0.20},
        "CHOPPY": {"buy": 0.30, "sell": -0.30},
        "HIGH_VOL": {"buy": 0.45, "sell": -0.45},
    }
    if calibration:
        thresholds_by_regime = calibration.get("thresholds_by_regime") or {}
        if isinstance(thresholds_by_regime.get(regime), dict):
            selected = thresholds_by_regime[regime]
            return {
                "buy": float(selected.get("buy", defaults[regime]["buy"])),
                "sell": float(selected.get("sell", defaults[regime]["sell"])),
            }
    return defaults.get(regime, defaults["CHOPPY"])


def build_signal_from_features(
    features: Dict[str, Any],
    *,
    symbol: Optional[str] = None,
    as_of: Optional[str] = None,
    lookback: int = 252,
    quality_score: Optional[int] = None,
    latest_close: Optional[float] = None,
    snapshot_meta: Optional[Dict[str, Any]] = None,
    mode_hint: Optional[str] = None,
) -> Dict[str, Any]:
    regime_label, regime_strength, signal_regime = classify_signal_regime(features)
    regime = str(features.get("signal_regime") or signal_regime or "CHOPPY").upper()
    if regime not in {"TRENDING", "CHOPPY", "HIGH_VOL"}:
        regime = signal_regime

    rsi14 = _safe_float(features.get("rsi14")) or 50.0
    mom_5 = _safe_float(features.get("mom_5")) or 0.0
    mom_21 = _safe_float(features.get("mom_21")) or 0.0
    mom_63 = _safe_float(features.get("mom_63")) or 0.0
    trend = _safe_float(features.get("trend_sma20_50")) or 0.0
    volume_z20 = _safe_float(features.get("volume_z20")) or 0.0
    sentiment = _safe_float(features.get("sentiment_score")) or 0.0
    maxdd = abs(_safe_float(features.get("maxdd_63d")) or 0.0)
    atr_pct = _safe_float(features.get("atr_pct")) or 0.0
    volatility_ann = _safe_float(features.get("volatility_ann")) or _safe_float(features.get("vol_63d")) or 0.0
    event_overhang_score = _safe_float(features.get("event_overhang_score")) or 0.0
    event_uncertainty_score = _safe_float(features.get("event_uncertainty_score")) or 0.0
    catalyst_burst_score = _safe_float(features.get("catalyst_burst_score")) or 0.0
    days_to_next_event = _safe_float(features.get("days_to_next_event"))
    earnings_window_flag = bool(features.get("earnings_window_flag"))
    post_event_instability_flag = bool(features.get("post_event_instability_flag"))
    implementation_fragility_score = _safe_float(features.get("implementation_fragility_score")) or 0.0
    liquidity_quality_score = _safe_float(features.get("liquidity_quality_score")) or 0.0
    friction_proxy_score = _safe_float(features.get("friction_proxy_score")) or 0.0
    execution_cleanliness_score = _safe_float(features.get("execution_cleanliness_score")) or 0.0
    breadth_confirmation_score = _safe_float(features.get("breadth_confirmation_score")) or 0.0
    internal_market_divergence_score = _safe_float(features.get("internal_market_divergence_score")) or 0.0
    leadership_concentration_score = _safe_float(features.get("leadership_concentration_score")) or 0.0
    benchmark_confirmation_score = _safe_float(features.get("benchmark_confirmation_score")) or 0.0
    sector_confirmation_score = _safe_float(features.get("sector_confirmation_score")) or 0.0
    macro_asset_alignment_score = _safe_float(features.get("macro_asset_alignment_score")) or 0.0
    cross_asset_conflict_score = _safe_float(features.get("cross_asset_conflict_score")) or 0.0
    cross_asset_divergence_score = _safe_float(features.get("cross_asset_divergence_score")) or 0.0
    market_stress_score = _safe_float(features.get("market_stress_score")) or 0.0
    spillover_risk_score = _safe_float(features.get("spillover_risk_score")) or 0.0
    correlation_breakdown_proxy = _safe_float(features.get("correlation_breakdown_proxy")) or 0.0
    volatility_shock_score = _safe_float(features.get("volatility_shock_score")) or 0.0
    stress_transition_score = _safe_float(features.get("stress_transition_score")) or 0.0
    unstable_environment_flag = bool(features.get("unstable_environment_flag"))
    defensive_regime_flag = bool(features.get("defensive_regime_flag"))

    rsi_sig = _clamp((rsi14 - 50.0) / 25.0, -1.0, 1.0)
    mom5_sig = _clamp(mom_5 / 0.10, -1.0, 1.0)
    mom21_sig = _clamp(mom_21 / 0.20, -1.0, 1.0)
    mom63_sig = _clamp(mom_63 / 0.30, -1.0, 1.0)
    trend_sig = _clamp(trend / 0.10, -1.0, 1.0)
    volume_sig = _clamp(volume_z20 / 3.0, -1.0, 1.0)
    sentiment_sig = _clamp(sentiment / 0.30, -1.0, 1.0)
    drawdown_pen = _clamp(maxdd / 0.35, 0.0, 1.0)
    vol_pen = _clamp((volatility_ann - 0.25) / 0.50, 0.0, 0.5)
    fragility_pen = _clamp((atr_pct - 0.04) / 0.08, 0.0, 0.4)
    event_penalty = _clamp(
        (
            0.50 * (event_overhang_score / 100.0)
            + 0.30 * (event_uncertainty_score / 100.0)
            + 0.20 * (catalyst_burst_score / 100.0)
        ),
        0.0,
        1.0,
    ) or 0.0
    liquidity_penalty = _clamp(
        (
            0.45 * (implementation_fragility_score / 100.0)
            + 0.30 * (friction_proxy_score / 100.0)
            + 0.25 * (100.0 - liquidity_quality_score) / 100.0
        ),
        0.0,
        1.0,
    ) or 0.0
    breadth_penalty = _clamp(
        (
            0.55 * (100.0 - breadth_confirmation_score) / 100.0
            + 0.25 * (internal_market_divergence_score / 100.0)
            + 0.20 * (leadership_concentration_score / 100.0)
        ),
        0.0,
        1.0,
    ) or 0.0
    cross_asset_penalty = _clamp(
        (
            0.45 * (cross_asset_conflict_score / 100.0)
            + 0.25 * (cross_asset_divergence_score / 100.0)
            + 0.15 * (100.0 - benchmark_confirmation_score) / 100.0
            + 0.15 * (100.0 - sector_confirmation_score) / 100.0
        ),
        0.0,
        1.0,
    ) or 0.0
    stress_penalty = _clamp(
        (
            0.35 * (market_stress_score / 100.0)
            + 0.25 * (spillover_risk_score / 100.0)
            + 0.20 * (correlation_breakdown_proxy / 100.0)
            + 0.20 * (volatility_shock_score / 100.0)
        ),
        0.0,
        1.0,
    ) or 0.0
    environment_support = _clamp(
        (
            0.38 * (breadth_confirmation_score / 100.0)
            + 0.24 * (benchmark_confirmation_score / 100.0)
            + 0.18 * (sector_confirmation_score / 100.0)
            + 0.20 * (macro_asset_alignment_score / 100.0)
        ),
        0.0,
        1.0,
    ) or 0.0

    short_component = _clamp(0.70 * mom5_sig + 0.30 * rsi_sig, -1.0, 1.0)
    mid_component = _clamp(0.60 * mom21_sig + 0.40 * trend_sig, -1.0, 1.0)
    long_component = _clamp(0.55 * mom63_sig + 0.45 * trend_sig, -1.0, 1.0)
    weights = _stack_weights_for_regime(regime)

    base_raw = (
        0.35 * mid_component
        + 0.30 * long_component
        + 0.15 * short_component
        + 0.10 * rsi_sig
        + 0.05 * volume_sig
        + 0.05 * sentiment_sig
        - 0.10 * drawdown_pen
    )
    stacked_raw = (
        weights["short"] * short_component
        + weights["mid"] * mid_component
        + weights["long"] * long_component
        + 0.08 * sentiment_sig
        + 0.04 * volume_sig
        - 0.10 * drawdown_pen
    )
    environment_penalty = _clamp(
        (
            0.28 * event_penalty
            + 0.24 * liquidity_penalty
            + 0.18 * breadth_penalty
            + 0.14 * cross_asset_penalty
            + 0.16 * stress_penalty
        ),
        0.0,
        0.45,
    ) or 0.0
    support_boost = _clamp((environment_support - 0.55) * 0.16, -0.05, 0.08) or 0.0
    base_score = _clamp(
        base_raw * (1.0 - vol_pen)
        - 0.05 * fragility_pen
        - environment_penalty
        + support_boost,
        -1.0,
        1.0,
    )
    stacked_score = _clamp(
        stacked_raw * (1.0 - vol_pen)
        - 0.08 * fragility_pen
        - environment_penalty
        + support_boost,
        -1.0,
        1.0,
    )

    score_mode = _resolve_score_mode(mode_hint)
    score = stacked_score if score_mode == "stacked" else base_score

    calibration_loaded, calibration = _load_calibration_for_symbol(symbol)
    thresholds = _thresholds_for_regime(regime, calibration)
    action = "HOLD"
    if score >= thresholds["buy"]:
        action = "BUY"
    elif score <= thresholds["sell"]:
        action = "SELL"

    quality_factor = _clamp((float(quality_score or 0) / 100.0) if quality_score is not None else 0.55, 0.1, 1.0)
    coverage_factor = 1.0
    if snapshot_meta and (snapshot_meta.get("coverage_status") in {"partial", "insufficient_history"}):
        coverage_factor = 0.75
    risk_factor = 0.65 if regime == "HIGH_VOL" else 0.85 if fragility_pen >= 0.15 else 1.0
    confidence_multiplier = _clamp(
        1.0
        - (
            0.24 * event_penalty
            + 0.22 * liquidity_penalty
            + 0.16 * breadth_penalty
            + 0.14 * cross_asset_penalty
            + 0.24 * stress_penalty
        ),
        0.30,
        1.0,
    ) or 1.0
    confidence = _clamp(abs(score) * quality_factor * coverage_factor * risk_factor, 0.0, 1.0)
    confidence = _clamp(confidence * confidence_multiplier, 0.0, 1.0)

    suppression_flags: List[str] = []
    if (event_penalty >= 0.45) or earnings_window_flag or post_event_instability_flag:
        suppression_flags.append("event_overhang")
    if liquidity_penalty >= 0.45 or (implementation_fragility_score >= 70):
        suppression_flags.append("implementation_fragility")
    if breadth_penalty >= 0.45 or (breadth_confirmation_score > 0 and breadth_confirmation_score < 45):
        suppression_flags.append("weak_breadth")
    if cross_asset_penalty >= 0.45 or (cross_asset_conflict_score >= 65):
        suppression_flags.append("cross_asset_conflict")
    if stress_penalty >= 0.45 or unstable_environment_flag or defensive_regime_flag:
        suppression_flags.append("market_stress")

    severe_suppression = (
        (event_overhang_score >= 82 and earnings_window_flag)
        or implementation_fragility_score >= 78
        or market_stress_score >= 78
        or len(suppression_flags) >= 3
    )
    if severe_suppression and action != "HOLD":
        action = "HOLD"
        score = _clamp(score * 0.55, -1.0, 1.0)
        confidence = _clamp(confidence * 0.72, 0.0, 1.0)

    event_penalties = {
        "event_overhang_score": event_overhang_score,
        "event_uncertainty_score": event_uncertainty_score,
        "catalyst_burst_score": catalyst_burst_score,
        "days_to_next_event": days_to_next_event,
        "earnings_window_flag": earnings_window_flag,
        "post_event_instability_flag": post_event_instability_flag,
        "penalty": round(float(event_penalty), 4),
    }
    liquidity_penalties = {
        "implementation_fragility_score": implementation_fragility_score,
        "liquidity_quality_score": liquidity_quality_score,
        "friction_proxy_score": friction_proxy_score,
        "execution_cleanliness_score": execution_cleanliness_score,
        "penalty": round(float(liquidity_penalty), 4),
    }
    breadth_penalties = {
        "breadth_confirmation_score": breadth_confirmation_score,
        "internal_market_divergence_score": internal_market_divergence_score,
        "leadership_concentration_score": leadership_concentration_score,
        "penalty": round(float(breadth_penalty), 4),
    }
    cross_asset_penalties = {
        "benchmark_confirmation_score": benchmark_confirmation_score,
        "sector_confirmation_score": sector_confirmation_score,
        "macro_asset_alignment_score": macro_asset_alignment_score,
        "cross_asset_conflict_score": cross_asset_conflict_score,
        "cross_asset_divergence_score": cross_asset_divergence_score,
        "penalty": round(float(cross_asset_penalty), 4),
    }
    stress_penalties = {
        "market_stress_score": market_stress_score,
        "spillover_risk_score": spillover_risk_score,
        "correlation_breakdown_proxy": correlation_breakdown_proxy,
        "volatility_shock_score": volatility_shock_score,
        "stress_transition_score": stress_transition_score,
        "unstable_environment_flag": unstable_environment_flag,
        "defensive_regime_flag": defensive_regime_flag,
        "penalty": round(float(stress_penalty), 4),
    }
    environment_penalties = {
        "combined_penalty": round(float(environment_penalty), 4),
        "support_boost": round(float(support_boost), 4),
    }

    notes: List[str] = []
    adjusted_confidence_notes: List[str] = []
    reason_codes: List[str] = []
    reason_details: Dict[str, str] = {}
    if trend_sig > 0:
        reason_codes.append("TREND_UP")
        reason_details["TREND_UP"] = "Trend structure remains constructive across the core moving-average stack."
        notes.append("Trend structure is positive.")
    elif trend_sig < 0:
        reason_codes.append("TREND_DOWN")
        reason_details["TREND_DOWN"] = "Trend structure remains negative across the core moving-average stack."
        notes.append("Trend structure is negative.")
    if mom21_sig > 0.2 or mom63_sig > 0.2:
        reason_codes.append("MOMENTUM_STRONG")
        reason_details["MOMENTUM_STRONG"] = "Momentum remains constructive across the intermediate horizons."
    elif mom21_sig < -0.2 or mom63_sig < -0.2:
        reason_codes.append("MOMENTUM_WEAK")
        reason_details["MOMENTUM_WEAK"] = "Momentum remains weak across the intermediate horizons."
    if sentiment_sig > 0.15:
        reason_codes.append("SENTIMENT_POSITIVE")
        reason_details["SENTIMENT_POSITIVE"] = "Sentiment is leaning positive relative to the recent baseline."
    elif sentiment_sig < -0.15:
        reason_codes.append("SENTIMENT_NEGATIVE")
        reason_details["SENTIMENT_NEGATIVE"] = "Sentiment is leaning negative relative to the recent baseline."
    if maxdd >= 0.15:
        reason_codes.append("DRAWDOWN_PRESSURE")
        reason_details["DRAWDOWN_PRESSURE"] = "Recent drawdown pressure is still constraining signal quality."
    if regime == "HIGH_VOL":
        reason_codes.append("HIGH_VOL_REGIME")
        reason_details["HIGH_VOL_REGIME"] = "The active regime remains high volatility, so confidence is suppressed."
        notes.append("Confidence reduced due to HIGH_VOL regime.")
    if "event_overhang" in suppression_flags:
        reason_codes.append("EVENT_OVERHANG")
        reason_details["EVENT_OVERHANG"] = "Event proximity or recent catalyst density is high enough that the setup is being treated as event-distorted rather than clean structural alpha."
        adjusted_confidence_notes.append("Confidence reduced because event proximity and catalyst density are elevated.")
    if "implementation_fragility" in suppression_flags:
        reason_codes.append("LIQUIDITY_FRAGILITY")
        reason_details["LIQUIDITY_FRAGILITY"] = "Liquidity quality, gap behavior, or implementation fragility is suppressing deployability even though the directional setup may look attractive."
        adjusted_confidence_notes.append("Implementation fragility is suppressing actionability and size confidence.")
    if "weak_breadth" in suppression_flags:
        reason_codes.append("WEAK_BREADTH")
        reason_details["WEAK_BREADTH"] = "Breadth and leadership internals are not confirming the setup cleanly."
        adjusted_confidence_notes.append("Breadth and participation are not broad enough to support a higher-confidence posture.")
    if "cross_asset_conflict" in suppression_flags:
        reason_codes.append("CROSS_ASSET_CONFLICT")
        reason_details["CROSS_ASSET_CONFLICT"] = "Sector, benchmark, or macro-asset context is contradicting the stock-level move."
        adjusted_confidence_notes.append("Cross-asset conflict is reducing trust in the raw directional score.")
    if "market_stress" in suppression_flags:
        reason_codes.append("MARKET_STRESS")
        reason_details["MARKET_STRESS"] = "Stress, spillover, or defensive-regime pressure is high enough that the setup is being handled more defensively."
        adjusted_confidence_notes.append("Market stress and spillover risk are suppressing confidence.")
    if severe_suppression:
        reason_codes.append("SEVERE_SUPPRESSION")
        reason_details["SEVERE_SUPPRESSION"] = "The canonical engine is overriding the raw directional impulse because combined event, liquidity, breadth, or stress signals are too hostile."
        adjusted_confidence_notes.append("The raw signal was overridden to HOLD because multiple realism filters tripped at once.")
    if not reason_codes:
        reason_codes.append("NEUTRAL_SCORE")
        reason_details["NEUTRAL_SCORE"] = "The canonical score remains close to neutral."
    notes.append(f"Regime: {regime}.")
    if calibration_loaded:
        notes.append("Using calibrated thresholds from FTIP_CALIBRATION_JSON_MAP/FTIP_CALIBRATION_JSON.")
    notes.append(f"Score mode: {score_mode.upper()} (canonical alpha core).")
    if suppression_flags:
        notes.append(
            "Suppression flags: " + ", ".join(flag.replace("_", " ") for flag in suppression_flags) + "."
        )
    notes.extend(adjusted_confidence_notes)

    atr_for_levels = atr_pct if atr_pct > 0 else 0.02
    entry_low = None
    entry_high = None
    stop_loss = None
    take_profit_1 = None
    take_profit_2 = None
    if latest_close is not None:
        entry_low = latest_close * (1.0 - 0.5 * atr_for_levels)
        entry_high = latest_close * (1.0 + 0.5 * atr_for_levels)
        if action == "BUY":
            stop_loss = latest_close * (1.0 - 1.5 * atr_for_levels)
            take_profit_1 = latest_close * (1.0 + 2.0 * atr_for_levels)
            take_profit_2 = latest_close * (1.0 + 3.0 * atr_for_levels)
        elif action == "SELL":
            stop_loss = latest_close * (1.0 + 1.5 * atr_for_levels)
            take_profit_1 = latest_close * (1.0 - 2.0 * atr_for_levels)
            take_profit_2 = latest_close * (1.0 - 3.0 * atr_for_levels)

    signal_payload = {
        "symbol": (symbol or "").strip().upper(),
        "as_of": as_of,
        "lookback": int(lookback),
        "effective_lookback": (
            int(snapshot_meta.get("available_history_bars"))
            if snapshot_meta and snapshot_meta.get("available_history_bars") is not None
            else int(lookback)
        ),
        "regime": regime,
        "thresholds": thresholds,
        "score": float(score),
        "signal": action,
        "confidence": float(confidence),
        "features": {k: v for k, v in features.items() if isinstance(v, (int, float)) and v is not None},
        "notes": notes,
        "score_mode": score_mode,
        "base_score": float(base_score),
        "stacked_score": float(stacked_score),
        "stacked_meta": {
            "stack_weights": weights,
            "components": {
                "short": short_component,
                "mid": mid_component,
                "long": long_component,
                "sentiment": sentiment_sig,
                "drawdown_penalty": drawdown_pen,
                "fragility_penalty": fragility_pen,
                "environment_support": environment_support,
            },
            "volatility_penalty": vol_pen,
            "environment_penalty": environment_penalty,
            "raw_before_penalties": stacked_raw,
        },
        "calibration_loaded": bool(calibration_loaded),
        "calibration_meta": (
            {
                "score_mode": score_mode,
                "base_score": float(base_score),
                "stacked_score": float(stacked_score),
                "symbol": (symbol or "").strip().upper() or None,
            }
            if calibration_loaded
            else None
        ),
        "reason_codes": reason_codes,
        "reason_details": reason_details,
        "suppression_flags": suppression_flags,
        "environment_penalties": environment_penalties,
        "event_penalties": event_penalties,
        "liquidity_penalties": liquidity_penalties,
        "breadth_penalties": breadth_penalties,
        "cross_asset_penalties": cross_asset_penalties,
        "stress_penalties": stress_penalties,
        "adjusted_confidence_notes": adjusted_confidence_notes,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "meta": {
            "signal_version": CANONICAL_SIGNAL_VERSION,
            "signal_schema_version": SIGNAL_SCHEMA_VERSION,
            "feature_version": CANONICAL_FEATURE_VERSION,
            "snapshot_id": snapshot_meta.get("snapshot_id") if snapshot_meta else None,
            "snapshot_version": snapshot_meta.get("snapshot_version") if snapshot_meta else None,
            "coverage_status": snapshot_meta.get("coverage_status") if snapshot_meta else None,
            "feature_hash": snapshot_meta.get("feature_hash") if snapshot_meta else None,
            "regime_label": regime_label,
            "regime_strength": regime_strength,
            "depth_adjustments": {
                "suppression_flags": suppression_flags,
                "environment_penalties": environment_penalties,
                "event_penalties": event_penalties,
                "liquidity_penalties": liquidity_penalties,
                "breadth_penalties": breadth_penalties,
                "cross_asset_penalties": cross_asset_penalties,
                "stress_penalties": stress_penalties,
                "adjusted_confidence_notes": adjusted_confidence_notes,
            },
        },
    }
    signal_payload["signal_hash"] = _hash_payload(signal_payload)
    return signal_payload


def build_canonical_signal(
    snapshot: Dict[str, Any],
    feature_payload: Dict[str, Any],
    *,
    quality_score: Optional[int] = None,
    mode_hint: Optional[str] = None,
) -> Dict[str, Any]:
    features = dict(feature_payload.get("features") or {})
    latest_close = _safe_float(features.get("last_close"))
    if latest_close is None:
        bars = list(snapshot.get("price_bars") or [])
        if bars:
            latest_close = _safe_float(bars[-1].get("close"))
    snapshot_meta = {
        "snapshot_id": snapshot.get("snapshot_id"),
        "snapshot_version": snapshot.get("snapshot_version"),
        "available_history_bars": snapshot.get("available_history_bars"),
        "coverage_status": (feature_payload.get("meta") or {}).get("coverage_status"),
        "feature_hash": (feature_payload.get("meta") or {}).get("feature_hash"),
    }
    return build_signal_from_features(
        features,
        symbol=snapshot.get("symbol"),
        as_of=snapshot.get("as_of_date"),
        lookback=int(snapshot.get("requested_lookback") or 252),
        quality_score=quality_score if quality_score is not None else _safe_float((snapshot.get("quality") or {}).get("quality_score")),
        latest_close=latest_close,
        snapshot_meta=snapshot_meta,
        mode_hint=mode_hint,
    )


def signals_daily_row(signal_payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(signal_payload.get("meta") or {})
    return {
        "action": signal_payload.get("signal"),
        "score": signal_payload.get("score"),
        "confidence": signal_payload.get("confidence"),
        "entry_low": signal_payload.get("entry_low"),
        "entry_high": signal_payload.get("entry_high"),
        "stop_loss": signal_payload.get("stop_loss"),
        "take_profit_1": signal_payload.get("take_profit_1"),
        "take_profit_2": signal_payload.get("take_profit_2"),
        "horizon_days": 21,
        "reason_codes": signal_payload.get("reason_codes") or [],
        "reason_details": signal_payload.get("reason_details") or {},
        "signal_version": SIGNAL_SCHEMA_VERSION,
        "effective_lookback": signal_payload.get("effective_lookback"),
        "regime": signal_payload.get("regime"),
        "thresholds": signal_payload.get("thresholds") or {},
        "score_mode": signal_payload.get("score_mode"),
        "base_score": signal_payload.get("base_score"),
        "stacked_score": signal_payload.get("stacked_score"),
        "snapshot_id": meta.get("snapshot_id"),
        "snapshot_version": meta.get("snapshot_version"),
        "signal_meta": {
            **meta,
            "signal_hash": signal_payload.get("signal_hash"),
            "signal_version": CANONICAL_SIGNAL_VERSION,
            "signal_schema_version": SIGNAL_SCHEMA_VERSION,
        },
    }
