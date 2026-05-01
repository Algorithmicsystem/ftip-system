from __future__ import annotations

from typing import Any, Dict

from .common import clamp, safe_float


_CLEANLINESS_BONUS = {
    "clean": 14.0,
    "mixed_clean": 8.0,
    "mixed": 2.0,
    "noisy": -8.0,
}


def build_execution_quality(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    vol = safe_float(snapshot.get("realized_vol_21d")) or 0.24
    atr_pct = safe_float(snapshot.get("atr_pct")) or 0.04
    gap_pct = abs(safe_float(snapshot.get("gap_pct")) or 0.0)
    gap_instability = safe_float(snapshot.get("gap_instability_10d")) or 0.22
    volume_anomaly = safe_float(snapshot.get("volume_anomaly")) or 1.0
    horizon_days = safe_float(snapshot.get("horizon_days")) or 21.0
    freshness = str(snapshot.get("freshness_status") or "unknown")
    urgency = str(snapshot.get("urgency_level") or "measured")
    patience = str(snapshot.get("patience_level") or "high")
    preferred_posture = str(snapshot.get("execution_preferred_posture") or "staged_watch")
    cleanliness = str(snapshot.get("signal_cleanliness") or "mixed_clean")

    friction_penalty = clamp(
        (vol * 145.0) + (atr_pct * 320.0) + (gap_pct * 420.0) + (gap_instability * 24.0)
        - max(0.0, volume_anomaly - 1.0) * 10.0,
        0.0,
        100.0,
    )
    turnover_penalty = 0.0
    if horizon_days <= 5:
        turnover_penalty += 64.0
    elif horizon_days <= 21:
        turnover_penalty += 42.0
    elif horizon_days <= 63:
        turnover_penalty += 28.0
    else:
        turnover_penalty += 18.0
    if urgency in {"high", "immediate"}:
        turnover_penalty += 8.0
    if patience in {"low", "very_low"}:
        turnover_penalty += 6.0
    if freshness in {"mixed_stale", "stale", "stale_but_usable"}:
        turnover_penalty += 10.0

    execution_quality_score = clamp(
        90.0
        - (friction_penalty * 0.52)
        - (turnover_penalty * 0.36)
        + _CLEANLINESS_BONUS.get(cleanliness, 4.0),
        0.0,
        100.0,
    )

    wait_for_better_entry_flag = (
        preferred_posture == "wait_for_confirmation"
        or execution_quality_score < 58.0
        or friction_penalty >= 52.0
    )
    confirmation_preferred_flag = (
        wait_for_better_entry_flag or preferred_posture in {"staged_watch", "wait_for_confirmation"}
    )

    return {
        "execution_quality_score": round(execution_quality_score, 2),
        "friction_penalty": round(friction_penalty, 2),
        "turnover_penalty": round(turnover_penalty, 2),
        "urgency_level": urgency,
        "patience_level": patience,
        "wait_for_better_entry_flag": wait_for_better_entry_flag,
        "confirmation_preferred_flag": confirmation_preferred_flag,
        "execution_cleanliness": cleanliness,
    }
