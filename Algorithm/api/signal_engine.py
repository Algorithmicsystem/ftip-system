from __future__ import annotations

from typing import Dict, Optional

from api.alpha import build_signal_from_features


def compute_daily_signal(
    features: Dict[str, Optional[float]],
    quality_score: int,
    latest_close: Optional[float],
) -> Dict[str, object]:
    payload = build_signal_from_features(
        dict(features),
        quality_score=quality_score,
        latest_close=latest_close,
        snapshot_meta={"coverage_status": "available", "available_history_bars": 252},
    )
    return {
        "action": payload["signal"],
        "score": float(payload["score"]),
        "confidence": float(payload["confidence"]),
        "entry_low": payload.get("entry_low"),
        "entry_high": payload.get("entry_high"),
        "stop_loss": payload.get("stop_loss"),
        "take_profit_1": payload.get("take_profit_1"),
        "take_profit_2": payload.get("take_profit_2"),
        "reason_codes": payload.get("reason_codes") or [],
        "reason_details": payload.get("reason_details") or {},
        "regime": payload.get("regime"),
        "thresholds": payload.get("thresholds") or {},
        "score_mode": payload.get("score_mode"),
        "base_score": payload.get("base_score"),
        "stacked_score": payload.get("stacked_score"),
        "effective_lookback": payload.get("effective_lookback"),
        "meta": payload.get("meta") or {},
    }
