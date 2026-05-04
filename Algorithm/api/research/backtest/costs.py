from __future__ import annotations

from typing import Any, Dict

from .common import clamp, safe_float


def estimate_trade_cost(
    record: Dict[str, Any],
    *,
    cost_model: Dict[str, Any],
) -> Dict[str, Any]:
    action = str(record.get("final_signal") or record.get("signal_action") or "HOLD").upper()
    if action == "HOLD":
        return {
            "fee_bps": 0.0,
            "spread_bps": 0.0,
            "slippage_bps": 0.0,
            "gap_risk_bps": 0.0,
            "turnover_penalty_bps": 0.0,
            "total_bps": 0.0,
            "cost_rate": 0.0,
            "liquidity_bucket": "hold",
        }

    features = dict(record.get("feature_vector") or {})
    signal_payload = dict(record.get("signal_payload") or {})
    suppression_count = len(signal_payload.get("suppression_flags") or record.get("suppression_flags") or [])

    liquidity_quality = safe_float(features.get("liquidity_quality_score")) or 50.0
    implementation_fragility = safe_float(features.get("implementation_fragility_score")) or 50.0
    friction_proxy = safe_float(features.get("friction_proxy_score")) or 50.0
    gap_risk = safe_float(features.get("overnight_gap_risk_score")) or 40.0
    stress = safe_float(features.get("market_stress_score")) or 40.0
    turnover_penalty_input = safe_float(record.get("turnover_penalty"))
    if turnover_penalty_input is None:
        turnover_penalty_input = (implementation_fragility / 100.0) * 14.0

    fee_bps = safe_float(cost_model.get("fee_bps")) or 1.0
    spread_bps = safe_float(cost_model.get("spread_bps")) or 2.0
    slippage_bps = safe_float(cost_model.get("slippage_bps")) or 5.0
    overnight_gap_risk_bps = safe_float(cost_model.get("overnight_gap_risk_bps")) or 1.0

    spread_adj = spread_bps * (
        1.0
        + max(0.0, 62.0 - liquidity_quality) / 110.0
        + stress / 400.0
    )
    slippage_adj = slippage_bps * (
        1.0
        + implementation_fragility / 100.0
        + friction_proxy / 220.0
    )
    gap_adj = overnight_gap_risk_bps * (1.0 + gap_risk / 80.0 + suppression_count / 8.0)
    turnover_penalty_bps = max(0.0, turnover_penalty_input)
    total_bps = fee_bps + spread_adj + slippage_adj + gap_adj + turnover_penalty_bps

    if implementation_fragility >= 72 or liquidity_quality <= 28:
        liquidity_bucket = "fragile"
    elif implementation_fragility >= 48 or liquidity_quality <= 48:
        liquidity_bucket = "average"
    else:
        liquidity_bucket = "clean"

    return {
        "fee_bps": round(fee_bps, 4),
        "spread_bps": round(spread_adj, 4),
        "slippage_bps": round(slippage_adj, 4),
        "gap_risk_bps": round(gap_adj, 4),
        "turnover_penalty_bps": round(turnover_penalty_bps, 4),
        "total_bps": round(clamp(total_bps, 0.0, 500.0), 4),
        "cost_rate": round(clamp(total_bps, 0.0, 500.0) / 10000.0, 8),
        "liquidity_bucket": liquidity_bucket,
    }

