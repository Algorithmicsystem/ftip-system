from __future__ import annotations
from api.assistant.phase3.common import clamp

_PESS_WINDOW_DAYS = 60  # pre-earnings announcement window (days)
_PESS_WARNING_THRESHOLD = 65.0
_PESS_HIGH_RISK_THRESHOLD = 80.0

def compute_pess(earnings_data: dict) -> float:
    """Pre-Earnings Stress Score (0-100). Higher = more earnings stress/risk."""
    eis_trend_delta = earnings_data.get("eis_trend_delta")           # EIS change over 2Q, bounded [-30, +30]
    accruals_acceleration = earnings_data.get("accruals_acceleration") # accruals change, bounded [-0.05, 0.10]
    guidance_revision_velocity = earnings_data.get("guidance_revision_velocity")  # z-score, bounded [-3, +3]
    insider_sell_ratio = earnings_data.get("insider_sell_ratio")     # 0-4, > 2x baseline = stress

    components = {}

    # Component 1: eis_trend_delta — falling EIS = stress, inverted
    if eis_trend_delta is not None:
        delta_bounded = clamp(float(eis_trend_delta), -30.0, 30.0)
        # invert: falling delta → higher PESS
        components["eis_trend_delta"] = ((30.0 - delta_bounded) / 60.0) * 100.0

    # Component 2: accruals_acceleration — rising accruals = stress
    if accruals_acceleration is not None:
        acc_bounded = clamp(float(accruals_acceleration), -0.05, 0.10)
        # normalize: rising = higher score
        components["accruals_acceleration"] = ((acc_bounded + 0.05) / 0.15) * 100.0

    # Component 3: guidance_revision_velocity — negative revisions = stress, inverted z-score
    if guidance_revision_velocity is not None:
        grv_bounded = clamp(float(guidance_revision_velocity), -3.0, 3.0)
        # invert: negative (downward revisions) → high PESS
        components["guidance_revision_velocity"] = ((3.0 - grv_bounded) / 6.0) * 100.0

    # Component 4: insider_sell_ratio — > 2x baseline = stress, bounded [0, 4]
    if insider_sell_ratio is not None:
        isr_bounded = clamp(float(insider_sell_ratio), 0.0, 4.0)
        components["insider_sell_ratio"] = (isr_bounded / 4.0) * 100.0

    if not components:
        return 50.0

    canonical_weights = {
        "eis_trend_delta": 0.35,
        "accruals_acceleration": 0.30,
        "guidance_revision_velocity": 0.20,
        "insider_sell_ratio": 0.15,
    }
    total_w = sum(canonical_weights[k] for k in components)
    if total_w <= 0:
        return 50.0
    pess = sum(components[k] * canonical_weights[k] for k in components) / total_w
    return round(clamp(pess, 0.0, 100.0), 2)


def evaluate_pess_flags(pess: float, days_to_earnings: int | None) -> dict:
    """Evaluate whether PESS triggers stress warnings based on earnings proximity."""
    in_window = days_to_earnings is not None and 5 <= int(days_to_earnings) <= _PESS_WINDOW_DAYS
    stress_flag = in_window and pess > _PESS_WARNING_THRESHOLD
    high_risk_flag = in_window and pess > _PESS_HIGH_RISK_THRESHOLD
    return {
        "earnings_stress_flag": stress_flag,
        "high_earnings_risk": high_risk_flag,
        "in_pre_earnings_window": in_window,
        "pess": round(pess, 2),
    }
