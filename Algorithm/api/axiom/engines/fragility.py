from __future__ import annotations

from typing import List, Optional

from api.assistant.phase3.common import bounded_score, clamp, mean
from api.axiom.contracts import AxiomEngineInput, EngineScore


def _rounded(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 2)


def _weighted_average(items: List[tuple[Optional[float], float]]) -> Optional[float]:
    usable = [(value, weight) for value, weight in items if value is not None and weight > 0]
    if not usable:
        return None
    total_weight = sum(weight for _value, weight in usable)
    if total_weight <= 0:
        return None
    return sum(float(value) * float(weight) for value, weight in usable) / total_weight


def score_critical_fragility(engine_input: AxiomEngineInput) -> EngineScore:
    candidate = engine_input.fragility
    volatility_instability_component = _weighted_average(
        [
            (candidate.volatility_stress_score, 0.28),
            (bounded_score(candidate.realized_vol_21d, low=0.12, high=0.6), 0.24),
            (bounded_score(candidate.realized_vol_63d, low=0.12, high=0.5), 0.16),
            (bounded_score(candidate.vol_of_vol_proxy, low=0.0, high=1.0), 0.16),
            (candidate.instability_score, 0.16),
        ]
    )
    gap_jump_risk_component = _weighted_average(
        [
            (candidate.gap_instability_10d and bounded_score(candidate.gap_instability_10d, low=0.0, high=2.0), 0.36),
            (candidate.overnight_gap_risk_score, 0.32),
            (bounded_score(candidate.abs_gap_mean_10d, low=0.0, high=0.05), 0.18),
            (bounded_score(abs(candidate.gap_pct or 0.0), low=0.0, high=0.08), 0.14),
        ]
    )
    drawdown_fragility_component = _weighted_average(
        [
            (candidate.drawdown_sensitivity_score, 0.28),
            (bounded_score(abs(candidate.maxdd_21d or 0.0), low=0.0, high=0.18), 0.22),
            (bounded_score(abs(candidate.maxdd_63d or 0.0), low=0.0, high=0.3), 0.22),
            (bounded_score(abs(candidate.maxdd_126d or 0.0), low=0.0, high=0.45), 0.14),
            (bounded_score(candidate.downside_asymmetry_21d, low=0.8, high=2.2), 0.14),
        ]
    )
    crowding_fragility_component = _weighted_average(
        [
            (candidate.narrative_crowding_score, 0.45),
            (candidate.event_overhang_score, 0.35),
            (candidate.event_uncertainty_score, 0.2),
        ]
    )
    liquidity_fragility_component = _weighted_average(
        [
            (candidate.implementation_fragility_score, 0.3),
            (candidate.tradability_caution_score, 0.18),
            (candidate.friction_proxy_score, 0.18),
            (candidate.anomaly_pressure_score, 0.14),
            (candidate.overnight_gap_risk_score, 0.12),
            (bounded_score(candidate.liquidity_quality_score, low=0.0, high=100.0, invert=True), 0.08),
        ]
    )
    regime_transition_risk_component = _weighted_average(
        [
            (candidate.regime_transition_score, 0.3),
            (candidate.regime_instability_score, 0.18),
            (candidate.market_stress_score, 0.22),
            (candidate.cross_asset_conflict_score, 0.16),
            (bounded_score(candidate.breadth_confirmation_score, low=0.0, high=100.0, invert=True), 0.14),
        ]
    )
    score = _weighted_average(
        [
            (volatility_instability_component, 0.23),
            (gap_jump_risk_component, 0.16),
            (drawdown_fragility_component, 0.2),
            (crowding_fragility_component, 0.12),
            (liquidity_fragility_component, 0.17),
            (regime_transition_risk_component, 0.12),
        ]
    )
    component_values = {
        "volatility_instability_component": _rounded(volatility_instability_component),
        "gap_jump_risk_component": _rounded(gap_jump_risk_component),
        "drawdown_fragility_component": _rounded(drawdown_fragility_component),
        "crowding_fragility_component": _rounded(crowding_fragility_component),
        "liquidity_fragility_component": _rounded(liquidity_fragility_component),
        "regime_transition_risk_component": _rounded(regime_transition_risk_component),
    }
    available_count = sum(1 for value in component_values.values() if value is not None)
    coverage = clamp(
        mean(
            [
                candidate.coverage_score,
                (available_count / max(len(component_values), 1)) * 100.0,
                100.0 if candidate.signal_fragility_score is not None else 55.0,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    confidence = clamp(
        mean(
            [
                coverage,
                candidate.signal_fragility_score,
                candidate.clean_setup_score and (100.0 - candidate.clean_setup_score),
                candidate.instability_score,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    flags: List[str] = list(candidate.suppression_flags)
    if candidate.event_risk_classification in {"high_event_risk", "event_distorted"}:
        flags.append("event_distortion")
    if (candidate.market_stress_score or 0.0) >= 65.0:
        flags.append("market_stress")
    if (candidate.cross_asset_conflict_score or 0.0) >= 60.0:
        flags.append("cross_asset_conflict")
    if (candidate.implementation_fragility_score or 0.0) >= 65.0:
        flags.append("implementation_fragility")
    if (candidate.regime_transition_score or 0.0) >= 65.0:
        flags.append("regime_transition")
    flags = sorted(set(flag for flag in flags if flag))

    if score is None:
        return EngineScore(
            score=None,
            confidence=round(confidence, 2),
            coverage=round(coverage, 2),
            status="unavailable" if coverage <= 0 else "partial",
            components={},
            flags=flags or ["fragility_unavailable"],
            summary="Critical Fragility cannot score the setup because the current bundle does not contain enough realized instability or implementation-risk evidence.",
        )

    status = "available" if coverage >= 65 and confidence >= 55 else "partial"
    summary_parts = [
        f"Critical Fragility reads {_rounded(score)} / 100, where higher means more path-risk and deployability drag.",
        f"Volatility instability is {_rounded(volatility_instability_component)}, drawdown fragility is {_rounded(drawdown_fragility_component)}, and liquidity fragility is {_rounded(liquidity_fragility_component)}.",
        f"Coverage is {round(coverage, 1)} / 100 and confidence is {round(confidence, 1)} / 100.",
    ]
    if flags:
        summary_parts.append(f"Primary fragility flags: {', '.join(flags[:4])}.")
    return EngineScore(
        score=round(score, 2),
        confidence=round(confidence, 2),
        coverage=round(coverage, 2),
        status=status,
        components={key: value for key, value in component_values.items() if value is not None},
        flags=flags,
        summary=" ".join(summary_parts),
    )
