from __future__ import annotations

from typing import Dict, List, Optional

from api.assistant.phase3.common import bounded_score, clamp, first_available, inverse_metric, mean, safe_float
from api.axiom.contracts import AxiomEngineInput, EngineScore


_VALUATION_WEIGHT = 0.18
_PROFITABILITY_WEIGHT = 0.24
_CASHFLOW_WEIGHT = 0.2
_BALANCE_WEIGHT = 0.2
_COMPLETENESS_WEIGHT = 0.18


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


def score_fundamental_reality(engine_input: AxiomEngineInput) -> EngineScore:
    candidate = engine_input.fundamental
    analyst_target_gap = None
    if candidate.latest_close not in (None, 0) and candidate.analyst_target_price is not None:
        analyst_target_gap = candidate.analyst_target_price / candidate.latest_close - 1.0
    growth_adjusted_pe = None
    if candidate.pe_ratio not in (None, 0) and candidate.revenue_growth_yoy is not None:
        growth_adjusted_pe = (candidate.revenue_growth_yoy * 100.0) / max(candidate.pe_ratio, 1.0)
    valuation_gap_component = _weighted_average(
        [
            (bounded_score(analyst_target_gap, low=-0.2, high=0.35), 0.45),
            (bounded_score(candidate.peg_ratio, low=0.6, high=3.0, invert=True), 0.25),
            (bounded_score(growth_adjusted_pe, low=-0.2, high=2.0), 0.3),
        ]
    )
    profitability_quality_component = _weighted_average(
        [
            (candidate.profitability_strength, 0.2),
            (bounded_score(candidate.gross_margin, low=0.15, high=0.7), 0.16),
            (bounded_score(candidate.operating_margin, low=-0.02, high=0.35), 0.24),
            (bounded_score(candidate.net_margin, low=-0.02, high=0.3), 0.16),
            (bounded_score(candidate.return_on_assets, low=0.0, high=0.18), 0.1),
            (bounded_score(candidate.return_on_equity, low=0.0, high=0.35), 0.14),
        ]
    )
    cashflow_quality_component = _weighted_average(
        [
            (candidate.cash_flow_durability, 0.35),
            (bounded_score(candidate.positive_fcf_ratio, low=0.0, high=1.0), 0.35),
            (bounded_score(candidate.free_cash_flow_margin, low=-0.05, high=0.3), 0.3),
        ]
    )
    balance_sheet_resilience_component = _weighted_average(
        [
            (candidate.balance_sheet_resilience, 0.34),
            (bounded_score(candidate.current_ratio, low=0.8, high=2.5), 0.2),
            (bounded_score(candidate.cash_ratio, low=0.1, high=1.5), 0.16),
            (bounded_score(inverse_metric(candidate.debt_to_equity, cap=3.0), low=0.0, high=1.0), 0.18),
            (bounded_score(inverse_metric(candidate.liabilities_to_assets, cap=1.0), low=0.0, high=1.0), 0.12),
        ]
    )
    data_completeness_component = _weighted_average(
        [
            (candidate.reporting_completeness_score, 0.3),
            (candidate.reporting_quality_proxy, 0.18),
            (candidate.provider_confidence, 0.16),
            (candidate.coverage_score, 0.18),
            (bounded_score(365.0 - min(candidate.filing_recency_days or 365.0, 365.0), low=0.0, high=365.0), 0.18),
        ]
    )
    score = _weighted_average(
        [
            (valuation_gap_component, _VALUATION_WEIGHT),
            (profitability_quality_component, _PROFITABILITY_WEIGHT),
            (cashflow_quality_component, _CASHFLOW_WEIGHT),
            (balance_sheet_resilience_component, _BALANCE_WEIGHT),
            (data_completeness_component, _COMPLETENESS_WEIGHT),
        ]
    )
    component_values = {
        "valuation_gap_component": _rounded(valuation_gap_component),
        "profitability_quality_component": _rounded(profitability_quality_component),
        "cashflow_quality_component": _rounded(cashflow_quality_component),
        "balance_sheet_resilience_component": _rounded(balance_sheet_resilience_component),
        "data_completeness_component": _rounded(data_completeness_component),
    }
    available_count = sum(1 for value in component_values.values() if value is not None)
    coverage = clamp(
        mean(
            [
                candidate.coverage_score,
                candidate.provider_confidence,
                (available_count / max(len(component_values), 1)) * 100.0,
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
                candidate.reporting_completeness_score,
                candidate.provider_confidence,
                data_completeness_component,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    flags: List[str] = []
    if candidate.filing_recency_days is not None and candidate.filing_recency_days > 180:
        flags.append("stale_filing_context")
    if candidate.coverage_score < 50:
        flags.append("thin_fundamental_coverage")
    if candidate.positive_fcf_ratio is not None and candidate.positive_fcf_ratio < 0.5:
        flags.append("weak_cashflow_durability")
    if candidate.debt_to_equity is not None and candidate.debt_to_equity > 1.5:
        flags.append("elevated_leverage")
    if not any(candidate.statement_coverage_flags.values()):
        flags.append("no_statement_backbone")

    if score is None:
        return EngineScore(
            score=None,
            confidence=round(confidence, 2),
            coverage=round(coverage, 2),
            status="unavailable" if coverage <= 0 else "partial",
            components={},
            flags=flags or ["fundamentals_unavailable"],
            summary="Fundamental Reality cannot score the setup because the current bundle does not contain enough filing or enrichment evidence.",
        )

    status = "available" if coverage >= 65 and confidence >= 55 else "partial"
    summary_parts = [
        f"Fundamental Reality reads {_rounded(score)} / 100.",
        f"Profitability quality is {_rounded(profitability_quality_component)} and balance-sheet resilience is {_rounded(balance_sheet_resilience_component)}.",
        f"Data completeness is {_rounded(data_completeness_component)} with coverage {round(coverage, 1)} / 100."
        if data_completeness_component is not None
        else f"Coverage is {round(coverage, 1)} / 100.",
    ]
    if candidate.weaknesses:
        summary_parts.append(f"Key weaknesses: {', '.join(candidate.weaknesses[:2])}.")
    elif candidate.strengths:
        summary_parts.append(f"Key strengths: {', '.join(candidate.strengths[:2])}.")
    return EngineScore(
        score=round(score, 2),
        confidence=round(confidence, 2),
        coverage=round(coverage, 2),
        status=status,
        components={key: value for key, value in component_values.items() if value is not None},
        flags=flags,
        summary=" ".join(summary_parts),
    )
