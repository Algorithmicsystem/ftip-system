from __future__ import annotations

from typing import Dict, List, Optional

from api.assistant.phase3.common import bounded_score, clamp, first_available, inverse_metric, mean, safe_float
from api.axiom.contracts import AxiomEngineInput, EngineScore
from api.axiom.engines.earnings_intelligence import compute_pess, evaluate_pess_flags


_VALUATION_WEIGHT = 0.14
_PROFITABILITY_WEIGHT = 0.22
_CASHFLOW_WEIGHT = 0.18
_BALANCE_WEIGHT = 0.18
_CAPS_WEIGHT = 0.14    # Rappaport CAPS replaces data_completeness
_EARNINGS_WEIGHT = 0.14


# ---------------------------------------------------------------------------
# 1.1 Penman-Schilit Earnings Integrity Score
# ---------------------------------------------------------------------------

def compute_eis(financials: dict) -> float:
    """Penman-Schilit Earnings Integrity Score (0–100, higher = higher earnings integrity).

    Grounded in Financial Statement Analysis (Penman), Financial Shenanigans
    (Schilit), Quality of Earnings (O'Glove).

    Formula: cash_earnings_ratio × 0.30 + accruals_quality × 0.40 + receivables_warning × 0.30
    """
    cfo = financials.get("cfo")
    net_income = financials.get("net_income")
    accruals_ratio = financials.get("accruals_ratio")       # (NI - CFO) / Assets
    receivables_growth = financials.get("receivables_growth")  # AR % change

    # Cash earnings ratio: CFO / |Net Income|; ≥1 is healthy
    if cfo is not None and net_income is not None and net_income != 0:
        cer = cfo / abs(net_income)
        cash_earnings_score = clamp((cer / 2.0) * 100.0, 0.0, 100.0)
    else:
        # Proxy via cash_flow_durability (already 0–100)
        cash_earnings_score = clamp(float(financials.get("cash_flow_durability") or 50.0), 0.0, 100.0)

    # Accruals quality: lower accruals ratio = higher integrity
    if accruals_ratio is not None:
        # bounded [-0.10, 0.15]; inverted so high accruals → low score
        accruals_score = clamp(((0.10 - float(accruals_ratio)) / 0.25) * 100.0, 0.0, 100.0)
    else:
        pfcf = float(financials.get("positive_fcf_ratio") or 0.5)
        accruals_score = clamp(pfcf * 100.0, 0.0, 100.0)

    # Receivables warning: rapid AR growth signals channel-stuffing
    if receivables_growth is not None:
        # bounded [0, 0.50]; inverted
        receivables_score = clamp(((0.20 - float(receivables_growth)) / 0.40) * 100.0, 0.0, 100.0)
    else:
        beat_rate = float(financials.get("earnings_beat_rate_4q") or 0.5)
        receivables_score = clamp(beat_rate * 100.0, 0.0, 100.0)

    base_eis = cash_earnings_score * 0.30 + accruals_score * 0.40 + receivables_score * 0.30
    base_eis = round(clamp(base_eis, 0.0, 100.0), 2)

    # Apply Schilit penalty when sufficient full-financial fields are present
    _SCHILIT_FIELDS = {
        "dso_change_yoy", "revenue_growth_yoy", "nonrecurring_income_pct",
        "capex_pct_revenue", "impairment_pct_assets", "related_party_revenue_pct",
    }
    if len(_SCHILIT_FIELDS & set(financials)) >= 2:
        try:
            from api.pe.schilit_analyzer import run_full_schilit_analysis
            schilit = run_full_schilit_analysis(financials)
            impact = float(schilit.get("composite_eis_impact") or 0.0)
            return round(clamp(base_eis - impact, 0.0, 100.0), 2)
        except Exception:
            pass

    return base_eis


# ---------------------------------------------------------------------------
# 1.2 Rappaport Competitive Advantage Period Score
# ---------------------------------------------------------------------------

def compute_caps(financials: dict, sector_context: dict) -> float:
    """Rappaport Competitive Advantage Period Score (0–100, higher = more durable moat).

    Grounded in Creating Shareholder Value (Rappaport).

    Formula: ROCE_spread × 0.40 + margin_stability × 0.25 + reinvestment_efficiency × 0.20
             + sector_advantage × 0.15
    """
    wacc = float(sector_context.get("wacc") or 0.08)
    roce = financials.get("return_on_capital_employed") or financials.get("return_on_equity")
    gross_margin = financials.get("gross_margin")
    operating_margin = financials.get("operating_margin")
    gross_margin_stability = financials.get("gross_margin_stability")  # std dev, lower = better
    revenue_growth = financials.get("revenue_growth_yoy")
    capex_to_revenue = financials.get("capex_to_revenue")
    sector_moat_score = float(sector_context.get("sector_moat_score") or 50.0)

    # ROCE spread: excess return above cost of capital, bounded [-0.05, 0.20]
    if roce is not None:
        spread = float(roce) - wacc
        roce_score = clamp(((spread + 0.05) / 0.25) * 100.0, 0.0, 100.0)
    else:
        roce_score = 50.0

    # Margin stability: lower std dev = more stable = higher score
    if gross_margin_stability is not None:
        margin_score = clamp((1.0 - float(gross_margin_stability) / 0.10) * 100.0, 0.0, 100.0)
    elif gross_margin is not None and operating_margin is not None and float(gross_margin) > 0:
        efficiency = float(operating_margin) / float(gross_margin)
        margin_score = clamp(efficiency * 100.0, 0.0, 100.0)
    else:
        margin_score = 50.0

    # Reinvestment efficiency: growth per unit of capex
    if revenue_growth is not None and capex_to_revenue is not None and float(capex_to_revenue) > 0:
        reinvest_score = clamp((float(revenue_growth) / float(capex_to_revenue)) * 50.0, 0.0, 100.0)
    elif revenue_growth is not None:
        reinvest_score = clamp((float(revenue_growth) + 0.10) / 0.40 * 100.0, 0.0, 100.0)
    else:
        reinvest_score = 50.0

    sector_adv_score = clamp(sector_moat_score, 0.0, 100.0)

    caps = (
        roce_score * 0.40
        + margin_score * 0.25
        + reinvest_score * 0.20
        + sector_adv_score * 0.15
    )
    return round(clamp(caps, 0.0, 100.0), 2)


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
            (bounded_score(candidate.quarterly_earnings_growth_yoy, low=-0.1, high=0.4), 0.1),
        ]
    )
    # EIS replaces raw earnings_quality — uses same fields as proxies
    _eis_inputs = {
        "cash_flow_durability": candidate.cash_flow_durability,
        "positive_fcf_ratio": candidate.positive_fcf_ratio,
        "earnings_beat_rate_4q": candidate.earnings_beat_rate_4q,
    }
    eis_score = compute_eis(_eis_inputs)
    earnings_quality_component = eis_score
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
    # CAPS replaces data_completeness — competitive advantage period assessment
    _caps_inputs = {
        "return_on_equity": candidate.return_on_equity,
        "gross_margin": candidate.gross_margin,
        "operating_margin": candidate.operating_margin,
        "revenue_growth_yoy": candidate.revenue_growth_yoy,
    }
    _sector = str(engine_input.source_context.get("symbol_meta", {}).get("sector") or "").lower()
    _SECTOR_WACC = {
        "technology": 0.09, "financials": 0.10, "healthcare": 0.08,
        "consumer discretionary": 0.09, "consumer staples": 0.07,
        "energy": 0.10, "industrials": 0.09, "utilities": 0.06,
        "real estate": 0.07, "communication services": 0.09, "materials": 0.09,
    }
    _SECTOR_MOAT = {
        "technology": 70.0, "healthcare": 65.0, "consumer staples": 62.0,
        "financials": 50.0, "energy": 40.0, "utilities": 45.0,
        "consumer discretionary": 55.0, "industrials": 48.0,
        "communication services": 60.0, "materials": 42.0, "real estate": 45.0,
    }
    _sector_ctx = {
        "wacc": _SECTOR_WACC.get(_sector, 0.08),
        "sector_moat_score": _SECTOR_MOAT.get(_sector, 50.0),
    }
    caps_component = compute_caps(_caps_inputs, _sector_ctx)
    _pess_data = engine_input.source_context.get("pess_data") or {}
    _pess = compute_pess(_pess_data)
    _days_to_earnings = engine_input.source_context.get("days_to_earnings")
    _pess_flags = evaluate_pess_flags(_pess, _days_to_earnings)
    # Apply earnings confidence penalty when PESS fires
    _pess_earnings_weight = _EARNINGS_WEIGHT
    if _pess_flags["earnings_stress_flag"]:
        _pess_earnings_weight = max(0.09, _EARNINGS_WEIGHT - 0.05)
    score = _weighted_average(
        [
            (valuation_gap_component, _VALUATION_WEIGHT),
            (profitability_quality_component, _PROFITABILITY_WEIGHT),
            (cashflow_quality_component, _CASHFLOW_WEIGHT),
            (balance_sheet_resilience_component, _BALANCE_WEIGHT),
            (caps_component, _CAPS_WEIGHT),
            (earnings_quality_component, _pess_earnings_weight),
        ]
    )
    component_values = {
        "valuation_gap_component": _rounded(valuation_gap_component),
        "profitability_quality_component": _rounded(profitability_quality_component),
        "cashflow_quality_component": _rounded(cashflow_quality_component),
        "balance_sheet_resilience_component": _rounded(balance_sheet_resilience_component),
        "caps_component": _rounded(caps_component),
        "earnings_quality_component": _rounded(earnings_quality_component),
        "pess_component": _rounded(_pess),
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
                caps_component,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    flags: List[str] = []
    if _pess_flags["earnings_stress_flag"]:
        flags.append("earnings_stress_pre_announcement")
    if _pess_flags["high_earnings_risk"]:
        flags.append("high_earnings_risk")
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
        f"Competitive advantage period (CAPS) is {_rounded(caps_component)} with coverage {round(coverage, 1)} / 100.",
    ]
    if earnings_quality_component is not None:
        summary_parts.append(f"Earnings integrity (EIS) is {_rounded(earnings_quality_component)}.")
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
