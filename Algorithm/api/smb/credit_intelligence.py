"""Phase 15.4: SMB Credit Intelligence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from api.assistant.phase3.common import clamp


@dataclass
class CreditIntelligence:
    entity_id: str
    borrowing_capacity_score: float
    estimated_max_loan_usd: float
    optimal_loan_structure: str
    rate_environment: str
    rate_environment_score: float
    credit_timing_recommendation: str
    dscr: float
    ltv_estimate: float


# ---------------------------------------------------------------------------
# Borrowing capacity
# ---------------------------------------------------------------------------

def compute_borrowing_capacity_score(
    financials: Dict[str, Any],
    credit_history: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Approximate credit quality from SMB financials."""
    ebitda = float(financials.get("ebitda") or 0.0)
    debt_service = float(financials.get("annual_debt_service") or 0.0)
    total_debt = float(financials.get("total_debt") or 0.0)
    ebitda_margin = float(financials.get("ebitda_margin") or 0.10)
    rev_growth = float(financials.get("revenue_growth_yoy") or 0.0)
    revenue = float(financials.get("revenue") or 0.0)

    # 1. DSCR (weight 0.35)
    if debt_service > 0 and ebitda > 0:
        dscr = ebitda / debt_service
    elif ebitda > 0:
        dscr = 3.0  # no debt = excellent
    else:
        dscr = 0.0

    if dscr > 1.5:
        dscr_score = 90.0
    elif dscr >= 1.25:
        dscr_score = 70.0
    elif dscr >= 1.0:
        dscr_score = 50.0
    else:
        dscr_score = 10.0

    # 2. Leverage ratio (weight 0.25): total_debt / EBITDA
    if ebitda > 0:
        leverage = total_debt / ebitda
    elif total_debt == 0:
        leverage = 0.0
    else:
        leverage = 5.0

    if leverage < 2.0:
        lev_score = 90.0
    elif leverage < 3.0:
        lev_score = 70.0
    elif leverage < 4.0:
        lev_score = 50.0
    else:
        lev_score = 30.0

    # 3. Profitability — EBITDA margin (weight 0.20)
    if ebitda_margin > 0.20:
        prof_score = 90.0
    elif ebitda_margin >= 0.15:
        prof_score = 75.0
    elif ebitda_margin >= 0.10:
        prof_score = 60.0
    else:
        prof_score = 40.0

    # 4. Revenue stability (weight 0.20)
    if rev_growth >= 0.05:
        rev_score = 80.0
    elif rev_growth >= 0.0:
        rev_score = 70.0
    else:
        rev_score = 30.0

    capacity_score = round(clamp(
        dscr_score * 0.35 + lev_score * 0.25 + prof_score * 0.20 + rev_score * 0.20,
        0.0, 100.0,
    ), 2)

    # Estimated max loan: ~3× annual EBITDA for strong borrowers
    max_loan = round(max(0.0, ebitda * 3.0 * (capacity_score / 100.0)), 2)

    # Loan structure
    if dscr > 1.5 and leverage < 2.0:
        structure = "term_loan"
    elif revenue > 0 and ebitda_margin < 0.10:
        structure = "revenue_based"
    elif leverage < 3.0:
        structure = "line_of_credit"
    else:
        structure = "sba"

    # Credit rating from capacity score
    if capacity_score >= 80: credit_rating = "excellent"
    elif capacity_score >= 65: credit_rating = "good"
    elif capacity_score >= 50: credit_rating = "adequate"
    elif capacity_score >= 35: credit_rating = "tight"
    else: credit_rating = "distressed"

    # DSCR interpretation
    if dscr >= 2.0:
        dscr_interpretation = f"DSCR of {dscr:.2f}x is strong — ample cash flow above debt obligations"
    elif dscr >= 1.25:
        dscr_interpretation = f"DSCR of {dscr:.2f}x is adequate — meeting debt service with limited buffer"
    elif dscr >= 1.0:
        dscr_interpretation = f"DSCR of {dscr:.2f}x is tight — barely covering debt obligations"
    else:
        dscr_interpretation = f"DSCR of {dscr:.2f}x is below 1.0 — unable to cover current debt service"

    # Max additional debt
    MIN_DSCR = 1.25
    if dscr <= MIN_DSCR or ebitda <= 0:
        max_additional_debt = 0.0
        max_additional_debt_formula = f"DSCR of {dscr:.2f}x provides no headroom above 1.25x minimum — additional debt not supported."
    else:
        headroom = dscr - MIN_DSCR
        annual_debt_service_est = ebitda / dscr if dscr > 0 else 0
        additional_capacity = headroom * annual_debt_service_est
        max_additional_debt = round(additional_capacity * 4, 2)
        max_additional_debt_formula = (
            f"DSCR of {dscr:.2f}x provides {headroom:.2f}x headroom above 1.25x minimum. "
            f"Existing debt service of ~${annual_debt_service_est:,.0f}/year means "
            f"${additional_capacity:,.0f}/year available for additional service, "
            f"supporting ~${max_additional_debt:,.0f} in additional debt at current rates."
        )

    # Lending recommendation
    if capacity_score >= 70:
        lending_recommendation = "Creditworthy — eligible for conventional term loans and lines of credit at market rates."
    elif capacity_score >= 50:
        lending_recommendation = "Adequate credit profile — eligible for SBA loans; expect standard covenant requirements."
    elif capacity_score >= 35:
        lending_recommendation = "Tight credit — revenue-based financing or secured lending only; reduce leverage before seeking new debt."
    else:
        lending_recommendation = "Distressed — lenders will require collateral; focus on cash flow improvement before borrowing."

    # Quick ratio proxy
    quick_ratio = round((revenue * 0.10) / max(total_debt * 0.20 + revenue * 0.05, 1.0), 2)
    # CCC proxy
    ccc_days = round(45.0 * (1.0 - ebitda_margin), 1)

    return {
        "borrowing_capacity_score": capacity_score,
        "dscr": round(dscr, 3),
        "leverage_ratio": round(leverage, 3),
        "estimated_max_loan_usd": max_loan,
        "optimal_loan_structure": structure,
        "component_scores": {
            "dscr_score": dscr_score,
            "leverage_score": lev_score,
            "profitability_score": prof_score,
            "revenue_stability_score": rev_score,
        },
        "credit_score": capacity_score,
        "credit_rating": credit_rating,
        "dscr_interpretation": dscr_interpretation,
        "max_additional_debt_usd": max_additional_debt,
        "max_additional_debt_formula": max_additional_debt_formula,
        "quick_ratio": quick_ratio,
        "cash_conversion_cycle_days": ccc_days,
        "lending_recommendation": lending_recommendation,
    }


# ---------------------------------------------------------------------------
# Rate environment
# ---------------------------------------------------------------------------

def compute_rate_environment_score(
    regime_label: str,
    ic_state: str = "INSUFFICIENT",
) -> Dict[str, Any]:
    """Rate environment for SMB borrowing (higher score = more favorable)."""
    regime = regime_label.upper()

    regime_map = {
        "RECOVERY": (75.0, "favorable", "Rates likely stable-to-falling — favorable time to borrow"),
        "TRENDING": (60.0, "neutral", "Economy strong; rates may rise — borrow at current fixed rates"),
        "CHOPPY": (55.0, "neutral", "Mixed rate signals — lock in fixed rate if borrowing"),
        "HIGH_VOL": (35.0, "elevated", "Credit tightening — expect higher spreads and stricter covenants"),
        "LIQUIDITY_FRACTURE": (15.0, "high", "Credit markets seizing — emergency liquidity only"),
        "COMPENSATION_CAPTURE": (30.0, "elevated", "Fed tightening — avoid variable rate debt"),
    }

    score, label, recommendation = regime_map.get(regime, (50.0, "neutral", "Monitor rate environment"))

    return {
        "score": score,
        "label": label,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Credit timing recommendation
# ---------------------------------------------------------------------------

def generate_credit_recommendation(
    capacity_score: float,
    rate_score: float,
    current_debt_level: float,
    revenue: float,
) -> str:
    """Matrix-based credit timing recommendation."""
    high_capacity = capacity_score > 70
    favorable_rates = rate_score > 60
    elevated_rates = rate_score < 40

    if not high_capacity:
        return "strengthen_balance_sheet"
    if favorable_rates:
        return "borrow_now"
    if elevated_rates:
        return "wait"
    return "borrow_now"  # high capacity + neutral rates


# ---------------------------------------------------------------------------
# Full credit intelligence
# ---------------------------------------------------------------------------

def build_credit_intelligence(
    entity_id: str,
    financials: Dict[str, Any],
    regime_label: str = "CHOPPY",
    credit_history: Optional[Dict] = None,
) -> CreditIntelligence:
    capacity_data = compute_borrowing_capacity_score(financials, credit_history)
    rate_data = compute_rate_environment_score(regime_label)

    revenue = float(financials.get("revenue") or 0.0) * 12
    total_debt = float(financials.get("total_debt") or 0.0)
    ltv = round(total_debt / max(revenue, 1.0), 4) if revenue > 0 else 0.0

    timing = generate_credit_recommendation(
        capacity_data["borrowing_capacity_score"],
        rate_data["score"],
        total_debt,
        revenue,
    )

    return CreditIntelligence(
        entity_id=entity_id,
        borrowing_capacity_score=capacity_data["borrowing_capacity_score"],
        estimated_max_loan_usd=capacity_data["estimated_max_loan_usd"],
        optimal_loan_structure=capacity_data["optimal_loan_structure"],
        rate_environment=rate_data["label"],
        rate_environment_score=rate_data["score"],
        credit_timing_recommendation=timing,
        dscr=capacity_data["dscr"],
        ltv_estimate=ltv,
    )
