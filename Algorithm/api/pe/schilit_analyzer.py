"""Phase 13.1: Full Schilit 7-Category Earnings Shenanigan Detection.

Implements all 7 categories from Howard Schilit's Financial Shenanigans
with explicit, verifiable detection rules and direct EIS impact scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp


@dataclass
class SchilitFlag:
    category: int
    category_name: str
    triggered: bool
    severity: str                # "low" | "medium" | "high" | "critical"
    evidence: List[str]
    impact_on_eis: float         # points to deduct from EIS (0–20)
    management_intent: str       # "likely_intentional" | "possible" | "unclear"


def _severity_from_count(count: int) -> str:
    if count == 0:
        return "low"
    if count == 1:
        return "low"
    if count == 2:
        return "medium"
    if count == 3:
        return "high"
    return "critical"


def _intent_from_count(count: int) -> str:
    if count >= 3:
        return "likely_intentional"
    if count == 2:
        return "possible"
    return "unclear"


# ---------------------------------------------------------------------------
# Category 1 — Recording Revenue Too Soon
# ---------------------------------------------------------------------------

def flag_category_1(financials: Dict) -> SchilitFlag:
    signals: List[str] = []
    dso_change = financials.get("dso_change_yoy")
    bill_and_hold = bool(financials.get("bill_and_hold_indicator", False))
    rec_growth = float(financials.get("receivables_growth") or 0.0)
    rev_growth = float(financials.get("revenue_growth_yoy") or 0.0)

    if dso_change is not None and float(dso_change) > 0.15:
        signals.append(f"DSO increased {float(dso_change)*100:.1f}% YoY (threshold: 15%)")
    if bill_and_hold:
        signals.append("Bill-and-hold arrangements detected")
    if rec_growth - rev_growth > 0.20:
        signals.append(
            f"Receivables growth ({rec_growth*100:.1f}%) exceeds revenue growth "
            f"({rev_growth*100:.1f}%) by >{20}pp (channel-stuffing signal)"
        )

    n = len(signals)
    return SchilitFlag(
        category=1,
        category_name="Recording Revenue Too Soon",
        triggered=n > 0,
        severity=_severity_from_count(n),
        evidence=signals,
        impact_on_eis=clamp(n * 5.0, 0.0, 20.0) if n > 0 else 0.0,
        management_intent=_intent_from_count(n),
    )


# ---------------------------------------------------------------------------
# Category 2 — Recording Bogus Revenue
# ---------------------------------------------------------------------------

def flag_category_2(financials: Dict) -> SchilitFlag:
    signals: List[str] = []
    rev_growth = float(financials.get("revenue_growth_yoy") or 0.0)
    sector_median_growth = float(financials.get("sector_revenue_growth") or 0.05)
    related_party_pct = financials.get("related_party_revenue_pct")
    ccc_change = financials.get("cash_conversion_cycle_change")  # positive = lengthening

    if sector_median_growth > 0 and rev_growth > sector_median_growth * 3.0:
        signals.append(
            f"Revenue growth ({rev_growth*100:.1f}%) is >{3}× sector median "
            f"({sector_median_growth*100:.1f}%)"
        )
    if ccc_change is not None and float(ccc_change) > 0 and rev_growth > 0:
        signals.append(
            f"Cash conversion cycle lengthening ({float(ccc_change):.1f} days) "
            "while revenue growing — no cash backing"
        )
    if related_party_pct is not None and float(related_party_pct) > 0.30:
        signals.append(
            f"Related-party revenue concentration {float(related_party_pct)*100:.1f}% "
            "> 30% threshold"
        )

    n = len(signals)
    return SchilitFlag(
        category=2,
        category_name="Recording Bogus Revenue",
        triggered=n > 0,
        severity=_severity_from_count(n),
        evidence=signals,
        impact_on_eis=clamp(n * 6.0, 0.0, 20.0) if n > 0 else 0.0,
        management_intent=_intent_from_count(n),
    )


# ---------------------------------------------------------------------------
# Category 3 — Boosting Income with One-Time Gains
# ---------------------------------------------------------------------------

def flag_category_3(financials: Dict) -> SchilitFlag:
    signals: List[str] = []
    nonrecurring_pct = financials.get("nonrecurring_income_pct")
    core_earnings = financials.get("core_earnings")
    reported_earnings = financials.get("reported_net_income") or financials.get("net_income")

    if nonrecurring_pct is not None and float(nonrecurring_pct) > 0.10:
        signals.append(
            f"Non-recurring income = {float(nonrecurring_pct)*100:.1f}% of reported "
            "net income (threshold: 10%)"
        )
    if core_earnings is not None and reported_earnings is not None and reported_earnings != 0:
        ratio = float(core_earnings) / abs(float(reported_earnings))
        if ratio < 0.85:
            signals.append(
                f"Core earnings / reported earnings = {ratio:.2f} "
                "(< 0.85 indicates one-time boosts inflating results)"
            )

    n = len(signals)
    return SchilitFlag(
        category=3,
        category_name="Boosting Income with One-Time Gains",
        triggered=n > 0,
        severity=_severity_from_count(n),
        evidence=signals,
        impact_on_eis=clamp(n * 4.0, 0.0, 20.0) if n > 0 else 0.0,
        management_intent=_intent_from_count(n),
    )


# ---------------------------------------------------------------------------
# Category 4 — Shifting Current Expenses to a Later Period
# ---------------------------------------------------------------------------

def flag_category_4(financials: Dict) -> SchilitFlag:
    signals: List[str] = []
    capex_pct = financials.get("capex_pct_revenue")
    sector_capex_pct = financials.get("sector_capex_pct_revenue")
    rd_capitalization_change = financials.get("rd_capitalization_change_yoy")
    depreciation_life_change = financials.get("depreciation_life_change_yoy")  # positive = extending
    prepaid_growth = float(financials.get("prepaid_expenses_growth") or 0.0)
    rev_growth = float(financials.get("revenue_growth_yoy") or 0.0)

    if capex_pct is not None and sector_capex_pct is not None:
        delta = float(capex_pct) - float(sector_capex_pct)
        if delta > 0.03:
            signals.append(
                f"CapEx/revenue ({float(capex_pct)*100:.1f}%) exceeds sector peer "
                f"({float(sector_capex_pct)*100:.1f}%) by >{3}pp — potential expense deferral"
            )
    if rd_capitalization_change is not None and float(rd_capitalization_change) > 0.10:
        signals.append(
            f"R&D capitalization increased {float(rd_capitalization_change)*100:.1f}% YoY "
            "(converting expense to asset)"
        )
    if depreciation_life_change is not None and float(depreciation_life_change) > 0:
        signals.append(
            f"Depreciation life extended by {float(depreciation_life_change):.1f} years "
            "(stretching write-down period)"
        )
    if prepaid_growth - rev_growth > 0.15:
        signals.append(
            f"Prepaid expenses growing ({prepaid_growth*100:.1f}%) faster than "
            f"revenue ({rev_growth*100:.1f}%) by >{15}pp"
        )

    n = len(signals)
    return SchilitFlag(
        category=4,
        category_name="Shifting Current Expenses to a Later Period",
        triggered=n > 0,
        severity=_severity_from_count(n),
        evidence=signals,
        impact_on_eis=clamp(n * 4.5, 0.0, 20.0) if n > 0 else 0.0,
        management_intent=_intent_from_count(n),
    )


# ---------------------------------------------------------------------------
# Category 5 — Employing Other Techniques to Hide Expenses
# ---------------------------------------------------------------------------

def flag_category_5(financials: Dict) -> SchilitFlag:
    signals: List[str] = []
    intangibles_pct_acquisition = financials.get("intangibles_pct_acquisition_price")
    opex_pct_change = financials.get("opex_pct_revenue_change_yoy")
    stock_comp_pct_change = financials.get("stock_comp_pct_salary_change_yoy")
    pension_discount_rate_change = financials.get("pension_discount_rate_change_yoy")

    if intangibles_pct_acquisition is not None and float(intangibles_pct_acquisition) > 0.70:
        signals.append(
            f"Intangibles allocated = {float(intangibles_pct_acquisition)*100:.1f}% of "
            "acquisition price (> 70% signals goodwill inflation)"
        )
    if opex_pct_change is not None and float(opex_pct_change) < -0.05:
        signals.append(
            f"OpEx/revenue declined {abs(float(opex_pct_change))*100:.1f}pp YoY "
            "while product complexity likely unchanged — potential expense hiding"
        )
    if stock_comp_pct_change is not None and float(stock_comp_pct_change) > 0.20:
        signals.append(
            f"Stock-based compensation grew {float(stock_comp_pct_change)*100:.1f}% YoY "
            "relative to salary — shifting reportable compensation"
        )
    if pension_discount_rate_change is not None and float(pension_discount_rate_change) > 0.005:
        signals.append(
            f"Pension discount rate raised {float(pension_discount_rate_change)*100:.2f}pp "
            "— reduces reported pension expense"
        )

    n = len(signals)
    return SchilitFlag(
        category=5,
        category_name="Employing Other Techniques to Hide Expenses",
        triggered=n > 0,
        severity=_severity_from_count(n),
        evidence=signals,
        impact_on_eis=clamp(n * 3.5, 0.0, 20.0) if n > 0 else 0.0,
        management_intent=_intent_from_count(n),
    )


# ---------------------------------------------------------------------------
# Category 6 — Shifting Current Income to a Later Period (Cookie Jar)
# ---------------------------------------------------------------------------

def flag_category_6(financials: Dict) -> SchilitFlag:
    signals: List[str] = []
    accrued_liabilities_growth = float(financials.get("accrued_liabilities_growth") or 0.0)
    rev_growth = float(financials.get("revenue_growth_yoy") or 0.0)
    reserve_reversal_pct = financials.get("reserve_reversal_pct_income")
    restructuring_reversal = bool(financials.get("restructuring_charge_reversal", False))
    warranty_reserve_change = financials.get("warranty_reserve_pct_change_yoy")

    if accrued_liabilities_growth - rev_growth > 0.20:
        signals.append(
            f"Accrued liabilities growing ({accrued_liabilities_growth*100:.1f}%) far "
            f"faster than revenue ({rev_growth*100:.1f}%) — reserve building signal"
        )
    if reserve_reversal_pct is not None and float(reserve_reversal_pct) > 0.05:
        signals.append(
            f"Reserve reversals = {float(reserve_reversal_pct)*100:.1f}% of income "
            "(cookie-jar reserve release)"
        )
    if restructuring_reversal:
        signals.append("Prior restructuring charges partially reversed — income smoothing signal")
    if warranty_reserve_change is not None and float(warranty_reserve_change) < -0.10:
        signals.append(
            f"Warranty reserves declined {abs(float(warranty_reserve_change))*100:.1f}% "
            "without corresponding product changes (under-reserving)"
        )

    n = len(signals)
    return SchilitFlag(
        category=6,
        category_name="Shifting Current Income to a Later Period",
        triggered=n > 0,
        severity=_severity_from_count(n),
        evidence=signals,
        impact_on_eis=clamp(n * 3.0, 0.0, 20.0) if n > 0 else 0.0,
        management_intent=_intent_from_count(n),
    )


# ---------------------------------------------------------------------------
# Category 7 — Shifting Future Expenses to Current (Big Bath)
# ---------------------------------------------------------------------------

def flag_category_7(financials: Dict) -> SchilitFlag:
    signals: List[str] = []
    impairment_pct = financials.get("impairment_pct_assets")
    ceo_transition = bool(financials.get("ceo_transition_recent", False))
    consecutive_restructuring = bool(financials.get("consecutive_restructuring_charges", False))
    inventory_writedown_pct = financials.get("inventory_writedown_pct")
    rev_growth = float(financials.get("revenue_growth_yoy") or 0.0)

    if impairment_pct is not None and float(impairment_pct) > 0.10 and ceo_transition:
        signals.append(
            f"Impairment charges = {float(impairment_pct)*100:.1f}% of assets AND "
            "recent CEO transition — classic big-bath write-down"
        )
    elif impairment_pct is not None and float(impairment_pct) > 0.10:
        signals.append(
            f"Impairment charges = {float(impairment_pct)*100:.1f}% of assets "
            "(large write-down, possible big bath)"
        )
    if consecutive_restructuring:
        signals.append(
            "Restructuring charges taken in consecutive periods — "
            "recurring 'one-time' charges signal ongoing expense shifting"
        )
    if (inventory_writedown_pct is not None and float(inventory_writedown_pct) > 0.05
            and rev_growth >= 0):
        signals.append(
            f"Inventory write-down = {float(inventory_writedown_pct)*100:.1f}% "
            "despite non-declining revenue — timing manipulation signal"
        )

    n = len(signals)
    return SchilitFlag(
        category=7,
        category_name="Shifting Future Expenses to Current Period (Big Bath)",
        triggered=n > 0,
        severity=_severity_from_count(n),
        evidence=signals,
        impact_on_eis=clamp(n * 2.5, 0.0, 20.0) if n > 0 else 0.0,
        management_intent=_intent_from_count(n),
    )


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

_CATEGORY_FNS = [
    flag_category_1,
    flag_category_2,
    flag_category_3,
    flag_category_4,
    flag_category_5,
    flag_category_6,
    flag_category_7,
]


def run_full_schilit_analysis(
    financials: Dict,
    sector_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run all 7 Schilit categories and return composite results."""
    if sector_context:
        merged = {**financials, **sector_context}
    else:
        merged = financials

    flags: Dict[int, SchilitFlag] = {}
    for fn in _CATEGORY_FNS:
        flag = fn(merged)
        flags[flag.category] = flag

    triggered = [f for f in flags.values() if f.triggered]
    triggered_count = len(triggered)

    severity_dist: Dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for f in triggered:
        severity_dist[f.severity] = severity_dist.get(f.severity, 0) + 1

    composite_impact = sum(f.impact_on_eis for f in triggered)
    schilit_score = round(clamp(100.0 - composite_impact, 0.0, 100.0), 2)

    most_serious: Optional[int] = None
    if triggered:
        most_serious = max(triggered, key=lambda f: f.impact_on_eis).category

    management_integrity = max(0.0, 100.0 - triggered_count * 15.0) if triggered_count > 0 else 100.0

    intentional_count = sum(
        1 for f in triggered if f.management_intent == "likely_intentional"
    )
    management_integrity = max(0.0, management_integrity - intentional_count * 10.0)

    if schilit_score > 80:
        recommendation = "clean"
    elif schilit_score > 60:
        recommendation = "watch"
    elif schilit_score > 40:
        recommendation = "investigate"
    else:
        recommendation = "avoid"

    return {
        "schilit_score": schilit_score,
        "triggered_flags": triggered_count,
        "categories": flags,
        "severity_distribution": severity_dist,
        "most_serious_category": most_serious,
        "composite_eis_impact": round(composite_impact, 2),
        "management_integrity_score": round(management_integrity, 2),
        "audit_risk_flag": triggered_count >= 3,
        "recommendation": recommendation,
    }
