"""Phase 25: Deal Attractiveness Score (DAS) engine using real fundamentals."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp


@dataclass
class DASResult:
    symbol: str
    das_score: float
    das_grade: str
    strategic_score: float
    financial_score: float
    operational_score: float
    risk_score: float
    investment_thesis: str
    key_strengths: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)


def _das_grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 70:
        return "B+"
    if score >= 60:
        return "B"
    if score >= 50:
        return "C+"
    if score >= 40:
        return "C"
    return "D"


def compute_das_score(symbol: str, fundamentals: Dict[str, Any]) -> DASResult:
    """4-component DAS (0-100). Each component 0-25."""
    fin = fundamentals or {}

    # --- Component 1: Strategic Fit (0-25) ---
    # Market cap size (mid-cap 2-20B preferred), sector attractiveness, revenue growth
    strategic = 0.0
    market_cap = fin.get("market_cap") or 0.0
    if 2e9 <= market_cap <= 20e9:
        strategic += 10.0
    elif 500e6 <= market_cap < 2e9:
        strategic += 7.0
    elif 20e9 < market_cap <= 80e9:
        strategic += 5.0
    rev_growth = fin.get("revenue_growth_yoy") or 0.0
    if rev_growth >= 0.15:
        strategic += 8.0
    elif rev_growth >= 0.08:
        strategic += 5.0
    elif rev_growth >= 0.0:
        strategic += 2.0
    sector = str(fin.get("sector") or "Unknown").lower()
    high_value_sectors = {"technology", "healthcare", "industrials", "consumer discretionary"}
    if any(s in sector for s in high_value_sectors):
        strategic += 7.0
    else:
        strategic += 3.0
    strategic = clamp(strategic, 0.0, 25.0)

    # --- Component 2: Financial Quality (0-25) ---
    financial = 0.0
    gross_margin = fin.get("gross_margin") or 0.0
    if gross_margin >= 0.50:
        financial += 10.0
    elif gross_margin >= 0.35:
        financial += 7.0
    elif gross_margin >= 0.20:
        financial += 4.0
    fcf_margin = fin.get("fcf_margin") or 0.0
    if fcf_margin >= 0.15:
        financial += 8.0
    elif fcf_margin >= 0.08:
        financial += 5.0
    elif fcf_margin >= 0.0:
        financial += 2.0
    eis = fin.get("axiom_eis") or 50.0
    financial += clamp((eis - 40.0) / 60.0 * 7.0, 0.0, 7.0)
    financial = clamp(financial, 0.0, 25.0)

    # --- Component 3: Operational Excellence (0-25) ---
    operational = 0.0
    op_margin = fin.get("op_margin") or 0.0
    if op_margin >= 0.20:
        operational += 10.0
    elif op_margin >= 0.12:
        operational += 7.0
    elif op_margin >= 0.05:
        operational += 4.0
    caps = fin.get("axiom_caps") or 50.0
    operational += clamp((caps - 40.0) / 60.0 * 8.0, 0.0, 8.0)
    roa = fin.get("return_on_assets") or 0.0
    if roa >= 0.12:
        operational += 7.0
    elif roa >= 0.06:
        operational += 4.0
    elif roa >= 0.0:
        operational += 1.0
    operational = clamp(operational, 0.0, 25.0)

    # --- Component 4: Risk Score (0-25, inverted — low risk = high score) ---
    risk = 20.0  # start optimistic
    dte = fin.get("debt_to_equity") or 0.0
    if dte > 200:
        risk -= 10.0
    elif dte > 100:
        risk -= 5.0
    elif dte > 50:
        risk -= 2.0
    dau = fin.get("axiom_dau")
    if dau is not None:
        if dau <= 30:
            risk += 5.0   # deeply depressed = value opportunity
        elif dau >= 75:
            risk -= 5.0   # high DAU = market crowded
    risk = clamp(risk, 0.0, 25.0)

    total = round(strategic + financial + operational + risk, 1)

    # Build narrative
    strengths: List[str] = []
    risks_list: List[str] = []
    if gross_margin >= 0.35:
        strengths.append(f"Strong gross margin ({gross_margin*100:.0f}%)")
    if fcf_margin >= 0.08:
        strengths.append(f"Healthy FCF generation ({fcf_margin*100:.0f}% FCF margin)")
    if rev_growth >= 0.08:
        strengths.append(f"Revenue growth {rev_growth*100:.0f}% YoY")
    if dte > 150:
        risks_list.append(f"High leverage (D/E {dte:.0f})")
    if fcf_margin < 0:
        risks_list.append("Negative free cash flow")
    if rev_growth and rev_growth < -0.05:
        risks_list.append(f"Revenue declining {abs(rev_growth)*100:.0f}% YoY")

    if total >= 70:
        thesis = f"{symbol} presents a compelling acquisition opportunity with strong financial quality and operational excellence."
    elif total >= 55:
        thesis = f"{symbol} meets PE acquisition criteria with moderate attractiveness; value creation requires operational improvement."
    else:
        thesis = f"{symbol} shows limited PE attractiveness; significant risks or weak fundamentals require careful evaluation."

    return DASResult(
        symbol=symbol,
        das_score=total,
        das_grade=_das_grade(total),
        strategic_score=strategic,
        financial_score=financial,
        operational_score=operational,
        risk_score=risk,
        investment_thesis=thesis,
        key_strengths=strengths[:3],
        key_risks=risks_list[:3],
    )
