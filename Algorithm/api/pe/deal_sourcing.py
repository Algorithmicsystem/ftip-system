"""Phase 13.2: Deal Sourcing Engine with proprietary Deal Attractiveness Score (DAS)."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from api import db
from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)


@dataclass
class DealCandidate:
    symbol: str
    company_name: str
    sector: str
    market_cap_estimate: float
    das_score: float
    das_components: Dict[str, float]
    fcf_yield: float
    caps_score: float
    valuation_discount_pct: float
    management_quality_signal: str   # "strong_buyer" | "neutral" | "seller"
    insider_activity: str
    axiom_dau: float
    schilit_score: Optional[float]
    exit_multiple_range: Dict[str, float]
    strategic_fit_themes: List[str]


# ---------------------------------------------------------------------------
# DAS computation
# ---------------------------------------------------------------------------

def compute_das(
    symbol: str,
    axiom_payload: Dict,
    financials: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Compute Deal Attractiveness Score (0–100) with component breakdown."""
    fin = financials or {}

    # --- FCF yield score ---
    fcf_yield = float(fin.get("fcf_yield") or 0.0)
    sector_median_fcf = float(fin.get("sector_median_fcf_yield") or 0.04)
    sector_std_fcf = float(fin.get("sector_std_fcf_yield") or 0.02)
    if sector_std_fcf > 0:
        fcf_z = (fcf_yield - sector_median_fcf) / sector_std_fcf
    else:
        fcf_z = 0.0
    fcf_yield_score = clamp(fcf_z * 20.0 + 50.0, 0.0, 100.0)

    # --- CAPS score (directly from AXIOM) ---
    caps_score = float(
        (axiom_payload.get("engine_scores", {})
         .get("fundamental_reality", {})
         .get("components", {})
         .get("caps_component")) or 50.0
    )

    # --- Valuation discount score ---
    eps_ttm = float(fin.get("eps_ttm") or 0.0)
    wacc = float(fin.get("wacc_estimate") or 0.09)
    g = float(fin.get("growth_rate_estimate") or 0.03)
    current_price = float(fin.get("current_price") or 1.0)
    if wacc > g and current_price > 0 and eps_ttm > 0:
        iv_estimate = (eps_ttm / wacc) * (1.0 + g)
        valuation_discount = (iv_estimate - current_price) / current_price
    else:
        valuation_discount = float(fin.get("valuation_discount") or 0.0)
    valuation_discount_score = clamp(valuation_discount * 100.0 + 50.0, 0.0, 100.0)

    # --- Management quality score ---
    insider_buys_6m = float(fin.get("insider_buys_6m") or 0.0)
    insider_sells_6m = float(fin.get("insider_sells_6m") or 0.0)
    buyback_yield = float(fin.get("buyback_yield") or 0.0)
    insider_buy_signal = insider_buys_6m > insider_sells_6m
    buyback_signal = buyback_yield > 0.02
    if insider_buy_signal and buyback_signal:
        management_quality_score = 70.0
        management_signal = "strong_buyer"
    elif insider_buy_signal or buyback_signal:
        management_quality_score = 50.0
        management_signal = "neutral"
    else:
        management_quality_score = 30.0
        management_signal = "seller"

    das = (
        fcf_yield_score * 0.30
        + caps_score * 0.25
        + valuation_discount_score * 0.25
        + management_quality_score * 0.20
    )
    das = round(clamp(das, 0.0, 100.0), 2)

    return {
        "das": das,
        "fcf_yield_score": round(fcf_yield_score, 2),
        "caps_score": round(caps_score, 2),
        "valuation_discount_score": round(valuation_discount_score, 2),
        "management_quality_score": round(management_quality_score, 2),
        "management_signal": management_signal,
        "fcf_yield": round(fcf_yield, 4),
        "valuation_discount_pct": round(valuation_discount * 100.0, 2),
        "insider_activity": "buying" if insider_buy_signal else "selling" if insider_sells_6m > insider_buys_6m else "neutral",
    }


# ---------------------------------------------------------------------------
# Strategic theme classification
# ---------------------------------------------------------------------------

def classify_strategic_themes(
    symbol: str,
    axiom_payload: Dict,
    sector: str,
) -> List[str]:
    """Assign strategic acquisition themes from AXIOM scores."""
    themes: List[str] = []

    engines = axiom_payload.get("engine_scores", {})
    fund = engines.get("fundamental_reality", {}).get("components", {})
    frag = engines.get("critical_fragility", {}).get("components", {})

    caps = float(fund.get("caps_component") or 50.0)
    eis = float(fund.get("earnings_quality_component") or 50.0)
    dau = float(axiom_payload.get("deployable_alpha_utility") or 50.0)
    pess = float(frag.get("pess_component") or 50.0)
    scps = float(frag.get("scps_component") or 50.0)
    rev_growth = float(axiom_payload.get("revenue_growth_yoy") or 0.0)
    dau_trend = float(axiom_payload.get("dau_trend_3m") or 0.0)

    _LARGE_SECTORS = {"Technology", "Healthcare", "Finance", "Consumer Discretionary", "Industrials"}
    if caps > 65 and sector in _LARGE_SECTORS:
        themes.append("consolidation_play")

    if dau < 50 and dau_trend > 0 and eis < 55:
        themes.append("turnaround")

    market_cap = float(axiom_payload.get("market_cap") or 0.0)
    if market_cap < 2_000_000_000 and caps > 65 and eis > 65:
        themes.append("bolt_on")

    if caps > 70 and rev_growth > 0.15:
        themes.append("platform_investment")

    if pess > 70 and dau < 40 and scps < 50:
        themes.append("distressed_opportunity")

    return themes


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

def screen_for_deal_candidates(
    min_das: float = 55.0,
    sectors: Optional[List[str]] = None,
    market_cap_range: Optional[Tuple[float, float]] = None,
    exclude_symbols: Optional[List[str]] = None,
) -> List[DealCandidate]:
    """Screen public company universe for PE-attractive acquisition candidates."""
    if not db.db_read_enabled():
        return []

    exclude = set(exclude_symbols or [])
    try:
        rows = db.safe_fetchall(
            """
            SELECT DISTINCT ON (symbol) symbol, payload, as_of_date
              FROM axiom_scores_daily
             ORDER BY symbol, as_of_date DESC
            """,
        ) or []
    except Exception as exc:
        logger.warning("deal_sourcing.screen_failed err=%s", exc)
        return []

    candidates: List[DealCandidate] = []
    for row in rows:
        sym = str(row[0])
        if sym in exclude:
            continue
        payload = row[1] if isinstance(row[1], dict) else {}

        meta = payload.get("symbol_meta") or {}
        sector = str(meta.get("sector") or payload.get("sector") or "Unknown")
        if sectors and sector not in sectors:
            continue

        company_name = str(meta.get("company_name") or sym)
        market_cap = float(meta.get("market_cap") or payload.get("market_cap") or 0.0)
        if market_cap_range:
            lo, hi = market_cap_range
            if not (lo <= market_cap <= hi):
                continue

        das_result = compute_das(sym, payload)
        score = das_result["das"]
        if score < min_das:
            continue

        axiom_dau = float(payload.get("deployable_alpha_utility") or 0.0)
        themes = classify_strategic_themes(sym, payload, sector)

        fund_comps = (payload.get("engine_scores", {})
                      .get("fundamental_reality", {})
                      .get("components", {}))
        caps = das_result["caps_score"]
        ev_ebitda_est = 12.0
        candidates.append(DealCandidate(
            symbol=sym,
            company_name=company_name,
            sector=sector,
            market_cap_estimate=market_cap,
            das_score=score,
            das_components={
                "fcf_yield_score": das_result["fcf_yield_score"],
                "caps_score": das_result["caps_score"],
                "valuation_discount_score": das_result["valuation_discount_score"],
                "management_quality_score": das_result["management_quality_score"],
            },
            fcf_yield=das_result["fcf_yield"],
            caps_score=caps,
            valuation_discount_pct=das_result["valuation_discount_pct"],
            management_quality_signal=das_result["management_signal"],
            insider_activity=das_result["insider_activity"],
            axiom_dau=axiom_dau,
            schilit_score=None,
            exit_multiple_range={"low": ev_ebitda_est * 0.85, "median": ev_ebitda_est, "high": ev_ebitda_est * 1.20},
            strategic_fit_themes=themes,
        ))

    candidates.sort(key=lambda c: c.das_score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# DealAttractivenessScore — 4-component model (Ilmanen framework)
# ---------------------------------------------------------------------------

@dataclass
class DealAttractivenessScore:
    symbol: str
    strategic_score: float     # 0-25
    financial_score: float     # 0-25
    operational_score: float   # 0-25
    risk_score: float          # 0-25
    total: float               # 0-100
    das_grade: str             # A/B/C/D/F
    investment_thesis: str
    key_strengths: List[str]
    key_risks: List[str]


def _das_grade(total: float) -> str:
    if total >= 85: return "A"
    if total >= 70: return "B"
    if total >= 55: return "C"
    if total >= 40: return "D"
    return "F"


def compute_deal_attractiveness_score(
    symbol: str,
    fundamentals: Dict,
    axiom_payload: Dict,
    market_data: Optional[Dict] = None,
) -> DealAttractivenessScore:
    """4-component DAS per Ilmanen Expected Returns framework."""
    fin = fundamentals or {}
    md = market_data or {}

    # Component 1: Strategic Fit (0-25)
    rev_growth = float(fin.get("revenue_growth_yoy") or 0.0)
    sector_growth = float(fin.get("sector_revenue_growth_median") or 0.05)
    if rev_growth > sector_growth * 1.5:
        market_position = 3
    elif rev_growth > sector_growth * 0.8:
        market_position = 2
    else:
        market_position = 1
    sector_tailwind = float(axiom_payload.get("macro_score") or 50.0) / 100.0
    strategic_score = round(min((market_position / 3.0) * 15 + sector_tailwind * 10, 25.0), 2)

    # Component 2: Financial Quality (0-25)
    nonrecurring_pct = float(fin.get("non_recurring_revenue_pct") or 0.05)
    revenue_quality = 1.0 - nonrecurring_pct
    ebitda_margin = float(fin.get("ebitda_margin") or 0.15)
    cash_conversion = float(fin.get("fcf_to_ebitda") or 0.7)
    sector_margin = float(fin.get("sector_ebitda_margin_median") or 0.15)
    margin_premium = (ebitda_margin - sector_margin) / max(sector_margin, 0.01)
    financial_score = round(min(
        revenue_quality * 8 + cash_conversion * 8 + clamp(margin_premium + 0.5, 0.0, 1.0) * 9,
        25.0,
    ), 2)

    # Component 3: Operational Excellence (0-25)
    gross_margin = float(fin.get("gross_margin") or 0.40)
    sector_gm = float(fin.get("sector_gross_margin_median") or 0.40)
    asset_turnover = float(fin.get("asset_turnover") or 1.0)
    asset_turnover_trend = float(fin.get("asset_turnover_change_yoy") or 0.0)
    sector_gm_std = max(sector_gm * 0.2, 0.01)
    gm_z = (gross_margin - sector_gm) / sector_gm_std
    operational_score = round(min(
        clamp((gm_z + 2) / 4.0, 0.0, 1.0) * 12
        + clamp(asset_turnover / 2.0, 0.0, 1.0) * 7
        + clamp((asset_turnover_trend + 0.1) / 0.2, 0.0, 1.0) * 6,
        25.0,
    ), 2)

    # Component 4: Risk (0-25, higher = lower risk)
    from api.pe.schilit_analyzer import SchilitForensicEngine
    schilit = SchilitForensicEngine().analyze(symbol, fin)
    schilit_pts = {"low": 25, "medium": 17, "high": 8, "critical": 0}.get(schilit.overall_risk, 12)
    regulatory_risk = float(fin.get("regulatory_risk_score") or 20.0)
    regulatory_adj = (100 - regulatory_risk) / 100.0 * 10
    customer_concentration = float(fin.get("customer_concentration_hhi") or 0.3)
    concentration_adj = (1 - customer_concentration) * 5
    risk_score = round(min(schilit_pts * 0.60 + regulatory_adj + concentration_adj, 25.0), 2)

    total = round(clamp(strategic_score + financial_score + operational_score + risk_score, 0.0, 100.0), 2)
    grade = _das_grade(total)

    # Narrative
    scores_map = {
        "Strategic Fit": strategic_score,
        "Financial Quality": financial_score,
        "Operational Excellence": operational_score,
        "Risk Profile": risk_score,
    }
    sorted_scores = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)
    key_strengths = [f"{k} ({v:.1f}/25)" for k, v in sorted_scores[:3] if v >= 12.5]
    key_risks = [f"{k} ({v:.1f}/25)" for k, v in sorted_scores[-3:] if v < 12.5][:3]

    if total >= 70:
        thesis = (f"{symbol} scores {grade} ({total:.0f}/100) on deal attractiveness. "
                  f"Strong {sorted_scores[0][0].lower()} and {sorted_scores[1][0].lower()} "
                  f"support a compelling acquisition thesis. "
                  f"Accounting quality is {schilit.overall_risk}.")
    else:
        thesis = (f"{symbol} scores {grade} ({total:.0f}/100) — "
                  f"below threshold for immediate acquisition interest. "
                  f"Primary concern: {sorted_scores[-1][0].lower()} ({sorted_scores[-1][1]:.1f}/25). "
                  f"Monitor for improvement before committing capital.")

    return DealAttractivenessScore(
        symbol=symbol,
        strategic_score=strategic_score,
        financial_score=financial_score,
        operational_score=operational_score,
        risk_score=risk_score,
        total=total,
        das_grade=grade,
        investment_thesis=thesis,
        key_strengths=key_strengths if key_strengths else [sorted_scores[0][0]],
        key_risks=key_risks if key_risks else ["No critical risks identified"],
    )
