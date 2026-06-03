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
