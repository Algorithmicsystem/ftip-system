"""Phase 4: PE Deal Flow Monitor — daily acquisition candidate screening."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from api import db
from api.assistant.phase3.common import clamp
from api.universe import AXIOM_UNIVERSE

logger = logging.getLogger(__name__)

AXIOM_ACQUISITION_CRITERIA = {
    "depressed_price_signal": lambda dau: dau <= 40,
    "financial_quality_minimum": 60,
    "schilit_risk_maximum": "medium",
    "strong_fundamentals": lambda ebitda_margin: ebitda_margin >= 0.15,
    "manageable_leverage": lambda net_debt_ebitda: net_debt_ebitda <= 3.5,
}


def score_acquisition_candidate(
    symbol: str,
    axiom_payload: Dict[str, Any],
    fundamentals: Dict[str, Any],
) -> float:
    """Score 0-100. Higher = better PE acquisition candidate."""
    score = 0.0
    fin = fundamentals or {}

    # 1. AXIOM signal (40 pts): SELL/depressed = opportunity
    dau = float(axiom_payload.get("dau") or axiom_payload.get("deployable_alpha_utility") or 50.0)
    if dau <= 35:
        score += 40
    elif dau <= 40:
        score += 30
    elif dau <= 50:
        score += 15

    # 2. Financial quality (30 pts)
    ebitda_margin = float(fin.get("ebitda_margin") or 0.10)
    fcf_yield = float(fin.get("fcf_to_ebitda") or fin.get("fcf_yield") or 0.5)
    score += min(ebitda_margin / 0.30 * 15, 15.0)
    score += min(fcf_yield * 15, 15.0)

    # 3. Accounting quality (20 pts)
    schilit_risk = str(axiom_payload.get("schilit_overall_risk") or "low")
    schilit_pts = {"low": 20, "medium": 12, "high": 4, "critical": 0}.get(schilit_risk, 10)
    score += schilit_pts

    # 4. IQ trend bonus (10 pts)
    iq_trend = str(axiom_payload.get("dossier_iq_trend") or "stable")
    if iq_trend == "improving":
        score += 10
    elif iq_trend == "stable":
        score += 5

    return round(min(score, 100.0), 1)


def _build_deal_rationale(
    symbol: str,
    score: float,
    axiom_payload: Dict[str, Any],
    fundamentals: Dict[str, Any],
) -> str:
    dau = float(axiom_payload.get("dau") or axiom_payload.get("deployable_alpha_utility") or 50.0)
    schilit = str(axiom_payload.get("schilit_overall_risk") or "low")
    ebitda_margin = float(fundamentals.get("ebitda_margin") or 0.10)
    parts = []
    if dau <= 40:
        parts.append(f"AXIOM SELL signal (DAU {dau:.0f}) indicates market pessimism = potential value")
    if ebitda_margin >= 0.15:
        parts.append(f"Strong EBITDA margin ({ebitda_margin*100:.0f}%) supports business quality")
    if schilit in ("low", "medium"):
        parts.append(f"Accounting quality {schilit} — clean financials reduce due diligence risk")
    return "; ".join(parts) if parts else "Meets minimum acquisition screening criteria"


def _load_axiom_payload(symbol: str, as_of_date: Optional[dt.date] = None) -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {}
    try:
        if as_of_date:
            row = db.safe_fetchone(
                "SELECT payload FROM axiom_scores_daily WHERE symbol = %s AND as_of_date <= %s ORDER BY as_of_date DESC LIMIT 1",
                (symbol, as_of_date),
            )
        else:
            row = db.safe_fetchone(
                "SELECT payload FROM axiom_scores_daily WHERE symbol = %s ORDER BY as_of_date DESC LIMIT 1",
                (symbol,),
            )
        if row and row[0]:
            import json
            p = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            dau = p.get("deployable_alpha_utility")
            return {"dau": dau, "deployable_alpha_utility": dau, **p}
    except Exception as exc:
        logger.debug("deal_flow.load_payload symbol=%s err=%s", symbol, exc)
    return {}


def _load_fundamentals(symbol: str, as_of_date: Optional[dt.date] = None) -> Dict[str, Any]:
    """Load fundamental data from AXIOM payload fundamental_reality engine scores."""
    payload = _load_axiom_payload(symbol, as_of_date)
    engine_scores = payload.get("engine_scores") or {}
    fund = (engine_scores.get("fundamental_reality") or {}).get("components") or {}
    return {
        "ebitda_margin": float(fund.get("ebitda_margin_score") or 0.0) / 100.0 * 0.30,
        "fcf_yield": float(fund.get("caps_component") or 50.0) / 100.0 * 0.05,
        "fcf_to_ebitda": float(fund.get("caps_component") or 50.0) / 200.0,
    }


def run_daily_deal_flow_screen(as_of_date: Optional[dt.date] = None) -> Dict[str, Any]:
    """Screen all AXIOM universe symbols for acquisition candidacy."""
    aod = as_of_date or dt.date.today()
    candidates = []

    for symbol in AXIOM_UNIVERSE:
        axiom_data = _load_axiom_payload(symbol, aod)
        fundamentals = _load_fundamentals(symbol, aod)
        score = score_acquisition_candidate(symbol, axiom_data, fundamentals)
        if score >= 30:
            candidates.append({
                "symbol": symbol,
                "acquisition_score": score,
                "dau": float(axiom_data.get("dau") or axiom_data.get("deployable_alpha_utility") or 50.0),
                "schilit_risk": str(axiom_data.get("schilit_overall_risk") or "unknown"),
                "das_grade": str(axiom_data.get("das_grade") or "unknown"),
                "rationale": _build_deal_rationale(symbol, score, axiom_data, fundamentals),
            })

    candidates.sort(key=lambda x: x["acquisition_score"], reverse=True)

    # Store results
    if db.db_write_enabled():
        import json
        try:
            db.safe_execute(
                """
                INSERT INTO deal_flow_scores (screen_date, candidates, universe_screened, candidates_found, created_at)
                VALUES (%s, %s::jsonb, %s, %s, now())
                ON CONFLICT (screen_date) DO UPDATE SET
                    candidates = EXCLUDED.candidates,
                    universe_screened = EXCLUDED.universe_screened,
                    candidates_found = EXCLUDED.candidates_found
                """,
                (aod, json.dumps(candidates[:10]), len(AXIOM_UNIVERSE), len(candidates)),
            )
        except Exception as exc:
            logger.debug("deal_flow.store_failed err=%s", exc)

    return {
        "as_of_date": aod.isoformat(),
        "candidates": candidates[:10],
        "universe_screened": len(AXIOM_UNIVERSE),
        "candidates_found": len(candidates),
    }
