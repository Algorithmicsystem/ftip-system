"""Phase 18.5a: Competitive Intelligence Routes."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from api.jobs.tenant_auth import require_tier

router = APIRouter(
    prefix="/competitive",
    tags=["competitive"],
)


@router.get("/{symbol}")
def get_competitive_report(symbol: str) -> Dict[str, Any]:
    from api.competitive.competitive_intelligence import generate_competitive_intelligence_report
    report = generate_competitive_intelligence_report(symbol.upper())
    return {
        "symbol": report.symbol,
        "as_of_date": report.as_of_date.isoformat(),
        "sector": report.sector,
        "competitor_count": report.competitor_count,
        "sector_dau_rank": report.sector_dau_rank,
        "sector_size": report.sector_size,
        "competitive_position": report.competitive_position,
        "market_share_momentum": report.market_share_momentum,
        "best_competitor": report.best_competitor,
        "most_dangerous_competitor": report.most_dangerous_competitor,
        "competitive_intelligence_score": report.competitive_intelligence_score,
        "competitors": [
            {
                "competitor_symbol": c.competitor_symbol,
                "relationship_type": c.relationship_type,
                "dau_advantage": c.dau_advantage,
                "competitive_position_score": c.competitive_position_score,
                "key_advantage": c.key_advantage,
                "key_vulnerability": c.key_vulnerability,
            }
            for c in report.competitors
        ],
    }


@router.get("/{symbol}/vs/{competitor}")
def get_head_to_head(symbol: str, competitor: str) -> Dict[str, Any]:
    from api.competitive.competitive_intelligence import (
        compute_competitor_profile,
    )
    # Load both payloads
    from api import db
    sym_payload: Dict[str, Any] = {}
    comp_payload: Dict[str, Any] = {}
    if db.db_read_enabled():
        try:
            r = db.safe_fetchone(
                "SELECT payload FROM axiom_scores_daily WHERE symbol=%s ORDER BY as_of_date DESC LIMIT 1",
                (symbol.upper(),),
            )
            sym_payload = r[0] if r and isinstance(r[0], dict) else {}
            r2 = db.safe_fetchone(
                "SELECT payload FROM axiom_scores_daily WHERE symbol=%s ORDER BY as_of_date DESC LIMIT 1",
                (competitor.upper(),),
            )
            comp_payload = r2[0] if r2 and isinstance(r2[0], dict) else {}
        except Exception:
            pass

    profile = compute_competitor_profile(symbol.upper(), competitor.upper(), sym_payload, comp_payload)

    # Build narrative
    sym_u = symbol.upper()
    comp_u = competitor.upper()
    pos = profile.competitive_position_score
    if pos > 60:
        stance = f"{sym_u} holds a clear competitive edge over {comp_u}"
    elif pos < 40:
        stance = f"{comp_u} currently outperforms {sym_u} on AXIOM metrics"
    else:
        stance = f"{sym_u} and {comp_u} are closely matched"

    adv_txt = (
        f"Key advantage: {profile.key_advantage.replace('_', ' ')} (+{abs(profile.dau_advantage):.1f} DAU pts)."
        if profile.key_advantage != "none" else "No clear AXIOM advantage identified."
    )
    vuln_txt = (
        f"Primary vulnerability: {profile.key_vulnerability.replace('_', ' ')}."
        if profile.key_vulnerability != "none" else ""
    )
    narrative = f"{stance}. {adv_txt} {vuln_txt}".strip()

    return {
        "symbol": profile.symbol,
        "competitor": profile.competitor_symbol,
        "dau_advantage": profile.dau_advantage,
        "eis_advantage": profile.eis_advantage,
        "caps_advantage": profile.caps_advantage,
        "fragility_advantage": profile.fragility_advantage,
        "momentum_advantage": profile.momentum_advantage,
        "competitive_position_score": profile.competitive_position_score,
        "key_advantage": profile.key_advantage,
        "key_vulnerability": profile.key_vulnerability,
        "head_to_head_narrative": narrative,
    }


@router.get("/{symbol}/sector-ranking")
def get_sector_ranking(symbol: str) -> Dict[str, Any]:
    from api.competitive.competitive_intelligence import generate_competitive_intelligence_report
    report = generate_competitive_intelligence_report(symbol.upper())
    return {
        "symbol": report.symbol,
        "sector": report.sector,
        "sector_dau_rank": report.sector_dau_rank,
        "sector_eis_rank": report.sector_eis_rank,
        "sector_caps_rank": report.sector_caps_rank,
        "sector_size": report.sector_size,
        "competitive_position": report.competitive_position,
        "competitive_intelligence_score": report.competitive_intelligence_score,
    }


@router.get("/{symbol}/management-quality")
def get_management_quality(symbol: str) -> Dict[str, Any]:
    from api import db
    from api.competitive.management_quality import compute_mqs
    payload: Dict[str, Any] = {}
    if db.db_read_enabled():
        try:
            r = db.safe_fetchone(
                "SELECT payload FROM axiom_scores_daily WHERE symbol=%s ORDER BY as_of_date DESC LIMIT 1",
                (symbol.upper(),),
            )
            payload = r[0] if r and isinstance(r[0], dict) else {}
        except Exception:
            pass
    mqs = compute_mqs(symbol.upper(), payload)
    return {
        "symbol": mqs.symbol,
        "mqs_score": mqs.mqs_score,
        "capital_allocation_score": mqs.capital_allocation_score,
        "guidance_accuracy_score": mqs.guidance_accuracy_score,
        "insider_alignment_score": mqs.insider_alignment_score,
        "management_integrity_signal": mqs.management_integrity_signal,
        "mqs_trend": mqs.mqs_trend,
        "red_flags": mqs.management_red_flags,
        "green_flags": mqs.management_green_flags,
    }


@router.get("/sector/{sector}/ranking")
def get_sector_dau_ranking(sector: str) -> Dict[str, Any]:
    """Return all symbols in the sector ranked by DAU."""
    from api import db
    from api.competitive.competitive_intelligence import AXIOM_SECTOR_GROUPS
    import datetime as _dt

    symbols = AXIOM_SECTOR_GROUPS.get(sector, [])
    ranked: List[Dict[str, Any]] = []

    if db.db_read_enabled() and symbols:
        try:
            rows = db.safe_fetchall(
                """
                SELECT DISTINCT ON (symbol) symbol,
                       (payload->>'deployable_alpha_utility')::numeric AS dau,
                       payload->>'regime_label' AS regime
                  FROM axiom_scores_daily
                 WHERE symbol = ANY(%s)
                 ORDER BY symbol, as_of_date DESC
                """,
                (symbols,),
            ) or []
            by_sym = {str(r[0]): {"dau": float(r[1] or 50), "regime": str(r[2] or "unknown")} for r in rows}
        except Exception:
            by_sym = {}

        for sym in symbols:
            info = by_sym.get(sym, {"dau": 50.0, "regime": "unknown"})
            ranked.append({"symbol": sym, "dau": info["dau"], "regime": info["regime"]})
    else:
        ranked = [{"symbol": sym, "dau": 50.0, "regime": "unknown"} for sym in symbols]

    ranked.sort(key=lambda x: x["dau"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return {
        "sector": sector,
        "symbol_count": len(ranked),
        "as_of_date": _dt.date.today().isoformat(),
        "rankings": ranked,
    }


@router.get("/sector/{sector}/management-rankings")
def get_sector_management_rankings(sector: str) -> List[Dict[str, Any]]:
    from api.competitive.management_quality import get_sector_mqs_rankings
    return get_sector_mqs_rankings(sector)
