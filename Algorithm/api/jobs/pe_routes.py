"""Phase 3 + Phase 13: PE and Corporate Intelligence API endpoints."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from api.jobs.tenant_auth import require_tier

router = APIRouter(
    prefix="/pe",
    tags=["pe"],
)


class EntityFinancialsIn(BaseModel):
    entity_id: str
    period_end: str          # ISO date string
    period_type: str = "quarterly"
    org_id: Optional[str] = None
    entity_name: Optional[str] = None
    sector: Optional[str] = None
    entry_date: Optional[str] = None
    entry_ev: Optional[float] = None
    target_exit_date: Optional[str] = None
    target_exit_multiple: Optional[float] = None
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    net_income: Optional[float] = None
    total_debt: Optional[float] = None
    cash: Optional[float] = None
    capex: Optional[float] = None
    free_cash_flow: Optional[float] = None
    headcount: Optional[int] = None
    arr: Optional[float] = None
    equity: Optional[float] = None


@router.post("/entity/financials")
def post_entity_financials(payload: EntityFinancialsIn) -> Dict[str, Any]:
    """Upsert periodic financials for a portfolio company."""
    from api.jobs.pe_intelligence import store_entity_financials
    from api import db
    period_end = dt.date.fromisoformat(payload.period_end)
    _meta_keys = {"entity_id", "period_end", "period_type", "org_id", "entity_name",
                  "sector", "entry_date", "entry_ev", "target_exit_date", "target_exit_multiple", "equity"}
    financials = {k: v for k, v in payload.model_dump().items() if k not in _meta_keys}
    ok = store_entity_financials(
        payload.entity_id, period_end, financials, payload.period_type
    )
    if payload.org_id and db.db_write_enabled():
        try:
            db.safe_execute(
                """
                INSERT INTO private_entities
                    (entity_id, org_id, entity_name, sector, entry_date, entry_ev,
                     target_exit_date, target_exit_multiple, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE)
                ON CONFLICT (entity_id) DO UPDATE SET
                    org_id                = EXCLUDED.org_id,
                    entity_name           = COALESCE(EXCLUDED.entity_name, private_entities.entity_name),
                    sector                = COALESCE(EXCLUDED.sector, private_entities.sector),
                    entry_date            = COALESCE(EXCLUDED.entry_date, private_entities.entry_date),
                    entry_ev              = COALESCE(EXCLUDED.entry_ev, private_entities.entry_ev),
                    target_exit_date      = COALESCE(EXCLUDED.target_exit_date, private_entities.target_exit_date),
                    target_exit_multiple  = COALESCE(EXCLUDED.target_exit_multiple, private_entities.target_exit_multiple),
                    updated_at            = now()
                """,
                (
                    payload.entity_id,
                    payload.org_id,
                    payload.entity_name,
                    payload.sector,
                    dt.date.fromisoformat(payload.entry_date) if payload.entry_date else None,
                    payload.entry_ev,
                    dt.date.fromisoformat(payload.target_exit_date) if payload.target_exit_date else None,
                    payload.target_exit_multiple,
                ),
            )
        except Exception:
            pass
    return {
        "status": "stored" if ok else "failed",
        "entity_id": payload.entity_id,
        "period_end": payload.period_end,
    }


@router.get("/entity/{entity_id}/health")
def get_entity_health(entity_id: str) -> Dict[str, Any]:
    """Return health score and component breakdown for a portfolio company."""
    from api.jobs.pe_intelligence import compute_entity_health
    return compute_entity_health(entity_id)


@router.get("/entity/{entity_id}/exit-timing")
def get_exit_timing(entity_id: str) -> Dict[str, Any]:
    """Return exit readiness score and recommendation for a portfolio company."""
    from api.jobs.pe_intelligence import compute_exit_timing
    return compute_exit_timing(entity_id)


@router.get("/portfolio/{org_id}/overview")
def get_portfolio_overview(org_id: str) -> Dict[str, Any]:
    """Return all active portfolio companies with health scores."""
    from api.jobs.pe_intelligence import get_portfolio_overview
    return get_portfolio_overview(org_id)


@router.get("/portfolio/{org_id}/stress-alerts")
def get_stress_alerts(org_id: str) -> Dict[str, Any]:
    """Return distressed portfolio companies (health < 40)."""
    from api.jobs.pe_intelligence import get_portfolio_stress_alerts
    return get_portfolio_stress_alerts(org_id)


# ---------------------------------------------------------------------------
# Phase 13: Schilit analyzer
# ---------------------------------------------------------------------------

class SchilitFinancialsIn(BaseModel):
    financials: Dict[str, Any]
    sector_context: Optional[Dict[str, Any]] = None


@router.get("/schilit/{entity_id}")
def get_schilit_analysis(entity_id: str) -> Dict[str, Any]:
    """Full Schilit 7-category analysis for a PE entity's most recent financials."""
    from api import db
    from api.pe.schilit_analyzer import run_full_schilit_analysis
    if not db.db_read_enabled():
        return {"entity_id": entity_id, "status": "db_disabled"}
    try:
        row = db.safe_fetchone(
            """
            SELECT revenue, ebitda, net_income, total_debt, cash, capex, free_cash_flow
              FROM private_entity_financials
             WHERE entity_id = %s ORDER BY period_end DESC LIMIT 1
            """,
            (entity_id,),
        )
        if not row:
            return {"entity_id": entity_id, "status": "no_data"}
        financials = {
            "revenue": float(row[0]) if row[0] is not None else None,
            "ebitda": float(row[1]) if row[1] is not None else None,
            "net_income": float(row[2]) if row[2] is not None else None,
            "total_debt": float(row[3]) if row[3] is not None else None,
            "cash": float(row[4]) if row[4] is not None else None,
            "capex": float(row[5]) if row[5] is not None else None,
            "free_cash_flow": float(row[6]) if row[6] is not None else None,
        }
        result = run_full_schilit_analysis(financials)
        result["categories"] = {
            k: {
                "category": v.category,
                "category_name": v.category_name,
                "triggered": v.triggered,
                "severity": v.severity,
                "evidence": v.evidence,
                "impact_on_eis": v.impact_on_eis,
                "management_intent": v.management_intent,
            }
            for k, v in result["categories"].items()
        }
        return {"entity_id": entity_id, "status": "ok", **result}
    except Exception as exc:
        return {"entity_id": entity_id, "status": "error", "error": str(exc)}


@router.get("/schilit/screen")
def screen_schilit(org_id: str = Query(...)) -> Dict[str, Any]:
    """Schilit screen for all portfolio entities — ranked by schilit_score (most suspicious first)."""
    from api import db
    from api.pe.schilit_analyzer import run_full_schilit_analysis
    if not db.db_read_enabled():
        return {"org_id": org_id, "entities": []}
    try:
        rows = db.safe_fetchall(
            """
            SELECT e.entity_id, e.entity_name,
                   f.revenue, f.ebitda, f.net_income
              FROM private_entities e
              LEFT JOIN LATERAL (
                  SELECT revenue, ebitda, net_income
                    FROM private_entity_financials
                   WHERE entity_id = e.entity_id
                   ORDER BY period_end DESC LIMIT 1
              ) f ON TRUE
             WHERE e.org_id = %s AND e.is_active = TRUE
            """,
            (org_id,),
        ) or []
        results = []
        for row in rows:
            eid, ename = str(row[0]), str(row[1] or row[0])
            fin = {
                "revenue": float(row[2]) if row[2] else None,
                "ebitda": float(row[3]) if row[3] else None,
                "net_income": float(row[4]) if row[4] else None,
            }
            r = run_full_schilit_analysis(fin)
            results.append({
                "entity_id": eid,
                "entity_name": ename,
                "schilit_score": r["schilit_score"],
                "triggered_flags": r["triggered_flags"],
                "recommendation": r["recommendation"],
                "audit_risk_flag": r["audit_risk_flag"],
            })
        results.sort(key=lambda x: x["schilit_score"])
        return {"org_id": org_id, "entities": results}
    except Exception as exc:
        return {"org_id": org_id, "entities": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Phase 13: Deal sourcing
# ---------------------------------------------------------------------------

@router.get("/deal-sourcing")
def get_deal_candidates(
    min_das: float = Query(55.0, ge=0, le=100),
    sector: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """Ranked PE-attractive acquisition candidates by DAS score."""
    from api.pe.deal_sourcing import screen_for_deal_candidates
    sectors = [sector] if sector else None
    candidates = screen_for_deal_candidates(min_das=min_das, sectors=sectors)[:limit]
    return {
        "count": len(candidates),
        "candidates": [
            {
                "symbol": c.symbol,
                "company_name": c.company_name,
                "sector": c.sector,
                "das_score": c.das_score,
                "das_components": c.das_components,
                "axiom_dau": c.axiom_dau,
                "strategic_fit_themes": c.strategic_fit_themes,
                "management_quality_signal": c.management_quality_signal,
            }
            for c in candidates
        ],
    }


@router.get("/deal-sourcing/{symbol}")
def get_deal_analysis(symbol: str) -> Dict[str, Any]:
    """Full DAS analysis for a specific public company."""
    from api import db
    from api.pe.deal_sourcing import compute_das, classify_strategic_themes
    if not db.db_read_enabled():
        return {"symbol": symbol, "status": "db_disabled"}
    try:
        row = db.safe_fetchone(
            "SELECT payload FROM axiom_scores_daily WHERE symbol = %s ORDER BY as_of_date DESC LIMIT 1",
            (symbol.upper(),),
        )
        payload = row[0] if row and row[0] else {}
        if isinstance(payload, str):
            import json
            payload = json.loads(payload)
        meta = payload.get("symbol_meta") or {}
        sector = str(meta.get("sector") or "Unknown")
        das_result = compute_das(symbol.upper(), payload)
        themes = classify_strategic_themes(symbol.upper(), payload, sector)
        return {
            "symbol": symbol.upper(),
            "sector": sector,
            **das_result,
            "strategic_fit_themes": themes,
        }
    except Exception as exc:
        return {"symbol": symbol, "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Phase 13: Comps
# ---------------------------------------------------------------------------

class CompsIn(BaseModel):
    entity_id: str
    financials: Dict[str, Any]
    sector: str = "Unknown"
    eis_score: float = 60.0
    caps_score: float = 60.0
    schilit_score: float = 100.0


@router.post("/entity/{entity_id}/comps")
def get_entity_comps(entity_id: str, payload: CompsIn) -> Dict[str, Any]:
    """Comparable transaction analysis for a PE entity."""
    from api.pe.comps_engine import run_comps_analysis
    result = run_comps_analysis(
        entity_id,
        payload.financials,
        payload.sector,
        payload.eis_score,
        payload.caps_score,
        payload.schilit_score,
    )
    return {
        "target_symbol": result.target_symbol,
        "comps_methodology": result.comps_methodology,
        "ev_ebitda_range": result.ev_ebitda_range,
        "ev_revenue_range": result.ev_revenue_range,
        "pe_ratio_range": result.pe_ratio_range,
        "quality_adjusted_multiple": result.quality_adjusted_multiple,
        "implied_valuation_range": result.implied_valuation_range,
        "upside_to_consensus": result.upside_to_consensus,
        "confidence": result.confidence,
    }


@router.get("/deal-sourcing/{symbol}/comps")
def get_public_company_comps(
    symbol: str,
    eis_score: float = Query(60.0),
    caps_score: float = Query(60.0),
    schilit_score: float = Query(100.0),
) -> Dict[str, Any]:
    """Comparable transaction analysis for a public company deal candidate."""
    from api import db
    from api.pe.comps_engine import run_comps_analysis
    if not db.db_read_enabled():
        return {"symbol": symbol, "status": "db_disabled"}
    try:
        row = db.safe_fetchone(
            "SELECT payload FROM axiom_scores_daily WHERE symbol = %s ORDER BY as_of_date DESC LIMIT 1",
            (symbol.upper(),),
        )
        payload = row[0] if row and row[0] else {}
        if isinstance(payload, str):
            import json
            payload = json.loads(payload)
        meta = payload.get("symbol_meta") or {}
        sector = str(meta.get("sector") or "Unknown")
        result = run_comps_analysis(
            symbol.upper(), {}, sector, eis_score, caps_score, schilit_score
        )
        return {
            "symbol": symbol.upper(),
            "sector": sector,
            "ev_ebitda_range": result.ev_ebitda_range,
            "implied_valuation_range": result.implied_valuation_range,
            "quality_adjusted_multiple": result.quality_adjusted_multiple,
            "confidence": result.confidence,
        }
    except Exception as exc:
        return {"symbol": symbol, "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Phase 13: Supply chain risk
# ---------------------------------------------------------------------------

@router.get("/portfolio/{org_id}/supply-chain-risks")
def get_supply_chain_risks(org_id: str) -> Dict[str, Any]:
    """Portfolio supply chain stress index via public market analogs."""
    from api import db
    from api.pe.supply_chain_risk import compute_supply_chain_stress_index
    if not db.db_read_enabled():
        return {"org_id": org_id, "portfolio_supply_chain_risk": 50.0, "entities": []}
    try:
        rows = db.safe_fetchall(
            "SELECT entity_id, entity_name, sector FROM private_entities WHERE org_id = %s AND is_active = TRUE",
            (org_id,),
        ) or []
        entities = [{"entity_id": str(r[0]), "entity_name": str(r[1] or r[0]), "sector": str(r[2] or "")} for r in rows]
        return compute_supply_chain_stress_index(org_id, entities)
    except Exception as exc:
        return {"org_id": org_id, "error": str(exc)}


@router.get("/portfolio/{org_id}/analog-alerts")
def get_analog_alerts(org_id: str) -> Dict[str, Any]:
    """Real-time stress alerts from public market analogs of portfolio companies."""
    from api.pe.supply_chain_risk import monitor_portfolio_analogs
    return monitor_portfolio_analogs(org_id)


# ---------------------------------------------------------------------------
# Prompt 4: DAS (4-component model) + Deal Flow
# ---------------------------------------------------------------------------

@router.get("/das/{symbol}")
def get_deal_attractiveness_score(symbol: str) -> Dict[str, Any]:
    """4-component Deal Attractiveness Score for a public company."""
    from api import db
    from api.pe.deal_sourcing import compute_deal_attractiveness_score
    if not db.db_read_enabled():
        return {"symbol": symbol, "status": "db_disabled"}
    try:
        row = db.safe_fetchone(
            "SELECT payload FROM axiom_scores_daily WHERE symbol = %s ORDER BY as_of_date DESC LIMIT 1",
            (symbol.upper(),),
        )
        import json as _json
        payload = (row[0] if row and row[0] else {})
        if isinstance(payload, str):
            payload = _json.loads(payload)
        das = compute_deal_attractiveness_score(symbol.upper(), {}, payload)
        result: Dict[str, Any] = {
            "symbol": das.symbol,
            "total": das.total,
            "das_score": das.total,
            "das_grade": das.das_grade,
            "strategic_score": das.strategic_score,
            "financial_score": das.financial_score,
            "operational_score": das.operational_score,
            "risk_score": das.risk_score,
            "investment_thesis": das.investment_thesis,
            "key_strengths": das.key_strengths,
            "key_risks": das.key_risks,
            "llm_enhanced": False,
        }
        # OpenAI PE analysis — additive
        from api import config as _cfg
        if _cfg.openai_api_key() and das.total is not None:
            try:
                from api.llm.openai_client import synthesize_pe_analysis
                dau_val = float(payload.get("deployable_alpha_utility") or 50.0)
                sig = "BUY" if dau_val >= 65 else ("SELL" if dau_val <= 40 else "HOLD")
                ai_text = synthesize_pe_analysis({
                    "symbol": symbol.upper(),
                    "signal_label": sig,
                    "dau": dau_val,
                    "das_score": float(das.total or 50),
                    "das_grade": das.das_grade or "B",
                    "schilit_risk": "moderate",
                    "investment_thesis": das.investment_thesis or "",
                })
                if ai_text:
                    result["ai_analysis"] = ai_text
                    result["llm_enhanced"] = True
            except Exception:
                pass
        return result
    except Exception as exc:
        return {"symbol": symbol, "status": "error", "error": str(exc)}


@router.get("/forensic/{symbol}")
def get_forensic_analysis(symbol: str) -> Dict[str, Any]:
    """Schilit forensic analysis for a public company using real yfinance fundamentals."""
    from api.pe.schilit_analyzer import SchilitForensicEngine
    from api.pe.fundamental_loader import load_company_fundamentals
    sym = symbol.upper()
    try:
        fund = load_company_fundamentals(sym)
        financials = {
            "revenue_growth_yoy": fund.get("revenue_growth_yoy") or 0.0,
            "gross_margin": fund.get("gross_margin") or 0.0,
            "op_margin": fund.get("op_margin") or 0.0,
        }
        engine = SchilitForensicEngine()
        report = engine.analyze(sym, financials)
        return {
            "symbol": sym,
            "overall_risk": report.overall_risk,
            "forensic_summary": report.forensic_summary,
            "red_flags": report.red_flags,
            "green_flags": report.green_flags,
            "eis_impact": report.eis_impact,
            "category_scores": report.category_scores,
            "report": {
                "overall_risk": report.overall_risk,
                "forensic_summary": report.forensic_summary,
            },
        }
    except Exception as exc:
        return {"symbol": sym, "status": "error", "error": str(exc)}


@router.get("/deal-flow")
def get_deal_flow(as_of_date: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    """Daily acquisition candidate screening across AXIOM universe."""
    from api.pe.deal_flow import run_daily_deal_flow_screen
    import datetime as _dt
    aod = _dt.date.fromisoformat(as_of_date) if as_of_date else None
    return run_daily_deal_flow_screen(aod)


@router.get("/portfolio/{org_id}/lp-report-v2")
def get_lp_report_v2(
    org_id: str,
    quarter: Optional[str] = None,
) -> Dict[str, Any]:
    """5-section structured LP report."""
    from api.pe.lp_reporting import build_structured_lp_report
    return build_structured_lp_report(org_id, quarter)


# ---------------------------------------------------------------------------
# Phase 13: LP reporting
# ---------------------------------------------------------------------------

@router.get("/portfolio/{org_id}/lp-report")
def get_lp_report(
    org_id: str,
    quarter: Optional[str] = None,
    format: str = Query("full", pattern="^(full|summary)$"),
) -> Dict[str, Any]:
    """Generate (or retrieve) quarterly LP report for a portfolio."""
    from api.pe.lp_reporting import generate_lp_report
    report = generate_lp_report(org_id, quarter)
    if format == "summary":
        return {
            "org_id": report.org_id,
            "report_quarter": report.report_quarter,
            "portfolio_summary": report.portfolio_summary,
            "risk_flags": report.risk_flags,
            "narrative_sections": report.narrative_sections,
        }
    return {
        "org_id": report.org_id,
        "report_quarter": report.report_quarter,
        "generated_at": report.generated_at.isoformat(),
        "portfolio_summary": report.portfolio_summary,
        "individual_company_reports": report.individual_company_reports,
        "value_creation_attribution": report.value_creation_attribution,
        "risk_flags": report.risk_flags,
        "exit_pipeline": report.exit_pipeline,
        "market_context": report.market_context,
        "narrative_sections": report.narrative_sections,
    }
