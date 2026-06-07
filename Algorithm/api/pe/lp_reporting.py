"""Phase 13.5: Automated LP Report Generation."""
from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)


@dataclass
class LPReport:
    org_id: str
    report_quarter: str
    generated_at: dt.datetime
    portfolio_summary: Dict[str, Any]
    individual_company_reports: List[Dict]
    value_creation_attribution: Dict[str, Any]
    risk_flags: List[str]
    exit_pipeline: Dict[str, Any]
    market_context: Dict[str, Any]
    narrative_sections: Dict[str, str]


def _health_bucket(score: Optional[float]) -> str:
    if score is None:
        return "watch"
    if score >= 75:
        return "strong"
    if score >= 55:
        return "healthy"
    if score >= 40:
        return "watch"
    return "stressed"


def generate_portfolio_summary(org_id: str, as_of_date: dt.date) -> Dict[str, Any]:
    """Build portfolio-level summary for the LP report."""
    from api.jobs.pe_intelligence import compute_entity_health, _load_recent_periods

    empty = {
        "total_companies": 0,
        "avg_health_score": 0.0,
        "health_distribution": {"strong": 0, "healthy": 0, "watch": 0, "stressed": 0},
        "avg_ebitda_margin": 0.0,
        "portfolio_revenue_growth": 0.0,
        "companies_on_target": 0,
        "companies_at_risk": 0,
        "aggregate_schilit_risk": "low",
        "unrealized_value_trend": "stable",
    }

    if not db.db_read_enabled():
        return empty

    try:
        rows = db.safe_fetchall(
            """
            SELECT entity_id, entity_name
              FROM private_entities
             WHERE org_id = %s AND is_active = TRUE
            """,
            (org_id,),
        ) or []
    except Exception as exc:
        logger.warning("lp_report.summary_failed org=%s err=%s", org_id, exc)
        return empty

    if not rows:
        return empty

    health_scores: List[float] = []
    ebitda_margins: List[float] = []
    rev_growths: List[float] = []
    dist: Dict[str, int] = {"strong": 0, "healthy": 0, "watch": 0, "stressed": 0}

    for row in rows:
        entity_id = str(row[0])
        health = compute_entity_health(entity_id)
        score = health.get("health_score")
        if score is not None:
            health_scores.append(score)
            bucket = _health_bucket(score)
            dist[bucket] = dist.get(bucket, 0) + 1

        periods = _load_recent_periods(entity_id, n=5)
        if periods:
            latest = periods[0]
            rev = latest.get("revenue")
            ebitda = latest.get("ebitda")
            if rev and rev > 0 and ebitda is not None:
                ebitda_margins.append(ebitda / rev)
            if len(periods) >= 4 and rev and periods[3].get("revenue"):
                prev_rev = periods[3]["revenue"]
                if prev_rev and prev_rev > 0:
                    rev_growths.append((rev - prev_rev) / prev_rev)

    total = len(rows)
    avg_health = round(sum(health_scores) / len(health_scores), 1) if health_scores else 0.0
    avg_margin = round(sum(ebitda_margins) / len(ebitda_margins), 4) if ebitda_margins else 0.0
    avg_growth = round(sum(rev_growths) / len(rev_growths), 4) if rev_growths else 0.0

    on_target = sum(1 for s in health_scores if s >= 65)
    at_risk = sum(1 for s in health_scores if s < 40)

    trend = "improving" if avg_growth > 0.05 else "declining" if avg_growth < -0.05 else "stable"
    schilit_risk = "high" if at_risk > total * 0.30 else "medium" if at_risk > 0 else "low"

    return {
        "total_companies": total,
        "avg_health_score": avg_health,
        "health_distribution": dist,
        "avg_ebitda_margin": avg_margin,
        "portfolio_revenue_growth": avg_growth,
        "companies_on_target": on_target,
        "companies_at_risk": at_risk,
        "aggregate_schilit_risk": schilit_risk,
        "unrealized_value_trend": trend,
    }


def generate_value_creation_attribution(org_id: str, periods: int = 4) -> Dict[str, Any]:
    """Attribute portfolio value change over the last N periods to specific drivers."""
    from api.jobs.pe_intelligence import _load_recent_periods

    zero = {
        "total_value_change": 0.0,
        "revenue_contribution": 0.0,
        "margin_contribution": 0.0,
        "leverage_contribution": 0.0,
        "multiple_contribution": 0.0,
    }

    if not db.db_read_enabled():
        return zero

    try:
        rows = db.safe_fetchall(
            "SELECT entity_id FROM private_entities WHERE org_id = %s AND is_active = TRUE",
            (org_id,),
        ) or []
    except Exception:
        return zero

    rev_contribs: List[float] = []
    margin_contribs: List[float] = []
    lev_contribs: List[float] = []

    for row in rows:
        entity_id = str(row[0])
        fin_periods = _load_recent_periods(entity_id, n=periods + 1)
        if len(fin_periods) < 2:
            continue
        current, prior = fin_periods[0], fin_periods[-1]

        # Revenue contribution
        cur_rev = current.get("revenue") or 0.0
        pri_rev = prior.get("revenue") or 0.0
        if pri_rev and pri_rev > 0:
            rev_contribs.append((cur_rev - pri_rev) / pri_rev)

        # Margin contribution
        cur_ebitda = current.get("ebitda") or 0.0
        pri_ebitda = prior.get("ebitda") or 0.0
        if cur_rev > 0 and pri_rev > 0:
            cur_margin = cur_ebitda / cur_rev
            pri_margin = pri_ebitda / pri_rev
            margin_contribs.append(cur_margin - pri_margin)

        # Leverage contribution (improvement = debt reduction)
        cur_debt = current.get("total_debt") or 0.0
        pri_debt = prior.get("total_debt") or 0.0
        if pri_debt and pri_debt > 0:
            lev_contribs.append(-(cur_debt - pri_debt) / pri_debt)

    n = max(len(rev_contribs), 1)
    rev_c = round(sum(rev_contribs) / n, 4) if rev_contribs else 0.0
    margin_c = round(sum(margin_contribs) / n, 4) if margin_contribs else 0.0
    lev_c = round(sum(lev_contribs) / n, 4) if lev_contribs else 0.0
    multiple_c = round(rev_c * 0.15, 4)
    total = round(rev_c + margin_c + lev_c + multiple_c, 4)

    return {
        "total_value_change": total,
        "revenue_contribution": rev_c,
        "margin_contribution": margin_c,
        "leverage_contribution": lev_c,
        "multiple_contribution": multiple_c,
    }


def generate_exit_pipeline(org_id: str) -> Dict[str, Any]:
    """Classify portfolio companies into exit readiness buckets."""
    from api.jobs.pe_intelligence import compute_exit_timing

    empty = {
        "ready_to_exit": [],
        "approaching_exit": [],
        "early_stage": [],
        "optimal_market_window": "unknown",
    }

    if not db.db_read_enabled():
        return empty

    try:
        rows = db.safe_fetchall(
            """
            SELECT entity_id, entity_name
              FROM private_entities
             WHERE org_id = %s AND is_active = TRUE
            """,
            (org_id,),
        ) or []
    except Exception:
        return empty

    ready, approaching, early = [], [], []

    for row in rows:
        entity_id, entity_name = str(row[0]), str(row[1] or row[0])
        timing = compute_exit_timing(entity_id)
        score = timing.get("exit_readiness_score")
        if score is None:
            early.append({"entity_id": entity_id, "entity_name": entity_name, "exit_readiness_score": None})
            continue
        item = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "exit_readiness_score": score,
            "recommendation": timing.get("recommendation"),
            "months_held": timing.get("months_held"),
        }
        if score > 80:
            ready.append(item)
        elif score >= 60:
            approaching.append(item)
        else:
            early.append(item)

    # Determine market window from SRI / regime (best effort)
    market_window = "monitor"
    try:
        sri_row = db.safe_fetchone(
            "SELECT sri FROM market_breadth_daily ORDER BY as_of_date DESC LIMIT 1"
        )
        if sri_row and sri_row[0] is not None:
            sri = float(sri_row[0])
            if sri < 30:
                market_window = "favorable"
            elif sri < 55:
                market_window = "monitor"
            else:
                market_window = "avoid_near_term"
    except Exception:
        pass

    return {
        "ready_to_exit": ready,
        "approaching_exit": approaching,
        "early_stage": early,
        "optimal_market_window": market_window,
    }


def _build_narratives(
    portfolio_summary: Dict,
    risk_flags: List[str],
    exit_pipeline: Dict,
    market_context: Dict,
) -> Dict[str, str]:
    n = portfolio_summary.get("total_companies", 0)
    avg_h = portfolio_summary.get("avg_health_score", 0.0)
    on_target = portfolio_summary.get("companies_on_target", 0)
    at_risk = portfolio_summary.get("companies_at_risk", 0)
    rev_growth = portfolio_summary.get("portfolio_revenue_growth", 0.0)

    executive_summary = (
        f"Portfolio of {n} companies averaged a health score of {avg_h:.1f}/100 "
        f"with {on_target} companies on target and {at_risk} requiring attention. "
        f"Weighted portfolio revenue growth was {rev_growth*100:.1f}% over the period."
    )

    regime = market_context.get("regime", "unknown")
    sri = market_context.get("sri", 50.0)
    market_ctx_text = (
        f"The current {regime} market regime with a Systemic Risk Index of {sri:.1f}/100 "
        f"({'elevated risk environment' if sri > 55 else 'stable environment'}). "
        "Portfolio positioning reflects the prevailing macro backdrop."
    )

    top_risks = risk_flags[:3] if risk_flags else ["No material risk flags identified"]
    risk_text = f"Key risks include: {', '.join(top_risks)}."

    ready = exit_pipeline.get("ready_to_exit", [])
    n_ready = len(ready)
    lead_entity = ready[0]["entity_name"] if ready else "None"
    window = exit_pipeline.get("optimal_market_window", "unknown")
    exit_text = (
        f"{n_ready} companies show exit readiness above 80, with {lead_entity} leading. "
        f"Market window assessment: {window}."
    )

    return {
        "executive_summary": executive_summary,
        "market_context": market_ctx_text,
        "risk_section": risk_text,
        "exit_outlook": exit_text,
    }


def generate_lp_report(org_id: str, quarter: Optional[str] = None) -> LPReport:
    """Assemble a full LP report from all intelligence components."""
    today = dt.date.today()
    if quarter is None:
        q = (today.month - 1) // 3 + 1
        quarter = f"{today.year}-Q{q}"

    portfolio_summary = generate_portfolio_summary(org_id, today)
    value_attribution = generate_value_creation_attribution(org_id)
    exit_pipeline = generate_exit_pipeline(org_id)

    # Market context from latest SRI + regime
    market_context: Dict[str, Any] = {"regime": "unknown", "sri": 50.0}
    if db.db_read_enabled():
        try:
            row = db.safe_fetchone(
                """
                SELECT sri FROM market_breadth_daily
                 ORDER BY as_of_date DESC LIMIT 1
                """
            )
            if row:
                market_context = {
                    "regime": "unknown",
                    "sri": float(row[0] or 50.0),
                }
        except Exception:
            pass

    # Risk flags
    risk_flags: List[str] = []
    if portfolio_summary.get("companies_at_risk", 0) > 0:
        risk_flags.append(
            f"{portfolio_summary['companies_at_risk']} portfolio companies with health score < 40"
        )
    if portfolio_summary.get("aggregate_schilit_risk") == "high":
        risk_flags.append("Elevated accounting quality risk in portfolio")
    if market_context.get("sri", 50.0) > 65:
        risk_flags.append(f"Systemic Risk Index elevated at {market_context['sri']:.1f}")

    # Individual company reports
    individual: List[Dict] = []
    if db.db_read_enabled():
        try:
            entity_rows = db.safe_fetchall(
                """
                SELECT entity_id, entity_name, sector
                  FROM private_entities
                 WHERE org_id = %s AND is_active = TRUE
                """,
                (org_id,),
            ) or []
            from api.jobs.pe_intelligence import compute_entity_health, compute_exit_timing
            for r in entity_rows:
                eid, ename, sector = str(r[0]), str(r[1] or r[0]), str(r[2] or "")
                health = compute_entity_health(eid)
                timing = compute_exit_timing(eid)
                individual.append({
                    "entity_id": eid,
                    "entity_name": ename,
                    "sector": sector,
                    "health_score": health.get("health_score"),
                    "exit_readiness_score": timing.get("exit_readiness_score"),
                    "alert": health.get("alert", False),
                })
        except Exception:
            pass

    narratives = _build_narratives(portfolio_summary, risk_flags, exit_pipeline, market_context)

    report = LPReport(
        org_id=org_id,
        report_quarter=quarter,
        generated_at=dt.datetime.now(),
        portfolio_summary=portfolio_summary,
        individual_company_reports=individual,
        value_creation_attribution=value_attribution,
        risk_flags=risk_flags,
        exit_pipeline=exit_pipeline,
        market_context=market_context,
        narrative_sections=narratives,
    )

    # Persist to DB
    if db.db_write_enabled():
        try:
            report_dict = {
                "org_id": org_id,
                "report_quarter": quarter,
                "portfolio_summary": portfolio_summary,
                "individual_company_reports": individual,
                "value_creation_attribution": value_attribution,
                "risk_flags": risk_flags,
                "exit_pipeline": exit_pipeline,
                "market_context": market_context,
                "narrative_sections": narratives,
            }
            db.safe_execute(
                """
                INSERT INTO lp_reports (report_id, org_id, report_quarter, report, generated_at)
                VALUES (%s, %s, %s, %s::jsonb, now())
                ON CONFLICT (report_id) DO NOTHING
                """,
                (str(uuid.uuid4()), org_id, quarter, json.dumps(report_dict)),
            )
        except Exception as exc:
            logger.warning("lp_report.persist_failed org=%s err=%s", org_id, exc)

    return report
