"""Phase 15.2: SMB Customer Concentration Risk."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db
from api.assistant.phase3.common import clamp


@dataclass
class CustomerConcentrationReport:
    entity_id: str
    top_customer_revenue_pct: float
    top_3_customer_revenue_pct: float
    concentration_risk_score: float
    concentration_label: str
    estimated_revenue_at_risk: float
    revenue_diversification_score: float
    customer_axiom_scores: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# HHI-based concentration
# ---------------------------------------------------------------------------

def compute_customer_concentration_score(
    customer_revenue_breakdown: Dict[str, float],
) -> Dict[str, Any]:
    """HHI-based customer concentration score (0–100)."""
    if not customer_revenue_breakdown:
        return {
            "concentration_risk_score": 0.0,
            "concentration_label": "safe",
            "top_customer_revenue_pct": 0.0,
            "top_3_customer_revenue_pct": 0.0,
        }

    total = sum(customer_revenue_breakdown.values())
    if total <= 0:
        return {
            "concentration_risk_score": 0.0,
            "concentration_label": "safe",
            "top_customer_revenue_pct": 0.0,
            "top_3_customer_revenue_pct": 0.0,
        }

    shares = {k: v / total for k, v in customer_revenue_breakdown.items()}
    hhi = sum(s ** 2 for s in shares.values())
    score = round(clamp(hhi * 100.0, 0.0, 100.0), 2)

    sorted_shares = sorted(shares.values(), reverse=True)
    top1_pct = sorted_shares[0] if sorted_shares else 0.0
    top3_pct = sum(sorted_shares[:3])

    if score < 10:
        label = "safe"
    elif score < 25:
        label = "watch"
    elif score < 50:
        label = "elevated"
    else:
        label = "critical"

    return {
        "concentration_risk_score": score,
        "concentration_label": label,
        "top_customer_revenue_pct": round(top1_pct, 4),
        "top_3_customer_revenue_pct": round(top3_pct, 4),
    }


# ---------------------------------------------------------------------------
# AXIOM monitoring of public customers
# ---------------------------------------------------------------------------

def compute_customer_axiom_monitoring(
    customer_ticker_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Pull AXIOM health data for any publicly traded customers."""
    if not customer_ticker_map or not db.db_read_enabled():
        return []

    results = []
    for customer_id, ticker in customer_ticker_map.items():
        try:
            row = db.safe_fetchone(
                """
                SELECT payload->>'deployable_alpha_utility',
                       payload->'engine_scores'->'critical_fragility'->>'score',
                       payload->'engine_scores'->'fundamental_reality'->'components'->>'eis_component'
                  FROM axiom_scores_daily
                 WHERE symbol = %s
                 ORDER BY as_of_date DESC
                 LIMIT 1
                """,
                (ticker,),
            )
        except Exception:
            row = None

        axiom_dau = float(row[0]) if row and row[0] is not None else None
        fragility = float(row[1]) if row and row[1] is not None else None
        eis = float(row[2]) if row and row[2] is not None else None

        alert = None
        if fragility is not None and fragility > 65:
            alert = "customer_financial_stress"
        elif eis is not None and eis < 40:
            alert = "earnings_quality_deteriorating"

        results.append({
            "customer_id": customer_id,
            "ticker": ticker,
            "axiom_dau": axiom_dau,
            "fragility_score": fragility,
            "eis_score": eis,
            "alert": alert,
        })

    return results


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------

def generate_concentration_alerts(
    report: CustomerConcentrationReport,
    axiom_monitoring: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    # Structural concentration risk
    if report.top_customer_revenue_pct > 0.50:
        alerts.append({
            "alert_type": "structural_risk",
            "severity": "critical",
            "message": (
                f"Largest customer represents {report.top_customer_revenue_pct*100:.1f}% of revenue — "
                "single customer departure could be existential"
            ),
            "recommended_action": "Aggressively diversify customer base; target largest customer < 30%",
        })

    if report.top_3_customer_revenue_pct > 0.60:
        alerts.append({
            "alert_type": "diversification_urgency",
            "severity": "high",
            "message": (
                f"Top 3 customers represent {report.top_3_customer_revenue_pct*100:.1f}% of revenue"
            ),
            "recommended_action": "Expand sales pipeline; target 10+ customers before next 12 months",
        })

    # AXIOM-driven alerts
    stressed_customers = [m for m in axiom_monitoring if m.get("alert")]
    if report.top_customer_revenue_pct > 0.30 and stressed_customers:
        for m in stressed_customers:
            alerts.append({
                "alert_type": "concentrated_customer_stress",
                "severity": "critical",
                "message": (
                    f"High concentration ({report.top_customer_revenue_pct*100:.1f}%) AND "
                    f"customer {m['ticker']} showing financial stress ({m['alert']})"
                ),
                "recommended_action": "Immediate customer health monitoring; build AR reserves",
            })

    return alerts


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def build_customer_concentration_report(
    entity_id: str,
    customer_revenue_breakdown: Dict[str, float],
    annual_revenue: float = 0.0,
    customer_ticker_map: Optional[Dict[str, str]] = None,
) -> CustomerConcentrationReport:
    conc = compute_customer_concentration_score(customer_revenue_breakdown)
    score = conc["concentration_risk_score"]
    top1_pct = conc["top_customer_revenue_pct"]
    top3_pct = conc["top_3_customer_revenue_pct"]

    estimated_revenue_at_risk = round(top1_pct * annual_revenue, 2)
    diversification_score = round(clamp(100.0 - score, 0.0, 100.0), 2)

    axiom_monitoring = compute_customer_axiom_monitoring(customer_ticker_map or {})

    report = CustomerConcentrationReport(
        entity_id=entity_id,
        top_customer_revenue_pct=top1_pct,
        top_3_customer_revenue_pct=top3_pct,
        concentration_risk_score=score,
        concentration_label=conc["concentration_label"],
        estimated_revenue_at_risk=estimated_revenue_at_risk,
        revenue_diversification_score=diversification_score,
        customer_axiom_scores=axiom_monitoring,
        alerts=[],
    )
    report.alerts = generate_concentration_alerts(report, axiom_monitoring)
    return report
