"""Phase 13.4: PE Portfolio Supply Chain Risk — public market analog monitoring."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from api import db
from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)


def find_public_market_analogs(
    entity_name: str,
    sector: str,
    revenue: float,
) -> List[Dict[str, Any]]:
    """Find public companies that are likely peers of a PE portfolio company."""
    if not db.db_read_enabled():
        return []

    try:
        rows = db.safe_fetchall(
            """
            SELECT DISTINCT ON (a.symbol)
                a.symbol,
                (a.payload->>'deployable_alpha_utility')::numeric AS dau,
                (a.payload->'engine_scores'->'critical_fragility'->>'score')::numeric AS fragility
              FROM axiom_scores_daily a
             WHERE a.payload->>'sector' = %s
             ORDER BY a.symbol, a.as_of_date DESC
            """,
            (sector,),
        ) or []
    except Exception as exc:
        logger.warning("supply_chain.analogs_failed entity=%s err=%s", entity_name, exc)
        return []

    analogs = []
    for row in rows:
        sym = str(row[0])
        dau = float(row[1]) if row[1] is not None else 50.0
        fragility = float(row[2]) if row[2] is not None else 50.0
        analogs.append({
            "symbol": sym,
            "company_name": sym,
            "match_score": round(dau / 100.0, 4),
            "current_dau": round(dau, 2),
            "fragility_score": round(fragility, 2),
        })

    analogs.sort(key=lambda x: x["current_dau"], reverse=True)
    return analogs[:5]


def _systemic_label(risk_score: float) -> str:
    if risk_score < 30:
        return "low"
    if risk_score < 55:
        return "moderate"
    if risk_score < 75:
        return "high"
    return "critical"


def compute_supply_chain_stress_index(
    org_id: str,
    portfolio_entities: List[Dict],
) -> Dict[str, Any]:
    """Aggregate supply chain stress across the portfolio using public analog fragility."""
    if not db.db_read_enabled():
        return {
            "org_id": org_id,
            "portfolio_supply_chain_risk": 50.0,
            "high_risk_entities": [],
            "supply_chain_stress_alerts": [],
            "systemic_portfolio_risk": "moderate",
        }

    try:
        from api.intelligence.intelligence_graph import GraphEdge, propagate_stress_through_graph

        entity_risks: List[float] = []
        high_risk: List[str] = []
        sc_alerts: List[Dict] = []

        for entity in portfolio_entities:
            entity_id = str(entity.get("entity_id", ""))
            entity_name = str(entity.get("entity_name", entity_id))
            sector = str(entity.get("sector", "Unknown"))
            revenue = float(entity.get("revenue") or 0.0)

            analogs = find_public_market_analogs(entity_name, sector, revenue)
            if not analogs:
                entity_risks.append(50.0)
                continue

            max_fragility = max(a["fragility_score"] for a in analogs)
            entity_risk = clamp(max_fragility, 0.0, 100.0)
            entity_risks.append(entity_risk)

            if max_fragility > 65:
                high_risk.append(entity_name)
                for a in analogs:
                    if a["fragility_score"] > 65:
                        sc_alerts.append({
                            "entity_id": entity_id,
                            "entity_name": entity_name,
                            "analog_symbol": a["symbol"],
                            "fragility_score": a["fragility_score"],
                            "severity": "high" if a["fragility_score"] > 80 else "medium",
                        })

        aggregate_risk = sum(entity_risks) / len(entity_risks) if entity_risks else 50.0
        aggregate_risk = clamp(aggregate_risk, 0.0, 100.0)

        return {
            "org_id": org_id,
            "portfolio_supply_chain_risk": round(aggregate_risk, 2),
            "high_risk_entities": high_risk,
            "supply_chain_stress_alerts": sc_alerts,
            "systemic_portfolio_risk": _systemic_label(aggregate_risk),
        }
    except Exception as exc:
        logger.warning("supply_chain.stress_index_failed org=%s err=%s", org_id, exc)
        return {
            "org_id": org_id,
            "portfolio_supply_chain_risk": 50.0,
            "high_risk_entities": [],
            "supply_chain_stress_alerts": [],
            "systemic_portfolio_risk": "moderate",
        }


def monitor_portfolio_analogs(org_id: str) -> Dict[str, Any]:
    """Check public analogs for each portfolio entity for stress signals."""
    if not db.db_read_enabled():
        return {
            "alerts": [],
            "clean_entities": [],
            "as_of_date": dt.date.today().isoformat(),
        }

    try:
        entity_rows = db.safe_fetchall(
            """
            SELECT entity_id, entity_name, sector, public_peer_symbol
              FROM private_entities
             WHERE org_id = %s AND is_active = TRUE
            """,
            (org_id,),
        ) or []
    except Exception as exc:
        logger.warning("supply_chain.monitor_failed org=%s err=%s", org_id, exc)
        return {"alerts": [], "clean_entities": [], "as_of_date": dt.date.today().isoformat()}

    alerts: List[Dict] = []
    clean: List[str] = []

    for row in entity_rows:
        entity_id, entity_name, sector, peer_sym = row[0], row[1], row[2], row[3]
        entity_name = entity_name or entity_id
        symbols_to_check = [peer_sym] if peer_sym else []
        if not symbols_to_check and sector:
            analogs = find_public_market_analogs(entity_name or "", sector or "", 0.0)
            symbols_to_check = [a["symbol"] for a in analogs[:3]]

        entity_has_alert = False
        for sym in symbols_to_check:
            try:
                row_axiom = db.safe_fetchone(
                    """
                    SELECT payload FROM axiom_scores_daily
                     WHERE symbol = %s ORDER BY as_of_date DESC LIMIT 1
                    """,
                    (sym,),
                )
                if not row_axiom or not row_axiom[0]:
                    continue
                payload = row_axiom[0] if isinstance(row_axiom[0], dict) else {}
                engines = payload.get("engine_scores", {})
                frag_comps = engines.get("critical_fragility", {}).get("components", {})
                fragility = float(frag_comps.get("mtrs_component") or
                                  engines.get("critical_fragility", {}).get("score") or 50.0)
                scps = float(frag_comps.get("scps_component") or 0.0)
                pess = float(frag_comps.get("pess_component") or 0.0)
                regime = str(payload.get("regime_label") or "")

                if fragility > 65:
                    alerts.append({
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "analog_symbol": sym,
                        "alert_type": "supply_chain_stress",
                        "severity": "high",
                        "description": f"Analog {sym} fragility={fragility:.1f} > 65",
                    })
                    entity_has_alert = True
                if scps > 70:
                    alerts.append({
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "analog_symbol": sym,
                        "alert_type": "bubble_risk",
                        "severity": "high",
                        "description": f"Analog {sym} Sornette score={scps:.1f} > 70",
                    })
                    entity_has_alert = True
                if pess > 65:
                    alerts.append({
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "analog_symbol": sym,
                        "alert_type": "earnings_stress",
                        "severity": "medium",
                        "description": f"Analog {sym} PESS={pess:.1f} > 65",
                    })
                    entity_has_alert = True
                if regime == "HIGH_VOL":
                    alerts.append({
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "analog_symbol": sym,
                        "alert_type": "regime_stress",
                        "severity": "medium",
                        "description": f"Analog {sym} in HIGH_VOL regime",
                    })
                    entity_has_alert = True
            except Exception:
                pass

        if not entity_has_alert:
            clean.append(entity_name)

    return {
        "alerts": alerts,
        "clean_entities": clean,
        "as_of_date": dt.date.today().isoformat(),
    }
