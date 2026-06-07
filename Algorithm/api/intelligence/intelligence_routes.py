"""Phase 12.5: Intelligence API routes."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from api import db
from api.intelligence.company_dossier import get_company_dossier, run_dossier_update_job
from api.intelligence.intelligence_graph import (
    build_symbol_sector_graph,
    compute_network_centrality,
    find_hidden_connections,
    propagate_stress_through_graph,
    seed_symbol_graph,
)
from api.intelligence.regime_playbook import (
    build_regime_playbook,
    get_regime_recommendation,
    update_regime_playbook,
)
from api.intelligence.regime_transitions import (
    compute_regime_transition_probabilities,
    identify_warning_signals,
)
from api.intelligence.signal_memory import (
    compute_signal_batting_average,
    get_signal_leaderboard,
    load_signal_performance_history,
    update_signal_performance_archive,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/intelligence", tags=["intelligence"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _moat_strength(days: int) -> str:
    if days < 90:
        return "building"
    if days < 365:
        return "established"
    if days < 730:
        return "strong"
    return "exceptional"


def compute_moat_score() -> Dict[str, Any]:
    """Aggregate moat metrics from the intelligence layer."""
    _empty = {
        "total_signals_archived": 0,
        "avg_signal_war": 0.0,
        "symbols_with_dossiers": 0,
        "avg_intelligence_quality_score": 0.0,
        "graph_nodes": 0,
        "graph_edges": 0,
        "regime_playbook_coverage": 0,
        "days_of_live_data": 0,
        "moat_strength": "building",
        "components": {
            "signal_war_database": 0.0,
            "regime_playbook_depth": 0.0,
            "dossier_completeness": 0.0,
            "connection_graph_density": 0.0,
        },
        "data_irreproducibility_estimate": 0.0,
    }
    if not db.db_read_enabled():
        return _empty

    try:
        sig_row = db.safe_fetchone(
            "SELECT COUNT(*), AVG(signal_war) FROM signal_performance_archive",
        )
        total_signals = int(sig_row[0] or 0) if sig_row else 0
        avg_war = round(float(sig_row[1] or 0.0), 4) if sig_row else 0.0

        dos_row = db.safe_fetchone(
            """
            SELECT COUNT(DISTINCT symbol),
                   AVG(CASE WHEN impact_score > 0 THEN impact_score END)
              FROM company_intelligence_archive
            """,
        )
        symbols_with_dossiers = int(dos_row[0] or 0) if dos_row else 0
        avg_iq = round(float(dos_row[1] or 0.0), 2) if dos_row else 0.0

        node_row = db.safe_fetchone("SELECT COUNT(*) FROM intelligence_graph_nodes")
        graph_nodes = int(node_row[0] or 0) if node_row else 0

        edge_row = db.safe_fetchone("SELECT COUNT(*) FROM intelligence_graph_edges")
        graph_edges = int(edge_row[0] or 0) if edge_row else 0

        pb_row = db.safe_fetchone("SELECT COUNT(*) FROM regime_playbook")
        pb_coverage = int(pb_row[0] or 0) if pb_row else 0

        first_row = db.safe_fetchone(
            "SELECT MIN(signal_date) FROM signal_performance_archive",
        )
        if first_row and first_row[0]:
            first_date = first_row[0]
            if isinstance(first_date, str):
                first_date = dt.date.fromisoformat(first_date)
            days_live = (dt.date.today() - first_date).days
        else:
            days_live = 0

        # Component scores (0–100 each)
        signal_war_component = min(total_signals / 500.0, 1.0) * 100.0
        playbook_component = min(pb_coverage / 5.0, 1.0) * 100.0
        dossier_component = min(symbols_with_dossiers / 30.0, 1.0) * 100.0
        graph_density = round(graph_edges / max(graph_nodes * (graph_nodes - 1) / 2, 1), 4) if graph_nodes > 1 else 0.0
        graph_component = min(graph_density * 200.0, 100.0)

        # Data irreproducibility: weighted composite of depth and uniqueness
        data_irr = round(
            signal_war_component * 0.35
            + playbook_component * 0.25
            + dossier_component * 0.25
            + graph_component * 0.15,
            2,
        )

        return {
            "total_signals_archived": total_signals,
            "avg_signal_war": avg_war,
            "symbols_with_dossiers": symbols_with_dossiers,
            "avg_intelligence_quality_score": avg_iq,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "regime_playbook_coverage": pb_coverage,
            "days_of_live_data": days_live,
            "moat_strength": _moat_strength(days_live),
            "components": {
                "signal_war_database": round(signal_war_component, 2),
                "regime_playbook_depth": round(playbook_component, 2),
                "dossier_completeness": round(dossier_component, 2),
                "connection_graph_density": round(graph_component, 2),
            },
            "data_irreproducibility_estimate": data_irr,
        }
    except Exception as exc:
        logger.warning("compute_moat_score_failed err=%s", exc)
        return _empty


# ---------------------------------------------------------------------------
# Signal memory routes
# ---------------------------------------------------------------------------

@router.get("/signals/batting-average/{symbol}")
def get_batting_average(symbol: str, lookback_days: int = Query(252, ge=21, le=1260)):
    return compute_signal_batting_average(symbol.upper(), lookback_days)


@router.get("/signals/leaderboard")
def get_leaderboard(limit: int = Query(10, ge=1, le=50)):
    return {"leaderboard": get_signal_leaderboard(limit)}


@router.get("/signals/history/{symbol}")
def get_signal_history(symbol: str, lookback_days: int = Query(252, ge=21, le=1260)):
    records = load_signal_performance_history(symbol.upper(), lookback_days)
    return {
        "symbol": symbol.upper(),
        "record_count": len(records),
        "records": [
            {
                "signal_date": str(r.signal_date),
                "signal_label": r.signal_label,
                "dau_at_signal": r.dau_at_signal,
                "regime_at_signal": r.regime_at_signal,
                "primary_factor_driver": r.primary_factor_driver,
                "horizon_21d_return": r.horizon_21d_return,
                "batting_average": r.batting_average,
                "signal_war": r.signal_war,
            }
            for r in records
        ],
    }


@router.post("/signals/archive")
def archive_signals(as_of_date: Optional[str] = None):
    aod = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    return update_signal_performance_archive(aod)


# ---------------------------------------------------------------------------
# Regime playbook routes
# ---------------------------------------------------------------------------

@router.get("/regime-playbook")
def get_playbook(lookback_days: int = Query(504, ge=63, le=2520)):
    playbook = build_regime_playbook(lookback_days)
    return {
        "regime_count": len(playbook),
        "regimes": {
            regime: {
                "recommended_signal_types": e.recommended_signal_types,
                "avoided_signal_types": e.avoided_signal_types,
                "factor_weights": e.factor_weights,
                "historical_accuracy": e.historical_accuracy,
                "sample_count": e.sample_count,
                "regime_duration_avg_days": e.regime_duration_avg_days,
                "transition_probability": e.transition_probability,
            }
            for regime, e in playbook.items()
        },
    }


@router.get("/regime-playbook/{regime_label}")
def get_regime_rec(regime_label: str):
    playbook = build_regime_playbook()
    return get_regime_recommendation(regime_label, playbook)


@router.post("/regime-playbook/update")
def update_playbook(as_of_date: Optional[str] = None):
    aod = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    return update_regime_playbook(aod)


@router.get("/regime-transitions/{regime_label}")
def get_regime_transitions(regime_label: str, lookback_days: int = Query(504, ge=63, le=2520)):
    return compute_regime_transition_probabilities(regime_label, lookback_days)


@router.get("/regime-warnings")
def get_regime_warnings(as_of_date: Optional[str] = None):
    aod = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    warnings = identify_warning_signals(aod)
    return {"as_of_date": aod.isoformat(), "warning_count": len(warnings), "warnings": warnings}


# ---------------------------------------------------------------------------
# Company dossier routes
# ---------------------------------------------------------------------------

@router.get("/dossier/{symbol}")
def get_dossier(symbol: str, lookback_days: int = Query(365, ge=30, le=1825)):
    return get_company_dossier(symbol.upper(), lookback_days)


@router.post("/dossier/update")
def update_dossier(as_of_date: Optional[str] = None):
    aod = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    return run_dossier_update_job(aod)


# ---------------------------------------------------------------------------
# Intelligence graph routes
# ---------------------------------------------------------------------------

@router.post("/graph/seed")
def seed_graph(symbols: List[str]):
    return seed_symbol_graph([s.upper() for s in symbols])


@router.get("/graph/connections/{symbol}")
def get_connections(symbol: str, max_hops: int = Query(3, ge=1, le=4)):
    if not db.db_read_enabled():
        return find_hidden_connections(symbol.upper(), [], max_hops)

    try:
        rows = db.safe_fetchall(
            "SELECT edge_id, source_node_id, target_node_id, edge_type, weight, direction FROM intelligence_graph_edges",
        ) or []
        from api.intelligence.intelligence_graph import GraphEdge
        edges = [
            GraphEdge(
                edge_id=str(r[0]), source_node_id=str(r[1]), target_node_id=str(r[2]),
                edge_type=str(r[3]), weight=float(r[4] or 1.0), direction=str(r[5] or "bidirectional"),
            )
            for r in rows
        ]
        return find_hidden_connections(symbol.upper(), edges, max_hops)
    except Exception as exc:
        logger.warning("graph_connections_failed err=%s", exc)
        return find_hidden_connections(symbol.upper(), [], max_hops)


@router.get("/graph/centrality")
def get_centrality():
    if not db.db_read_enabled():
        return {"centrality": {}}

    try:
        rows = db.safe_fetchall(
            "SELECT edge_id, source_node_id, target_node_id, edge_type, weight, direction FROM intelligence_graph_edges",
        ) or []
        from api.intelligence.intelligence_graph import GraphEdge
        edges = [
            GraphEdge(
                edge_id=str(r[0]), source_node_id=str(r[1]), target_node_id=str(r[2]),
                edge_type=str(r[3]), weight=float(r[4] or 1.0), direction=str(r[5] or "bidirectional"),
            )
            for r in rows
        ]
        return {"centrality": compute_network_centrality(edges)}
    except Exception as exc:
        logger.warning("graph_centrality_failed err=%s", exc)
        return {"centrality": {}}


# ---------------------------------------------------------------------------
# Signal WAR endpoints (full formula)
# ---------------------------------------------------------------------------

@router.get("/signal-war/{symbol}")
def get_signal_war(symbol: str, lookback_days: int = Query(252, ge=21, le=1260)):
    stats = compute_signal_batting_average(symbol.upper(), lookback_days)
    return {
        "symbol": symbol.upper(),
        "signal_war": stats.get("signal_war"),
        "batting_average_21d": stats.get("batting_average_21d"),
        "slugging_21d": stats.get("slugging_21d"),
        "war_ic_component": stats.get("war_ic_component"),
        "league_avg": stats.get("league_avg"),
        "sample_count": stats.get("sample_count", 0),
        "regime_breakdown": stats.get("regime_breakdown", {}),
    }


@router.get("/signal-war/leaderboard")
def get_war_leaderboard(limit: int = Query(10, ge=1, le=50)):
    return {"leaderboard": get_signal_leaderboard(limit)}


# ---------------------------------------------------------------------------
# Moat score
# ---------------------------------------------------------------------------

@router.get("/moat-score")
def get_moat_score():
    return compute_moat_score()
