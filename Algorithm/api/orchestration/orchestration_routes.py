"""Phase 17.5: Intelligence Orchestration Routes."""
from __future__ import annotations

import datetime as dt
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel

from api.jobs.tenant_auth import require_tier
from fastapi import Depends

orch_router = APIRouter(
    prefix="/orchestration",
    tags=["orchestration"],
    dependencies=[Depends(require_tier("enterprise"))],
)

intel_router = APIRouter(
    prefix="/intelligence",
    tags=["intelligence"],
    dependencies=[Depends(require_tier("enterprise"))],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PipelineRunIn(BaseModel):
    as_of_date: Optional[str] = None
    symbols: Optional[List[str]] = None
    stages_to_run: Optional[List[str]] = None


class SelfImprovementTriggerIn(BaseModel):
    force_retrain: bool = False
    min_new_samples: int = 20


# ---------------------------------------------------------------------------
# Pipeline routes
# ---------------------------------------------------------------------------

_pipeline_run_id: Optional[str] = None
_pipeline_lock = threading.Lock()


@orch_router.post("/pipeline/run")
def trigger_pipeline_run(body: PipelineRunIn, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger full pipeline asynchronously — returns run_id immediately."""
    import uuid
    run_id = str(uuid.uuid4())

    as_of_date = None
    if body.as_of_date:
        try:
            as_of_date = dt.date.fromisoformat(body.as_of_date)
        except Exception:
            pass

    def _run():
        global _pipeline_run_id
        from api.orchestration.pipeline_orchestrator import run_full_pipeline
        result = run_full_pipeline(
            as_of_date=as_of_date,
            symbols=body.symbols,
            stages_to_run=body.stages_to_run,
        )
        with _pipeline_lock:
            _pipeline_run_id = result.run_id

    background_tasks.add_task(_run)
    return {"status": "triggered", "run_id": run_id}


@orch_router.get("/pipeline/status")
def get_pipeline_status() -> Dict[str, Any]:
    from api.orchestration.pipeline_orchestrator import get_pipeline_status as _get_status
    status = _get_status()
    return status or {"status": "no_pipeline_runs_found"}


@orch_router.get("/pipeline/history")
def get_pipeline_history() -> List[Dict[str, Any]]:
    from api.orchestration.pipeline_orchestrator import get_pipeline_history as _get_hist
    return _get_hist(limit=10)


# ---------------------------------------------------------------------------
# Health routes
# ---------------------------------------------------------------------------

@orch_router.get("/health")
def get_system_health() -> Dict[str, Any]:
    from api.orchestration.system_health import SystemHealth, compute_system_health
    health = compute_system_health()
    return {
        "as_of": health.as_of.isoformat(),
        "overall_health_score": health.overall_health_score,
        "overall_status": health.overall_status,
        "database_health": health.database_health,
        "data_freshness": health.data_freshness,
        "provider_health": health.provider_health,
        "ml_model_health": health.ml_model_health,
        "signal_quality": health.signal_quality,
        "pipeline_health": health.pipeline_health,
        "sri_level": health.sri_level,
        "active_alerts": health.active_alerts,
    }


@orch_router.get("/health/history")
def get_health_history(lookback_days: int = Query(default=30)) -> List[Dict[str, Any]]:
    from api.orchestration.system_health import get_health_history as _hist
    return _hist(lookback_days=lookback_days)


@orch_router.get("/health/alerts")
def get_active_alerts() -> Dict[str, Any]:
    from api.orchestration.system_health import compute_system_health
    health = compute_system_health()
    return {"active_alerts": health.active_alerts, "count": len(health.active_alerts)}


# ---------------------------------------------------------------------------
# Self-improvement routes
# ---------------------------------------------------------------------------

@orch_router.get("/self-improvement/status")
def get_self_improvement_status() -> Dict[str, Any]:
    from api.orchestration.self_improvement import SelfImprovementStatus, check_self_improvement_status
    s = check_self_improvement_status()
    return {
        "last_model_training": s.last_model_training.isoformat() if s.last_model_training else None,
        "last_model_version": s.last_model_version,
        "training_sample_count": s.training_sample_count,
        "current_psi_score": s.current_psi_score,
        "drift_warning": s.drift_warning,
        "weight_optimization_pending": s.weight_optimization_pending,
        "effective_breadth": s.effective_breadth,
        "amqs_score": s.amqs_score,
        "next_recommended_action": s.next_recommended_action,
    }


@orch_router.post("/self-improvement/trigger")
def trigger_self_improvement_endpoint(body: SelfImprovementTriggerIn) -> Dict[str, Any]:
    from api.orchestration.self_improvement import trigger_self_improvement
    return trigger_self_improvement(
        force_retrain=body.force_retrain,
        min_new_samples=body.min_new_samples,
    )


@orch_router.get("/self-improvement/history")
def get_self_improvement_history(lookback_days: int = Query(default=90)) -> List[Dict[str, Any]]:
    from api.orchestration.self_improvement import get_improvement_history
    return get_improvement_history(lookback_days=lookback_days)


# ---------------------------------------------------------------------------
# Universal intelligence routes
# ---------------------------------------------------------------------------

@intel_router.get("/universal/{symbol}")
def get_universal_intelligence(symbol: str) -> Dict[str, Any]:
    from api.universal.intelligence_api import (
        UniversalIntelligenceResponse,
        assemble_universal_intelligence,
        cache_universal_response,
    )
    resp = assemble_universal_intelligence(symbol.upper())
    cache_universal_response(symbol.upper(), resp)
    return {
        "symbol": resp.symbol,
        "as_of_date": resp.as_of_date.isoformat(),
        "signal_label": resp.signal_label,
        "dau": resp.dau,
        "ml_adjusted_dau": resp.ml_adjusted_dau,
        "analyst_rating": resp.analyst_rating,
        "conviction": resp.conviction,
        "regime_label": resp.regime_label,
        "regime_strength": resp.regime_strength,
        "systemic_risk_index": resp.systemic_risk_index,
        "ic_state": resp.ic_state,
        "intelligence_quality_score": resp.intelligence_quality_score,
        "eis_score": resp.eis_score,
        "caps_score": resp.caps_score,
        "fragility_score": resp.fragility_score,
        "scps_score": resp.scps_score,
        "bfs_score": resp.bfs_score,
        "factor_composite_score": resp.factor_composite_score,
        "osms_score": resp.osms_score,
        "ias_score": resp.ias_score,
        "pess_score": resp.pess_score,
        "var_1d_99": resp.var_1d_99,
        "sri": resp.sri,
        "primary_driver": resp.primary_driver,
        "primary_conclusion": resp.primary_conclusion,
        "top_supporting_evidence": resp.top_supporting_evidence,
        "top_risk": resp.top_risk,
        "signal_batting_average": resp.signal_batting_average,
        "dossier_event_count": resp.dossier_event_count,
        "moat_score": resp.moat_score,
        "data_freshness_hours": resp.data_freshness_hours,
        "staleness_warning": resp.staleness_warning,
    }


@intel_router.get("/universal/batch")
def get_universal_intelligence_batch(
    symbols: str = Query(..., description="Comma-separated symbols, max 30"),
) -> List[Dict[str, Any]]:
    from api.universal.intelligence_api import (
        assemble_universal_intelligence,
        cache_universal_response,
    )
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:30]
    results = []
    for sym in symbol_list:
        resp = assemble_universal_intelligence(sym)
        cache_universal_response(sym, resp)
        results.append({
            "symbol": resp.symbol,
            "as_of_date": resp.as_of_date.isoformat(),
            "signal_label": resp.signal_label,
            "dau": resp.dau,
            "analyst_rating": resp.analyst_rating,
            "conviction": resp.conviction,
            "data_freshness_hours": resp.data_freshness_hours,
            "staleness_warning": resp.staleness_warning,
        })
    return results
