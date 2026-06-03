"""Phase 11.6: Risk Framework Routes.

All risk endpoints under /axiom/risk/.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api import db, security
from api.axiom.risk.var_engine import compute_historical_var, compute_portfolio_var
from api.axiom.risk.stress_engine import (
    SCENARIO_PARAMETERS,
    run_sornette_scenario,
    run_stress_test,
)
from api.axiom.risk.correlation_monitor import compute_rolling_correlation_matrix, detect_correlation_spike
from api.axiom.risk.drawdown_engine import compute_drawdown_series, compute_expected_recovery_time
from api.axiom.risk.systemic_risk import compute_sri, get_sri_history

router = APIRouter(
    prefix="/axiom/risk",
    tags=["risk"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)
logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACK = 252


class PortfolioRequest(BaseModel):
    positions: Dict[str, float] = Field(default_factory=dict)
    portfolio_id: str = "default"
    horizon_days: int = 5
    confidence: float = 0.99


class StressTestRequest(BaseModel):
    positions: Dict[str, float] = Field(default_factory=dict)


def _load_symbol_returns(symbol: str, lookback: int = _DEFAULT_LOOKBACK) -> List[float]:
    if not db.db_read_enabled():
        return []
    try:
        since = dt.date.today() - dt.timedelta(days=lookback + 30)
        rows = db.safe_fetchall(
            """
            SELECT return_pct
              FROM signal_pnl_daily
             WHERE symbol = %s
               AND signal_date >= %s
               AND return_pct IS NOT NULL
               AND horizon_days = 1
             ORDER BY signal_date ASC
             LIMIT %s
            """,
            (symbol, since, lookback),
        )
        return [float(r[0]) for r in (rows or [])]
    except Exception:
        return []


def _load_axiom_scores(symbol: str, as_of_date: Optional[dt.date] = None) -> Dict:
    if not db.db_read_enabled():
        return {}
    try:
        import json
        aod = as_of_date or dt.date.today()
        row = db.safe_fetchone(
            """
            SELECT payload
              FROM axiom_scores_daily
             WHERE symbol = %s AND as_of_date <= %s
             ORDER BY as_of_date DESC LIMIT 1
            """,
            (symbol, aod),
        )
        if not row:
            return {}
        p = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
        return p
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/var/{symbol}")
def get_symbol_var(
    symbol: str,
    confidence: float = Query(default=0.99),
    horizon_days: int = Query(default=5),
) -> Dict[str, Any]:
    returns = _load_symbol_returns(symbol.upper())
    payload = _load_axiom_scores(symbol.upper())
    mtrs_score = float(
        (payload.get("engine_scores") or {})
        .get("critical_fragility", {})
        .get("score") or 50.0
    )
    var = compute_historical_var(returns, confidence=confidence, horizon_days=horizon_days, mtrs_score=mtrs_score)
    return {"symbol": symbol.upper(), "confidence": confidence, "horizon_days": horizon_days, **var}


@router.post("/portfolio")
def compute_portfolio_risk(req: PortfolioRequest) -> Dict[str, Any]:
    positions = {s.upper(): w for s, w in req.positions.items()}
    symbol_returns = {sym: _load_symbol_returns(sym) for sym in positions}
    axiom_scores = {sym: _load_axiom_scores(sym) for sym in positions}

    var_result = compute_portfolio_var(
        positions, symbol_returns, confidence=req.confidence, horizon_days=req.horizon_days
    )
    stress_result = run_stress_test(positions, axiom_scores)
    corr_result = compute_rolling_correlation_matrix(symbol_returns)

    return {
        "portfolio_id": req.portfolio_id,
        "as_of_date": dt.date.today().isoformat(),
        "var": var_result,
        "stress_test": stress_result,
        "correlation": corr_result,
    }


@router.get("/stress-test/scenarios")
def get_stress_scenarios() -> Dict[str, Any]:
    return {
        "scenarios": {
            k: {
                "historical_drawdown_pct": v["historical_drawdown_pct"],
                "recovery_days": v["recovery_days"],
                "sector_impact": v["sector_impact"],
            }
            for k, v in SCENARIO_PARAMETERS.items()
        }
    }


@router.post("/stress-test")
def run_portfolio_stress_test(req: StressTestRequest) -> Dict[str, Any]:
    positions = {s.upper(): w for s, w in req.positions.items()}
    axiom_scores = {sym: _load_axiom_scores(sym) for sym in positions}
    return run_stress_test(positions, axiom_scores)


@router.get("/correlation")
def get_correlation(
    lookback_days: int = Query(default=21),
) -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {"correlation_matrix": {}, "correlation_regime": "unknown"}

    try:
        since = dt.date.today() - dt.timedelta(days=lookback_days + 5)
        rows = db.safe_fetchall(
            """
            SELECT symbol, return_pct
              FROM signal_pnl_daily
             WHERE signal_date >= %s
               AND return_pct IS NOT NULL
               AND horizon_days = 1
             ORDER BY symbol, signal_date
            """,
            (since,),
        ) or []
        symbol_returns: Dict[str, List[float]] = {}
        for r in rows:
            symbol_returns.setdefault(str(r[0]), []).append(float(r[1]))

        if len(symbol_returns) < 2:
            return {"correlation_matrix": {}, "correlation_regime": "unknown"}

        matrix_result = compute_rolling_correlation_matrix(symbol_returns, window=lookback_days)
        spike = detect_correlation_spike(
            matrix_result["avg_pairwise_correlation"],
            historical_avg=0.30,
        )
        return {**matrix_result, "spike_detection": spike}
    except Exception as exc:
        logger.warning("correlation_endpoint_error err=%s", exc)
        return {"error": str(exc)}


@router.get("/drawdown/{symbol}")
def get_symbol_drawdown(
    symbol: str,
    lookback_days: int = Query(default=252),
    regime_label: str = Query(default="TRENDING"),
) -> Dict[str, Any]:
    returns = _load_symbol_returns(symbol.upper(), lookback=lookback_days)
    if not returns:
        return {"symbol": symbol.upper(), "error": "no_return_data"}

    dd = compute_drawdown_series(returns)
    recovery = compute_expected_recovery_time(
        dd["current_drawdown_pct"],
        historical_cagr=dd["calmar_ratio"] * abs(dd["max_drawdown_pct"]) if dd["max_drawdown_pct"] < 0 else 0.10,
        regime_label=regime_label,
    )
    return {"symbol": symbol.upper(), "drawdown": dd, "recovery_estimate": recovery}


@router.get("/sri")
def get_sri(
    as_of_date: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    aod = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    return compute_sri(aod)


@router.get("/sri/history")
def get_sri_history_endpoint(
    lookback_days: int = Query(default=63),
) -> Dict[str, Any]:
    history = get_sri_history(lookback_days=lookback_days)
    return {"lookback_days": lookback_days, "history": history}
