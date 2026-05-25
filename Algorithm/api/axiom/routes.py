"""Session 12: AXIOM sizing API.

POST /axiom/size  — compute a fractional-Kelly position weight from stored
                    or inline AXIOM scores.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api import db, security
from api.axiom.sizer import KellySizeResult, compute_kelly_size

router = APIRouter(
    prefix="/axiom",
    tags=["axiom"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

logger = logging.getLogger(__name__)

_SAFE_FLOAT_SENTINEL = object()


def _sf(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        import math
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SizeRequest(BaseModel):
    symbol: str
    as_of_date: Optional[str] = None

    # --- optional inline score overrides (skip DB lookup if all provided) ---
    dau: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    fragility_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    liquidity_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    research_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    overall_confidence: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    deployability_tier: Optional[str] = None

    # --- calibration & IC (loaded from DB if not provided) ---
    hit_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ic_state: Optional[str] = None

    # --- portfolio context ---
    max_weight: float = Field(default=0.10, ge=0.001, le=1.0)
    portfolio_heat: float = Field(default=0.0, ge=0.0, le=1.0)
    fractional_kelly: float = Field(default=0.5, gt=0.0, le=1.0)


class SizeResponse(BaseModel):
    symbol: str
    as_of_date: str
    suggested_weight: float
    suggested_weight_pct: str
    kelly_gross_weight: float
    fractional_kelly_applied: float
    ic_kelly_multiplier: float
    fragility_penalty_applied: float
    active_constraint: str
    size_band: str
    deployability_tier: str
    ic_state: str
    downside_flags: list
    rationale: str
    inputs: dict
    data_source: str   # "db" | "inline" | "partial"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_axiom_scores(symbol: str, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
    if not db.db_read_enabled():
        return None
    row = db.safe_fetchone(
        """
        SELECT payload
        FROM axiom_scores_daily
        WHERE symbol = %s AND as_of_date = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (symbol, as_of_date),
    )
    if not row or not row[0]:
        return None
    payload = row[0]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return None
    return payload


def _load_ic_state(as_of_date: dt.date, horizon: str = "21d") -> Optional[str]:
    if not db.db_read_enabled():
        return None
    row = db.safe_fetchone(
        """
        SELECT ic_state
        FROM signal_ic_daily
        WHERE score_field = 'composite'
          AND horizon_label = %s
          AND as_of_date <= %s
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (horizon, as_of_date),
    )
    if row and row[0]:
        return str(row[0])
    return None


def _load_hit_rate(horizon: str = "21d") -> Optional[float]:
    """Load the most recent calibration hit rate for the 21d horizon."""
    if not db.db_read_enabled():
        return None
    row = db.safe_fetchone(
        """
        SELECT payload->'diagnostics'->'overall_outcome_metrics'->>'hit_rate'
        FROM axiom_calibration_snapshots
        WHERE horizon_label = %s
        ORDER BY as_of_date DESC NULLS LAST, created_at DESC
        LIMIT 1
        """,
        (horizon,),
    )
    if row and row[0] is not None:
        try:
            return float(row[0])
        except (TypeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/size", response_model=SizeResponse)
def axiom_size(req: SizeRequest) -> SizeResponse:
    as_of_date_str = req.as_of_date or dt.date.today().isoformat()
    try:
        as_of_date = dt.date.fromisoformat(as_of_date_str)
    except ValueError:
        as_of_date = dt.date.today()
        as_of_date_str = as_of_date.isoformat()

    # ------------------------------------------------------------------
    # 1. Resolve scores: inline override → DB lookup → safe defaults
    # ------------------------------------------------------------------
    data_source = "inline"
    db_payload: Optional[Dict[str, Any]] = None

    inline_complete = all(
        v is not None for v in [req.dau, req.fragility_score, req.deployability_tier]
    )

    if not inline_complete:
        db_payload = _load_axiom_scores(req.symbol, as_of_date)
        if db_payload:
            data_source = "db" if not inline_complete else "partial"
        else:
            data_source = "inline"

    def _from_db(key: str, subkey: Optional[str] = None, default: float = 0.0) -> float:
        if db_payload is None:
            return default
        if subkey:
            block = db_payload.get(key) or {}
            if isinstance(block, dict):
                return _sf(block.get(subkey), default)
            return default
        return _sf(db_payload.get(key), default)

    def _engine(engine_name: str, default: float = 50.0) -> float:
        if db_payload is None:
            return default
        eng = (db_payload.get("engine_scores") or {}).get(engine_name) or {}
        return _sf(eng.get("score"), default)

    dau              = req.dau              if req.dau              is not None else _from_db("deployable_alpha_utility", default=0.0)
    fragility_score  = req.fragility_score  if req.fragility_score  is not None else _engine("critical_fragility", default=50.0)
    liquidity_score  = req.liquidity_score  if req.liquidity_score  is not None else _engine("liquidity_convexity", default=50.0)
    research_score   = req.research_score   if req.research_score   is not None else _engine("research_integrity", default=50.0)
    overall_conf     = req.overall_confidence if req.overall_confidence is not None else _from_db("overall_confidence", default=50.0)
    deploy_tier      = req.deployability_tier if req.deployability_tier is not None else (
                          str(db_payload.get("deployability_tier") or "monitor_only") if db_payload else "monitor_only"
                       )

    if data_source == "db" and any(v is not None for v in [req.dau, req.fragility_score, req.deployability_tier]):
        data_source = "partial"

    # ------------------------------------------------------------------
    # 2. Resolve IC state and hit rate
    # ------------------------------------------------------------------
    ic_state = req.ic_state or _load_ic_state(as_of_date) or "INSUFFICIENT"
    hit_rate = req.hit_rate if req.hit_rate is not None else _load_hit_rate()

    # ------------------------------------------------------------------
    # 3. Run sizer
    # ------------------------------------------------------------------
    result: KellySizeResult = compute_kelly_size(
        symbol=req.symbol,
        as_of_date=as_of_date_str,
        dau=dau,
        fragility_score=fragility_score,
        liquidity_score=liquidity_score,
        research_score=research_score,
        overall_confidence=overall_conf,
        deployability_tier=deploy_tier,
        hit_rate=hit_rate,
        ic_state=ic_state,
        fractional_kelly=req.fractional_kelly,
        max_weight=req.max_weight,
        portfolio_heat=req.portfolio_heat,
    )

    logger.info(
        "axiom.size symbol=%s weight=%.4f tier=%s ic=%s constraint=%s",
        req.symbol, result.suggested_weight, result.deployability_tier,
        result.ic_state, result.active_constraint,
    )

    return SizeResponse(
        symbol=result.symbol,
        as_of_date=result.as_of_date,
        suggested_weight=result.suggested_weight,
        suggested_weight_pct=f"{result.suggested_weight * 100:.2f}%",
        kelly_gross_weight=result.kelly_gross_weight,
        fractional_kelly_applied=result.fractional_kelly_applied,
        ic_kelly_multiplier=result.ic_kelly_multiplier,
        fragility_penalty_applied=result.fragility_penalty_applied,
        active_constraint=result.active_constraint,
        size_band=result.size_band,
        deployability_tier=result.deployability_tier,
        ic_state=result.ic_state,
        downside_flags=result.downside_flags,
        rationale=result.rationale,
        inputs=result.inputs,
        data_source=data_source,
    )
