"""AXIOM sizing and memo API.

POST /axiom/size             — fractional-Kelly position weight
POST /axiom/memo             — generate IC memo with lineage hash
GET  /axiom/memo/{memo_id}   — retrieve stored memo
GET  /axiom/memo/verify/{h}  — retrieve by lineage hash (tamper-check)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api import db, security
from api.axiom.sizer import KellySizeResult, compute_kelly_size
from api.axiom.memo import (
    AxiomMemo,
    build_memo,
    compute_lineage_hash,
    build_canonical_inputs,
    store_memo,
    load_memo_by_id,
    load_memo_by_hash,
)
from api.axiom.screener import screen_universe

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
    active_constraint: str
    size_band: str
    deployability_tier: str
    ic_state: str
    downside_flags: list
    rationale: str
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
        active_constraint=result.active_constraint,
        size_band=result.size_band,
        deployability_tier=result.deployability_tier,
        ic_state=result.ic_state,
        downside_flags=result.downside_flags,
        rationale=result.rationale,
        data_source=data_source,
    )


# ---------------------------------------------------------------------------
# Memo helpers
# ---------------------------------------------------------------------------

def _load_signal_context(symbol: str, as_of_date: dt.date) -> Dict[str, str]:
    """Load signal direction and regime from prosperity_signals_daily."""
    if not db.db_read_enabled():
        return {}
    try:
        row = db.safe_fetchone(
            """
            SELECT signal, regime
            FROM prosperity_signals_daily
            WHERE symbol = %s AND as_of = %s
            ORDER BY lookback DESC
            LIMIT 1
            """,
            (symbol, as_of_date),
        )
        if not row:
            return {}
        return {
            "signal_label": str(row[0]) if row[0] else "HOLD",
            "regime_label": str(row[1]) if row[1] else "unknown",
        }
    except Exception:
        return {}


def _load_breadth_state(as_of_date: dt.date) -> Optional[str]:
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            "SELECT breadth_state FROM market_breadth_daily WHERE as_of_date = %s",
            (as_of_date,),
        )
        return str(row[0]) if row and row[0] else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Memo request / response models
# ---------------------------------------------------------------------------

class MemoRequest(BaseModel):
    symbol: str
    as_of_date: Optional[str] = None

    # Optional inline score overrides (same pattern as SizeRequest)
    dau: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    fragility_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    liquidity_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    research_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    overall_confidence: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    deployability_tier: Optional[str] = None

    # Signal context (loaded from DB if not provided)
    signal_label: Optional[str] = None
    regime_label: Optional[str] = None
    breadth_state: Optional[str] = None

    # Calibration & IC
    hit_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ic_state: Optional[str] = None

    # Portfolio context
    max_weight: float = Field(default=0.10, ge=0.001, le=1.0)
    portfolio_heat: float = Field(default=0.0, ge=0.0, le=1.0)
    fractional_kelly: float = Field(default=0.5, gt=0.0, le=1.0)

    store: bool = True


# ---------------------------------------------------------------------------
# Memo endpoints
# ---------------------------------------------------------------------------

@router.post("/memo")
def axiom_memo(req: MemoRequest) -> Dict[str, Any]:
    as_of_date_str = req.as_of_date or dt.date.today().isoformat()
    try:
        as_of_date = dt.date.fromisoformat(as_of_date_str)
    except ValueError:
        as_of_date = dt.date.today()
        as_of_date_str = as_of_date.isoformat()

    # ------------------------------------------------------------------
    # 1. Resolve AXIOM scores (same logic as /axiom/size)
    # ------------------------------------------------------------------
    data_source = "inline"
    db_payload: Optional[Dict[str, Any]] = None

    inline_complete = all(
        v is not None for v in [req.dau, req.fragility_score, req.deployability_tier]
    )
    if not inline_complete:
        db_payload = _load_axiom_scores(req.symbol, as_of_date)
        if db_payload:
            data_source = "db"

    def _from_db(key: str, default: float = 0.0) -> float:
        if db_payload is None:
            return default
        return _sf(db_payload.get(key), default)

    def _engine(engine_name: str, default: float = 50.0) -> float:
        if db_payload is None:
            return default
        eng = (db_payload.get("engine_scores") or {}).get(engine_name) or {}
        return _sf(eng.get("score"), default)

    dau             = req.dau             if req.dau             is not None else _from_db("deployable_alpha_utility", 0.0)
    fragility_score = req.fragility_score if req.fragility_score is not None else _engine("critical_fragility", 50.0)
    liquidity_score = req.liquidity_score if req.liquidity_score is not None else _engine("liquidity_convexity", 50.0)
    research_score  = req.research_score  if req.research_score  is not None else _engine("research_integrity", 50.0)
    overall_conf    = req.overall_confidence if req.overall_confidence is not None else _from_db("overall_confidence", 50.0)
    deploy_tier     = req.deployability_tier if req.deployability_tier is not None else (
                        str(db_payload.get("deployability_tier") or "monitor_only") if db_payload else "monitor_only"
                      )

    if data_source == "db" and any(
        v is not None for v in [req.dau, req.fragility_score, req.deployability_tier]
    ):
        data_source = "partial"

    engine_scores = (db_payload.get("engine_scores") or None) if db_payload else None

    # ------------------------------------------------------------------
    # 2. Resolve IC + calibration
    # ------------------------------------------------------------------
    ic_state = req.ic_state or _load_ic_state(as_of_date) or "INSUFFICIENT"
    hit_rate = req.hit_rate if req.hit_rate is not None else _load_hit_rate()

    # ------------------------------------------------------------------
    # 3. Resolve signal context
    # ------------------------------------------------------------------
    sig_ctx = _load_signal_context(req.symbol, as_of_date) if (
        req.signal_label is None or req.regime_label is None
    ) else {}

    signal_label  = req.signal_label  or sig_ctx.get("signal_label",  "HOLD")
    regime_label  = req.regime_label  or sig_ctx.get("regime_label",  "unknown")
    breadth_state = req.breadth_state or _load_breadth_state(as_of_date) or "NEUTRAL"

    # ------------------------------------------------------------------
    # 4. Run Kelly sizer
    # ------------------------------------------------------------------
    sizing: KellySizeResult = compute_kelly_size(
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

    # ------------------------------------------------------------------
    # 5. Compute conviction score
    # ------------------------------------------------------------------
    from api.jobs.alerts import compute_conviction_score
    conviction_score = compute_conviction_score(
        dau=dau,
        signal_label=signal_label,
        regime_label=regime_label,
        breadth_state=breadth_state,
        ic_state=ic_state,
    )

    # ------------------------------------------------------------------
    # 6. Build memo with lineage hash
    # ------------------------------------------------------------------
    memo: AxiomMemo = build_memo(
        symbol=req.symbol,
        as_of_date=as_of_date_str,
        dau=dau,
        fragility_score=fragility_score,
        liquidity_score=liquidity_score,
        research_score=research_score,
        overall_confidence=overall_conf,
        deployability_tier=deploy_tier,
        ic_state=ic_state,
        hit_rate=hit_rate,
        fractional_kelly=req.fractional_kelly,
        max_weight=req.max_weight,
        signal_label=signal_label,
        regime_label=regime_label,
        breadth_state=breadth_state,
        conviction_score=conviction_score,
        suggested_weight=sizing.suggested_weight,
        kelly_gross_weight=sizing.kelly_gross_weight,
        fractional_kelly_applied=sizing.fractional_kelly_applied,
        ic_kelly_multiplier=sizing.ic_kelly_multiplier,
        fragility_penalty_applied=sizing.fragility_penalty_applied,
        active_constraint=sizing.active_constraint,
        size_band=sizing.size_band,
        downside_flags=sizing.downside_flags,
        rationale=sizing.rationale,
        data_source=data_source,
        engine_scores=engine_scores,
    )

    stored = False
    if req.store and db.db_write_enabled():
        stored = store_memo(memo)

    logger.info(
        "axiom.memo symbol=%s signal=%s conviction=%.1f weight=%.4f hash=%s",
        memo.symbol, memo.signal_label, memo.conviction_score,
        memo.suggested_weight, memo.lineage_hash[:12],
    )

    return {
        "status": "ok",
        "memo_id": memo.memo_id,
        "lineage_hash": memo.lineage_hash,
        "stored": stored,
        "memo": memo.memo_body,
    }


@router.get("/memo/verify/{lineage_hash}")
def axiom_memo_by_hash(lineage_hash: str) -> Dict[str, Any]:
    row = load_memo_by_hash(lineage_hash)
    if not row:
        raise HTTPException(status_code=404, detail="memo_not_found")
    return {"status": "ok", "verified": True, "memo": row}


@router.get("/memo/{memo_id}")
def axiom_memo_by_id(memo_id: str) -> Dict[str, Any]:
    row = load_memo_by_id(memo_id)
    if not row:
        raise HTTPException(status_code=404, detail="memo_not_found")
    return {"status": "ok", "memo": row}


# ---------------------------------------------------------------------------
# Screener endpoint
# ---------------------------------------------------------------------------

class ScreenRequest(BaseModel):
    as_of_date: Optional[str] = None
    min_dau: float = Field(default=0.0, ge=0.0, le=100.0)
    signal_filter: Optional[list] = None     # e.g. ["BUY"] or ["BUY","HOLD"]
    min_conviction: float = Field(default=0.0, ge=0.0, le=100.0)
    max_weight: float = Field(default=0.10, ge=0.001, le=1.0)
    fractional_kelly: float = Field(default=0.5, gt=0.0, le=1.0)
    limit: int = Field(default=50, ge=1, le=200)


@router.post("/screen")
def axiom_screen(req: ScreenRequest) -> Dict[str, Any]:
    as_of_date_str = req.as_of_date or dt.date.today().isoformat()
    try:
        as_of_date = dt.date.fromisoformat(as_of_date_str)
    except ValueError:
        as_of_date = dt.date.today()

    signal_filter = (
        [s.upper() for s in req.signal_filter if s]
        if req.signal_filter
        else None
    )

    result = screen_universe(
        as_of_date,
        min_dau=req.min_dau,
        signal_filter=signal_filter,
        min_conviction=req.min_conviction,
        max_weight=req.max_weight,
        fractional_kelly=req.fractional_kelly,
        limit=req.limit,
    )

    logger.info(
        "axiom.screen date=%s screened=%d returned=%d ic=%s breadth=%s",
        as_of_date, result.get("total_screened", 0), result.get("count", 0),
        result.get("ic_state"), result.get("breadth_state"),
    )

    return result
