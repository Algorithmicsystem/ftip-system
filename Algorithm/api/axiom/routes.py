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

from fastapi import APIRouter, Depends, HTTPException, Query, Request
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
from api.axiom.sanitizer import sanitize_engine_breakdown

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
def axiom_screen(req: ScreenRequest, request: Request = None) -> Dict[str, Any]:
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

    # Sanitize proprietary sub-component fields for free-tier callers
    tenant = getattr(request.state, "tenant", None) if request else None
    tenant_tier = (tenant or {}).get("tier", "free")
    if tenant_tier not in ("pro", "enterprise"):
        symbols_out = []
        for sym_entry in result.get("symbols") or []:
            payload = sym_entry.get("payload") or {}
            sym_entry = dict(sym_entry)
            sym_entry["payload"] = sanitize_engine_breakdown(payload)
            symbols_out.append(sym_entry)
        result = dict(result)
        result["symbols"] = symbols_out

    return result


# ---------------------------------------------------------------------------
# Phase 21: Portfolio Allocation Engine
# ---------------------------------------------------------------------------

class AllocateRequest(BaseModel):
    as_of_date: Optional[str] = None
    max_position_weight: float = Field(default=0.10, ge=0.001, le=1.0)
    max_sector_concentration: float = Field(default=0.30, ge=0.01, le=1.0)
    max_portfolio_heat: float = Field(default=1.0, ge=0.01, le=1.0)
    min_dau: float = Field(default=0.0, ge=0.0, le=100.0)
    min_conviction: float = Field(default=0.0, ge=0.0, le=100.0)
    fractional_kelly: float = Field(default=0.5, gt=0.0, le=1.0)
    limit: int = Field(default=20, ge=1, le=100)
    correlation_threshold: float = Field(default=0.80, ge=0.0, le=1.0)


@router.post("/allocate")
def axiom_allocate(req: AllocateRequest) -> Dict[str, Any]:
    """Build a sector-capped, heat-limited portfolio allocation.

    Runs the conviction screener across the universe and applies:
    - per-position Kelly weight cap
    - per-sector concentration limit
    - portfolio heat (total invested %) cap

    Returns a ranked allocation with sector breakdown and rejected positions.
    """
    from api.axiom.allocator import build_portfolio_allocation

    as_of_date_str = req.as_of_date or dt.date.today().isoformat()
    try:
        as_of_date = dt.date.fromisoformat(as_of_date_str)
    except ValueError:
        as_of_date = dt.date.today()

    result = build_portfolio_allocation(
        as_of_date,
        max_position_weight=req.max_position_weight,
        max_sector_concentration=req.max_sector_concentration,
        max_portfolio_heat=req.max_portfolio_heat,
        min_dau=req.min_dau,
        min_conviction=req.min_conviction,
        fractional_kelly=req.fractional_kelly,
        limit=req.limit,
        correlation_threshold=req.correlation_threshold,
    )

    logger.info(
        "axiom.allocate date=%s positions=%d weight_total=%.3f ic=%s",
        as_of_date,
        result.get("position_count", 0),
        result.get("portfolio_weight_total", 0.0),
        result.get("ic_state"),
    )

    return result


# ---------------------------------------------------------------------------
# Phase 29: Conviction Velocity
# ---------------------------------------------------------------------------

def _linear_slope(values: list) -> float:
    """Least-squares slope of values vs integer index [0, 1, ..., n-1]."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def _compute_conviction_trends(
    as_of_date: dt.date,
    *,
    window: int = 5,
    min_dau: float = 0.0,
    limit: int = 50,
) -> Dict[str, Any]:
    from api import db
    if not db.db_read_enabled():
        return {"status": "db_disabled", "trends": []}

    since = as_of_date - dt.timedelta(days=window + 3)
    rows = db.safe_fetchall(
        """
        SELECT
            symbol,
            as_of_date,
            (payload->>'overall_confidence')::numeric  AS confidence,
            (payload->>'deployable_alpha_utility')::numeric AS dau
        FROM axiom_scores_daily
        WHERE as_of_date BETWEEN %s AND %s
          AND (payload->>'overall_confidence') IS NOT NULL
        ORDER BY symbol, as_of_date ASC
        """,
        (since, as_of_date),
    )

    # Group by symbol
    by_sym: Dict[str, list] = {}
    dau_by_sym: Dict[str, float] = {}
    for sym, date, conf, dau in (rows or []):
        by_sym.setdefault(sym, []).append(float(conf or 0))
        dau_by_sym[sym] = float(dau or 0)

    trends = []
    for sym, confs in by_sym.items():
        dau = dau_by_sym.get(sym, 0.0)
        if dau < min_dau:
            continue
        slope = round(_linear_slope(confs[-window:]), 4)
        latest = confs[-1]
        earliest = confs[0]
        trends.append({
            "symbol": sym,
            "conviction_velocity": slope,
            "latest_confidence": round(latest, 2),
            "change_over_window": round(latest - earliest, 2),
            "dau": round(dau, 2),
            "trend": "accelerating" if slope > 1.0 else ("decelerating" if slope < -1.0 else "stable"),
            "window_days": len(confs[-window:]),
        })

    trends.sort(key=lambda x: x["conviction_velocity"], reverse=True)
    return {
        "status": "ok",
        "as_of_date": as_of_date.isoformat(),
        "window": window,
        "symbol_count": len(trends),
        "trends": trends[:limit],
    }


@router.get("/allocate/replay")
def allocate_replay(
    start_date: str = Query(...),
    end_date: Optional[str] = Query(default=None),
    max_position_weight: float = Query(default=0.10, ge=0.001, le=1.0),
    max_sector_concentration: float = Query(default=0.30, ge=0.01, le=1.0),
    max_portfolio_heat: float = Query(default=1.0, ge=0.01, le=1.0),
    fractional_kelly: float = Query(default=0.5, gt=0.0, le=1.0),
    horizon_days: int = Query(default=21, ge=1, le=63),
) -> Dict[str, Any]:
    """Replay allocator over a date range and evaluate against realized P&L.

    For each trading date in [start_date, end_date]:
    1. Runs build_portfolio_allocation to get the day's positions
    2. Looks up forward returns from signal_pnl_daily at horizon_days
    3. Computes a synthetic portfolio return as weighted-average return

    Returns an equity curve, Sharpe ratio, max drawdown, and win rate.
    """
    from api import db
    from api.axiom.allocator import build_portfolio_allocation

    try:
        start = dt.date.fromisoformat(start_date)
        end = dt.date.fromisoformat(end_date) if end_date else dt.date.today()
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    if (end - start).days > 365:
        return {"status": "error", "message": "Date range cannot exceed 365 days."}

    if not db.db_read_enabled():
        return {"status": "db_disabled", "equity_curve": [], "summary": {}}

    daily_returns: list = []
    equity = 1.0
    equity_curve = []
    winning_days = 0

    current = start
    while current <= end:
        alloc = build_portfolio_allocation(
            current,
            max_position_weight=max_position_weight,
            max_sector_concentration=max_sector_concentration,
            max_portfolio_heat=max_portfolio_heat,
            fractional_kelly=fractional_kelly,
        )

        allocations = alloc.get("allocations") or []
        if not allocations:
            equity_curve.append({"date": current.isoformat(), "equity": round(equity, 6), "day_return": 0.0, "positions": 0})
            current += dt.timedelta(days=1)
            continue

        symbols = [a["symbol"] for a in allocations]
        weights = {a["symbol"]: a["suggested_weight"] for a in allocations}

        pnl_rows = db.safe_fetchall(
            """
            SELECT symbol, return_pct
            FROM signal_pnl_daily
            WHERE symbol = ANY(%s)
              AND signal_date = %s
              AND horizon_days = %s
              AND return_pct IS NOT NULL
            """,
            (symbols, current, horizon_days),
        ) or []

        pnl_map = {r[0]: float(r[1]) for r in pnl_rows}

        if pnl_map:
            weighted_ret = sum(
                weights.get(sym, 0.0) * ret / max(sum(weights.values()), 1e-8)
                for sym, ret in pnl_map.items()
            )
            day_return = round(weighted_ret, 4)
            equity = round(equity * (1 + day_return / 100), 6)
            daily_returns.append(day_return)
            if day_return > 0:
                winning_days += 1
        else:
            day_return = 0.0

        equity_curve.append({
            "date": current.isoformat(),
            "equity": equity,
            "day_return": day_return,
            "positions": len(allocations),
            "covered": len(pnl_map),
        })
        current += dt.timedelta(days=1)

    # Summary stats
    n = len(daily_returns)
    avg_ret = sum(daily_returns) / n if n else 0.0
    std_ret = (sum((r - avg_ret) ** 2 for r in daily_returns) / n) ** 0.5 if n > 1 else 0.0
    sharpe = round((avg_ret / std_ret) * (252 ** 0.5), 3) if std_ret > 0 else None

    max_dd = 0.0
    peak = 1.0
    for point in equity_curve:
        eq = point["equity"]
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return {
        "status": "ok",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "trading_days_with_pnl": n,
        "equity_curve": equity_curve,
        "summary": {
            "final_equity": round(equity, 6),
            "total_return_pct": round((equity - 1.0) * 100, 3),
            "avg_daily_return_pct": round(avg_ret, 4),
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": round(max_dd * 100, 3),
            "win_rate": round(winning_days / n, 3) if n else None,
            "winning_days": winning_days,
            "total_days": n,
        },
    }


@router.get("/conviction/trends")
def conviction_trends(
    as_of_date: Optional[str] = Query(default=None),
    window: int = Query(default=5, ge=2, le=30),
    min_dau: float = Query(default=0.0, ge=0.0, le=100.0),
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, Any]:
    """Return conviction velocity (5-day confidence slope) per symbol."""
    date = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    return _compute_conviction_trends(date, window=window, min_dau=min_dau, limit=limit)


@router.get("/allocation/kelly-status")
def kelly_status(as_of_date: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    """Return current Kelly sizer IC source, value, and effective position sizing mode.

    Exposes whether the sizer is using live IC, a bootstrap prior, or the default.
    """
    aod = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    from api.axiom.screener import _load_ic_state_bulk

    ic_state = _load_ic_state_bulk(aod)

    # Load numeric IC value from signal_ic_daily
    ic_value: Optional[float] = None
    ic_source = "insufficient"
    sample_count = 0

    if db.db_read_enabled():
        try:
            row = db.safe_fetchone(
                """
                SELECT ic_value, sample_size FROM signal_ic_daily
                 WHERE score_field IN ('axiom_composite', 'composite')
                   AND horizon_label = '21d'
                   AND as_of_date <= %s
                   AND ic_value IS NOT NULL
                 ORDER BY as_of_date DESC LIMIT 1
                """,
                (aod,),
            )
            if row and row[0] is not None:
                ic_value = float(row[0])
                sample_count = int(row[1] or 0)
                ic_source = "signal_ic_daily"
        except Exception:
            pass

    if ic_value is None:
        ic_value = 0.04  # Grinold-Kahn conservative prior
        ic_source = "bootstrap_prior"

    confidence_factor = min(sample_count / 50.0, 1.0)
    adjusted_ic = ic_value * confidence_factor

    from api.axiom.sizer import _IC_KELLY_MULTIPLIER
    ic_mult = _IC_KELLY_MULTIPLIER.get(ic_state, 0.25)

    kelly_mode = (
        "live_ic" if ic_source == "signal_ic_daily"
        else "bootstrap_prior" if ic_source == "bootstrap_prior"
        else "insufficient_default"
    )

    return {
        "ic_source": ic_source,
        "ic_value": round(ic_value, 4),
        "ic_state": ic_state,
        "confidence_factor": round(confidence_factor, 4),
        "adjusted_ic": round(adjusted_ic, 4),
        "ic_kelly_multiplier": ic_mult,
        "position_size_pct": round(ic_mult * 0.5 * 0.10 * 100, 2),  # half-kelly × 10% budget
        "kelly_mode": kelly_mode,
        "sample_count": sample_count,
    }
