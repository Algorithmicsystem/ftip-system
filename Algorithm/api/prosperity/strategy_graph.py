from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api import config, db
from api.ops import metrics_tracker
from api.prosperity import ingest, query
from api.prosperity.strategy_graph_db import (
    latest_ensemble,
    latest_strategies,
    upsert_ensemble_row,
    upsert_strategy_rows,
)
from ftip.strategy_graph import compute_strategy_graph

logger = logging.getLogger("prosperity.strategy_graph")
router = APIRouter()


class StrategyGraphRunRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, min_length=1)
    from_date: dt.date
    to_date: dt.date
    as_of_date: dt.date
    lookback: int = Field(252, ge=5, le=5000)
    concurrency: int = Field(3, ge=1)
    force_refresh: bool = False
    persist: bool = True
    audit_no_lookahead: bool = True
    audit_max_rows: int = Field(3, ge=1, le=50)
    compute_features_if_missing: bool = True


class StrategyGraphRunResult(BaseModel):
    status: str
    trace_id: Optional[str]
    requested: Dict[str, Any]
    result: Dict[str, Any]
    timings: Dict[str, float]


class StrategyGraphLatestResponse(BaseModel):
    data: Dict[str, Any]


class StrategyGraphListResponse(BaseModel):
    data: List[Dict[str, Any]]


class _Normalizer:
    @staticmethod
    def symbols(symbols: Optional[List[str]]) -> List[str]:
        syms = sorted(
            {(s or "").strip().upper() for s in (symbols or []) if s and s.strip()}
        )
        if not syms:
            default = config.env("FTIP_UNIVERSE_DEFAULT", "AAPL,MSFT").split(",")
            syms = sorted({s.strip().upper() for s in default if s.strip()})
        if not syms:
            raise HTTPException(status_code=400, detail="symbols required")
        return syms


@router.post("/run", response_model=StrategyGraphRunResult)
async def run_strategy_graph(req: StrategyGraphRunRequest, request: Request):
    trace_id = getattr(request.state, "trace_id", None)
    if req.from_date > req.to_date:
        raise HTTPException(
            status_code=400, detail="from_date must be on/before to_date"
        )
    if req.to_date > req.as_of_date:
        raise HTTPException(
            status_code=400, detail="as_of_date must be on/after to_date"
        )

    symbols = _Normalizer.symbols(req.symbols)
    concurrency = min(max(req.concurrency, 1), 5)
    requested = {**req.model_dump(), "symbols": symbols, "concurrency": concurrency}
    timings: Dict[str, float] = {}
    rows_written = {"strategies": 0, "ensembles": 0}
    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []

    for sym in symbols:
        errors: List[str] = []
        try:
            if db.db_enabled() and db.db_write_enabled():
                ingest.ingest_bars(
                    sym, req.from_date, req.to_date, force_refresh=req.force_refresh
                )
            bars = (
                query.fetch_bars(sym, req.from_date, req.as_of_date)
                if db.db_enabled()
                else []
            )
            from api.main import Candle

            candles: List[Candle] = [
                Candle(
                    timestamp=b["date"] if isinstance(b, dict) else b.timestamp,
                    close=float(b["close"]),
                    volume=(
                        float(b.get("volume"))
                        if isinstance(b, dict) and b.get("volume") is not None
                        else None
                    ),
                )
                for b in bars
            ]
            if not candles:
                # fallback: try to synthesize a minimal history using ingest helper
                from api.main import massive_fetch_daily_bars

                fetched = massive_fetch_daily_bars(
                    sym, req.from_date.isoformat(), req.as_of_date.isoformat()
                )
                for bar in fetched:
                    candles.append(
                        Candle(timestamp=bar.timestamp, close=float(bar.close))
                    )
            result = compute_strategy_graph(
                sym,
                req.as_of_date,
                req.lookback,
                candles,
                audit_no_lookahead=req.audit_no_lookahead,
            )
            if req.persist and db.db_enabled() and db.db_write_enabled():
                strat_rows = []
                for strat in result["strategies"]:
                    strat_rows.append(
                        {
                            "symbol": sym,
                            "as_of_date": req.as_of_date,
                            "lookback": req.lookback,
                            "strategy_id": strat.get("strategy_id"),
                            "strategy_version": strat.get("version"),
                            "regime": result.get("regime"),
                            "raw_score": strat.get("raw_score"),
                            "normalized_score": strat.get("normalized_score"),
                            "signal": strat.get("signal"),
                            "confidence": strat.get("confidence"),
                            "rationale": strat.get("rationale"),
                            "feature_contributions": strat.get("feature_contributions"),
                            "meta": {"regime_meta": result.get("regime_meta")},
                        }
                    )
                rows_written["strategies"] += upsert_strategy_rows(strat_rows)
                ensemble = result.get("ensemble") or {}
                upsert_ensemble_row(
                    {
                        "symbol": sym,
                        "as_of_date": req.as_of_date,
                        "lookback": req.lookback,
                        "regime": result.get("regime"),
                        "ensemble_method": ensemble.get("ensemble_method"),
                        "final_signal": ensemble.get("final_signal"),
                        "final_score": ensemble.get("final_score"),
                        "final_confidence": ensemble.get("final_confidence"),
                        "thresholds": ensemble.get("thresholds"),
                        "risk_overlay_applied": ensemble.get("risk_overlay_applied"),
                        "strategies_used": ensemble.get("strategies_used"),
                        "audit": result.get("audit"),
                        "hashes": result.get("hashes"),
                    }
                )
                rows_written["ensembles"] += 1
            symbols_ok.append(sym)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(str(exc))
            symbols_failed.append({"symbol": sym, "reason": "; ".join(errors)})
            logger.exception(
                "strategy_graph.run.failed", extra={"symbol": sym, "trace_id": trace_id}
            )
            continue

    status = "ok" if not symbols_failed else "partial"
    metrics_tracker.record_run(
        "strategy_graph",
        trace_id,
        status,
        timings=timings,
        rows_written={
            "strategies": rows_written.get("strategies"),
            "ensembles": rows_written.get("ensembles"),
        },
    )

    return {
        "status": status,
        "trace_id": trace_id,
        "requested": requested,
        "result": {
            "symbols_ok": symbols_ok,
            "symbols_failed": symbols_failed,
            "rows_written": rows_written,
        },
        "timings": timings,
    }


@router.get("/latest/ensemble", response_model=StrategyGraphLatestResponse)
async def get_latest_ensemble(symbol: str, lookback: int = 252):
    if not db.db_enabled() or not db.db_read_enabled():
        raise HTTPException(status_code=503, detail="database disabled")
    res = latest_ensemble(symbol.upper(), lookback)
    if not res:
        raise HTTPException(status_code=404, detail="not found")
    return {"data": res}


@router.get("/latest/strategies", response_model=StrategyGraphListResponse)
async def get_latest_strategies(symbol: str, lookback: int = 252):
    if not db.db_enabled() or not db.db_read_enabled():
        raise HTTPException(status_code=503, detail="database disabled")
    res = latest_strategies(symbol.upper(), lookback)
    if not res:
        raise HTTPException(status_code=404, detail="not found")
    return {"data": res}
