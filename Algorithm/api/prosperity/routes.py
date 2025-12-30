from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from api import config, db
from api.prosperity import ingest, query
from api.prosperity.strategy_graph import router as strategy_graph_router
from api.prosperity.models import (
    BarsIngestBulkRequest,
    BarsIngestRequest,
    BarsResponse,
    FeaturesComputeRequest,
    HealthResponse,
    SignalsComputeRequest,
    SnapshotRunRequest,
    UniverseUpsertRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)

router.include_router(strategy_graph_router, prefix="/strategy_graph")


def _require_db_enabled(write: bool = False, read: bool = False) -> None:
    if not db.db_enabled():
        raise HTTPException(status_code=503, detail="database disabled")
    if write and not db.db_write_enabled():
        raise HTTPException(status_code=503, detail="database writes disabled")
    if read and not db.db_read_enabled():
        raise HTTPException(status_code=503, detail="database reads disabled")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        db_enabled=db.db_enabled(),
        db_write_enabled=db.db_write_enabled(),
        db_read_enabled=db.db_read_enabled(),
    )


@router.post("/bootstrap")
async def bootstrap(request: Request):
    token = config.env("PROSPERITY_ADMIN_TOKEN")
    if token and request.headers.get("x-admin-token") != token:
        raise HTTPException(status_code=403, detail="forbidden")

    if not db.db_enabled():
        return {"status": "ok", "db_enabled": False, "migrated": False, "versions": []}

    from api import migrations

    versions = migrations.ensure_schema()
    db.ensure_schema()
    return {"status": "ok", "db_enabled": True, "migrated": bool(versions), "versions": versions}


@router.post("/universe/upsert")
async def universe_upsert(req: UniverseUpsertRequest):
    _require_db_enabled(write=True)
    count, symbols = ingest.upsert_universe(req.symbols)
    return {"count": count, "symbols": symbols}


@router.post("/bars/ingest")
async def bars_ingest(req: BarsIngestRequest):
    _require_db_enabled(write=True)
    return ingest.ingest_bars(req.symbol, req.from_date, req.to_date, force_refresh=req.force_refresh)


@router.post("/bars/ingest_bulk")
async def bars_ingest_bulk(req: BarsIngestBulkRequest):
    _require_db_enabled(write=True)
    return ingest.ingest_bars_bulk(req.symbols, req.from_date, req.to_date, concurrency=req.concurrency, force_refresh=req.force_refresh)


@router.get("/bars", response_model=BarsResponse)
async def bars(symbol: str, from_date: dt.date, to_date: dt.date):
    _require_db_enabled(read=True)
    data = query.fetch_bars(symbol.upper(), from_date, to_date)
    return BarsResponse(symbol=symbol.upper(), from_date=from_date, to_date=to_date, data=data)


@router.post("/features/compute")
async def features_compute(req: FeaturesComputeRequest):
    _require_db_enabled(write=True)
    return ingest.compute_and_store_features(req.symbol, req.as_of_date, req.lookback)


@router.post("/signals/compute")
async def signals_compute(req: SignalsComputeRequest):
    _require_db_enabled(write=True)
    return ingest.compute_and_store_signal(req.symbol, req.as_of_date, req.lookback)


def _normalize_symbols(symbols: List[str]) -> List[str]:
    cleaned = sorted({(s or "").strip().upper() for s in symbols if s and s.strip()})
    if not cleaned:
        raise HTTPException(status_code=400, detail="symbols required")
    return cleaned


@router.post("/snapshot/run")
async def snapshot_run(req: SnapshotRunRequest, request: Request):
    _require_db_enabled(write=True, read=True)
    trace_id = getattr(request.state, "trace_id", None)
    if req.from_date > req.to_date:
        raise HTTPException(status_code=400, detail="from_date must be on/before to_date")
    if req.to_date > req.as_of_date:
        raise HTTPException(status_code=400, detail="as_of_date must be on/after to_date")

    symbols = req.symbols or ((config.env("FTIP_UNIVERSE_DEFAULT", "") or "AAPL,MSFT").split(","))
    symbols = _normalize_symbols(symbols)
    concurrency = min(max(req.concurrency, 1), 5)

    requested = {
        "symbols": symbols,
        "from_date": req.from_date.isoformat(),
        "to_date": req.to_date.isoformat(),
        "as_of_date": req.as_of_date.isoformat(),
        "lookback": req.lookback,
        "concurrency": concurrency,
        "force_refresh": bool(req.force_refresh),
    }

    timings: Dict[str, float] = {}
    rows_written = {"signals": 0, "features": 0}
    strategy_graph_rows = {"strategies": 0, "ensembles": 0}
    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []

    t0 = time.time()
    try:
        ingest.upsert_universe(symbols)
    except Exception as exc:  # pragma: no cover - guarded via validation
        raise HTTPException(status_code=503, detail=f"failed to upsert universe: {exc}")
    timings["upsert_universe"] = time.time() - t0

    for sym in symbols:
        sym_start = time.time()
        errors: List[str] = []

        try:
            ingest.ingest_bars(sym, req.from_date, req.to_date, force_refresh=req.force_refresh)
        except Exception as exc:
            errors.append(f"bars: {exc}")

        if not errors:
            try:
                ingest.compute_and_store_features(sym, req.as_of_date, req.lookback)
                rows_written["features"] += 1
            except Exception as exc:
                errors.append(f"features: {exc}")

        if not errors:
            try:
                ingest.compute_and_store_signal(sym, req.as_of_date, req.lookback)
                rows_written["signals"] += 1
            except Exception as exc:
                errors.append(f"signals: {exc}")

        if not errors and req.compute_strategy_graph:
            try:
                from api.main import Candle
                bars = query.fetch_bars(sym, req.from_date, req.as_of_date)
                candles = [
                    Candle(
                        timestamp=b["date"],
                        close=float(b["close"]),
                        volume=float(b["volume"]) if b.get("volume") is not None else None,
                    )
                    for b in bars
                ]
                from ftip.strategy_graph import compute_strategy_graph
                from api.prosperity.strategy_graph_db import upsert_ensemble_row, upsert_strategy_rows

                res = compute_strategy_graph(sym, req.as_of_date, req.lookback, candles)
                strat_rows = []
                for strat in res.get("strategies", []):
                    strat_rows.append(
                        {
                            "symbol": sym,
                            "as_of_date": req.as_of_date,
                            "lookback": req.lookback,
                            "strategy_id": strat.get("strategy_id"),
                            "strategy_version": strat.get("version"),
                            "regime": res.get("regime"),
                            "raw_score": strat.get("raw_score"),
                            "normalized_score": strat.get("normalized_score"),
                            "signal": strat.get("signal"),
                            "confidence": strat.get("confidence"),
                            "rationale": strat.get("rationale"),
                            "feature_contributions": strat.get("feature_contributions"),
                            "meta": {"regime_meta": res.get("regime_meta")},
                        }
                    )
                strategy_graph_rows["strategies"] += upsert_strategy_rows(strat_rows)
                ens = res.get("ensemble") or {}
                upsert_ensemble_row(
                    {
                        "symbol": sym,
                        "as_of_date": req.as_of_date,
                        "lookback": req.lookback,
                        "regime": res.get("regime"),
                        "ensemble_method": ens.get("ensemble_method"),
                        "final_signal": ens.get("final_signal"),
                        "final_score": ens.get("final_score"),
                        "final_confidence": ens.get("final_confidence"),
                        "thresholds": ens.get("thresholds"),
                        "risk_overlay_applied": ens.get("risk_overlay_applied"),
                        "strategies_used": ens.get("strategies_used"),
                        "audit": res.get("audit"),
                        "hashes": res.get("hashes"),
                    }
                )
                strategy_graph_rows["ensembles"] += 1
            except Exception as exc:
                errors.append(f"strategy_graph: {exc}")

        duration = time.time() - sym_start
        if errors:
            symbols_failed.append({"symbol": sym, "reason": "; ".join(errors)})
            logger.warning(
                "[prosperity.snapshot] symbol failed", extra={"symbol": sym, "errors": errors, "duration_sec": duration, "trace_id": trace_id}
            )
        else:
            symbols_ok.append(sym)
            logger.info(
                "[prosperity.snapshot] symbol complete", extra={"symbol": sym, "duration_sec": duration, "trace_id": trace_id}
            )

    timings["total"] = time.time() - t0

    result_payload = {
        "symbols_ok": symbols_ok,
        "symbols_failed": symbols_failed,
        "rows_written": rows_written,
    }
    if req.compute_strategy_graph:
        result_payload["strategy_graph_rows"] = strategy_graph_rows

    return {
        "status": "ok",
        "trace_id": trace_id,
        "requested": requested,
        "result": result_payload,
        "timings": timings,
    }


@router.get("/latest/signal")
async def latest_signal(symbol: str, lookback: int = 252):
    _require_db_enabled(read=True)
    try:
        res = query.latest_signal(symbol.upper(), lookback)
    except db.DBError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    if not res:
        raise HTTPException(status_code=404, detail="not found")
    return res


@router.get("/latest/features")
async def latest_features(symbol: str, lookback: int = 252):
    _require_db_enabled(read=True)
    try:
        res = query.latest_features(symbol.upper(), lookback)
    except db.DBError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    if not res:
        raise HTTPException(status_code=404, detail="not found")
    return res
