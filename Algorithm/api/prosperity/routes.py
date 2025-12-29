from __future__ import annotations

import datetime as dt
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from api import config, db
from api.prosperity import ingest, query
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


@router.post("/snapshot/run")
async def snapshot_run(req: SnapshotRunRequest):
    _require_db_enabled(write=True, read=True)
    symbols = req.symbols or ((config.env("FTIP_UNIVERSE_DEFAULT", "") or "AAPL,MSFT").split(","))
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    timings: Dict[str, float] = {}

    t0 = time.time()
    ingest.upsert_universe(symbols)
    timings["upsert_universe"] = time.time() - t0

    t1 = time.time()
    bars_res = ingest.ingest_bars_bulk(symbols, req.from_date, req.to_date, concurrency=req.concurrency, force_refresh=req.force_refresh)
    timings["bars"] = time.time() - t1

    t2 = time.time()
    feat_res = ingest.compute_features_bulk(symbols, req.as_of_date, req.lookback)
    timings["features"] = time.time() - t2

    t3 = time.time()
    sig_res = ingest.compute_signals_bulk(symbols, req.as_of_date, req.lookback)
    timings["signals"] = time.time() - t3

    return {
        "status": "ok",
        "bars": bars_res,
        "features": feat_res,
        "signals": sig_res,
        "timings": timings,
    }


@router.get("/latest/signal")
async def latest_signal(symbol: str, lookback: int = 252):
    _require_db_enabled(read=True)
    res = query.latest_signal(symbol.upper(), lookback)
    if not res:
        raise HTTPException(status_code=404, detail="not found")
    return res


@router.get("/latest/features")
async def latest_features(symbol: str, lookback: int = 252):
    _require_db_enabled(read=True)
    res = query.latest_features(symbol.upper(), lookback)
    if not res:
        raise HTTPException(status_code=404, detail="not found")
    return res
