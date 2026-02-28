from __future__ import annotations

import datetime as dt
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from api import db
from api.data import service
from api.data.models import (
    CorpActionsIngestRequest,
    DataVersionCreateRequest,
    DataVersionResponse,
    FundamentalsIngestRequest,
    FundamentalsQueryResponse,
    NewsIngestRequest,
    NewsQueryResponse,
    PricesIngestDailyRequest,
    PricesQueryDailyResponse,
    SymbolsUpsertRequest,
    UniverseSetRequest,
)

router = APIRouter(prefix="/data", tags=["data"])


def _require_db_enabled(write: bool = False, read: bool = False) -> None:
    if not db.db_enabled():
        raise HTTPException(status_code=503, detail="database disabled")
    if write and not db.db_write_enabled():
        raise HTTPException(status_code=503, detail="database writes disabled")
    if read and not db.db_read_enabled():
        raise HTTPException(status_code=503, detail="database reads disabled")


def _to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@router.post("/version/create", response_model=DataVersionResponse)
async def create_version(req: DataVersionCreateRequest) -> DataVersionResponse:
    _require_db_enabled(write=True)
    try:
        payload = service.record_data_version(
            req.source_name, req.source_snapshot_hash, req.notes
        )
    except (db.DBError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return DataVersionResponse(**payload)


@router.post("/symbols/upsert")
async def symbols_upsert(req: SymbolsUpsertRequest):
    _require_db_enabled(write=True)
    try:
        count = service.upsert_symbols([_to_dict(item) for item in req.items])
    except (db.DBError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"count": count}


@router.post("/universe/set")
async def set_universe(req: UniverseSetRequest):
    _require_db_enabled(write=True)
    try:
        count = service.set_universe(
            req.universe_name,
            req.symbols,
            start_ts=req.start_ts,
            end_ts=req.end_ts,
            data_version_id=req.data_version_id,
        )
    except (db.DBError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"count": count}


@router.post("/prices/ingest_daily")
async def ingest_prices_daily(req: PricesIngestDailyRequest):
    _require_db_enabled(write=True)
    count = service.ingest_prices_daily(
        req.data_version_id, [_to_dict(item) for item in req.items]
    )
    return {"count": count}


@router.get("/prices/query_daily", response_model=PricesQueryDailyResponse)
async def query_prices_daily(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    as_of_ts: dt.datetime,
    adjusted: bool = False,
) -> PricesQueryDailyResponse:
    _require_db_enabled(read=True)
    items = service.query_prices_daily(symbol, start_date, end_date, as_of_ts, adjusted)
    return PricesQueryDailyResponse(items=items)


@router.post("/corp_actions/ingest")
async def ingest_corp_actions(req: CorpActionsIngestRequest):
    _require_db_enabled(write=True)
    count = service.ingest_corp_actions(
        req.data_version_id, [_to_dict(item) for item in req.items]
    )
    return {"count": count}


@router.post("/fundamentals/ingest_pit")
async def ingest_fundamentals(req: FundamentalsIngestRequest):
    _require_db_enabled(write=True)
    count = service.ingest_fundamentals(
        req.data_version_id, [_to_dict(item) for item in req.items]
    )
    return {"count": count}


@router.get("/fundamentals/query_pit", response_model=FundamentalsQueryResponse)
async def query_fundamentals(
    symbol: str,
    as_of_ts: dt.datetime,
    metric_keys: Optional[str] = Query(default=None),
) -> FundamentalsQueryResponse:
    _require_db_enabled(read=True)
    parsed_keys: Optional[List[str]] = None
    if metric_keys:
        parsed_keys = [k.strip() for k in metric_keys.split(",") if k.strip()]
    items = service.query_latest_fundamentals(symbol, as_of_ts, metric_keys=parsed_keys)
    return FundamentalsQueryResponse(items=items)


@router.post("/news/ingest")
async def ingest_news(req: NewsIngestRequest):
    _require_db_enabled(write=True)
    count = service.ingest_news(
        req.data_version_id, [_to_dict(item) for item in req.items]
    )
    return {"count": count}


@router.get("/news/query", response_model=NewsQueryResponse)
async def query_news(
    symbol: str,
    as_of_ts: dt.datetime,
    start_ts: Optional[dt.datetime] = None,
    end_ts: Optional[dt.datetime] = None,
    limit: int = 200,
) -> NewsQueryResponse:
    _require_db_enabled(read=True)
    items = service.query_news(
        symbol, as_of_ts, start_ts=start_ts, end_ts=end_ts, limit=limit
    )
    return NewsQueryResponse(items=items)
