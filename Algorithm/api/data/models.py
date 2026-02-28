from __future__ import annotations

import datetime as dt
from typing import List, Optional

from pydantic import BaseModel, Field


class DataVersionCreateRequest(BaseModel):
    source_name: str
    source_snapshot_hash: str
    notes: str = ""


class DataVersionResponse(BaseModel):
    id: int
    source_name: str
    source_snapshot_hash: str
    code_sha: str
    notes: str
    created_at: dt.datetime


class SymbolItem(BaseModel):
    symbol: str
    country: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None


class SymbolsUpsertRequest(BaseModel):
    items: List[SymbolItem] = Field(default_factory=list)


class UniverseSetRequest(BaseModel):
    universe_name: str = "default"
    symbols: List[str] = Field(default_factory=list)
    start_ts: Optional[dt.datetime] = None
    end_ts: Optional[dt.datetime] = None
    data_version_id: Optional[int] = None


class PriceDailyItem(BaseModel):
    symbol: str
    date: dt.date
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    currency: Optional[str] = None
    as_of_ts: Optional[dt.datetime] = None


class PricesIngestDailyRequest(BaseModel):
    data_version_id: int
    items: List[PriceDailyItem] = Field(default_factory=list)


class PricesQueryDailyResponse(BaseModel):
    items: List[dict]


class CorpActionItem(BaseModel):
    symbol: str
    action_type: str
    effective_date: dt.date
    factor: Optional[float] = None
    value: Optional[float] = None
    announced_ts: dt.datetime
    as_of_ts: Optional[dt.datetime] = None


class CorpActionsIngestRequest(BaseModel):
    data_version_id: int
    items: List[CorpActionItem] = Field(default_factory=list)


class FundamentalItem(BaseModel):
    symbol: str
    metric_key: str
    metric_value: float
    period_end: dt.date
    published_ts: dt.datetime
    as_of_ts: Optional[dt.datetime] = None


class FundamentalsIngestRequest(BaseModel):
    data_version_id: int
    items: List[FundamentalItem] = Field(default_factory=list)


class FundamentalsQueryResponse(BaseModel):
    items: List[dict]


class NewsItem(BaseModel):
    symbol: str
    published_ts: dt.datetime
    source: str
    credibility: float
    headline: str
    full_text: Optional[str] = None
    as_of_ts: Optional[dt.datetime] = None


class NewsIngestRequest(BaseModel):
    data_version_id: int
    items: List[NewsItem] = Field(default_factory=list)


class NewsQueryResponse(BaseModel):
    items: List[dict]
