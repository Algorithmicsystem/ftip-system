from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UniverseUpsertRequest(BaseModel):
    symbols: List[str]
    metadata: Optional[Dict[str, Any]] = None


class BarsIngestRequest(BaseModel):
    symbol: str
    from_date: date
    to_date: date
    force_refresh: bool = False


class BarsIngestBulkRequest(BaseModel):
    symbols: List[str]
    from_date: date
    to_date: date
    force_refresh: bool = False
    concurrency: int = Field(3, ge=1)


class BarsResponse(BaseModel):
    symbol: str
    from_date: date
    to_date: date
    data: List[Dict[str, Any]]


class FeaturesComputeRequest(BaseModel):
    symbol: str
    as_of_date: date
    lookback: int


class SignalsComputeRequest(FeaturesComputeRequest):
    pass


class SnapshotRunRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, min_length=1)
    from_date: date
    to_date: date
    as_of_date: date
    lookback: int = Field(252, ge=5, le=5000)
    concurrency: int = Field(3, ge=1)
    force_refresh: bool = False
    compute_strategy_graph: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
    db_enabled: bool
    db_write_enabled: bool
    db_read_enabled: bool


class CoverageRow(BaseModel):
    symbol: str
    first_date: Optional[date] = None
    last_date: Optional[date] = None
    bars: int = 0
    missing_days_estimate: Optional[int] = None


class CoverageResponse(BaseModel):
    db_enabled: bool
    coverage: List[CoverageRow]


class BacktestAudit(BaseModel):
    request: Dict[str, Any]
    response: Dict[str, Any]


class SnapshotResult(BaseModel):
    status: str
    bars: Dict[str, Any]
    features: Dict[str, Any]
    signals: Dict[str, Any]
    timings: Dict[str, float]


class ProsperityBacktestRequest(BaseModel):
    symbols: List[str]
    start_date: date
    end_date: date
    lookback_days: int = Field(63, ge=5, le=5000)
    costs_bps: float = Field(5.0, ge=0.0)
