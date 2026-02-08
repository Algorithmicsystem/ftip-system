from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from api.backtest import service

router = APIRouter(tags=["backtest"])


class CostModel(BaseModel):
    fee_bps: float = Field(1, ge=0)
    slippage_bps: float = Field(5, ge=0)


class BacktestRunRequest(BaseModel):
    symbol: Optional[str] = None
    universe: str = "sp500"
    date_start: str
    date_end: str
    horizon: str
    risk_mode: str
    signal_version_hash: str = "auto"
    cost_model: CostModel


class BacktestRunResponse(BaseModel):
    run_id: str
    status: str


@router.post("/backtest/run", response_model=BacktestRunResponse)
def run_backtest(req: BacktestRunRequest) -> BacktestRunResponse:
    result = service.run_backtest(
        symbol=req.symbol,
        universe=req.universe,
        date_start=req.date_start,
        date_end=req.date_end,
        horizon=req.horizon,
        risk_mode=req.risk_mode,
        signal_version_hash=req.signal_version_hash,
        cost_model=req.cost_model.model_dump(),
    )
    return BacktestRunResponse(**result)


@router.get("/backtest/results")
def backtest_results(run_id: str = Query(...)) -> Dict[str, Any]:
    return service.fetch_results(run_id)


@router.get("/backtest/equity-curve")
def backtest_equity_curve(run_id: str = Query(...)) -> Dict[str, Any]:
    points = service.fetch_equity_curve(run_id)
    return {"points": points}
