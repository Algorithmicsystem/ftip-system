from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    citations: Optional[List[str]] = None


class SessionResponse(BaseModel):
    session_id: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class ExplainSignalRequest(BaseModel):
    symbol: str
    as_of: str
    lookback: int = 252


class ExplainBacktestRequest(BaseModel):
    symbols: List[str]
    from_date: str
    to_date: str
    lookback: int = 252
    rebalance_every: int = 21
    trading_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    max_weight: Optional[float] = None
    min_trade_delta: float = 0.0005
    max_turnover_per_rebalance: float = 0.25
    allow_shorts: bool = False


class TitleSessionRequest(BaseModel):
    session_id: str
    hint: Optional[str] = None


class TitleSessionResponse(BaseModel):
    session_id: str
    title: str


class AnalyzeRequest(BaseModel):
    symbol: str
    horizon: str
    risk_mode: str


class TopPicksRequest(BaseModel):
    universe: str
    horizon: str
    risk_mode: str
    limit: int = 10


class NarrateRequest(BaseModel):
    payload: Dict[str, Any]
    user_message: str


class NarrateResponse(BaseModel):
    headline: str
    summary: str
    bullets: List[str]
    disclaimer: str
    followups: List[str]
