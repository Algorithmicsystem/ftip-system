from __future__ import annotations

import datetime as dt
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class NarratorSignalRequest(BaseModel):
    symbol: str
    as_of: dt.date
    lookback: int = 252
    include_features: bool = True
    include_db_history_days: int = Field(default=0, ge=0)
    style: Literal["concise", "detailed", "memo"] = "concise"


class NarratorSignalResponse(BaseModel):
    symbol: str
    as_of: dt.date
    signal: str
    narrative: str
    bullets: List[str]
    disclaimer: str


class NarratorPortfolioRequest(BaseModel):
    symbols: List[str]
    from_date: dt.date
    to_date: dt.date
    lookback: int = 252
    rebalance_every: int = 5
    max_weight: Optional[float] = None
    style: Literal["concise", "detailed", "memo"] = "memo"
    include_backtest: bool = True


class NarratorPortfolioResponse(BaseModel):
    narrative: str
    bullets: List[str]
    disclaimer: str
    performance: Optional[Dict[str, float]] = None


class NarratorAskRequest(BaseModel):
    question: str
    symbols: List[str]
    as_of: dt.date
    lookback: int = 252
    context: Optional[Dict[str, object]] = None


class NarratorAskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, str]]
    disclaimer: str


__all__ = [
    "NarratorSignalRequest",
    "NarratorSignalResponse",
    "NarratorPortfolioRequest",
    "NarratorPortfolioResponse",
    "NarratorAskRequest",
    "NarratorAskResponse",
]
