from __future__ import annotations

import datetime as dt
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Side = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]
FillStatus = Literal["FILLED", "PARTIAL", "NONE"]


class ImpactModelConfig(BaseModel):
    impact_k: float = Field(15.0, ge=0)
    median_rv_reference: float = Field(0.02, gt=0)
    max_impact_bps: float = Field(200.0, ge=0)


class CostModel(BaseModel):
    fee_bps: float = Field(1.0, ge=0)
    slippage_bps: float = Field(5.0, ge=0)
    spread_bps: float = Field(2.0, ge=0)
    impact: ImpactModelConfig = Field(default_factory=ImpactModelConfig)
    participation_rate: float = Field(0.02, ge=0, le=1)
    max_adv_pct: float = Field(0.1, ge=0, le=1)
    volatility_window: int = Field(20, ge=2)
    use_close_to_close_vol: bool = True
    allow_market_orders: bool = True
    allow_limit_orders: bool = True
    limit_fill_probability: float = Field(0.65, ge=0, le=1)
    overnight_gap_risk_bps: float = Field(1.0, ge=0)
    seed: int = 42

    @classmethod
    def from_legacy(cls, payload: dict) -> "CostModel":
        return cls(
            fee_bps=float(payload.get("fee_bps", 1.0)),
            slippage_bps=float(payload.get("slippage_bps", 5.0)),
            spread_bps=float(payload.get("spread_bps", 2.0)),
            impact=payload.get("impact") or ImpactModelConfig(),
            participation_rate=float(payload.get("participation_rate", 0.02)),
            max_adv_pct=float(payload.get("max_adv_pct", 0.1)),
            volatility_window=int(payload.get("volatility_window", 20)),
            use_close_to_close_vol=bool(payload.get("use_close_to_close_vol", True)),
            allow_market_orders=bool(payload.get("allow_market_orders", True)),
            allow_limit_orders=bool(payload.get("allow_limit_orders", True)),
            limit_fill_probability=float(payload.get("limit_fill_probability", 0.65)),
            overnight_gap_risk_bps=float(payload.get("overnight_gap_risk_bps", 1.0)),
            seed=int(payload.get("seed", 42)),
        )


class MarketStateInputs(BaseModel):
    date: dt.date
    close: float = Field(..., gt=0)
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    currency: Optional[str] = None
    adv_20: Optional[float] = Field(default=None, ge=0)
    rv_20: Optional[float] = Field(default=None, ge=0)
    spread_proxy_bps: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "MarketStateInputs":
        if self.high < self.low:
            raise ValueError("high must be >= low")
        return self


class ExecutionPlan(BaseModel):
    symbol: str
    date: dt.date
    side: Side
    notional: float = Field(..., gt=0)
    order_type: OrderType = "MARKET"
    limit_price: Optional[float] = Field(default=None, gt=0)
    time_in_force_days: int = Field(1, ge=1)
    participation_rate_override: Optional[float] = Field(default=None, ge=0, le=1)


class ExecutionResult(BaseModel):
    filled_qty: float
    filled_notional: float
    avg_fill_price: float
    fees_paid: float
    slippage_paid: float
    impact_paid: float
    spread_paid: float
    overnight_risk_paid: float
    total_cost: float
    effective_price: float
    fill_status: FillStatus
    reason: str


class ConstraintConfig(BaseModel):
    max_position_weight: float = Field(1.0, ge=0)
    max_sector_weight: Optional[float] = Field(default=None, ge=0)
    max_beta_exposure: Optional[float] = Field(default=None, ge=0)
    max_gross_exposure: float = Field(1.0, ge=0)
    max_turnover_per_rebalance: float = Field(1.0, ge=0)
    min_trade_notional: float = Field(0.0, ge=0)
    max_trade_notional: Optional[float] = Field(default=None, ge=0)
    cooldown_days: int = Field(0, ge=0)
