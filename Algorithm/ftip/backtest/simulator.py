from dataclasses import dataclass
from typing import Optional

from ..friction import CostModel, ExecutionPlan, FrictionEngine, MarketStateInputs

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    volatility: float


class Portfolio:
    """Simple portfolio container for backtest results."""

    def __init__(self, initial_capital: float = 1.0):
        self.initial_capital = initial_capital


class BacktestSimulator:
    def __init__(
        self,
        leverage_limit: float = 1.0,
        position_limit: float = 1.0,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        trading_days: int = 252,
    ):
        self.leverage_limit = leverage_limit
        self.position_limit = position_limit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.trading_days = trading_days

    def run(
        self,
        prices: pd.Series,
        weights: pd.Series,
        friction_model: Optional[CostModel] = None,
    ) -> BacktestResult:
        returns = prices.pct_change().fillna(0.0)
        weights = weights.reindex(prices.index).fillna(0.0)

        equity = 1.0
        peak = 1.0
        equity_curve = []
        current_weight = 0.0

        friction_engine = FrictionEngine(friction_model) if friction_model else None

        for date, r in returns.items():
            w = float(weights.loc[date])
            w = np.clip(w, -self.position_limit, self.position_limit)

            if abs(w) > self.leverage_limit:
                w = np.sign(w) * self.leverage_limit

            if self.stop_loss is not None and r <= -abs(self.stop_loss):
                w = 0.0

            equity *= 1 + current_weight * r
            peak = max(peak, equity)

            if self.trailing_stop is not None:
                drawdown = 1 - equity / peak
                if drawdown >= self.trailing_stop:
                    w = 0.0

            if friction_engine is not None:
                turnover = abs(w - current_weight)
                if turnover > 0:
                    px = float(prices.loc[date])
                    market = MarketStateInputs(
                        date=pd.Timestamp(date).date(),
                        close=px,
                        open=px,
                        high=px,
                        low=px,
                        volume=1_000_000.0,
                        adv_20=1_000_000.0,
                        rv_20=max(abs(r), 0.01),
                    )
                    plan = ExecutionPlan(
                        symbol="PORT",
                        date=pd.Timestamp(date).date(),
                        side="BUY" if w >= current_weight else "SELL",
                        notional=turnover,
                        order_type="MARKET",
                    )
                    exec_result = friction_engine.simulate(market, plan)
                    equity -= exec_result.total_cost

            current_weight = w
            equity_curve.append((date, equity))

        equity_curve = pd.Series({d: v for d, v in equity_curve})
        portfolio_returns = equity_curve.pct_change().fillna(0.0)

        total_return = equity_curve.iloc[-1] - 1
        annual_return = (1 + portfolio_returns.mean()) ** self.trading_days - 1
        volatility = portfolio_returns.std() * np.sqrt(self.trading_days)
        sharpe = (
            0.0
            if volatility == 0
            else portfolio_returns.mean() / volatility * np.sqrt(self.trading_days)
        )
        max_drawdown = (1 - equity_curve / equity_curve.cummax()).max()

        return BacktestResult(
            equity_curve=equity_curve,
            returns=portfolio_returns,
            total_return=total_return,
            annual_return=annual_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            volatility=volatility,
        )


def evaluate_backtest(
    prices: pd.Series,
    weights: pd.Series,
    friction_model: Optional[CostModel] = None,
    **kwargs,
) -> BacktestResult:
    simulator = BacktestSimulator(**kwargs)
    return simulator.run(prices, weights, friction_model=friction_model)
