from dataclasses import dataclass
from typing import Optional

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

    def run(self, prices: pd.Series, weights: pd.Series) -> BacktestResult:
        returns = prices.pct_change().fillna(0.0)
        weights = weights.reindex(prices.index).fillna(0.0)
        equity = 1.0
        equity_curve = []
        peak = equity

        current_weight = 0.0
        for date, r in returns.items():
            w = weights.loc[date]
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
            current_weight = w
            equity_curve.append((date, equity))

        equity_curve = pd.Series({d: v for d, v in equity_curve})
        portfolio_returns = equity_curve.pct_change().fillna(0.0)

        total_return = equity_curve.iloc[-1] - 1
        annual_return = (1 + portfolio_returns.mean()) ** self.trading_days - 1
        volatility = portfolio_returns.std() * np.sqrt(self.trading_days)
        sharpe = 0.0 if volatility == 0 else portfolio_returns.mean() / volatility * np.sqrt(self.trading_days)
        max_drawdown = self._max_drawdown(equity_curve)

        return BacktestResult(
            equity_curve=equity_curve,
            returns=portfolio_returns,
            total_return=total_return,
            annual_return=annual_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            volatility=volatility,
        )

    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> float:
        rolling_max = equity_curve.cummax()
        drawdowns = 1 - equity_curve / rolling_max
        return drawdowns.max()


def evaluate_backtest(prices: pd.Series, weights: pd.Series, **kwargs) -> BacktestResult:
    simulator = BacktestSimulator(**kwargs)
    return simulator.run(prices, weights)
