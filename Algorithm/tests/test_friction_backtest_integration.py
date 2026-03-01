import numpy as np
import pandas as pd

from ftip.backtest import evaluate_backtest
from ftip.friction import CostModel


def test_backtest_with_friction_underperforms_frictionless() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = pd.Series(100 + np.linspace(0, 5, len(dates)), index=dates)
    weights = pd.Series([0.0, 1.0] * 15, index=dates)

    frictionless = evaluate_backtest(prices, weights)
    with_friction = evaluate_backtest(
        prices,
        weights,
        friction_model=CostModel(fee_bps=2, slippage_bps=8, spread_bps=3, seed=42),
    )

    assert with_friction.total_return <= frictionless.total_return
