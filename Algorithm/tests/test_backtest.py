import pandas as pd

from ftip.backtest import BacktestSimulator, evaluate_backtest


def test_backtest_simulator_runs(sample_data):
    weights = pd.Series(0.5, index=sample_data.index)
    sim = BacktestSimulator(
        leverage_limit=1.0, position_limit=1.0, stop_loss=0.05, trailing_stop=0.2
    )
    result = sim.run(sample_data["close"], weights)
    assert result.equity_curve.iloc[0] == 1.0
    assert result.equity_curve.iloc[-1] > 0
    assert 0 <= result.max_drawdown <= 1


def test_evaluate_backtest_helper(sample_data):
    weights = pd.Series(0.2, index=sample_data.index)
    result = evaluate_backtest(sample_data["close"], weights)
    assert result.total_return == result.equity_curve.iloc[-1] - 1
    assert len(result.returns) == len(sample_data)
