from ftip.system import FTIPSystem

def test_system_run_all(sample_data):
    system = FTIPSystem()
    results = system.run_all(sample_data)
    for key in ["features", "labels", "scores", "structural_alpha", "superfactor", "backtest"]:
        assert key in results
    assert len(results["features"]) == len(sample_data)
    assert results["backtest"].total_return == results["backtest"].equity_curve.iloc[-1] - 1
