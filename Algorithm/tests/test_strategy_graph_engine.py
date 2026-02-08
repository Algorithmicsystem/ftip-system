import datetime as dt

from api.main import Candle
from ftip.strategy_graph import compute_strategy_graph


def test_strategy_graph_engine_basic():
    as_of = dt.date(2024, 1, 25)
    candles = [
        Candle(
            timestamp=(dt.date(2023, 12, 20) + dt.timedelta(days=i)).isoformat(),
            close=100 + 0.5 * i,
        )
        for i in range(40)
    ]
    result = compute_strategy_graph("TEST", as_of, 30, candles)

    assert result["audit"]["last_candle_used"] == as_of.isoformat()
    assert result["audit"]["no_lookahead_ok"] is True
    assert result["regime"] in {"TRENDING", "CHOPPY", "HIGH_VOL", "RISK_OFF"}
    assert len(result["strategies"]) == 5
    for strat in result["strategies"]:
        assert strat["signal"] in {"BUY", "SELL", "HOLD"}
        assert 0.0 <= float(strat["confidence"]) <= 1.0

    ensemble = result["ensemble"]
    assert ensemble["final_signal"] in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= float(ensemble["final_confidence"]) <= 1.0
    assert result["hashes"]["strategies_hash"] != ""
    assert result["hashes"]["ensemble_hash"] != ""
