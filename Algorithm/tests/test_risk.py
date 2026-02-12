from ftip.risk import correlation_guard_stub, volatility_targeting


def test_volatility_targeting_weights_sum_to_one():
    weights = {"AAPL": 1.0, "MSFT": 1.0, "NVDA": 1.0}
    vol = {"AAPL": 0.2, "MSFT": 0.3, "NVDA": 0.4}

    out = volatility_targeting(weights, vol, target_vol=0.2, max_weight=0.6)

    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_volatility_targeting_respects_max_weight():
    weights = {"AAPL": 1.0, "MSFT": 1.0, "NVDA": 1.0}
    vol = {"AAPL": 0.1, "MSFT": 3.0, "NVDA": 3.5}

    out = volatility_targeting(weights, vol, target_vol=0.25, max_weight=0.4)

    assert out
    assert max(out.values()) <= 0.4 + 1e-9


def test_volatility_targeting_is_deterministic_and_zero_when_empty():
    weights = {"MSFT": 1.0, "AAPL": 1.0}
    vol = {"MSFT": 0.5, "AAPL": 0.5}

    out1 = volatility_targeting(weights, vol, target_vol=0.2, max_weight=0.8)
    out2 = volatility_targeting(weights, vol, target_vol=0.2, max_weight=0.8)

    assert out1 == out2
    assert volatility_targeting({}, {}, target_vol=0.2, max_weight=0.8) == {}
    assert correlation_guard_stub(out1) == out1
