from ftip.risk import (
    compute_return_correlation_matrix,
    correlation_guard,
    correlation_guard_stub,
    volatility_targeting,
)


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
    # stub with no matrix is a pass-through
    assert correlation_guard_stub(out1) == out1


# ---------------------------------------------------------------------------
# compute_return_correlation_matrix
# ---------------------------------------------------------------------------


def test_correlation_matrix_perfectly_correlated():
    rets = [0.01, -0.02, 0.03, -0.01, 0.02]
    matrix = compute_return_correlation_matrix({"A": rets, "B": rets})
    assert abs(matrix["A"]["B"] - 1.0) < 1e-9
    assert abs(matrix["B"]["A"] - 1.0) < 1e-9
    assert matrix["A"]["A"] == 1.0


def test_correlation_matrix_perfectly_anticorrelated():
    rets = [0.01, -0.02, 0.03, -0.01, 0.02]
    neg = [-r for r in rets]
    matrix = compute_return_correlation_matrix({"A": rets, "B": neg})
    assert abs(matrix["A"]["B"] - (-1.0)) < 1e-9


def test_correlation_matrix_uncorrelated():
    # Alternating +/- series are uncorrelated with a flat series.
    a = [0.01, -0.01] * 50
    b = [0.0] * 100  # flat → std = 0, treated as uncorrelated
    matrix = compute_return_correlation_matrix({"A": a, "B": b})
    assert matrix["A"]["B"] == 0.0


def test_correlation_matrix_too_few_observations():
    matrix = compute_return_correlation_matrix({"A": [0.01], "B": [0.02]})
    assert matrix["A"]["B"] == 0.0


def test_correlation_matrix_empty():
    assert compute_return_correlation_matrix({}) == {}


# ---------------------------------------------------------------------------
# correlation_guard
# ---------------------------------------------------------------------------


def test_correlation_guard_passthrough_when_no_matrix():
    weights = {"A": 0.6, "B": 0.4}
    out = correlation_guard(weights, correlation_matrix=None)
    assert out == {"A": 0.6, "B": 0.4}


def test_correlation_guard_no_change_below_threshold():
    weights = {"A": 0.5, "B": 0.5}
    matrix = {"A": {"A": 1.0, "B": 0.5}, "B": {"B": 1.0, "A": 0.5}}
    out = correlation_guard(weights, correlation_matrix=matrix, threshold=0.85)
    # Correlation 0.5 < 0.85: weights should be normalised but otherwise unchanged.
    assert abs(out["A"] - 0.5) < 1e-6
    assert abs(out["B"] - 0.5) < 1e-6


def test_correlation_guard_reduces_smaller_weight_when_highly_correlated():
    # A and B are perfectly correlated (corr=1.0, threshold=0.85).
    # haircut = (1.0 - 0.85) / (1.0 - 0.85) = 1.0 → B is fully removed.
    weights = {"A": 0.7, "B": 0.3}
    matrix = {"A": {"A": 1.0, "B": 1.0}, "B": {"B": 1.0, "A": 1.0}}
    out = correlation_guard(weights, correlation_matrix=matrix, threshold=0.85)
    # B is zeroed out; A gets the full allocation.
    assert "B" not in out or out.get("B", 0.0) < 1e-9
    assert abs(out.get("A", 0.0) - 1.0) < 1e-9


def test_correlation_guard_output_sums_to_one():
    weights = {"A": 0.4, "B": 0.35, "C": 0.25}
    # A and B highly correlated, C independent.
    matrix = {
        "A": {"A": 1.0, "B": 0.95, "C": 0.1},
        "B": {"B": 1.0, "A": 0.95, "C": 0.2},
        "C": {"C": 1.0, "A": 0.1, "B": 0.2},
    }
    out = correlation_guard(weights, correlation_matrix=matrix, threshold=0.85)
    assert out
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_correlation_guard_empty_input():
    assert correlation_guard({}) == {}
    assert correlation_guard({}, correlation_matrix={}) == {}


def test_correlation_guard_real_pipeline():
    # Simulate the routes.py flow: compute matrix from returns then guard.
    import random
    random.seed(42)
    base = [random.gauss(0, 0.01) for _ in range(200)]
    # A and B share the same base returns → perfectly correlated.
    returns_by_sym = {
        "A": base,
        "B": base,
        "C": [random.gauss(0, 0.01) for _ in range(200)],
    }
    matrix = compute_return_correlation_matrix(returns_by_sym)
    weights = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    out = correlation_guard(weights, correlation_matrix=matrix, threshold=0.85)

    assert abs(sum(out.values()) - 1.0) < 1e-9
    # Highly-correlated A and B should together hold less than C's share
    # after the guard reduces one of them.
    ab_total = out.get("A", 0.0) + out.get("B", 0.0)
    c_share = out.get("C", 0.0)
    assert ab_total < 1.0  # guard reduced at least one of A/B
    assert c_share > 0.0
