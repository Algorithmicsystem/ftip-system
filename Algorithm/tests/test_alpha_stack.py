from ftip.alpha_kernel import StructuralAlphaKernel
from ftip.superfactor import SuperfactorModel
from ftip.features import FeatureEngineer

from api.alpha.canonical_features import _rsi


def test_rsi_wilder_ema_known_values():
    # Alternating series: 14 deltas exactly = 7 gains of +1, 7 losses of -1.
    # seed avg_gain = 7/14 = 0.5, seed avg_loss = 7/14 = 0.5 (no further bars)
    # RS = 1.0  →  RSI = 50.0
    closes = [10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0,
              11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0]  # 15 values
    result = _rsi(closes, window=14)
    assert result is not None
    assert abs(result - 50.0) < 0.01


def test_rsi_all_gains_returns_100():
    # Monotonically rising series: all deltas are gains → avg_loss stays 0 → RSI = 100
    closes = [float(i) for i in range(1, 20)]
    result = _rsi(closes, window=14)
    assert result == 100.0


def test_rsi_all_losses_returns_0():
    # Monotonically falling series: avg_gain stays 0 → RS = 0 → RSI = 0
    closes = [float(i) for i in range(20, 1, -1)]
    result = _rsi(closes, window=14)
    assert result is not None
    assert result < 1.0


def test_rsi_insufficient_data_returns_none():
    closes = [1.0, 2.0, 3.0]
    assert _rsi(closes, window=14) is None


def test_rsi_uses_full_history_not_last_window():
    # 29 closes: first 15 all rise (+1), last 14 all fall (-1).
    # Seed (first 14 deltas) = all gains → avg_gain=1, avg_loss=0.
    # Wilder's EMA then decays avg_gain and builds avg_loss over 14 more steps.
    # A naive SMA of only the last 14 bars (all -1) would give RSI = 0.
    # Wilder's EMA carries forward the initial gain seed → RSI > 0.
    closes = list(range(1, 16)) + [14 - i for i in range(14)]
    result = _rsi(closes, window=14)
    assert result is not None
    assert result > 0.0, f"Wilder EMA should give RSI > 0 (gains in history), got {result}"
    assert result < 50.0, f"Recent losses should push RSI below 50, got {result}"


def test_structural_alpha_kernel(sample_data):
    features = FeatureEngineer().build_feature_matrix(sample_data)
    kernel = StructuralAlphaKernel(n_factors=2)
    kernel.fit(features)
    alpha = kernel.structural_alpha(features)
    assert alpha.name == "structural_alpha"
    assert len(alpha) == len(sample_data)


def test_superfactor_model(sample_data):
    features = FeatureEngineer().build_feature_matrix(sample_data)
    model = SuperfactorModel()
    superfactor = model.fit_transform(features)
    assert superfactor.name == "superfactor_alpha"
    assert len(superfactor) == len(sample_data)
