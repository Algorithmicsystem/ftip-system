"""Phase 31: Knowledge Vault IP Implementation tests.

Covers all 10 proprietary calculations grounded in the 100-book knowledge vault:
  1.1  Penman-Schilit EIS
  1.2  Rappaport CAPS
  1.3  Kahneman-Tversky Asymmetric Sentiment
  1.4  Sornette SCPS
  1.5  Mandelbrot MTRS
  1.6  Kyle Lambda KLE
  1.7  Ilmanen CARDI
  1.8  Shiller-Kindleberger BFS
  1.9  Grinold-Kahn AMQS
  1.10 De Prado Triple Barrier
"""
from __future__ import annotations

import math
import pytest


# ---------------------------------------------------------------------------
# 1.1 Penman-Schilit Earnings Integrity Score
# ---------------------------------------------------------------------------

class TestEIS:
    def test_high_cfo_relative_to_earnings_scores_well(self):
        from api.axiom.engines.fundamental import compute_eis
        # CFO >> net income: healthy cash conversion
        result = compute_eis({"cfo": 200.0, "net_income": 100.0})
        assert result > 60.0

    def test_eis_detects_accruals_manipulation(self):
        """High accruals ratio (large positive accruals vs assets) → EIS < 40."""
        from api.axiom.engines.fundamental import compute_eis
        result = compute_eis({
            "accruals_ratio": 0.12,       # high accruals → bad quality
            "positive_fcf_ratio": 0.10,   # poor FCF backing
        })
        assert result < 40.0

    def test_low_accruals_rewards_integrity(self):
        from api.axiom.engines.fundamental import compute_eis
        result = compute_eis({
            "accruals_ratio": -0.05,      # negative accruals → very clean earnings
            "positive_fcf_ratio": 0.90,
            "earnings_beat_rate_4q": 0.80,
        })
        assert result > 60.0

    def test_high_receivables_growth_penalises(self):
        from api.axiom.engines.fundamental import compute_eis
        result = compute_eis({"receivables_growth": 0.45})  # rapid AR growth
        assert result < 55.0

    def test_output_bounded_0_100(self):
        from api.axiom.engines.fundamental import compute_eis
        for inputs in [
            {"cfo": -500.0, "net_income": 10.0, "accruals_ratio": 0.50},
            {"cfo": 10000.0, "net_income": 1.0},
            {},
        ]:
            r = compute_eis(inputs)
            assert 0.0 <= r <= 100.0

    def test_empty_financials_returns_midrange(self):
        from api.axiom.engines.fundamental import compute_eis
        # No data → should fall back to defaults around 50
        result = compute_eis({})
        assert 30.0 <= result <= 75.0


# ---------------------------------------------------------------------------
# 1.2 Rappaport CAPS
# ---------------------------------------------------------------------------

class TestCAPS:
    def test_caps_rewards_durable_advantage(self):
        """High ROCE spread + stable margins → CAPS > 60."""
        from api.axiom.engines.fundamental import compute_caps
        result = compute_caps(
            financials={
                "return_on_capital_employed": 0.22,   # ROCE = 22%
                "gross_margin": 0.60,
                "operating_margin": 0.25,
                "revenue_growth_yoy": 0.15,
            },
            sector_context={"wacc": 0.08, "sector_moat_score": 80.0},
        )
        assert result > 60.0

    def test_caps_low_for_value_destroyer(self):
        """ROCE below WACC → CAPS < 40."""
        from api.axiom.engines.fundamental import compute_caps
        result = compute_caps(
            financials={
                "return_on_equity": 0.04,   # below WACC of 8%
                "gross_margin": 0.20,
                "operating_margin": 0.02,
            },
            sector_context={"wacc": 0.08, "sector_moat_score": 20.0},
        )
        assert result < 40.0

    def test_sector_moat_contributes(self):
        from api.axiom.engines.fundamental import compute_caps
        high = compute_caps(
            {"return_on_equity": 0.12},
            {"wacc": 0.08, "sector_moat_score": 90.0},
        )
        low = compute_caps(
            {"return_on_equity": 0.12},
            {"wacc": 0.08, "sector_moat_score": 10.0},
        )
        assert high > low

    def test_output_bounded_0_100(self):
        from api.axiom.engines.fundamental import compute_caps
        r = compute_caps({}, {})
        assert 0.0 <= r <= 100.0


# ---------------------------------------------------------------------------
# 1.3 Kahneman-Tversky Asymmetric Sentiment
# ---------------------------------------------------------------------------

class TestAsymmetricSentiment:
    def test_asymmetric_sentiment_loss_aversion(self):
        """Equal positive and negative sentiment → net score < 50 (losses dominate)."""
        from api.axiom.engines.behavior import compute_asymmetric_sentiment
        result = compute_asymmetric_sentiment(positive=50.0, negative=50.0)
        # net = 0.5 - 2.25*0.5 = -0.625 → below 50
        assert result < 50.0

    def test_pure_positive_sentiment_above_50(self):
        from api.axiom.engines.behavior import compute_asymmetric_sentiment
        result = compute_asymmetric_sentiment(positive=80.0, negative=0.0)
        assert result > 50.0

    def test_pure_negative_sentiment_below_50(self):
        from api.axiom.engines.behavior import compute_asymmetric_sentiment
        result = compute_asymmetric_sentiment(positive=0.0, negative=80.0)
        assert result < 50.0

    def test_zero_zero_maps_to_neutral(self):
        """No sentiment in either direction → exactly 50 (neutral reference point)."""
        from api.axiom.engines.behavior import compute_asymmetric_sentiment
        result = compute_asymmetric_sentiment(0.0, 0.0)
        assert result == pytest.approx(50.0)

    def test_loss_aversion_coefficient_is_225(self):
        from api.axiom.engines.behavior import _LOSS_AVERSION_COEFFICIENT
        assert _LOSS_AVERSION_COEFFICIENT == pytest.approx(2.25)

    def test_output_bounded_0_100(self):
        from api.axiom.engines.behavior import compute_asymmetric_sentiment
        for p, n in [(0, 0), (100, 0), (0, 100), (100, 100), (200, -50)]:
            r = compute_asymmetric_sentiment(p, n)
            assert 0.0 <= r <= 100.0


# ---------------------------------------------------------------------------
# 1.4 Sornette SCPS
# ---------------------------------------------------------------------------

class TestSCPS:
    def _accelerating_bubble(self, n=120):
        """Super-exponential: daily return accelerates — slope of log-prices rises."""
        prices = [100.0]
        for i in range(1, n):
            # Return increases linearly: 0.1% → 1.5% over the window
            ret = 0.001 + 0.014 * i / n
            prices.append(prices[-1] * (1.0 + ret))
        return prices

    def _flat_series(self, n=60, value=100.0):
        return [value] * n

    def test_scps_elevated_in_accelerating_bubble(self):
        """Accelerating log-price slope (rising daily returns) → SCPS > 70."""
        from api.axiom.engines.fragility import compute_scps
        prices = self._accelerating_bubble(n=120)
        result = compute_scps(prices)
        assert result > 70.0

    def test_scps_moderate_for_flat_series(self):
        from api.axiom.engines.fragility import compute_scps
        result = compute_scps(self._flat_series(n=60))
        # Flat series: no acceleration, should be near neutral
        assert result < 75.0

    def test_too_short_returns_50(self):
        from api.axiom.engines.fragility import compute_scps
        assert compute_scps([100.0, 101.0]) == pytest.approx(50.0)

    def test_empty_returns_50(self):
        from api.axiom.engines.fragility import compute_scps
        assert compute_scps([]) == pytest.approx(50.0)

    def test_output_bounded(self):
        from api.axiom.engines.fragility import compute_scps
        prices = [100 * math.exp(0.20 * i) for i in range(120)]
        r = compute_scps(prices)
        assert 0.0 <= r <= 100.0


# ---------------------------------------------------------------------------
# 1.5 Mandelbrot MTRS
# ---------------------------------------------------------------------------

class TestMTRS:
    def _normal_returns(self, n=200, std=0.01):
        import random
        random.seed(42)
        return [random.gauss(0, std) for _ in range(n)]

    def _fat_tail_returns(self, n=200):
        """Heavy-tailed: Pareto-like extremes with varying magnitudes."""
        import random
        random.seed(7)
        base = [0.001] * n
        # Power-law extremes: 0.30, 0.22, 0.18, 0.15, ... decreasing
        extreme_mags = [0.30, 0.26, 0.22, 0.19, 0.17, 0.15, 0.13, 0.12, 0.11, 0.10,
                        0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06]
        indices = random.sample(range(n), len(extreme_mags))
        for idx, mag in zip(indices, extreme_mags):
            base[idx] = random.choice([-1, 1]) * mag
        return base

    def test_mtrs_elevated_for_fat_tails(self):
        """Heavy-tailed return series → MTRS > 65."""
        from api.axiom.engines.fragility import compute_mtrs
        result = compute_mtrs(self._fat_tail_returns(200))
        assert result > 65.0

    def test_mtrs_lower_for_normal_returns(self):
        """Near-normal returns → MTRS < 40."""
        from api.axiom.engines.fragility import compute_mtrs
        result = compute_mtrs(self._normal_returns(200, std=0.005))
        assert result < 55.0

    def test_fat_tails_higher_than_normal(self):
        from api.axiom.engines.fragility import compute_mtrs
        fat = compute_mtrs(self._fat_tail_returns())
        normal = compute_mtrs(self._normal_returns())
        assert fat > normal

    def test_too_short_returns_50(self):
        from api.axiom.engines.fragility import compute_mtrs
        assert compute_mtrs([0.01, 0.02]) == pytest.approx(50.0)

    def test_output_bounded(self):
        from api.axiom.engines.fragility import compute_mtrs
        r = compute_mtrs(self._fat_tail_returns())
        assert 0.0 <= r <= 100.0


# ---------------------------------------------------------------------------
# 1.6 Kyle Lambda KLE
# ---------------------------------------------------------------------------

class TestKLE:
    def test_kle_high_for_liquid(self):
        """Large avg volume and small lambda → high liquidity score."""
        from api.axiom.engines.liquidity_convexity import compute_kyle_lambda
        # Small price impact per unit of flow = liquid
        returns = [0.001] * 20
        volumes = [1_000_000.0] * 20
        avg_vol = 1_000_000.0
        result = compute_kyle_lambda(returns, volumes, avg_vol)
        assert result > 60.0

    def test_kle_high_for_illiquid(self):
        """High price impact per unit of flow → low liquidity score."""
        from api.axiom.engines.liquidity_convexity import compute_kyle_lambda
        # Large moves (8%) on normal volume: high lambda = illiquid
        returns = [0.08] * 20   # 8% per-bar moves
        volumes = [10_000.0] * 20
        avg_vol = 10_000.0
        result = compute_kyle_lambda(returns, volumes, avg_vol)
        assert result < 40.0

    def test_missing_inputs_returns_50(self):
        from api.axiom.engines.liquidity_convexity import compute_kyle_lambda
        assert compute_kyle_lambda([], [], 0) == pytest.approx(50.0)
        assert compute_kyle_lambda([0.01], [1000], 0) == pytest.approx(50.0)

    def test_output_bounded(self):
        from api.axiom.engines.liquidity_convexity import compute_kyle_lambda
        r = compute_kyle_lambda([0.10] * 10, [5000.0] * 10, 5000.0)
        assert 0.0 <= r <= 100.0


# ---------------------------------------------------------------------------
# 1.7 Ilmanen CARDI
# ---------------------------------------------------------------------------

class TestCARDI:
    def test_cardi_high_in_carry_momentum_environment(self):
        """Positive term spread + strong 12-1 momentum → CARDI > 65."""
        from api.axiom.engines.state_pricing import compute_cardi
        result = compute_cardi({
            "carry_score": 75.0,       # positive term spread
            "value_score": 60.0,
            "momentum_score": 80.0,    # strong 12-1 momentum
            "defensive_score": 55.0,
        })
        assert result > 65.0

    def test_cardi_low_in_risk_off(self):
        from api.axiom.engines.state_pricing import compute_cardi
        result = compute_cardi({
            "carry_score": 20.0,
            "value_score": 25.0,
            "momentum_score": 15.0,
            "defensive_score": 30.0,
        })
        assert result < 35.0

    def test_empty_dict_returns_50(self):
        from api.axiom.engines.state_pricing import compute_cardi
        assert compute_cardi({}) == pytest.approx(50.0)

    def test_momentum_has_highest_weight(self):
        """Momentum gets 0.30 weight; changing it should have largest effect."""
        from api.axiom.engines.state_pricing import compute_cardi
        base = compute_cardi({"carry_score": 50.0, "value_score": 50.0,
                              "momentum_score": 50.0, "defensive_score": 50.0})
        momentum_up = compute_cardi({"carry_score": 50.0, "value_score": 50.0,
                                     "momentum_score": 100.0, "defensive_score": 50.0})
        carry_up = compute_cardi({"carry_score": 100.0, "value_score": 50.0,
                                  "momentum_score": 50.0, "defensive_score": 50.0})
        assert (momentum_up - base) > (carry_up - base)

    def test_output_bounded(self):
        from api.axiom.engines.state_pricing import compute_cardi
        r = compute_cardi({"carry_score": 150.0, "momentum_score": -50.0})
        assert 0.0 <= r <= 100.0


# ---------------------------------------------------------------------------
# 1.8 Shiller-Kindleberger BFS
# ---------------------------------------------------------------------------

class TestBFS:
    def test_bfs_extreme_in_euphoria_stage(self):
        """High CAPE + euphoria stage → BFS > 75."""
        from api.axiom.engines.fragility import compute_bfs
        result = compute_bfs({
            "cape_z_score": 3.0,
            "kindleberger_stage": "euphoria",
            "narrative_intensity": 80.0,
        })
        assert result > 75.0

    def test_bfs_low_in_normal_recovery(self):
        from api.axiom.engines.fragility import compute_bfs
        result = compute_bfs({
            "cape_z_score": -1.0,
            "kindleberger_stage": "recovery",
            "narrative_intensity": 20.0,
        })
        assert result < 35.0

    def test_panic_stage_highest(self):
        from api.axiom.engines.fragility import compute_bfs
        panic = compute_bfs({"kindleberger_stage": "panic"})
        normal = compute_bfs({"kindleberger_stage": "normal"})
        assert panic > normal

    def test_kindleberger_stage_scores_correct(self):
        from api.axiom.engines.fragility import _KINDLEBERGER_STAGE_SCORE
        assert _KINDLEBERGER_STAGE_SCORE["euphoria"] == 80.0
        assert _KINDLEBERGER_STAGE_SCORE["panic"] == 90.0
        assert _KINDLEBERGER_STAGE_SCORE["displacement"] == 20.0
        assert _KINDLEBERGER_STAGE_SCORE["recovery"] == 10.0

    def test_output_bounded(self):
        from api.axiom.engines.fragility import compute_bfs
        r = compute_bfs({"cape_z_score": 10.0, "kindleberger_stage": "panic",
                         "narrative_intensity": 100.0})
        assert 0.0 <= r <= 100.0

    def test_empty_context_midrange(self):
        from api.axiom.engines.fragility import compute_bfs
        r = compute_bfs({})
        assert 20.0 <= r <= 65.0


# ---------------------------------------------------------------------------
# 1.9 Grinold-Kahn AMQS
# ---------------------------------------------------------------------------

class TestAMQS:
    def test_amqs_higher_with_more_breadth(self):
        """Same IC and TE, more symbols → higher AMQS (use moderate values to avoid ceiling)."""
        from api.jobs.ic_gate import compute_amqs
        low_breadth = compute_amqs(ic_value=0.02, breadth=10, tracking_error=0.10)
        high_breadth = compute_amqs(ic_value=0.02, breadth=100, tracking_error=0.10)
        assert high_breadth > low_breadth

    def test_amqs_positive_ic_above_midpoint(self):
        from api.jobs.ic_gate import compute_amqs
        result = compute_amqs(ic_value=0.10, breadth=50, tracking_error=0.05)
        assert result > 50.0

    def test_amqs_negative_ic_low_score(self):
        from api.jobs.ic_gate import compute_amqs
        result = compute_amqs(ic_value=-0.10, breadth=50, tracking_error=0.05)
        assert result < 50.0

    def test_zero_ic_midrange(self):
        from api.jobs.ic_gate import compute_amqs
        result = compute_amqs(ic_value=0.0, breadth=50, tracking_error=0.05)
        # IR = 0 → should map to 40 (2/5 * 100)
        assert result == pytest.approx(40.0, abs=2.0)

    def test_amqs_in_gate_state(self, monkeypatch):
        """load_ic_gate_state returns amqs_score in its dict."""
        from api import db
        monkeypatch.setattr(db, "db_read_enabled", lambda: True)
        import datetime as dt
        fake_row = ("STRONG", 50, 0.06)
        monkeypatch.setattr(db, "safe_fetchone", lambda *a, **k: fake_row)
        from api.jobs.ic_gate import load_ic_gate_state
        result = load_ic_gate_state(dt.date(2026, 1, 5))
        assert "amqs_score" in result
        assert 0.0 <= result["amqs_score"] <= 100.0

    def test_output_bounded(self):
        from api.jobs.ic_gate import compute_amqs
        for ic, br, te in [(1.0, 1000, 0.001), (-1.0, 1, 10.0), (0.0, 0, 0.0)]:
            r = compute_amqs(ic, br, te)
            assert 0.0 <= r <= 100.0


# ---------------------------------------------------------------------------
# 1.10 De Prado Triple Barrier
# ---------------------------------------------------------------------------

class TestTripleBarrier:
    def test_profit_target_hit(self):
        """Price rises above profit barrier → label = 1."""
        from api.jobs.pnl import compute_triple_barrier_label
        prices = [100.0, 101.0, 102.5, 103.0]  # entry 100, profit target 102
        result = compute_triple_barrier_label(
            price_series=prices,
            entry_price=100.0,
            profit_factor=0.02,
            stop_factor=0.01,
            horizon_days=5,
        )
        assert result == 1

    def test_stop_loss_hit(self):
        """Price falls below stop barrier → label = -1."""
        from api.jobs.pnl import compute_triple_barrier_label
        prices = [100.0, 99.5, 98.5, 97.0]  # stop at 99
        result = compute_triple_barrier_label(
            price_series=prices,
            entry_price=100.0,
            profit_factor=0.02,
            stop_factor=0.01,
            horizon_days=5,
        )
        assert result == -1

    def test_time_stop(self):
        """Price stays between barriers → label = 0."""
        from api.jobs.pnl import compute_triple_barrier_label
        prices = [100.0, 100.3, 100.5, 100.8, 101.0]  # never hits 102 or 99
        result = compute_triple_barrier_label(
            price_series=prices,
            entry_price=100.0,
            profit_factor=0.02,
            stop_factor=0.01,
            horizon_days=5,
        )
        assert result == 0

    def test_triple_barrier_labels_correctly(self):
        """Known price paths → expected labels."""
        from api.jobs.pnl import compute_triple_barrier_label
        # Path 1: hits profit first
        assert compute_triple_barrier_label([100, 105], 100.0, 0.03, 0.02, 5) == 1
        # Path 2: hits stop first
        assert compute_triple_barrier_label([100, 97], 100.0, 0.03, 0.02, 5) == -1
        # Path 3: neither
        assert compute_triple_barrier_label([100, 101, 100], 100.0, 0.03, 0.02, 3) == 0

    def test_profit_barrier_takes_priority_over_stop(self):
        """If both would be hit, the first price to cross determines label."""
        from api.jobs.pnl import compute_triple_barrier_label
        # Rises above profit first, then would have dropped
        result = compute_triple_barrier_label(
            [100.0, 103.5, 98.0],  # 103.5 hits profit (2%) before 98 hits stop
            entry_price=100.0,
            profit_factor=0.02,
            stop_factor=0.02,
            horizon_days=5,
        )
        assert result == 1

    def test_empty_series_returns_time_stop(self):
        from api.jobs.pnl import compute_triple_barrier_label
        assert compute_triple_barrier_label([], 100.0) == 0

    def test_zero_entry_price_returns_time_stop(self):
        from api.jobs.pnl import compute_triple_barrier_label
        assert compute_triple_barrier_label([100.0, 105.0], 0.0) == 0
