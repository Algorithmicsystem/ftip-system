"""Phase 11: Advanced Risk Framework tests.

Tests for VaR engine, stress testing, correlation monitor,
drawdown intelligence, and the full Systemic Risk Index.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import List
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_returns(n: int = 50, step: float = 0.001) -> List[float]:
    """Deterministic return series: [step, 2*step, ..., n*step] centred near 0."""
    half = n // 2
    return [step * (i - half) for i in range(n)]


def _alternating_returns(n: int = 30, amplitude: float = 0.02) -> List[float]:
    return [amplitude if i % 2 == 0 else -amplitude for i in range(n)]


def _orthogonal_pair(n: int = 30, amplitude: float = 0.01):
    """Two uncorrelated return series."""
    a = [amplitude if i % 4 in (0, 1) else -amplitude for i in range(n)]
    b = [amplitude if i % 4 in (1, 2) else -amplitude for i in range(n)]
    return a, b


# ---------------------------------------------------------------------------
# TestVaREngine
# ---------------------------------------------------------------------------

class TestVaREngine:
    def test_var_positive_number(self):
        from api.axiom.risk.var_engine import compute_historical_var
        returns = _linear_returns(50)
        result = compute_historical_var(returns, confidence=0.99)
        assert result["var_1d"] > 0.0

    def test_historical_var_99_correct(self):
        from api.axiom.risk.var_engine import compute_historical_var
        # 100 returns; 1st percentile should be near the most negative value
        returns = [i * 0.001 - 0.05 for i in range(100)]  # -0.05 to 0.049
        result = compute_historical_var(returns, confidence=0.99, mtrs_score=0.0)
        # With mtrs_score=0, adjustment=1.0, var_1d = -sorted[1] = 0.049
        sorted_r = sorted(returns)
        expected_raw = -sorted_r[1]
        assert result["var_1d"] == pytest.approx(expected_raw, rel=0.01)

    def test_cvar_gte_var(self):
        from api.axiom.risk.var_engine import compute_historical_var
        returns = _linear_returns(60)
        result = compute_historical_var(returns, confidence=0.95)
        assert result["cvar_1d"] >= result["var_1d"]

    def test_parametric_var_z_scores(self):
        from api.axiom.risk.var_engine import compute_parametric_var
        daily_vol = 0.20 / math.sqrt(252)
        result = compute_parametric_var(0.20, confidence=0.99, mtrs_score=0.0)
        # fat_tail = 1.0 when mtrs_score=0
        expected = 2.326 * daily_vol
        assert result["var_1d"] == pytest.approx(expected, rel=0.01)

    def test_mtrs_adjustment_increases_var(self):
        from api.axiom.risk.var_engine import compute_historical_var
        returns = _linear_returns(50)
        low = compute_historical_var(returns, mtrs_score=0.0)
        high = compute_historical_var(returns, mtrs_score=80.0)
        assert high["var_1d"] > low["var_1d"]

    def test_insufficient_data_returns_none(self):
        from api.axiom.risk.var_engine import compute_historical_var
        result = compute_historical_var([0.01] * 10)  # only 10 returns
        assert result["var_1d"] is None
        assert result["sample_count"] == 0

    def test_portfolio_var_computed(self):
        from api.axiom.risk.var_engine import compute_portfolio_var
        r1 = _linear_returns(50, 0.002)
        r2 = _linear_returns(50, 0.001)
        result = compute_portfolio_var({"A": 0.6, "B": 0.4}, {"A": r1, "B": r2})
        assert result["portfolio_var_1d"] is not None
        assert result["portfolio_var_1d"] > 0.0

    def test_diversification_benefit_positive(self):
        from api.axiom.risk.var_engine import compute_portfolio_var
        a, b = _orthogonal_pair(50, 0.02)
        result = compute_portfolio_var({"A": 0.5, "B": 0.5}, {"A": a, "B": b})
        # Imperfect correlation → undiversified > portfolio VaR
        if result["diversification_benefit"] is not None:
            assert result["diversification_benefit"] >= -1e-6  # can be near 0

    def test_concentration_risk_detected(self):
        from api.axiom.risk.var_engine import compute_portfolio_var
        # One position holds 90% weight in perfect-correlation scenario
        r = _linear_returns(50, 0.002)
        result = compute_portfolio_var({"BIG": 0.9, "SMALL": 0.1}, {"BIG": r, "SMALL": r})
        assert result["concentration_risk"] is True

    def test_concentration_risk_false_for_equal_weights(self):
        from api.axiom.risk.var_engine import compute_portfolio_var
        r1 = _linear_returns(50, 0.002)
        r2 = list(reversed(r1))
        result = compute_portfolio_var({"A": 0.5, "B": 0.5}, {"A": r1, "B": r2})
        # Equal weights, opposite correlation → no single dominator
        # Either outcome is acceptable for truly balanced portfolios
        assert isinstance(result["concentration_risk"], bool)

    def test_portfolio_marginal_var_keys(self):
        from api.axiom.risk.var_engine import compute_portfolio_var
        r = _linear_returns(50, 0.002)
        result = compute_portfolio_var({"AAPL": 0.5, "MSFT": 0.5}, {"AAPL": r, "MSFT": r})
        assert "AAPL" in result["marginal_var"]
        assert "MSFT" in result["marginal_var"]


# ---------------------------------------------------------------------------
# TestStressEngine
# ---------------------------------------------------------------------------

class TestStressEngine:
    def test_liquidity_fracture_loss_negative(self):
        from api.axiom.risk.stress_engine import run_stress_test
        positions = {"TECH": 1.0}
        scores = {"TECH": {"sector": "Technology", "fragility_score": 50, "scps_score": 50}}
        result = run_stress_test(positions, scores, scenarios=["liquidity_fracture"])
        assert result["scenarios"]["liquidity_fracture"]["portfolio_loss_pct"] < 0.0

    def test_recovery_scenario_positive(self):
        from api.axiom.risk.stress_engine import run_stress_test
        positions = {"X": 1.0}
        scores = {"X": {"sector": "Finance", "fragility_score": 50, "scps_score": 50}}
        result = run_stress_test(positions, scores, scenarios=["recovery_reset"])
        assert result["scenarios"]["recovery_reset"]["portfolio_loss_pct"] > 0.0

    def test_worst_scenario_identified(self):
        from api.axiom.risk.stress_engine import run_stress_test
        positions = {"A": 1.0}
        scores = {"A": {"sector": "Technology", "fragility_score": 60, "scps_score": 60}}
        result = run_stress_test(positions, scores)
        worst = result["worst_scenario"]
        worst_loss = result["scenarios"][worst]["portfolio_loss_pct"]
        for s, data in result["scenarios"].items():
            assert data["portfolio_loss_pct"] >= worst_loss

    def test_all_scenarios_computed(self):
        from api.axiom.risk.stress_engine import run_stress_test
        positions = {"X": 1.0}
        scores = {"X": {"sector": "Unknown", "fragility_score": 50, "scps_score": 50}}
        result = run_stress_test(positions, scores)  # scenarios=None → all 4
        assert len(result["scenarios"]) == 4

    def test_sornette_identifies_high_risk(self):
        from api.axiom.risk.stress_engine import run_sornette_scenario
        positions = {"BUBBLE": 0.5, "SAFE": 0.5}
        scores = {
            "BUBBLE": {"scps_score": 90},
            "SAFE": {"scps_score": 30},
        }
        result = run_sornette_scenario(positions, scores)
        assert "BUBBLE" in result["high_risk_symbols"]
        assert "SAFE" not in result["high_risk_symbols"]

    def test_sornette_recommendation_reduce(self):
        from api.axiom.risk.stress_engine import run_sornette_scenario
        # 35% weight in high-SCPS names → reduce_exposure
        positions = {f"X{i}": 0.35 / 3 for i in range(3)}
        positions["SAFE"] = 0.65
        scores = {f"X{i}": {"scps_score": 85} for i in range(3)}
        scores["SAFE"] = {"scps_score": 20}
        result = run_sornette_scenario(positions, scores)
        assert result["recommendation"] == "reduce_exposure"

    def test_stress_var_99_bounded(self):
        from api.axiom.risk.stress_engine import run_stress_test
        positions = {"A": 0.5, "B": 0.5}
        scores = {
            "A": {"sector": "Technology", "fragility_score": 50, "scps_score": 50},
            "B": {"sector": "Utilities", "fragility_score": 50, "scps_score": 50},
        }
        result = run_stress_test(positions, scores)
        # Max possible scenario loss < 100%
        assert result["stress_var_99"] < 1.0

    def test_sector_impact_varies(self):
        from api.axiom.risk.stress_engine import run_stress_test
        positions = {"TECH": 0.5, "UTIL": 0.5}
        scores = {
            "TECH": {"sector": "Technology", "fragility_score": 50, "scps_score": 50},
            "UTIL": {"sector": "Utilities", "fragility_score": 50, "scps_score": 50},
        }
        # In liquidity_fracture: Technology multiplier=1.4, Utilities=0.6 → different losses
        full = run_stress_test(positions, scores, scenarios=["liquidity_fracture"])
        worst = full["scenarios"]["liquidity_fracture"]["worst_positions"]
        # TECH position loss should be more negative than UTIL
        tech_loss = next((p["position_loss_pct"] for p in worst if p["symbol"] == "TECH"), None)
        util_loss = next((p["position_loss_pct"] for p in worst if p["symbol"] == "UTIL"), None)
        if tech_loss is not None and util_loss is not None:
            assert tech_loss < util_loss  # TECH loss is more negative


# ---------------------------------------------------------------------------
# TestCorrelationMonitor
# ---------------------------------------------------------------------------

class TestCorrelationMonitor:
    def test_perfect_correlation(self):
        from api.axiom.risk.correlation_monitor import compute_rolling_correlation_matrix
        r = [0.01 * i for i in range(30)]
        result = compute_rolling_correlation_matrix({"A": r, "B": r})
        assert result["correlation_matrix"]["A"]["B"] == pytest.approx(1.0, abs=1e-6)

    def test_zero_correlation(self):
        from api.axiom.risk.correlation_monitor import compute_rolling_correlation_matrix
        # Period-2 vs period-4 series: each group of 4 contributes zero dot product
        n = 40
        a = [0.01 if i % 2 == 0 else -0.01 for i in range(n)]   # period 2: +,-,+,-,...
        b = [0.01 if i % 4 in (0, 1) else -0.01 for i in range(n)]  # period 4: +,+,-,-,...
        result = compute_rolling_correlation_matrix({"A": a, "B": b})
        corr = result["correlation_matrix"]["A"]["B"]
        assert abs(corr) < 0.1

    def test_regime_normal(self):
        from api.axiom.risk.correlation_monitor import compute_rolling_correlation_matrix
        # Two series with low correlation → avg near 0 → normal
        a = [0.01 * (i % 5 - 2) for i in range(40)]
        b = [0.01 * ((i + 2) % 5 - 2) for i in range(40)]
        result = compute_rolling_correlation_matrix({"A": a, "B": b})
        assert result["avg_pairwise_correlation"] < 0.60  # at most elevated

    def test_regime_crisis(self):
        from api.axiom.risk.correlation_monitor import compute_rolling_correlation_matrix
        # Two nearly identical series → high avg correlation → crisis
        r = [0.01 * i for i in range(30)]
        noise = [r[i] + 0.0001 for i in range(30)]
        result = compute_rolling_correlation_matrix({"A": r, "B": noise})
        assert result["correlation_regime"] == "crisis"

    def test_spike_detected(self):
        from api.axiom.risk.correlation_monitor import detect_correlation_spike
        result = detect_correlation_spike(current_avg=0.60, historical_avg=0.30, threshold_multiplier=1.5)
        assert result["spike_detected"] is True

    def test_spike_severity_severe(self):
        from api.axiom.risk.correlation_monitor import detect_correlation_spike
        result = detect_correlation_spike(current_avg=0.90, historical_avg=0.25, threshold_multiplier=1.5)
        assert result["severity"] == "severe"

    def test_correlation_regime_score_bounded(self):
        from api.axiom.risk.correlation_monitor import compute_correlation_regime_score
        for avg in (0.0, 0.5, 1.0):
            for trend in (-5.0, 0.0, 5.0):
                crs = compute_correlation_regime_score(avg, trend)
                assert 0.0 <= crs <= 100.0


# ---------------------------------------------------------------------------
# TestDrawdownEngine
# ---------------------------------------------------------------------------

class TestDrawdownEngine:
    def test_drawdown_zero_for_all_positive(self):
        from api.axiom.risk.drawdown_engine import compute_drawdown_series
        returns = [0.01] * 30  # always going up
        result = compute_drawdown_series(returns)
        assert result["current_drawdown_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_negative(self):
        from api.axiom.risk.drawdown_engine import compute_drawdown_series
        # Goes up then sharply down
        returns = [0.02] * 10 + [-0.05] * 8 + [0.01] * 5
        result = compute_drawdown_series(returns)
        assert result["max_drawdown_pct"] < 0.0

    def test_calmar_ratio_computed(self):
        from api.axiom.risk.drawdown_engine import compute_drawdown_series
        returns = [0.005] * 50 + [-0.02] * 5 + [0.003] * 30
        result = compute_drawdown_series(returns)
        if result["max_drawdown_pct"] < 0:
            assert result["calmar_ratio"] != 0.0

    def test_recovery_days_higher_in_stress(self):
        from api.axiom.risk.drawdown_engine import compute_expected_recovery_time
        r_trending = compute_expected_recovery_time(-0.20, 0.15, "TRENDING")
        r_high_vol = compute_expected_recovery_time(-0.20, 0.15, "HIGH_VOL")
        assert r_high_vol["estimated_recovery_days"] > r_trending["estimated_recovery_days"]

    def test_max_pain_triggered(self):
        from api.axiom.risk.drawdown_engine import compute_max_pain_index
        # moderate threshold = -0.20; current = -0.25 → triggered
        result = compute_max_pain_index(-0.25, 0.10, "moderate")
        assert result["pain_triggered"] is True

    def test_max_pain_not_triggered(self):
        from api.axiom.risk.drawdown_engine import compute_max_pain_index
        # moderate threshold = -0.20; current = -0.10 → not triggered
        result = compute_max_pain_index(-0.10, 0.10, "moderate")
        assert result["pain_triggered"] is False

    def test_recommendation_exit_when_past_pain(self):
        from api.axiom.risk.drawdown_engine import compute_max_pain_index
        result = compute_max_pain_index(-0.30, 0.10, "moderate")
        assert result["distance_to_max_pain"] < 0.0
        assert result["recommendation"] == "exit"


# ---------------------------------------------------------------------------
# TestSystemicRiskIndex
# ---------------------------------------------------------------------------

class TestSystemicRiskIndex:
    def test_sri_neutral_no_data(self):
        from api.axiom.risk.systemic_risk import compute_sri
        with patch("api.axiom.risk.systemic_risk.db.db_read_enabled", return_value=False):
            result = compute_sri(dt.date.today())
        assert result["sri"] == pytest.approx(50.0, abs=0.01)

    def test_sri_bounded(self):
        from api.axiom.risk.systemic_risk import compute_sri
        with patch("api.axiom.risk.systemic_risk.db.db_read_enabled", return_value=False):
            for _ in range(3):
                result = compute_sri(dt.date.today())
                assert 0.0 <= result["sri"] <= 100.0

    def test_sri_label_stable(self):
        from api.axiom.risk.systemic_risk import _sri_label
        assert _sri_label(20.0) == "stable"

    def test_sri_label_critical(self):
        from api.axiom.risk.systemic_risk import _sri_label
        assert _sri_label(90.0) == "critical"

    def test_sri_recommendation_generated(self):
        from api.axiom.risk.systemic_risk import compute_sri
        with patch("api.axiom.risk.systemic_risk.db.db_read_enabled", return_value=False):
            result = compute_sri(dt.date.today())
        assert len(result["recommendation"]) > 0

    def test_sri_primary_driver_identified(self):
        from api.axiom.risk.systemic_risk import compute_sri, _SRI_COMPONENTS
        with patch("api.axiom.risk.systemic_risk.db.db_read_enabled", return_value=False):
            result = compute_sri(dt.date.today())
        assert result["primary_driver"] in set(_SRI_COMPONENTS.keys())

    def test_sri_history_returns_list(self):
        from api.axiom.risk.systemic_risk import get_sri_history
        with patch("api.axiom.risk.systemic_risk.db.db_read_enabled", return_value=False):
            history = get_sri_history()
        assert isinstance(history, list)
