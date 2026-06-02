"""Phase 32: Intelligence Linkage Graph Expansion tests.

Covers Phase 2 implementation:
  2.1  Regime Analog Library (find_regime_analogs)
  2.2  Company-Macro Sensitivity (compute_ols_beta, store/load)
  2.3  Cross-Sector Linkage Intelligence (get_peers_with_axiom, get_stress_propagation)
"""
from __future__ import annotations

import datetime as dt
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analog_row(
    analog_id="analog_test_001",
    ref_date=dt.date(2022, 1, 5),
    regime="liquidity_fracture",
    macro_ctx=None,
    ret_30d=None,
    ret_90d=None,
    vix=23.0,
    cape=38.0,
    ic_state="WEAK",
    description="Test event",
):
    return (
        analog_id,
        ref_date,
        regime,
        macro_ctx or {"rate_env": "hiking"},
        ret_30d or {"Technology": -18.0, "Energy": 12.0},
        ret_90d or {"Technology": -28.0, "Energy": 30.0},
        vix,
        cape,
        ic_state,
        description,
    )


def _peer_row(
    linked="MSFT",
    link_type="sector_peer",
    strength=0.80,
    last_validated=dt.date(2024, 1, 1),
    method="rolling_correlation",
    dau=70.0,
    confidence=0.75,
    tier="TIER_1",
    regime="trend_confirmation",
):
    return (linked, link_type, strength, last_validated, method, dau, confidence, tier, regime)


# ---------------------------------------------------------------------------
# 2.1 — find_regime_analogs
# ---------------------------------------------------------------------------

class TestFindRegimeAnalogs:
    def test_returns_empty_when_db_disabled(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
        result = mod.find_regime_analogs("liquidity_fracture")
        assert result == []

    def test_returns_empty_on_db_error(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: (_ for _ in ()).throw(Exception("db error")))
        result = mod.find_regime_analogs("liquidity_fracture")
        assert result == []

    def test_returns_analogs_with_expected_keys(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [_analog_row()])
        result = mod.find_regime_analogs("liquidity_fracture")
        assert len(result) == 1
        a = result[0]
        assert "analog_id" in a
        assert "reference_date" in a
        assert "similarity_score" in a
        assert "following_30d_return_by_sector" in a
        assert "following_90d_return_by_sector" in a

    def test_vix_proximity_lowers_score_for_distant_vix(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        # Two analogs: one with vix=20 (close), one with vix=50 (far)
        rows = [
            _analog_row(analog_id="close", vix=20.0, ref_date=dt.date(2022, 1, 5)),
            _analog_row(analog_id="far", vix=50.0, ref_date=dt.date(2021, 1, 5)),
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.find_regime_analogs("liquidity_fracture", vix_current=22.0)
        scores = {a["analog_id"]: a["similarity_score"] for a in result}
        assert scores["close"] > scores["far"]

    def test_cape_proximity_lowers_score_for_distant_cape(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        rows = [
            _analog_row(analog_id="close_cape", cape=30.0, ref_date=dt.date(2022, 1, 5)),
            _analog_row(analog_id="far_cape", cape=5.0, ref_date=dt.date(2021, 1, 5)),
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.find_regime_analogs("liquidity_fracture", cape_current=28.0)
        scores = {a["analog_id"]: a["similarity_score"] for a in result}
        assert scores["close_cape"] > scores["far_cape"]

    def test_limit_respected(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        rows = [_analog_row(analog_id=f"a{i}", ref_date=dt.date(2020 + i, 1, 1)) for i in range(10)]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.find_regime_analogs("liquidity_fracture", limit=3)
        assert len(result) <= 3

    def test_similarity_score_bounded_0_100(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        rows = [_analog_row(vix=100.0, cape=100.0)]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.find_regime_analogs("x", vix_current=10.0, cape_current=10.0)
        for a in result:
            assert 0.0 <= a["similarity_score"] <= 100.0

    def test_recent_analog_scores_higher_than_old(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        rows = [
            _analog_row(analog_id="recent", ref_date=dt.date(2023, 6, 1), vix=23.0, cape=38.0),
            _analog_row(analog_id="old",    ref_date=dt.date(2001, 6, 1), vix=23.0, cape=38.0),
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.find_regime_analogs("liquidity_fracture", vix_current=23.0, cape_current=38.0)
        scores = {a["analog_id"]: a["similarity_score"] for a in result}
        assert scores["recent"] > scores["old"]

    def test_reference_date_is_iso_string(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [_analog_row()])
        result = mod.find_regime_analogs("liquidity_fracture")
        assert isinstance(result[0]["reference_date"], str)
        dt.date.fromisoformat(result[0]["reference_date"])  # must be valid ISO


# ---------------------------------------------------------------------------
# 2.2 — compute_ols_beta
# ---------------------------------------------------------------------------

class TestComputeOlsBeta:
    def test_returns_zero_beta_for_short_series(self):
        from api.jobs.regime_analogs import compute_ols_beta
        result = compute_ols_beta([0.01, 0.02], [0.01, 0.02])
        assert result["beta"] == 0.0
        assert result["r_squared"] == 0.0

    def test_perfect_correlation_gives_beta_one(self):
        from api.jobs.regime_analogs import compute_ols_beta
        series = [i * 0.01 for i in range(50)]
        result = compute_ols_beta(series, series)
        assert abs(result["beta"] - 1.0) < 0.01
        assert result["r_squared"] > 0.99

    def test_flat_factor_returns_zero(self):
        from api.jobs.regime_analogs import compute_ols_beta
        returns = [0.01 * i for i in range(20)]
        factor = [1.0] * 20  # no variation → var=0
        result = compute_ols_beta(returns, factor)
        assert result["beta"] == 0.0

    def test_negative_correlation(self):
        from api.jobs.regime_analogs import compute_ols_beta
        n = 50
        factor = [0.01 * i for i in range(n)]
        returns = [-0.02 * i for i in range(n)]
        result = compute_ols_beta(returns, factor)
        assert result["beta"] < 0.0

    def test_r_squared_bounded(self):
        from api.jobs.regime_analogs import compute_ols_beta
        import random
        random.seed(42)
        returns = [random.gauss(0, 0.01) for _ in range(100)]
        factor = [random.gauss(0, 0.01) for _ in range(100)]
        result = compute_ols_beta(returns, factor)
        assert 0.0 <= result["r_squared"] <= 1.0


# ---------------------------------------------------------------------------
# 2.2 — store_macro_sensitivity / load_macro_sensitivity
# ---------------------------------------------------------------------------

class TestMacroSensitivityStore:
    def test_store_returns_false_when_write_disabled(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: False)
        result = mod.store_macro_sensitivity("AAPL", "vix", 0.5, 0.3, dt.date(2024, 1, 1))
        assert result is False

    def test_store_returns_true_on_success(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        executed = []
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda sql, params: executed.append(params))
        result = mod.store_macro_sensitivity("AAPL", "vix", 0.5, 0.3, dt.date(2024, 1, 1))
        assert result is True
        assert len(executed) == 1
        params = executed[0]
        assert params[0] == "AAPL"
        assert params[1] == "vix"

    def test_store_returns_false_on_exception(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda *a: (_ for _ in ()).throw(Exception("fail")))
        result = mod.store_macro_sensitivity("AAPL", "vix", 0.5, 0.3, dt.date(2024, 1, 1))
        assert result is False

    def test_load_returns_empty_when_db_disabled(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
        result = mod.load_macro_sensitivity("AAPL")
        assert result == []

    def test_load_deserializes_rows(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        rows = [
            ("vix", 0.45, 0.32, 252, dt.date(2024, 6, 1)),
            ("gdp_growth", -0.12, 0.15, 252, dt.date(2024, 6, 1)),
        ]
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.load_macro_sensitivity("AAPL")
        assert len(result) == 2
        assert result[0]["macro_factor"] == "vix"
        assert result[0]["sensitivity_beta"] == pytest.approx(0.45)
        assert result[1]["macro_factor"] == "gdp_growth"
        assert result[1]["sensitivity_beta"] == pytest.approx(-0.12)

    def test_load_estimated_at_is_iso_string(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        rows = [("vix", 0.5, 0.3, 252, dt.date(2024, 6, 1))]
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.load_macro_sensitivity("AAPL")
        dt.date.fromisoformat(result[0]["estimated_at"])


# ---------------------------------------------------------------------------
# 2.3 — get_peers_with_axiom
# ---------------------------------------------------------------------------

class TestGetPeersWithAxiom:
    def test_returns_empty_when_db_disabled(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
        result = mod.get_peers_with_axiom("AAPL", dt.date(2024, 1, 1))
        assert result == []

    def test_returns_empty_on_db_error(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: (_ for _ in ()).throw(Exception("err")))
        result = mod.get_peers_with_axiom("AAPL", dt.date(2024, 1, 1))
        assert result == []

    def test_deserializes_peer_rows(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [_peer_row()])
        result = mod.get_peers_with_axiom("AAPL", dt.date(2024, 1, 1))
        assert len(result) == 1
        p = result[0]
        assert p["linked_symbol"] == "MSFT"
        assert p["link_type"] == "sector_peer"
        assert p["linkage_strength"] == pytest.approx(0.80)
        assert p["dau"] == pytest.approx(70.0)
        assert p["regime_label"] == "trend_confirmation"

    def test_handles_null_optional_fields(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        row = ("GOOG", "competitor", None, None, None, None, None, None, None)
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [row])
        result = mod.get_peers_with_axiom("AAPL", dt.date(2024, 1, 1))
        assert len(result) == 1
        assert result[0]["linkage_strength"] is None
        assert result[0]["dau"] is None

    def test_returns_multiple_peers(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        rows = [
            _peer_row(linked="MSFT", link_type="sector_peer", strength=0.85),
            _peer_row(linked="GOOG", link_type="sector_peer", strength=0.70),
            _peer_row(linked="SPY",  link_type="benchmark",   strength=0.50),
        ]
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.get_peers_with_axiom("AAPL", dt.date(2024, 1, 1))
        assert len(result) == 3


# ---------------------------------------------------------------------------
# 2.3 — get_stress_propagation
# ---------------------------------------------------------------------------

class TestGetStressPropagation:
    def test_returns_db_disabled_status(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
        result = mod.get_stress_propagation("AAPL", dt.date(2024, 1, 1))
        assert result["status"] == "db_disabled"
        assert result["propagation"] == []

    def test_returns_ok_with_propagation(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        # Fragility payload is stored as JSON object string with "score" key
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: ('{"score": 75.0}',))
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [_peer_row(strength=0.80, dau=70.0)])
        result = mod.get_stress_propagation("AAPL", dt.date(2024, 1, 1))
        assert result["status"] == "ok"
        assert result["symbol"] == "AAPL"
        assert result["source_fragility"] == pytest.approx(75.0)
        assert len(result["propagation"]) == 1

    def test_transmitted_stress_calculation(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: ('{"score": 80.0}',))
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [_peer_row(strength=0.60)])
        result = mod.get_stress_propagation("AAPL", dt.date(2024, 1, 1))
        transmitted = result["propagation"][0]["transmitted_stress"]
        assert transmitted == pytest.approx(48.0, abs=0.5)

    def test_stress_alert_triggered_when_high_transmitted_and_high_dau(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: ('{"score": 90.0}',))
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [_peer_row(strength=0.80, dau=70.0)])
        result = mod.get_stress_propagation("AAPL", dt.date(2024, 1, 1))
        # transmitted = 90 * 0.8 = 72 > 50, dau=70 > 60 → alert
        assert result["propagation"][0]["stress_alert"] is True

    def test_stress_alert_not_triggered_when_low_dau(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: ('{"score": 90.0}',))
        # DAU below threshold
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [_peer_row(strength=0.80, dau=40.0)])
        result = mod.get_stress_propagation("AAPL", dt.date(2024, 1, 1))
        assert result["propagation"][0]["stress_alert"] is False

    def test_no_source_fragility_transmitted_stress_is_none(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: None)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [_peer_row()])
        result = mod.get_stress_propagation("AAPL", dt.date(2024, 1, 1))
        assert result["source_fragility"] is None
        assert result["propagation"][0]["transmitted_stress"] is None
        assert result["propagation"][0]["stress_alert"] is False

    def test_propagation_sorted_by_transmitted_stress_desc(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: ('{"score": 100.0}',))
        rows = [
            _peer_row(linked="LOW_PEER",  strength=0.30, dau=40.0),
            _peer_row(linked="HIGH_PEER", strength=0.90, dau=80.0),
            _peer_row(linked="MID_PEER",  strength=0.60, dau=65.0),
        ]
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: rows)
        result = mod.get_stress_propagation("AAPL", dt.date(2024, 1, 1))
        stresses = [p["transmitted_stress"] for p in result["propagation"]]
        assert stresses == sorted(stresses, reverse=True)

    def test_as_of_date_included_in_response(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: None)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [])
        result = mod.get_stress_propagation("AAPL", dt.date(2024, 6, 15))
        assert result["as_of_date"] == "2024-06-15"


# ---------------------------------------------------------------------------
# API endpoint smoke tests
# ---------------------------------------------------------------------------

class TestRegimeAnalogsEndpoint:
    def test_ops_regime_analogs_endpoint_exists(self):
        from api.ops import router
        paths = {getattr(r, "path", None) for r in router.routes}
        assert "/ops/regime/analogs" in paths

    def test_regime_analogs_returns_structure(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [_analog_row()])
        from api.ops import regime_analogs
        result = regime_analogs(regime_label="liquidity_fracture")
        assert result["regime_label"] == "liquidity_fracture"
        assert "count" in result
        assert "analogs" in result


class TestLinkageEndpoints:
    def test_linkage_router_registered(self):
        from api.jobs.linkage_routes import router
        paths = {getattr(r, "path", None) for r in router.routes}
        assert "/linkage/peers/{symbol}" in paths
        assert "/linkage/stress-propagation/{symbol}" in paths

    def test_peers_endpoint_returns_structure(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [_peer_row()])
        from api.jobs.linkage_routes import get_peers
        result = get_peers("AAPL", as_of_date="2024-01-01")
        assert result["symbol"] == "AAPL"
        assert result["as_of_date"] == "2024-01-01"
        assert result["count"] == 1
        assert len(result["peers"]) == 1

    def test_peers_endpoint_uppercases_symbol(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
        from api.jobs.linkage_routes import get_peers
        result = get_peers("aapl", as_of_date=None)
        assert result["symbol"] == "AAPL"

    def test_stress_propagation_endpoint_returns_structure(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: ('{"score": 60.0}',))
        monkeypatch.setattr(mod.db, "safe_fetchall", lambda *a, **k: [_peer_row()])
        from api.jobs.linkage_routes import get_stress_propagation
        result = get_stress_propagation("AAPL", as_of_date="2024-01-01")
        assert result["status"] == "ok"
        assert result["symbol"] == "AAPL"

    def test_stress_propagation_endpoint_uppercases_symbol(self, monkeypatch):
        import api.jobs.regime_analogs as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
        from api.jobs.linkage_routes import get_stress_propagation
        result = get_stress_propagation("msft", as_of_date=None)
        assert result["symbol"] == "MSFT"
