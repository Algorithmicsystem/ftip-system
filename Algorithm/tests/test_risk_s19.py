"""Session 19: Portfolio risk overlay tests."""
from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import security


def _db_env(monkeypatch):
    monkeypatch.setenv("FTIP_API_KEY", "secret")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_WRITE_ENABLED", "1")
    security.reset_auth_cache()


AUTH = {"X-FTIP-API-Key": "secret"}
_DATE = dt.date(2025, 1, 2)

_PORTFOLIO = [
    {"symbol": "NVDA", "weight": 0.20},
    {"symbol": "AAPL", "weight": 0.15},
    {"symbol": "MSFT", "weight": 0.10},
]


# ---------------------------------------------------------------------------
# compute_concentration (pure, no DB)
# ---------------------------------------------------------------------------

class TestComputeConcentration:
    def test_equal_weights_low_hhi(self):
        from api.jobs.risk import compute_concentration
        positions = [{"symbol": f"S{i}", "weight": 0.10} for i in range(10)]
        result = compute_concentration(positions)
        assert result["hhi"] == pytest.approx(0.10, rel=1e-3)
        assert result["concentration_state"] == "LOW"

    def test_single_position_max_hhi(self):
        from api.jobs.risk import compute_concentration
        result = compute_concentration([{"symbol": "A", "weight": 1.0}])
        assert result["hhi"] == pytest.approx(1.0, rel=1e-3)
        assert result["concentration_state"] == "HIGH"

    def test_empty_positions_returns_unknown(self):
        from api.jobs.risk import compute_concentration
        result = compute_concentration([])
        assert result["concentration_state"] == "UNKNOWN"
        assert result["hhi"] == 0.0

    def test_top3_weight_correct(self):
        from api.jobs.risk import compute_concentration
        positions = [
            {"symbol": "A", "weight": 0.30},
            {"symbol": "B", "weight": 0.25},
            {"symbol": "C", "weight": 0.20},
            {"symbol": "D", "weight": 0.10},
            {"symbol": "E", "weight": 0.05},
        ]
        result = compute_concentration(positions)
        assert result["top3_weight"] == pytest.approx(0.75, rel=1e-3)

    def test_effective_n_inverse_of_hhi(self):
        from api.jobs.risk import compute_concentration
        positions = [{"symbol": f"S{i}", "weight": 0.25} for i in range(4)]
        result = compute_concentration(positions)
        assert abs(result["effective_n"] - result["hhi"] ** -1) < 0.1

    def test_moderate_state_threshold(self):
        from api.jobs.risk import compute_concentration
        # HHI = 0.25 (4 positions with different weights) → MODERATE
        positions = [
            {"symbol": "A", "weight": 0.40},
            {"symbol": "B", "weight": 0.30},
            {"symbol": "C", "weight": 0.20},
            {"symbol": "D", "weight": 0.10},
        ]
        result = compute_concentration(positions)
        hhi = result["hhi"]
        assert 0.20 < hhi <= 0.33
        assert result["concentration_state"] == "MODERATE"

    def test_zero_weights_excluded(self):
        from api.jobs.risk import compute_concentration
        positions = [
            {"symbol": "A", "weight": 0.50},
            {"symbol": "B", "weight": 0.00},
        ]
        result = compute_concentration(positions)
        assert result["hhi"] == pytest.approx(0.25, rel=1e-3)


# ---------------------------------------------------------------------------
# compute_sector_exposure (pure, no DB)
# ---------------------------------------------------------------------------

class TestComputeSectorExposure:
    def test_aggregates_by_sector(self):
        from api.jobs.risk import compute_sector_exposure
        positions = [
            {"symbol": "NVDA", "weight": 0.20},
            {"symbol": "AMD",  "weight": 0.10},
            {"symbol": "JNJ",  "weight": 0.15},
        ]
        sector_map = {"NVDA": "technology", "AMD": "technology", "JNJ": "healthcare"}
        result = compute_sector_exposure(positions, sector_map)
        assert result["technology"] == pytest.approx(0.30, rel=1e-3)
        assert result["healthcare"] == pytest.approx(0.15, rel=1e-3)

    def test_unknown_sector_for_missing_symbols(self):
        from api.jobs.risk import compute_sector_exposure
        positions = [{"symbol": "XYZ", "weight": 0.10}]
        result = compute_sector_exposure(positions, {})
        assert "unknown" in result
        assert result["unknown"] == pytest.approx(0.10, rel=1e-3)

    def test_sorted_descending(self):
        from api.jobs.risk import compute_sector_exposure
        positions = [
            {"symbol": "A", "weight": 0.10},
            {"symbol": "B", "weight": 0.30},
        ]
        sector_map = {"A": "healthcare", "B": "technology"}
        result = compute_sector_exposure(positions, sector_map)
        sectors = list(result.keys())
        assert sectors[0] == "technology"


# ---------------------------------------------------------------------------
# _risk_score_and_flags (pure, no DB)
# ---------------------------------------------------------------------------

class TestRiskScoreAndFlags:
    def _base_inputs(self):
        from api.jobs.risk import compute_concentration, compute_sector_exposure
        positions = [
            {"symbol": "NVDA", "weight": 0.20},
            {"symbol": "AAPL", "weight": 0.15},
        ]
        conc = compute_concentration(positions)
        sector_exp = compute_sector_exposure(positions, {"NVDA": "technology", "AAPL": "technology"})
        return conc, sector_exp, positions

    def test_clean_portfolio_low_risk(self):
        from api.jobs.risk import _risk_score_and_flags
        conc, sect, positions = self._base_inputs()
        axiom = {
            "NVDA": {"regime_label": "fundamental_convergence", "fragility_score": 25.0},
            "AAPL": {"regime_label": "fundamental_convergence", "fragility_score": 20.0},
        }
        score, state, flags = _risk_score_and_flags(conc, sect, axiom, positions, [], "MODERATE")
        assert state in ("LOW", "MODERATE")
        assert not any(f["flag"] == "veto_regime" for f in flags)

    def test_veto_regime_adds_flag(self):
        from api.jobs.risk import _risk_score_and_flags
        conc, sect, positions = self._base_inputs()
        axiom = {
            "NVDA": {"regime_label": "euphoria_critical", "fragility_score": 30.0},
            "AAPL": {"regime_label": "fundamental_convergence", "fragility_score": 20.0},
        }
        score, state, flags = _risk_score_and_flags(conc, sect, axiom, positions, [], "MODERATE")
        assert any(f["flag"] == "veto_regime" for f in flags)
        assert score > 0

    def test_degraded_ic_adds_flag(self):
        from api.jobs.risk import _risk_score_and_flags
        conc, sect, positions = self._base_inputs()
        axiom = {
            "NVDA": {"regime_label": "fundamental_convergence", "fragility_score": 20.0},
            "AAPL": {"regime_label": "fundamental_convergence", "fragility_score": 20.0},
        }
        score, state, flags = _risk_score_and_flags(conc, sect, axiom, positions, [], "DEGRADED")
        assert any(f["flag"] == "ic_degraded" for f in flags)
        assert score >= 15

    def test_high_fragility_adds_flag(self):
        from api.jobs.risk import _risk_score_and_flags
        conc, sect, positions = self._base_inputs()
        axiom = {
            "NVDA": {"regime_label": "fundamental_convergence", "fragility_score": 85.0},
            "AAPL": {"regime_label": "fundamental_convergence", "fragility_score": 20.0},
        }
        score, state, flags = _risk_score_and_flags(conc, sect, axiom, positions, [], "MODERATE")
        assert any(f["flag"] == "high_fragility" and f["symbol"] == "NVDA" for f in flags)

    def test_regime_flips_increase_score(self):
        from api.jobs.risk import _risk_score_and_flags, compute_concentration, compute_sector_exposure
        positions = [{"symbol": "NVDA", "weight": 0.15}]
        conc = compute_concentration(positions)
        sect = compute_sector_exposure(positions, {})
        axiom = {"NVDA": {"regime_label": "fundamental_convergence", "fragility_score": 20.0}}
        score_no_flip, _, _ = _risk_score_and_flags(conc, sect, axiom, positions, [], "MODERATE")
        flips = [{"symbol": "NVDA", "prev_regime": "fundamental_convergence", "curr_regime": "euphoria_critical"}]
        score_with_flip, _, _ = _risk_score_and_flags(conc, sect, axiom, positions, flips, "MODERATE")
        assert score_with_flip > score_no_flip

    def test_critical_state_above_threshold(self):
        from api.jobs.risk import _risk_score_and_flags
        # Max out: high HHI + veto regimes + DEGRADED IC + sector concentration + many flips
        from api.jobs.risk import compute_concentration, compute_sector_exposure
        positions = [{"symbol": "NVDA", "weight": 1.0}]  # HHI=1 → +25
        conc = compute_concentration(positions)  # state=HIGH
        sect = compute_sector_exposure(positions, {"NVDA": "technology"})
        # Sector = 100% technology → +15
        # veto regime (weight=1.0) → +30
        # DEGRADED IC → +15
        # 3+ flips → +15
        axiom = {"NVDA": {"regime_label": "euphoria_critical", "fragility_score": 20.0}}
        flips = [
            {"symbol": "A", "prev_regime": "x", "curr_regime": "y"},
            {"symbol": "B", "prev_regime": "x", "curr_regime": "y"},
            {"symbol": "C", "prev_regime": "x", "curr_regime": "y"},
        ]
        score, state, _ = _risk_score_and_flags(conc, sect, axiom, positions, flips, "DEGRADED")
        assert state == "CRITICAL"
        assert score >= 65

    def test_score_capped_at_100(self):
        from api.jobs.risk import _risk_score_and_flags, compute_concentration, compute_sector_exposure
        positions = [{"symbol": "A", "weight": 1.0}]
        conc = compute_concentration(positions)
        sect = compute_sector_exposure(positions, {"A": "technology"})
        axiom = {"A": {"regime_label": "euphoria_critical", "fragility_score": 90.0}}
        flips = [{"symbol": f"S{i}", "prev_regime": "x", "curr_regime": "y"} for i in range(5)]
        score, _, _ = _risk_score_and_flags(conc, sect, axiom, positions, flips, "DEGRADED")
        assert score <= 100.0


# ---------------------------------------------------------------------------
# detect_regime_flips
# ---------------------------------------------------------------------------

class TestDetectRegimeFlips:
    def test_no_change_returns_empty(self):
        from api.jobs.risk import detect_regime_flips
        with patch("api.jobs.risk.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.side_effect = [
                [("NVDA", "fundamental_convergence")],   # current
                [("NVDA", "fundamental_convergence", _DATE - dt.timedelta(days=5))],  # prev
            ]
            flips = detect_regime_flips(["NVDA"], _DATE)
        assert flips == []

    def test_flip_detected(self):
        from api.jobs.risk import detect_regime_flips
        with patch("api.jobs.risk.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.side_effect = [
                [("NVDA", "euphoria_critical")],          # current
                [("NVDA", "fundamental_convergence", _DATE - dt.timedelta(days=7))],  # prev
            ]
            flips = detect_regime_flips(["NVDA"], _DATE)
        assert len(flips) == 1
        assert flips[0]["symbol"] == "NVDA"
        assert flips[0]["prev_regime"] == "fundamental_convergence"
        assert flips[0]["curr_regime"] == "euphoria_critical"

    def test_db_disabled_returns_empty(self):
        from api.jobs.risk import detect_regime_flips
        with patch("api.jobs.risk.db") as mock_db:
            mock_db.db_read_enabled.return_value = False
            flips = detect_regime_flips(["NVDA"], _DATE)
        assert flips == []

    def test_empty_symbols_returns_empty(self):
        from api.jobs.risk import detect_regime_flips
        with patch("api.jobs.risk.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            flips = detect_regime_flips([], _DATE)
        assert flips == []


# ---------------------------------------------------------------------------
# compute_portfolio_risk (mocked DB)
# ---------------------------------------------------------------------------

class TestComputePortfolioRisk:
    def _run(self, portfolio=None, **kwargs):
        from api.jobs.risk import compute_portfolio_risk
        portfolio = portfolio or _PORTFOLIO
        with patch("api.jobs.risk.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            # sector_map
            mock_db.safe_fetchall.side_effect = [
                # _load_sector_map
                [("NVDA", "technology"), ("AAPL", "technology"), ("MSFT", "technology")],
                # _load_axiom_data
                [
                    ("NVDA", "fundamental_convergence", 82.0, "live_candidate", 0.78, 28.0),
                    ("AAPL", "fundamental_convergence", 70.0, "live_candidate", 0.72, 22.0),
                    ("MSFT", "compensation_capture",    65.0, "live_candidate", 0.68, 18.0),
                ],
                # detect_regime_flips — current
                [
                    ("NVDA", "fundamental_convergence"),
                    ("AAPL", "fundamental_convergence"),
                    ("MSFT", "compensation_capture"),
                ],
                # detect_regime_flips — prev
                [
                    ("NVDA", "fundamental_convergence", _DATE - dt.timedelta(days=10)),
                    ("AAPL", "fundamental_convergence", _DATE - dt.timedelta(days=10)),
                    ("MSFT", "compensation_capture",    _DATE - dt.timedelta(days=10)),
                ],
            ]
            mock_db.safe_fetchone.side_effect = [
                ("MODERATE",),  # IC state
                ("EXPANDING",), # breadth state
            ]
            return compute_portfolio_risk(_DATE, portfolio, **kwargs)

    def test_returns_required_keys(self):
        result = self._run()
        for k in ("status", "as_of_date", "portfolio_size", "gross_weight",
                  "concentration", "sector_exposure", "risk_flags",
                  "risk_score", "risk_state", "per_symbol",
                  "regime_flips", "ic_state", "breadth_state"):
            assert k in result, f"missing key: {k}"

    def test_status_is_ok(self):
        result = self._run()
        assert result["status"] == "ok"

    def test_portfolio_size_correct(self):
        result = self._run()
        assert result["portfolio_size"] == 3

    def test_per_symbol_count_matches_portfolio(self):
        result = self._run()
        assert len(result["per_symbol"]) == 3

    def test_symbols_uppercased(self):
        from api.jobs.risk import compute_portfolio_risk
        lower_portfolio = [{"symbol": "nvda", "weight": 0.15}]
        with patch("api.jobs.risk.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.side_effect = [[], [], [], []]
            mock_db.safe_fetchone.side_effect = [("MODERATE",), ("EXPANDING",)]
            result = compute_portfolio_risk(_DATE, lower_portfolio)
        assert result["per_symbol"][0]["symbol"] == "NVDA"

    def test_gross_weight_is_sum_of_weights(self):
        result = self._run()
        expected = round(0.20 + 0.15 + 0.10, 4)
        assert result["gross_weight"] == pytest.approx(expected, rel=1e-3)

    def test_no_flags_for_clean_portfolio(self):
        result = self._run()
        veto_flags = [f for f in result["risk_flags"] if f["flag"] == "veto_regime"]
        assert len(veto_flags) == 0

    def test_veto_regime_detected(self):
        from api.jobs.risk import compute_portfolio_risk
        portfolio = [{"symbol": "NVDA", "weight": 0.25}]
        with patch("api.jobs.risk.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.side_effect = [
                [],  # sector_map
                [("NVDA", "euphoria_critical", 75.0, "monitor_only", 0.45, 30.0)],  # axiom
                [("NVDA", "euphoria_critical")],  # regime flips current
                [],  # regime flips prev
            ]
            mock_db.safe_fetchone.side_effect = [("MODERATE",), ("EXPANDING",)]
            result = compute_portfolio_risk(_DATE, portfolio)
        assert any(f["flag"] == "veto_regime" for f in result["risk_flags"])
        assert result["risk_state"] in ("MODERATE", "HIGH", "CRITICAL")

    def test_regime_aligned_field(self):
        result = self._run()
        nvda = next(p for p in result["per_symbol"] if p["symbol"] == "NVDA")
        assert nvda["regime_aligned"] is True
        assert nvda["regime_veto"] is False

    def test_as_of_date_propagated(self):
        result = self._run()
        assert result["as_of_date"] == _DATE.isoformat()

    def test_no_regime_flips_for_stable_portfolio(self):
        result = self._run()
        assert result["regime_flips"] == []


# ---------------------------------------------------------------------------
# Route contract tests
# ---------------------------------------------------------------------------

class TestRiskRoutes:
    def test_overlay_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.post("/jobs/risk/overlay", json={"portfolio": [{"symbol": "NVDA", "weight": 0.2}]})
        assert resp.status_code == 401

    def test_overlay_route_in_openapi(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        assert "/jobs/risk/overlay" in resp.json().get("paths", {})

    def test_overlay_returns_ok(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app

        def fake_risk(as_of_date, portfolio, **kwargs):
            return {
                "status": "ok",
                "as_of_date": as_of_date.isoformat(),
                "portfolio_size": len(portfolio),
                "gross_weight": sum(p["weight"] for p in portfolio),
                "concentration": {"hhi": 0.08, "effective_n": 12.5,
                                  "top3_weight": 0.45, "max_single_weight": 0.20,
                                  "concentration_state": "LOW"},
                "sector_exposure": {"technology": 0.45},
                "sector_concentration_state": "MODERATE",
                "regime_breakdown": {"fundamental_convergence": 3},
                "regime_flips": [],
                "risk_flags": [],
                "ic_state": "MODERATE",
                "breadth_state": "EXPANDING",
                "risk_score": 8.0,
                "risk_state": "LOW",
                "per_symbol": [],
            }

        monkeypatch.setattr("api.jobs.risk.compute_portfolio_risk", fake_risk)
        client = TestClient(app)
        resp = client.post(
            "/jobs/risk/overlay",
            json={
                "portfolio": [
                    {"symbol": "NVDA", "weight": 0.20},
                    {"symbol": "AAPL", "weight": 0.15},
                ],
                "as_of_date": "2025-01-02",
            },
            headers=AUTH,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["risk_state"] == "LOW"
        assert "concentration" in data

    def test_empty_portfolio_returns_zero_size(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app

        def fake_risk(as_of_date, portfolio, **kwargs):
            return {"status": "ok", "as_of_date": as_of_date.isoformat(),
                    "portfolio_size": 0, "gross_weight": 0.0,
                    "concentration": {}, "sector_exposure": {}, "sector_concentration_state": "LOW",
                    "regime_breakdown": {}, "regime_flips": [], "risk_flags": [],
                    "ic_state": "INSUFFICIENT", "breadth_state": "NEUTRAL",
                    "risk_score": 0.0, "risk_state": "LOW", "per_symbol": []}

        monkeypatch.setattr("api.jobs.risk.compute_portfolio_risk", fake_risk)
        client = TestClient(app)
        resp = client.post(
            "/jobs/risk/overlay",
            json={"portfolio": []},
            headers=AUTH,
        )
        assert resp.status_code == 200
        assert resp.json()["portfolio_size"] == 0

    def test_invalid_weight_rejected(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.post(
            "/jobs/risk/overlay",
            json={"portfolio": [{"symbol": "NVDA", "weight": 1.5}]},  # > 1.0
            headers=AUTH,
        )
        assert resp.status_code == 422
