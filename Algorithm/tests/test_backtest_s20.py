"""Session 20: AXIOM backtest harness tests."""
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
_D0 = dt.date(2024, 1, 2)
_D1 = dt.date(2024, 1, 31)


# ---------------------------------------------------------------------------
# _find_horizon_price (pure)
# ---------------------------------------------------------------------------

class TestFindHorizonPrice:
    def _fn(self):
        from api.jobs.axiom_backtest import _find_horizon_price
        return _find_horizon_price

    def test_exact_match(self):
        fn = self._fn()
        dpm = {dt.date(2024, 1, 23): 150.0}
        result = fn(dpm, dt.date(2024, 1, 2), 21)
        assert result == pytest.approx(150.0)

    def test_skips_weekend_finds_monday(self):
        fn = self._fn()
        # target = Jan 2 + 5 = Jan 7 (Sunday); next trading day is Jan 8
        dpm = {dt.date(2024, 1, 8): 100.0}
        result = fn(dpm, dt.date(2024, 1, 2), 5)
        assert result == pytest.approx(100.0)

    def test_returns_none_when_no_price_within_8_days(self):
        fn = self._fn()
        result = fn({}, dt.date(2024, 1, 2), 21)
        assert result is None

    def test_prefers_exact_over_nearby(self):
        fn = self._fn()
        target = dt.date(2024, 1, 2) + dt.timedelta(days=21)
        dpm = {
            target: 200.0,
            target + dt.timedelta(days=1): 201.0,
        }
        result = fn(dpm, dt.date(2024, 1, 2), 21)
        assert result == pytest.approx(200.0)

    def test_max_lookahead_7_days(self):
        fn = self._fn()
        target = dt.date(2024, 1, 2) + dt.timedelta(days=21)
        # 8 days after target — should NOT be found
        dpm = {target + dt.timedelta(days=8): 999.0}
        result = fn(dpm, dt.date(2024, 1, 2), 21)
        assert result is None


# ---------------------------------------------------------------------------
# compute_backtest_stats (pure, no DB)
# ---------------------------------------------------------------------------

class TestComputeBacktestStats:
    def _fn(self):
        from api.jobs.axiom_backtest import compute_backtest_stats
        return compute_backtest_stats

    def _make_signal(self, sym, date, label="BUY", dau=0.6, regime="fundamental_convergence"):
        return {
            "symbol": sym,
            "as_of_date": date,
            "regime_label": regime,
            "dau": dau,
            "signal_label": label,
        }

    def _make_prices(self, sym, dates_prices):
        return {sym: dict(zip(dates_prices[0::2], dates_prices[1::2]))}

    def test_empty_signals_returns_empty(self):
        fn = self._fn()
        result = fn([], {}, 21, ["BUY", "SELL"])
        assert result["total_signals"] == 0
        assert result["hit_rate"] is None

    def test_single_buy_win(self):
        fn = self._fn()
        sig = self._make_signal("AAPL", dt.date(2024, 1, 2))
        p0_date = dt.date(2024, 1, 2)
        p1_date = dt.date(2024, 1, 23)
        price_map = {"AAPL": {p0_date: 100.0, p1_date: 110.0}}
        result = fn([sig], price_map, 21, ["BUY", "SELL"])
        assert result["total_signals"] == 1
        assert result["hit_rate"] == pytest.approx(1.0)
        assert result["avg_return_pct"] == pytest.approx(10.0, rel=1e-3)

    def test_single_buy_loss(self):
        fn = self._fn()
        sig = self._make_signal("AAPL", dt.date(2024, 1, 2))
        price_map = {"AAPL": {
            dt.date(2024, 1, 2): 100.0,
            dt.date(2024, 1, 23): 90.0,
        }}
        result = fn([sig], price_map, 21, ["BUY", "SELL"])
        assert result["hit_rate"] == pytest.approx(0.0)
        assert result["avg_return_pct"] == pytest.approx(-10.0, rel=1e-3)

    def test_single_sell_win(self):
        fn = self._fn()
        # SELL is short; price goes DOWN = adjusted return > 0 = hit
        sig = self._make_signal("TSLA", dt.date(2024, 1, 2), label="SELL")
        price_map = {"TSLA": {
            dt.date(2024, 1, 2): 200.0,
            dt.date(2024, 1, 23): 180.0,  # down 10%
        }}
        result = fn([sig], price_map, 21, ["BUY", "SELL"])
        assert result["hit_rate"] == pytest.approx(1.0)
        assert result["avg_return_pct"] == pytest.approx(10.0, rel=1e-3)

    def test_hold_excluded_from_results(self):
        fn = self._fn()
        sig = self._make_signal("AAPL", dt.date(2024, 1, 2), label="HOLD")
        price_map = {"AAPL": {
            dt.date(2024, 1, 2): 100.0,
            dt.date(2024, 1, 23): 110.0,
        }}
        result = fn([sig], price_map, 21, ["BUY", "SELL"])
        assert result["total_signals"] == 0

    def test_missing_entry_price_excluded(self):
        fn = self._fn()
        sig = self._make_signal("AAPL", dt.date(2024, 1, 2))
        # Only exit price, no entry
        price_map = {"AAPL": {dt.date(2024, 1, 23): 110.0}}
        result = fn([sig], price_map, 21, ["BUY", "SELL"])
        assert result["total_signals"] == 0

    def test_missing_exit_price_excluded(self):
        fn = self._fn()
        sig = self._make_signal("AAPL", dt.date(2024, 1, 2))
        # Only entry price, no exit
        price_map = {"AAPL": {dt.date(2024, 1, 2): 100.0}}
        result = fn([sig], price_map, 21, ["BUY", "SELL"])
        assert result["total_signals"] == 0

    def test_by_regime_grouped(self):
        fn = self._fn()
        sigs = [
            self._make_signal("A", dt.date(2024, 1, 2), regime="bull"),
            self._make_signal("B", dt.date(2024, 1, 2), regime="bull"),
            self._make_signal("C", dt.date(2024, 1, 2), regime="bear"),
        ]
        price_map = {
            "A": {dt.date(2024, 1, 2): 100.0, dt.date(2024, 1, 23): 110.0},
            "B": {dt.date(2024, 1, 2): 100.0, dt.date(2024, 1, 23): 90.0},
            "C": {dt.date(2024, 1, 2): 50.0, dt.date(2024, 1, 23): 55.0},
        }
        result = fn(sigs, price_map, 21, ["BUY", "SELL"])
        assert "bull" in result["by_regime"]
        assert result["by_regime"]["bull"]["n"] == 2
        assert "bear" in result["by_regime"]
        assert result["by_regime"]["bear"]["n"] == 1

    def test_equity_curve_shape(self):
        fn = self._fn()
        sigs = [
            self._make_signal("A", dt.date(2024, 1, 2)),
            self._make_signal("B", dt.date(2024, 1, 10)),
        ]
        price_map = {
            "A": {dt.date(2024, 1, 2): 100.0, dt.date(2024, 1, 23): 110.0},
            "B": {dt.date(2024, 1, 10): 100.0, dt.date(2024, 1, 31): 105.0},
        }
        result = fn(sigs, price_map, 21, ["BUY", "SELL"])
        assert len(result["equity_curve"]) == 2
        # All equity values should be > 1.0 since both are wins
        for v in result["equity_curve"].values():
            assert v > 1.0

    def test_portfolio_returns_length_matches_equity_curve(self):
        fn = self._fn()
        sigs = [
            self._make_signal("A", dt.date(2024, 1, 2)),
            self._make_signal("B", dt.date(2024, 1, 10)),
        ]
        price_map = {
            "A": {dt.date(2024, 1, 2): 100.0, dt.date(2024, 1, 23): 110.0},
            "B": {dt.date(2024, 1, 10): 100.0, dt.date(2024, 1, 31): 105.0},
        }
        result = fn(sigs, price_map, 21, ["BUY", "SELL"])
        assert len(result["portfolio_returns"]) == len(result["equity_curve"])

    def test_by_signal_keys(self):
        fn = self._fn()
        sigs = [
            self._make_signal("A", dt.date(2024, 1, 2), label="BUY"),
            self._make_signal("B", dt.date(2024, 1, 2), label="SELL"),
        ]
        price_map = {
            "A": {dt.date(2024, 1, 2): 100.0, dt.date(2024, 1, 23): 110.0},
            "B": {dt.date(2024, 1, 2): 100.0, dt.date(2024, 1, 23): 90.0},
        }
        result = fn(sigs, price_map, 21, ["BUY", "SELL"])
        assert "BUY" in result["by_signal"]
        assert "SELL" in result["by_signal"]

    def test_sharpe_computed_for_2plus_dates(self):
        fn = self._fn()
        sigs = [
            self._make_signal("A", dt.date(2024, 1, 2)),
            self._make_signal("B", dt.date(2024, 1, 10)),
        ]
        price_map = {
            "A": {dt.date(2024, 1, 2): 100.0, dt.date(2024, 1, 23): 110.0},
            "B": {dt.date(2024, 1, 10): 100.0, dt.date(2024, 1, 31): 105.0},
        }
        result = fn(sigs, price_map, 21, ["BUY", "SELL"])
        assert result["sharpe"] is not None

    def test_sharpe_none_for_single_date(self):
        fn = self._fn()
        sig = self._make_signal("A", dt.date(2024, 1, 2))
        price_map = {"A": {dt.date(2024, 1, 2): 100.0, dt.date(2024, 1, 23): 110.0}}
        result = fn([sig], price_map, 21, ["BUY", "SELL"])
        assert result["sharpe"] is None

    def test_max_drawdown_zero_for_all_wins(self):
        fn = self._fn()
        sigs = [self._make_signal("A", dt.date(2024, 1, 2) + dt.timedelta(days=i)) for i in range(3)]
        price_map = {}
        for i, sig in enumerate(sigs):
            d = sig["as_of_date"]
            exit_d = d + dt.timedelta(days=21)
            price_map[f"A"] = price_map.get("A", {})
            price_map["A"][d] = 100.0
            price_map["A"][exit_d] = 110.0
        result = fn(sigs, price_map, 21, ["BUY", "SELL"])
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-6)

    def test_equity_curve_keys_are_iso_strings(self):
        fn = self._fn()
        sig = self._make_signal("A", dt.date(2024, 1, 2))
        price_map = {"A": {dt.date(2024, 1, 2): 100.0, dt.date(2024, 1, 23): 110.0}}
        result = fn([sig], price_map, 21, ["BUY", "SELL"])
        for k in result["equity_curve"]:
            # Should parse as ISO date
            dt.date.fromisoformat(k)


# ---------------------------------------------------------------------------
# load_signals_for_backtest (mocked DB)
# ---------------------------------------------------------------------------

class TestLoadSignalsForBacktest:
    def test_returns_empty_when_db_disabled(self, monkeypatch):
        monkeypatch.setenv("FTIP_DB_READ_ENABLED", "0")
        from api.jobs.axiom_backtest import load_signals_for_backtest
        result = load_signals_for_backtest(_D0, _D1)
        assert result == []

    def test_maps_rows_correctly(self, monkeypatch):
        _db_env(monkeypatch)
        rows = [
            ("AAPL", dt.date(2024, 1, 5), "bull", 0.75, "BUY"),
            ("TSLA", dt.date(2024, 1, 5), "bear", None, "SELL"),
        ]
        with patch("api.db.safe_fetchall", return_value=rows):
            from api.jobs.axiom_backtest import load_signals_for_backtest
            result = load_signals_for_backtest(_D0, _D1)
        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["dau"] == pytest.approx(0.75)
        assert result[0]["signal_label"] == "BUY"
        assert result[1]["dau"] is None

    def test_none_dau_preserved(self, monkeypatch):
        _db_env(monkeypatch)
        rows = [("MSFT", dt.date(2024, 1, 5), "unknown", None, "HOLD")]
        with patch("api.db.safe_fetchall", return_value=rows):
            from api.jobs.axiom_backtest import load_signals_for_backtest
            result = load_signals_for_backtest(_D0, _D1)
        assert result[0]["dau"] is None


# ---------------------------------------------------------------------------
# load_prices_for_backtest (mocked DB)
# ---------------------------------------------------------------------------

class TestLoadPricesForBacktest:
    def test_empty_symbols_returns_empty(self, monkeypatch):
        _db_env(monkeypatch)
        from api.jobs.axiom_backtest import load_prices_for_backtest
        result = load_prices_for_backtest([], _D0, _D1)
        assert result == {}

    def test_builds_nested_map(self, monkeypatch):
        _db_env(monkeypatch)
        rows = [
            ("AAPL", dt.date(2024, 1, 2), 150.0),
            ("AAPL", dt.date(2024, 1, 3), 152.0),
            ("MSFT", dt.date(2024, 1, 2), 300.0),
        ]
        with patch("api.db.safe_fetchall", return_value=rows):
            from api.jobs.axiom_backtest import load_prices_for_backtest
            result = load_prices_for_backtest(["AAPL", "MSFT"], _D0, _D1)
        assert result["AAPL"][dt.date(2024, 1, 2)] == pytest.approx(150.0)
        assert result["AAPL"][dt.date(2024, 1, 3)] == pytest.approx(152.0)
        assert result["MSFT"][dt.date(2024, 1, 2)] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# store_backtest_run (mocked DB)
# ---------------------------------------------------------------------------

class TestStoreBacktestRun:
    def test_returns_false_when_write_disabled(self, monkeypatch):
        monkeypatch.setenv("FTIP_DB_WRITE_ENABLED", "0")
        from api.jobs.axiom_backtest import store_backtest_run
        result = store_backtest_run("rid", _D0, _D1, 21, 0.0, ["BUY"], {"total_signals": 5})
        assert result is False

    def test_calls_exec1_and_returns_true(self, monkeypatch):
        _db_env(monkeypatch)
        with patch("api.db.exec1") as mock_exec:
            from api.jobs.axiom_backtest import store_backtest_run
            result = store_backtest_run(
                "test-run-id", _D0, _D1, 21, 0.0, ["BUY", "SELL"],
                {"total_signals": 10, "hit_rate": 0.6, "avg_return_pct": 2.5,
                 "sharpe": 1.2, "max_drawdown": 0.05, "spearman_ic": 0.3},
            )
        assert result is True
        mock_exec.assert_called_once()

    def test_exec1_exception_returns_false(self, monkeypatch):
        _db_env(monkeypatch)
        with patch("api.db.exec1", side_effect=Exception("db error")):
            from api.jobs.axiom_backtest import store_backtest_run
            result = store_backtest_run("rid", _D0, _D1, 21, 0.0, ["BUY"], {})
        assert result is False


# ---------------------------------------------------------------------------
# load_recent_runs (mocked DB)
# ---------------------------------------------------------------------------

class TestLoadRecentRuns:
    def test_returns_empty_when_db_disabled(self, monkeypatch):
        monkeypatch.setenv("FTIP_DB_READ_ENABLED", "0")
        from api.jobs.axiom_backtest import load_recent_runs
        result = load_recent_runs()
        assert result == []

    def test_maps_rows_to_dicts(self, monkeypatch):
        _db_env(monkeypatch)
        rows = [(
            "run-001", dt.date(2024, 1, 2), dt.date(2024, 1, 31), 21, 0.0,
            ["BUY", "SELL"], 100, 0.62, 1.5, 0.9, 0.12, 0.25,
            dt.datetime(2024, 2, 1, 12, 0, 0),
        )]
        with patch("api.db.safe_fetchall", return_value=rows):
            from api.jobs.axiom_backtest import load_recent_runs
            result = load_recent_runs(5)
        assert len(result) == 1
        r = result[0]
        assert r["run_id"] == "run-001"
        assert r["total_signals"] == 100
        assert r["hit_rate"] == pytest.approx(0.62)
        assert r["sharpe"] == pytest.approx(0.9)
        assert isinstance(r["from_date"], str)

    def test_exception_returns_empty(self, monkeypatch):
        _db_env(monkeypatch)
        with patch("api.db.safe_fetchall", side_effect=Exception("oops")):
            from api.jobs.axiom_backtest import load_recent_runs
            result = load_recent_runs()
        assert result == []


# ---------------------------------------------------------------------------
# run_axiom_backtest full pipeline (mocked DB)
# ---------------------------------------------------------------------------

class TestRunAxiomBacktest:
    def _run(self, signals=None, prices=None):
        from api.jobs.axiom_backtest import run_axiom_backtest
        with patch("api.jobs.axiom_backtest.load_signals_for_backtest", return_value=signals or []), \
             patch("api.jobs.axiom_backtest.load_prices_for_backtest", return_value=prices or {}), \
             patch("api.jobs.axiom_backtest.store_backtest_run", return_value=True):
            return run_axiom_backtest(_D0, _D1, horizon_days=21, min_dau=0.0,
                                      signal_filter=["BUY", "SELL"], store=True)

    def test_status_ok(self):
        result = self._run()
        assert result["status"] == "ok"

    def test_empty_signals_returns_zero(self):
        result = self._run()
        assert result["total_signals"] == 0
        assert result["stored"] is False

    def test_with_signals_stored_true(self):
        signals = [{
            "symbol": "AAPL", "as_of_date": dt.date(2024, 1, 2),
            "regime_label": "bull", "dau": 0.8, "signal_label": "BUY",
        }]
        prices = {"AAPL": {
            dt.date(2024, 1, 2): 100.0,
            dt.date(2024, 1, 23): 110.0,
        }}
        result = self._run(signals=signals, prices=prices)
        assert result["total_signals"] == 1
        assert result["stored"] is True

    def test_response_includes_all_fields(self):
        result = self._run()
        for key in ["run_id", "from_date", "to_date", "horizon_days", "min_dau",
                    "signal_filter", "stored", "total_signals", "hit_rate",
                    "sharpe", "max_drawdown"]:
            assert key in result

    def test_from_to_dates_in_response(self):
        result = self._run()
        assert result["from_date"] == _D0.isoformat()
        assert result["to_date"] == _D1.isoformat()


# ---------------------------------------------------------------------------
# Route contract tests
# ---------------------------------------------------------------------------

class TestBacktestRoutes:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        _db_env(monkeypatch)

    @pytest.fixture
    def client(self):
        from api.main import app
        return TestClient(app, raise_server_exceptions=True)

    def test_run_requires_auth(self, client):
        resp = client.post("/jobs/axiom-backtest/run", json={
            "from_date": "2024-01-01", "to_date": "2024-01-31"
        })
        assert resp.status_code in (401, 403)

    def test_runs_requires_auth(self, client):
        resp = client.get("/jobs/axiom-backtest/runs")
        assert resp.status_code in (401, 403)

    def test_run_invalid_date_format(self, client):
        resp = client.post(
            "/jobs/axiom-backtest/run",
            json={"from_date": "not-a-date", "to_date": "2024-01-31"},
            headers=AUTH,
        )
        assert resp.status_code == 422

    def test_run_from_after_to_rejected(self, client):
        resp = client.post(
            "/jobs/axiom-backtest/run",
            json={"from_date": "2024-02-01", "to_date": "2024-01-01"},
            headers=AUTH,
        )
        assert resp.status_code == 422

    def test_run_returns_ok_status(self, client):
        with patch("api.jobs.axiom_backtest.load_signals_for_backtest", return_value=[]), \
             patch("api.jobs.axiom_backtest.load_prices_for_backtest", return_value={}):
            resp = client.post(
                "/jobs/axiom-backtest/run",
                json={"from_date": "2024-01-01", "to_date": "2024-01-31"},
                headers=AUTH,
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_runs_returns_ok_status(self, client):
        with patch("api.jobs.axiom_backtest.load_recent_runs", return_value=[]):
            resp = client.get("/jobs/axiom-backtest/runs", headers=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "runs" in data

    def test_run_openapi_registered(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        assert "/jobs/axiom-backtest/run" in paths
        assert "/jobs/axiom-backtest/runs" in paths

    def test_horizon_days_out_of_range_rejected(self, client):
        resp = client.post(
            "/jobs/axiom-backtest/run",
            json={"from_date": "2024-01-01", "to_date": "2024-01-31", "horizon_days": 0},
            headers=AUTH,
        )
        assert resp.status_code == 422
