"""Session 18: Signal P&L tracker tests."""
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
_DATE = dt.date(2025, 2, 1)


# ---------------------------------------------------------------------------
# spearman_ic unit tests
# ---------------------------------------------------------------------------

class TestSpearmanIC:
    def test_perfect_positive_correlation(self):
        from api.jobs.pnl import spearman_ic
        scores  = [1.0, 2.0, 3.0, 4.0, 5.0]
        returns = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(spearman_ic(scores, returns) - 1.0) < 1e-9

    def test_perfect_negative_correlation(self):
        from api.jobs.pnl import spearman_ic
        scores  = [1.0, 2.0, 3.0, 4.0, 5.0]
        returns = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert abs(spearman_ic(scores, returns) + 1.0) < 1e-9

    def test_too_few_pairs_returns_zero(self):
        from api.jobs.pnl import spearman_ic
        assert spearman_ic([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_uncorrelated_returns_near_zero(self):
        from api.jobs.pnl import spearman_ic
        import math
        scores  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        returns = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
        ic = spearman_ic(scores, returns)
        assert -1.0 <= ic <= 1.0


# ---------------------------------------------------------------------------
# compute_signal_pnl unit tests
# ---------------------------------------------------------------------------

class TestComputeSignalPnl:
    def _signals_row(self, sym="NVDA", signal="BUY", score=0.7,
                     lookback=252, dau=75.0, regime="fundamental_convergence"):
        return (sym, signal, score, lookback, dau, regime, 0.8)

    def _bars(self, sym, signal_date, as_of_date, p0=100.0, p1=105.0):
        return [
            (sym, signal_date, p0),
            (sym, as_of_date, p1),
        ]

    def test_db_disabled_returns_empty(self):
        from api.jobs.pnl import compute_signal_pnl
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = False
            result = compute_signal_pnl(_DATE)
        assert result == []

    def test_no_signals_returns_empty(self):
        from api.jobs.pnl import compute_signal_pnl
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.return_value = []
            result = compute_signal_pnl(_DATE)
        assert result == []

    def test_buy_signal_positive_return_is_hit(self):
        from api.jobs.pnl import compute_signal_pnl
        signal_date = _DATE - dt.timedelta(days=5)
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.side_effect = [
                [self._signals_row(signal="BUY")],       # signals query
                self._bars("NVDA", signal_date, _DATE),  # prices query
            ]
            rows = compute_signal_pnl(_DATE, horizons=[5])
        assert len(rows) == 1
        assert rows[0]["hit"] is True
        assert rows[0]["return_pct"] == pytest.approx(5.0, rel=1e-3)

    def test_buy_signal_negative_return_is_miss(self):
        from api.jobs.pnl import compute_signal_pnl
        signal_date = _DATE - dt.timedelta(days=5)
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.side_effect = [
                [self._signals_row(signal="BUY")],
                self._bars("NVDA", signal_date, _DATE, p0=100.0, p1=95.0),
            ]
            rows = compute_signal_pnl(_DATE, horizons=[5])
        assert rows[0]["hit"] is False
        assert rows[0]["return_pct"] < 0

    def test_sell_signal_negative_return_is_hit(self):
        from api.jobs.pnl import compute_signal_pnl
        signal_date = _DATE - dt.timedelta(days=5)
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.side_effect = [
                [self._signals_row(signal="SELL")],
                self._bars("NVDA", signal_date, _DATE, p0=100.0, p1=92.0),
            ]
            rows = compute_signal_pnl(_DATE, horizons=[5])
        assert rows[0]["hit"] is True

    def test_hold_signal_has_no_hit(self):
        from api.jobs.pnl import compute_signal_pnl
        signal_date = _DATE - dt.timedelta(days=5)
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.side_effect = [
                [self._signals_row(signal="HOLD")],
                self._bars("NVDA", signal_date, _DATE),
            ]
            rows = compute_signal_pnl(_DATE, horizons=[5])
        assert rows[0]["hit"] is None

    def test_missing_price_yields_none_return(self):
        from api.jobs.pnl import compute_signal_pnl
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.side_effect = [
                [self._signals_row()],
                [],   # no bars
            ]
            rows = compute_signal_pnl(_DATE, horizons=[5])
        assert rows[0]["return_pct"] is None
        assert rows[0]["hit"] is None

    def test_multiple_horizons_produce_separate_rows(self):
        from api.jobs.pnl import compute_signal_pnl
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            # One signal query + one bars query per horizon
            mock_db.safe_fetchall.side_effect = [
                # horizon 5
                [self._signals_row()],
                self._bars("NVDA", _DATE - dt.timedelta(days=5), _DATE),
                # horizon 21
                [self._signals_row()],
                self._bars("NVDA", _DATE - dt.timedelta(days=21), _DATE),
            ]
            rows = compute_signal_pnl(_DATE, horizons=[5, 21])
        horizons_seen = {r["horizon_days"] for r in rows}
        assert horizons_seen == {5, 21}

    def test_row_contains_required_keys(self):
        from api.jobs.pnl import compute_signal_pnl
        signal_date = _DATE - dt.timedelta(days=5)
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.side_effect = [
                [self._signals_row()],
                self._bars("NVDA", signal_date, _DATE),
            ]
            rows = compute_signal_pnl(_DATE, horizons=[5])
        r = rows[0]
        for k in ("symbol", "signal_date", "horizon_days", "signal_label",
                  "return_pct", "hit", "computed_at", "price_at_signal", "price_at_horizon"):
            assert k in r, f"missing key: {k}"

    def test_return_pct_calculation(self):
        from api.jobs.pnl import compute_signal_pnl
        signal_date = _DATE - dt.timedelta(days=5)
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.side_effect = [
                [self._signals_row(signal="BUY")],
                self._bars("NVDA", signal_date, _DATE, p0=200.0, p1=210.0),
            ]
            rows = compute_signal_pnl(_DATE, horizons=[5])
        # (210/200 - 1) * 100 = 5.0
        assert rows[0]["return_pct"] == pytest.approx(5.0, rel=1e-3)


# ---------------------------------------------------------------------------
# store_signal_pnl unit tests
# ---------------------------------------------------------------------------

class TestStoreSignalPnl:
    def _make_row(self, symbol="NVDA", signal_date=None, horizon=5):
        signal_date = signal_date or _DATE - dt.timedelta(days=horizon)
        return {
            "symbol": symbol,
            "signal_date": signal_date,
            "horizon_days": horizon,
            "lookback": 252,
            "signal_label": "BUY",
            "signal_score": 0.7,
            "dau": 75.0,
            "regime_label": "fundamental_convergence",
            "price_at_signal": 100.0,
            "horizon_date": _DATE,
            "price_at_horizon": 105.0,
            "return_pct": 5.0,
            "hit": True,
            "computed_at": _DATE,
        }

    def test_write_disabled_returns_zero(self):
        from api.jobs.pnl import store_signal_pnl
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_write_enabled.return_value = False
            result = store_signal_pnl([self._make_row()])
        assert result == 0

    def test_empty_rows_returns_zero(self):
        from api.jobs.pnl import store_signal_pnl
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_write_enabled.return_value = True
            result = store_signal_pnl([])
        assert result == 0

    def test_success_returns_count(self):
        from api.jobs.pnl import store_signal_pnl
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_write_enabled.return_value = True
            mock_db.exec1.return_value = None
            result = store_signal_pnl([self._make_row(), self._make_row("AAPL", horizon=21)])
        assert result == 2

    def test_exception_per_row_does_not_abort(self):
        from api.jobs.pnl import store_signal_pnl
        call_count = [0]
        def maybe_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("db error")
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_write_enabled.return_value = True
            mock_db.exec1.side_effect = maybe_fail
            result = store_signal_pnl([self._make_row("NVDA"), self._make_row("AAPL")])
        assert result == 1  # second row succeeds


# ---------------------------------------------------------------------------
# load_pnl_summary unit tests
# ---------------------------------------------------------------------------

def _make_summary_rows(*specs):
    """Build fake DB rows: (sym, signal_date, horizon, label, score, dau, regime, ret_pct, hit, computed)."""
    rows = []
    for spec in specs:
        sym, label, horizon, ret_pct, hit = spec
        rows.append((
            sym, _DATE - dt.timedelta(days=horizon + 1), horizon,
            label, 0.7, 75.0, "fundamental_convergence",
            ret_pct, hit, _DATE,
        ))
    return rows


class TestLoadPnlSummary:
    def _summary(self, rows, **kwargs):
        from api.jobs.pnl import load_pnl_summary
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchall.return_value = rows
            return load_pnl_summary(_DATE, **kwargs)

    def test_db_disabled_returns_early(self):
        from api.jobs.pnl import load_pnl_summary
        with patch("api.jobs.pnl.db") as mock_db:
            mock_db.db_read_enabled.return_value = False
            result = load_pnl_summary(_DATE)
        assert result["status"] == "db_disabled"

    def test_empty_db_returns_ok(self):
        result = self._summary([])
        assert result["status"] == "ok"
        assert result["total_rows"] == 0
        assert result["rows_with_return"] == 0

    def test_hit_rate_buy_signals(self):
        rows = _make_summary_rows(
            ("NVDA", "BUY", 5, 2.0, True),
            ("AAPL", "BUY", 5, 1.5, True),
            ("TSLA", "BUY", 5, -1.0, False),
        )
        result = self._summary(rows)
        buy = result["by_signal"]["BUY"]
        assert buy["n"] == 3
        assert abs(buy["hit_rate"] - 2/3) < 0.01

    def test_avg_return_by_horizon(self):
        rows = _make_summary_rows(
            ("NVDA", "BUY", 5, 3.0, True),
            ("AAPL", "BUY", 5, 1.0, True),
        )
        result = self._summary(rows)
        h5 = result["by_horizon"]["5"]
        assert abs(h5["avg_return_pct"] - 2.0) < 0.01

    def test_none_return_excluded_from_stats(self):
        rows = _make_summary_rows(
            ("NVDA", "BUY", 5, 2.0, True),
            ("AAPL", "BUY", 5, None, None),
        )
        result = self._summary(rows)
        assert result["rows_with_return"] == 1
        assert result["by_signal"]["BUY"]["n"] == 1

    def test_spearman_ic_computed_when_enough_rows(self):
        rows = _make_summary_rows(
            ("A", "BUY", 5, 5.0, True),
            ("B", "BUY", 5, 3.0, True),
            ("C", "BUY", 5, 1.0, True),
            ("D", "SELL", 5, -1.0, True),
            ("E", "SELL", 5, -3.0, True),
        )
        result = self._summary(rows)
        assert result["spearman_ic"] is not None
        assert -1.0 <= result["spearman_ic"] <= 1.0

    def test_spearman_ic_none_when_too_few(self):
        rows = _make_summary_rows(
            ("A", "BUY", 5, 2.0, True),
            ("B", "BUY", 5, 1.0, True),
        )
        result = self._summary(rows)
        assert result["spearman_ic"] is None

    def test_recent_list_capped_at_20(self):
        rows = _make_summary_rows(
            *[("S{}".format(i), "BUY", 5, float(i), True) for i in range(25)]
        )
        result = self._summary(rows)
        assert len(result["recent"]) <= 20

    def test_recent_has_required_keys(self):
        rows = _make_summary_rows(("NVDA", "BUY", 5, 2.0, True))
        result = self._summary(rows)
        r = result["recent"][0]
        for k in ("symbol", "signal_date", "horizon_days", "signal_label", "return_pct", "hit"):
            assert k in r

    def test_by_horizon_and_by_signal_populated(self):
        rows = _make_summary_rows(
            ("A", "BUY",  5,  2.0, True),
            ("B", "SELL", 21, -1.5, True),
        )
        result = self._summary(rows)
        assert "5" in result["by_horizon"]
        assert "21" in result["by_horizon"]
        assert "BUY" in result["by_signal"]
        assert "SELL" in result["by_signal"]


# ---------------------------------------------------------------------------
# Route contract tests
# ---------------------------------------------------------------------------

class TestPnLRoutes:
    def test_compute_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.post("/jobs/pnl/compute", json={})
        assert resp.status_code == 401

    def test_summary_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.get("/jobs/pnl/summary")
        assert resp.status_code == 401

    def test_compute_route_in_openapi(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json().get("paths", {})
        assert "/jobs/pnl/compute" in paths
        assert "/jobs/pnl/summary" in paths

    def test_compute_returns_digest(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app

        def fake_compute(as_of_date, horizons):
            return []

        def fake_store(rows):
            return 0

        monkeypatch.setattr("api.jobs.pnl.compute_signal_pnl", fake_compute)
        monkeypatch.setattr("api.jobs.pnl.store_signal_pnl", fake_store)
        client = TestClient(app)
        resp = client.post(
            "/jobs/pnl/compute",
            json={"as_of_date": "2025-01-02"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["as_of_date"] == "2025-01-02"
        assert "total_rows" in data
        assert "stored" in data

    def test_summary_db_disabled_response(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        monkeypatch.setattr("api.jobs.pnl.db.db_read_enabled", lambda: False)
        client = TestClient(app)
        resp = client.get("/jobs/pnl/summary", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["status"] == "db_disabled"

    def test_summary_default_date_fallback(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        captured = {}

        def fake_summary(as_of_date, lookback_days=90):
            captured["date"] = as_of_date
            return {"status": "ok", "as_of_date": as_of_date.isoformat(),
                    "lookback_days": lookback_days, "total_rows": 0,
                    "rows_with_return": 0, "by_horizon": {}, "by_signal": {},
                    "spearman_ic": None, "recent": []}

        monkeypatch.setattr("api.jobs.pnl.load_pnl_summary", fake_summary)
        client = TestClient(app)
        resp = client.get("/jobs/pnl/summary", headers=AUTH)
        assert resp.status_code == 200
        import datetime as _dt
        assert captured["date"] == _dt.date.today() - _dt.timedelta(days=1)
