"""Phase 30: Allocator Replay tests."""
from __future__ import annotations

import datetime as dt

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alloc_result(symbols=("NVDA", "AAPL"), weight=0.05):
    return {
        "status": "ok",
        "allocations": [
            {"symbol": s, "suggested_weight": weight, "sector": "Technology",
             "signal_label": "BUY", "dau": 80.0, "conviction_score": 60.0,
             "suggested_weight_pct": "5.00%", "size_band": "small",
             "deployability_tier": "live_candidate", "ic_state": "STRONG",
             "active_constraint": "kelly", "downside_flags": [], "rank": i + 1}
            for i, s in enumerate(symbols)
        ],
        "portfolio_weight_total": weight * len(symbols),
        "position_count": len(symbols),
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_replay_db_disabled(monkeypatch):
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    from fastapi.testclient import TestClient
    from api.main import app
    monkeypatch.setenv("FTIP_API_KEY", "secret")
    client = TestClient(app)
    resp = client.get(
        "/axiom/allocate/replay?start_date=2026-01-05&end_date=2026-01-07",
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "db_disabled"


def test_replay_date_range_too_large(monkeypatch):
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setenv("FTIP_API_KEY", "secret")
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    resp = client.get(
        "/axiom/allocate/replay?start_date=2020-01-01&end_date=2026-01-01",
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"


def test_replay_with_pnl_data(monkeypatch):
    from api import db
    import api.axiom.allocator as allocmod
    from api.axiom import screener as scr

    monkeypatch.setattr(db, "db_read_enabled", lambda: True)

    # Mock allocator to return two positions on each call
    import api.axiom.routes as routes_mod
    monkeypatch.setattr(
        routes_mod,
        "_compute_conviction_trends",
        lambda *a, **kw: {"status": "ok", "trends": []},
    )

    alloc_call_count = [0]

    def fake_build(date, **kwargs):
        alloc_call_count[0] += 1
        return _alloc_result()

    from api.axiom import allocator as alloc_mod
    monkeypatch.setattr(alloc_mod, "build_portfolio_allocation", fake_build)

    # Patch the routes module's import of build_portfolio_allocation
    import api.axiom.routes as routes
    monkeypatch.setattr(
        "api.axiom.allocator.build_portfolio_allocation",
        fake_build,
    )

    # PnL rows: NVDA +2%, AAPL +1% on each date
    pnl_rows = [("NVDA", 2.0), ("AAPL", 1.0)]
    monkeypatch.setattr(db, "safe_fetchall", lambda *_a, **_k: pnl_rows)
    monkeypatch.setattr(db, "safe_fetchone", lambda *_a, **_k: None)

    monkeypatch.setenv("FTIP_API_KEY", "secret")
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.get(
        "/axiom/allocate/replay?start_date=2026-01-05&end_date=2026-01-07",
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "equity_curve" in body
    assert "summary" in body
    summary = body["summary"]
    assert "final_equity" in summary
    assert "sharpe_ratio" in summary
    assert "max_drawdown_pct" in summary
    assert "win_rate" in summary


def test_replay_equity_grows_with_positive_returns(monkeypatch):
    from api import db
    from api.axiom import allocator as alloc_mod

    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(
        alloc_mod, "build_portfolio_allocation",
        lambda *a, **kw: _alloc_result(("NVDA",), weight=0.10),
    )
    monkeypatch.setattr(db, "safe_fetchall", lambda *_a, **_k: [("NVDA", 5.0)])
    monkeypatch.setattr(db, "safe_fetchone", lambda *_a, **_k: None)

    monkeypatch.setenv("FTIP_API_KEY", "secret")
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    resp = client.get(
        "/axiom/allocate/replay?start_date=2026-01-05&end_date=2026-01-06",
        headers={"X-FTIP-API-Key": "secret"},
    )
    body = resp.json()
    assert body["summary"]["final_equity"] > 1.0
    assert body["summary"]["total_return_pct"] > 0.0


def test_replay_summary_win_rate_all_positive(monkeypatch):
    from api import db
    from api.axiom import allocator as alloc_mod

    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(
        alloc_mod, "build_portfolio_allocation",
        lambda *a, **kw: _alloc_result(("NVDA",), weight=0.10),
    )
    monkeypatch.setattr(db, "safe_fetchall", lambda *_a, **_k: [("NVDA", 3.0)])
    monkeypatch.setattr(db, "safe_fetchone", lambda *_a, **_k: None)

    monkeypatch.setenv("FTIP_API_KEY", "secret")
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    resp = client.get(
        "/axiom/allocate/replay?start_date=2026-01-05&end_date=2026-01-07",
        headers={"X-FTIP-API-Key": "secret"},
    )
    body = resp.json()
    assert body["summary"]["win_rate"] == pytest.approx(1.0)
