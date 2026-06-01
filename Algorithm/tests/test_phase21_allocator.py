"""Phase 21: Portfolio Allocation Engine tests."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

import pytest

from api.axiom.allocator import build_portfolio_allocation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_rows(symbols_sectors):
    """Build fake DB rows: (symbol, payload, signal_label, sector)."""
    rows = []
    for sym, sector, dau, signal in symbols_sectors:
        payload = {
            "deployable_alpha_utility": dau,
            "overall_confidence": 65.0,
            "deployability_tier": "live_candidate",
            "regime_label": "trending",
            "engine_scores": {
                "critical_fragility": {"score": 30.0},
                "liquidity_convexity": {"score": 70.0},
                "research_integrity": {"score": 70.0},
            },
        }
        rows.append((sym, payload, signal, sector))
    return rows


def _patch_db(monkeypatch, rows, ic_state="STRONG"):
    import api.axiom.allocator as mod
    from api.axiom import screener as scr

    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(mod.db, "safe_fetchall", lambda *_a, **_k: rows)
    monkeypatch.setattr(mod.db, "safe_fetchone", lambda *_a, **_k: None)
    monkeypatch.setattr(scr, "_load_ic_state_bulk", lambda _d: ic_state)
    monkeypatch.setattr(scr, "_load_breadth_state_bulk", lambda _d: "EXPANDING")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_allocation_returns_sorted_positions(monkeypatch):
    rows = _mock_rows([
        ("NVDA", "Technology", 85.0, "BUY"),
        ("AAPL", "Technology", 72.0, "BUY"),
        ("JPM",  "Financials",  60.0, "BUY"),
    ])
    _patch_db(monkeypatch, rows)

    result = build_portfolio_allocation(
        dt.date(2026, 1, 5),
        max_position_weight=0.10,
        max_sector_concentration=0.30,
        max_portfolio_heat=1.0,
    )

    assert result["status"] == "ok"
    allocs = result["allocations"]
    assert len(allocs) >= 1
    # ranks are ascending
    for i, a in enumerate(allocs, start=1):
        assert a["rank"] == i
    # weights are non-negative
    for a in allocs:
        assert a["suggested_weight"] >= 0.0


def test_sector_concentration_cap_is_enforced(monkeypatch):
    """Four Technology symbols — sector cap so tight (0.015) that at most one fits fully.
    After the first symbol uses ~0.01, the second triggers the cap."""
    rows = _mock_rows([
        ("NVDA",  "Technology", 90.0, "BUY"),
        ("AAPL",  "Technology", 85.0, "BUY"),
        ("MSFT",  "Technology", 80.0, "BUY"),
        ("GOOGL", "Technology", 75.0, "BUY"),
    ])
    _patch_db(monkeypatch, rows)

    result = build_portfolio_allocation(
        dt.date(2026, 1, 5),
        max_position_weight=0.10,
        max_sector_concentration=0.015,   # very tight → second symbol overflows
        max_portfolio_heat=1.0,
    )

    tech_weight = result["sector_breakdown"].get("Technology", 0.0)
    assert tech_weight <= 0.015 + 1e-6
    # At least one symbol should be rejected due to sector cap
    rejected_reasons = [r["rejection_reason"] for r in result["rejected"]]
    assert "sector_cap_exceeded" in rejected_reasons


def test_portfolio_heat_cap_is_enforced(monkeypatch):
    """Portfolio heat capped at 0.15 with 0.10/position → max 1 full + partial."""
    rows = _mock_rows([
        ("NVDA", "Technology", 90.0, "BUY"),
        ("AAPL", "Healthcare", 85.0, "BUY"),
        ("MSFT", "Financials", 80.0, "BUY"),
    ])
    _patch_db(monkeypatch, rows)

    result = build_portfolio_allocation(
        dt.date(2026, 1, 5),
        max_position_weight=0.10,
        max_sector_concentration=1.0,
        max_portfolio_heat=0.15,
    )

    assert result["portfolio_weight_total"] <= 0.15 + 1e-6


def test_returns_sector_breakdown(monkeypatch):
    rows = _mock_rows([
        ("NVDA", "Technology", 80.0, "BUY"),
        ("JPM",  "Financials", 75.0, "BUY"),
    ])
    _patch_db(monkeypatch, rows)

    result = build_portfolio_allocation(dt.date(2026, 1, 5))

    breakdown = result["sector_breakdown"]
    assert isinstance(breakdown, dict)
    # Every sector in allocations appears in breakdown
    alloc_sectors = {a["sector"] for a in result["allocations"]}
    for sec in alloc_sectors:
        assert sec in breakdown


def test_db_disabled_returns_empty(monkeypatch):
    import api.axiom.allocator as mod
    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)

    result = build_portfolio_allocation(dt.date(2026, 1, 5))

    assert result["status"] == "db_disabled"
    assert result["allocations"] == []
    assert result["position_count"] == 0


def test_allocate_endpoint_returns_200(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app

    rows = _mock_rows([
        ("NVDA", "Technology", 85.0, "BUY"),
        ("JPM",  "Financials", 70.0, "BUY"),
    ])

    import api.axiom.allocator as mod
    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(mod.db, "safe_fetchall", lambda *_a, **_k: rows)
    monkeypatch.setattr(mod.db, "safe_fetchone", lambda *_a, **_k: None)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    client = TestClient(app)
    resp = client.post(
        "/axiom/allocate",
        json={"as_of_date": "2026-01-05", "max_portfolio_heat": 0.5},
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "allocations" in body
    assert "sector_breakdown" in body
    assert "portfolio_weight_total" in body
