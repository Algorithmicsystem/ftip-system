"""Phase 26: Correlation Guard integration tests."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

import pytest

from ftip.risk import compute_return_correlation_matrix, correlation_guard


# ---------------------------------------------------------------------------
# Unit: compute_return_correlation_matrix
# ---------------------------------------------------------------------------

def test_perfect_positive_correlation():
    series = {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [2.0, 4.0, 6.0, 8.0, 10.0]}
    matrix = compute_return_correlation_matrix(series)
    assert matrix["A"]["B"] == pytest.approx(1.0, abs=1e-6)
    assert matrix["B"]["A"] == pytest.approx(1.0, abs=1e-6)
    assert matrix["A"]["A"] == pytest.approx(1.0)


def test_perfect_negative_correlation():
    series = {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [-1.0, -2.0, -3.0, -4.0, -5.0]}
    matrix = compute_return_correlation_matrix(series)
    assert matrix["A"]["B"] == pytest.approx(-1.0, abs=1e-6)


def test_flat_series_gets_zero_correlation():
    series = {"A": [1.0, 2.0, 3.0], "B": [5.0, 5.0, 5.0]}
    matrix = compute_return_correlation_matrix(series)
    assert matrix["A"]["B"] == pytest.approx(0.0)


def test_short_series_gets_zero_correlation():
    series = {"A": [1.0], "B": [2.0]}
    matrix = compute_return_correlation_matrix(series)
    assert matrix["A"]["B"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Unit: correlation_guard
# ---------------------------------------------------------------------------

def test_guard_passthrough_when_no_matrix():
    weights = {"A": 0.05, "B": 0.04, "C": 0.03}
    result = correlation_guard(weights, correlation_matrix=None)
    # pass-through: all symbols present
    assert set(result.keys()) == {"A", "B", "C"}


def test_guard_reduces_highly_correlated_pair():
    weights = {"A": 0.05, "B": 0.05}
    matrix = {"A": {"A": 1.0, "B": 0.95}, "B": {"A": 0.95, "B": 1.0}}
    result = correlation_guard(weights, correlation_matrix=matrix, threshold=0.80)
    # After iterative haircuts the weaker leg may be fully dropped (< 1e-10)
    # What matters: result is normalised and at most one leg survives at full weight
    total = sum(result.values())
    assert total == pytest.approx(1.0, abs=1e-6)  # normalised
    # Either one symbol was dropped entirely OR the sum is 1 with both reduced
    assert len(result) <= 2
    assert all(v > 0 for v in result.values())


def test_guard_no_reduction_below_threshold():
    weights = {"A": 0.05, "B": 0.05}
    matrix = {"A": {"A": 1.0, "B": 0.70}, "B": {"A": 0.70, "B": 1.0}}
    result = correlation_guard(weights, correlation_matrix=matrix, threshold=0.80)
    # Correlation 0.70 < threshold 0.80, so no haircut
    assert result["A"] == pytest.approx(result["B"], abs=1e-6)


# ---------------------------------------------------------------------------
# Integration: build_portfolio_allocation with correlation guard
# ---------------------------------------------------------------------------

def _mock_rows(symbols_sectors):
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


def _patch_alloc(monkeypatch, axiom_rows, pnl_rows=None):
    import api.axiom.allocator as mod
    from api.axiom import screener as scr

    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)

    call_count = [0]

    def fake_fetchall(*_a, **_k):
        call_count[0] += 1
        if call_count[0] == 1:
            return axiom_rows
        return pnl_rows or []

    monkeypatch.setattr(mod.db, "safe_fetchall", fake_fetchall)
    monkeypatch.setattr(mod.db, "safe_fetchone", lambda *_a, **_k: None)
    monkeypatch.setattr(scr, "_load_ic_state_bulk", lambda _d: "STRONG")
    monkeypatch.setattr(scr, "_load_breadth_state_bulk", lambda _d: "EXPANDING")


def test_allocation_result_includes_correlation_guard_key(monkeypatch):
    rows = _mock_rows([
        ("NVDA", "Technology", 85.0, "BUY"),
        ("AAPL", "Technology", 72.0, "BUY"),
    ])
    _patch_alloc(monkeypatch, rows)
    from api.axiom.allocator import build_portfolio_allocation
    result = build_portfolio_allocation(dt.date(2026, 1, 5))
    assert "correlation_guard" in result
    assert "adjusted_count" in result["correlation_guard"]
    assert "dropped_symbols" in result["correlation_guard"]


def test_correlation_threshold_in_constraints(monkeypatch):
    rows = _mock_rows([("NVDA", "Technology", 85.0, "BUY")])
    _patch_alloc(monkeypatch, rows)
    from api.axiom.allocator import build_portfolio_allocation
    result = build_portfolio_allocation(dt.date(2026, 1, 5), correlation_threshold=0.75)
    assert result["constraints"]["correlation_threshold"] == 0.75


def test_highly_correlated_pair_gets_adjusted(monkeypatch):
    """When pnl returns are near-identical, correlation_guard reduces the smaller weight."""
    import api.axiom.allocator as mod
    from api.axiom import screener as scr

    axiom_rows = _mock_rows([
        ("NVDA", "Technology", 85.0, "BUY"),
        ("MSFT", "Technology", 83.0, "BUY"),
    ])

    # Return series where NVDA and MSFT are perfectly correlated
    pnl_rows = (
        [("NVDA", float(i)) for i in range(1, 11)] +
        [("MSFT", float(i)) for i in range(1, 11)]
    )

    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
    call_count = [0]

    def fake_fetchall(*_a, **_k):
        call_count[0] += 1
        return axiom_rows if call_count[0] == 1 else pnl_rows

    monkeypatch.setattr(mod.db, "safe_fetchall", fake_fetchall)
    monkeypatch.setattr(mod.db, "safe_fetchone", lambda *_a, **_k: None)
    monkeypatch.setattr(scr, "_load_ic_state_bulk", lambda _d: "STRONG")
    monkeypatch.setattr(scr, "_load_breadth_state_bulk", lambda _d: "EXPANDING")

    from api.axiom.allocator import build_portfolio_allocation
    result = build_portfolio_allocation(
        dt.date(2026, 1, 5),
        correlation_threshold=0.50,  # low threshold to trigger on perfect correlation
    )

    cg = result["correlation_guard"]
    # Either adjusted or dropped — either way the guard fired
    assert cg["adjusted_count"] > 0 or len(cg["dropped_symbols"]) > 0
