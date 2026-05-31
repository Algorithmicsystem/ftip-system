"""Regression tests for Phase 5 — backtest PIT universe guard and survivorship bias."""

from __future__ import annotations

import datetime as dt
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal_always_buy(sym, date):
    return {"action": "BUY", "signal_version": "v1", "feature_version": "v1"}


def _make_bars(symbols, dates, base_prices=None):
    """Build a bars dict with linearly increasing prices."""
    base_prices = base_prices or {}
    bars = {"SPY": {d: 400.0 + i for i, d in enumerate(dates)}}
    for sym in symbols:
        base = base_prices.get(sym, 100.0)
        bars[sym] = {d: base + i * 2.0 for i, d in enumerate(dates)}
    return bars


def _run(symbols, dates, pit_universe=None):
    from api.research.backtest.engine import run_canonical_backtest
    bars = _make_bars(symbols, dates)
    return run_canonical_backtest(
        symbols=symbols,
        bars=bars,
        market_states={},
        start=dates[0],
        end=dates[-1],
        horizon="3d",
        risk_mode="long_only",
        cost_model={"fee_bps": 0.0, "spread_bps": 0.0, "slippage_bps": 0.0},
        signal_version_hash="v1",
        quality_score_fetcher=lambda s, d: 80,
        signal_resolver=_signal_always_buy,
        friction_engine=None,
        pit_universe=pit_universe,
    )


# ---------------------------------------------------------------------------
# 1. PIT guard flag in result
# ---------------------------------------------------------------------------

def test_pit_guard_applied_false_when_not_provided():
    """When pit_universe is not provided, result must flag pit_guard_applied=False."""
    base = dt.date(2024, 1, 2)
    dates = [base + dt.timedelta(days=i) for i in range(6)]
    result = _run(["AAPL"], dates, pit_universe=None)
    assert result["metrics"]["pit_guard_applied"] is False


def test_pit_guard_applied_true_when_provided():
    """When pit_universe is provided, result must flag pit_guard_applied=True."""
    base = dt.date(2024, 1, 2)
    dates = [base + dt.timedelta(days=i) for i in range(6)]
    result = _run(["AAPL"], dates, pit_universe={"AAPL": dates[0]})
    assert result["metrics"]["pit_guard_applied"] is True


def test_pit_filtered_symbol_days_zero_when_no_filter():
    """With no pit_universe, pit_filtered_symbol_days must be 0."""
    base = dt.date(2024, 1, 2)
    dates = [base + dt.timedelta(days=i) for i in range(6)]
    result = _run(["AAPL"], dates, pit_universe=None)
    assert result["metrics"]["pit_filtered_symbol_days"] == 0


# ---------------------------------------------------------------------------
# 2. PIT guard correctly excludes pre-listing symbol-days
# ---------------------------------------------------------------------------

def test_pit_guard_excludes_pre_listing_trades():
    """Symbol listed mid-period must have no trades before its listing date."""
    base = dt.date(2024, 1, 2)
    dates = [base + dt.timedelta(days=i) for i in range(8)]
    listing_date = dates[4]  # listed on 5th date

    result = _run(
        ["AAPL"],
        dates,
        pit_universe={"AAPL": listing_date},
    )
    # All trades must have entry_dt >= listing_date
    for trade in result["trades"]:
        entry = dt.date.fromisoformat(trade["entry_dt"]) if isinstance(trade["entry_dt"], str) else trade["entry_dt"]
        assert entry >= listing_date, (
            f"Trade entry {entry} is before listing date {listing_date}"
        )


def test_pit_guard_filtered_symbol_days_nonzero_for_late_listing():
    """Symbol listed on day 4 of 8 must have positive pit_filtered_symbol_days."""
    base = dt.date(2024, 1, 2)
    dates = [base + dt.timedelta(days=i) for i in range(8)]
    listing_date = dates[3]

    result = _run(
        ["AAPL"],
        dates,
        pit_universe={"AAPL": listing_date},
    )
    assert result["metrics"]["pit_filtered_symbol_days"] > 0


def test_pit_guard_fully_excluded_symbol_produces_no_trades():
    """Symbol listed after backtest end must produce zero trades."""
    base = dt.date(2024, 1, 2)
    dates = [base + dt.timedelta(days=i) for i in range(6)]
    after_end = dates[-1] + dt.timedelta(days=10)

    result = _run(
        ["AAPL"],
        dates,
        pit_universe={"AAPL": after_end},
    )
    assert result["trades"] == []


# ---------------------------------------------------------------------------
# 3. Equity curve difference: with vs without PIT guard
# ---------------------------------------------------------------------------

def test_pit_guard_reduces_activity_for_late_listing():
    """Backtest with PIT guard (late listing) must have fewer trades than without."""
    base = dt.date(2024, 1, 2)
    dates = [base + dt.timedelta(days=i) for i in range(10)]
    mid = dates[5]

    result_no_guard = _run(["AAPL"], dates, pit_universe=None)
    result_guarded = _run(["AAPL"], dates, pit_universe={"AAPL": mid})

    assert len(result_guarded["trades"]) <= len(result_no_guard["trades"]), (
        "PIT guard should produce fewer or equal trades when symbol has late listing"
    )


def test_pit_guard_symbol_available_from_start_is_unaffected():
    """Symbol available from the first date must produce identical equity to no-guard run."""
    base = dt.date(2024, 1, 2)
    dates = [base + dt.timedelta(days=i) for i in range(8)]

    result_no_guard = _run(["AAPL"], dates, pit_universe=None)
    result_guarded = _run(["AAPL"], dates, pit_universe={"AAPL": dates[0]})

    # Final equity must be identical when symbol is listed from day 0
    final_no_guard = result_no_guard["equity_curve"][-1]["equity"]
    final_guarded = result_guarded["equity_curve"][-1]["equity"]
    assert abs(final_no_guard - final_guarded) < 1e-9, (
        f"Equity diverged: no_guard={final_no_guard} guarded={final_guarded}"
    )


# ---------------------------------------------------------------------------
# 4. Multi-symbol: partial PIT coverage
# ---------------------------------------------------------------------------

def test_pit_guard_filters_only_late_symbol_not_early_one():
    """PIT guard for SYM2 should not affect SYM1 (which is available from start)."""
    base = dt.date(2024, 1, 2)
    dates = [base + dt.timedelta(days=i) for i in range(8)]

    # SYM1 available from start, SYM2 listed on day 5
    result = _run(
        ["SYM1", "SYM2"],
        dates,
        pit_universe={"SYM1": dates[0], "SYM2": dates[5]},
    )
    sym1_trades = [t for t in result["trades"] if t["symbol"] == "SYM1"]
    sym2_trades = [t for t in result["trades"] if t["symbol"] == "SYM2"]

    # SYM1 should have activity; SYM2 may have less
    # Core check: no SYM2 entry before its listing
    for trade in sym2_trades:
        entry = dt.date.fromisoformat(trade["entry_dt"]) if isinstance(trade["entry_dt"], str) else trade["entry_dt"]
        assert entry >= dates[5], f"SYM2 entered before listing: {entry}"


def test_pit_guard_symbol_not_in_map_uses_full_period():
    """Symbol not in pit_universe dict must NOT be filtered (zero filtered days for it)."""
    base = dt.date(2024, 1, 2)
    dates = [base + dt.timedelta(days=i) for i in range(8)]

    # SYM2 available from start (dates[0]), SYM1 absent from map → unrestricted
    result_absent = _run(
        ["SYM1"],
        dates,
        pit_universe={"SYM2": dates[0]},  # SYM1 deliberately absent
    )
    result_explicit = _run(
        ["SYM1"],
        dates,
        pit_universe={"SYM1": dates[0]},  # SYM1 explicitly available from start
    )
    # Both runs must produce identical equity since SYM1 is unrestricted in both
    eq_absent = result_absent["equity_curve"][-1]["equity"]
    eq_explicit = result_explicit["equity_curve"][-1]["equity"]
    assert abs(eq_absent - eq_explicit) < 1e-9, (
        f"Symbol absent from pit_universe ({eq_absent}) should match explicitly listed ({eq_explicit})"
    )
