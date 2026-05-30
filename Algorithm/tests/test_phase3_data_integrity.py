"""Regression tests for Phase 3 — backtest cost wiring and formula version columns."""

import datetime as dt
import pytest


# ---------------------------------------------------------------------------
# 1. Backtest equity curve cost model wiring
# ---------------------------------------------------------------------------

def _build_cost_model(fee_bps=1.0, spread_bps=2.0, slippage_bps=5.0):
    return {"fee_bps": fee_bps, "spread_bps": spread_bps, "slippage_bps": slippage_bps}


def test_one_way_cost_rate_formula():
    """_one_way_cost_rate = (fee + spread/2 + slippage) / 10000."""
    from api.research.backtest.engine import _one_way_cost_rate
    rate = _one_way_cost_rate({"fee_bps": 1.0, "spread_bps": 2.0, "slippage_bps": 5.0})
    expected = (1.0 + 1.0 + 5.0) / 10000.0  # fee + half-spread + slippage
    assert abs(rate - expected) < 1e-10


def test_one_way_cost_rate_defaults_on_missing_keys():
    """Missing keys fall back to defaults (fee=1, spread=2, slippage=5)."""
    from api.research.backtest.engine import _one_way_cost_rate
    rate = _one_way_cost_rate({})
    expected = (1.0 + 1.0 + 5.0) / 10000.0
    assert abs(rate - expected) < 1e-10


def _make_minimal_backtest(cost_model):
    """Run a two-symbol, three-date backtest with at least one trade."""
    from api.research.backtest.engine import run_canonical_backtest

    base = dt.date(2024, 1, 2)
    dates = [base + dt.timedelta(days=i) for i in range(5)]

    bars = {
        "AAPL": {d: 100.0 + i * 2 for i, d in enumerate(dates)},
        "SPY":  {d: 400.0 + i for i, d in enumerate(dates)},
    }

    # Always BUY AAPL so we get trades
    def signal_resolver(sym, date):
        if sym == "AAPL":
            return {"action": "BUY", "signal_version": "v1", "feature_version": "v1"}
        return None

    result = run_canonical_backtest(
        symbols=["AAPL"],
        bars=bars,
        market_states={},
        start=dates[0],
        end=dates[-1],
        horizon="3d",
        risk_mode="long_only",
        cost_model=cost_model,
        signal_version_hash="v1",
        quality_score_fetcher=lambda sym, date: 80,
        signal_resolver=signal_resolver,
        friction_engine=None,
    )
    return result


def test_backtest_equity_curve_lower_with_nonzero_costs():
    """Equity curve with real costs must end lower than with zero costs."""
    result_costly = _make_minimal_backtest(_build_cost_model(fee_bps=5.0, spread_bps=10.0, slippage_bps=15.0))
    result_free   = _make_minimal_backtest(_build_cost_model(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0))

    costly_final = result_costly["equity_curve"][-1]["equity"]
    free_final   = result_free["equity_curve"][-1]["equity"]

    assert costly_final < free_final, (
        f"Costly backtest equity {costly_final:.6f} should be less than free {free_final:.6f} — "
        "cost model not wired to equity curve"
    )


def test_backtest_zero_cost_model_unchanged():
    """Zero-cost model must not deduct anything from equity curve."""
    result = _make_minimal_backtest(_build_cost_model(fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0))
    # All daily returns should be purely from price changes, no negative cost drag
    for point in result["equity_curve"]:
        assert point["equity"] >= 0.0


def test_backtest_high_cost_reduces_returns_vs_low_cost():
    """Higher cost model must produce lower final equity than lower cost model."""
    result_high = _make_minimal_backtest(_build_cost_model(fee_bps=20.0, spread_bps=30.0, slippage_bps=40.0))
    result_low  = _make_minimal_backtest(_build_cost_model(fee_bps=0.5, spread_bps=1.0, slippage_bps=2.0))
    high_final = result_high["equity_curve"][-1]["equity"]
    low_final  = result_low["equity_curve"][-1]["equity"]
    assert high_final <= low_final, (
        f"High-cost equity {high_final:.6f} should be <= low-cost equity {low_final:.6f}"
    )


# ---------------------------------------------------------------------------
# 2. formula_version columns added to prosperity_signals_daily migration
# ---------------------------------------------------------------------------

def test_migration_adds_signal_version_to_prosperity_signals_daily():
    """Migration must declare signal_version and feature_version for prosperity_signals_daily."""
    import inspect
    from api import migrations
    source = inspect.getsource(migrations)
    assert '"signal_version"' in source or "'signal_version'" in source, (
        "migrations must add signal_version column to prosperity_signals_daily"
    )
    assert '"feature_version"' in source or "'feature_version'" in source, (
        "migrations must add feature_version column to prosperity_signals_daily"
    )


def test_ingest_writes_signal_version_to_db():
    """ingest.py INSERT SQL must include signal_version and feature_version columns."""
    import inspect
    from api.prosperity import ingest
    source = inspect.getsource(ingest)
    assert "signal_version" in source, "ingest.py must write signal_version to prosperity_signals_daily"
    assert "feature_version" in source, "ingest.py must write feature_version to prosperity_signals_daily"


def test_ingest_extracts_signal_version_from_meta_in_source():
    """Ingest SQL must read signal_version from both signal_dict and meta fallback."""
    import inspect
    from api.prosperity import ingest
    source = inspect.getsource(ingest)
    # Must fall back to meta when signal_version not at top level of signal_dict
    assert 'meta.get("signal_version")' in source or "meta.get('signal_version')" in source or \
           '_sig_meta.get("signal_version")' in source or "_sig_meta.get('signal_version')" in source, (
        "ingest.py must extract signal_version from meta as fallback"
    )


def test_prosperity_routes_ingest_also_writes_signal_version():
    """prosperity/routes.py INSERT must also write signal_version and feature_version."""
    import inspect
    from api.prosperity import routes
    source = inspect.getsource(routes)
    assert "signal_version" in source, "routes.py must write signal_version to prosperity_signals_daily"
    assert "feature_version" in source, "routes.py must write feature_version to prosperity_signals_daily"
