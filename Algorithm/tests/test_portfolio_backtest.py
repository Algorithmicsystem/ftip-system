import datetime as dt

from fastapi import HTTPException

from api import main


def _make_candles(num_days: int = 1200, *, start: dt.date = dt.date(2020, 1, 1), slope: float = 0.1):
    candles = []
    for i in range(num_days):
        day = start + dt.timedelta(days=i)
        candles.append(main.Candle(timestamp=day.isoformat(), close=100.0 + i * slope))
    return candles


def test_portfolio_backtest_skips_failed_symbols(monkeypatch):
    monkeypatch.setenv("FTIP_PORTFOLIO_MIN_SYMBOLS", "2")

    good_candles = _make_candles()

    def fake_fetch(symbol: str, from_date: str, to_date: str):
        if symbol == "FAIL":
            raise HTTPException(status_code=429, detail="rate limited")
        return list(good_candles)

    monkeypatch.setattr(main, "massive_fetch_daily_bars_cached", fake_fetch)

    req = main.PortfolioBacktestRequest(
        symbols=["AAA", "BBB", "FAIL"],
        from_date="2023-06-01",
        to_date="2023-12-31",
        include_equity_curve=False,
    )

    resp = main.backtest_portfolio(req)

    assert resp.audit is not None
    skipped = resp.audit.get("skipped_symbols") or []
    assert any(item.get("symbol") == "FAIL" for item in skipped)
    assert resp.total_return is not None
    assert resp.annual_return is not None


def test_portfolio_backtest_beta_computed(monkeypatch):
    good_candles = _make_candles(slope=0.1)
    spy_candles = _make_candles(slope=0.05)  # SPY rises at half the rate

    def fake_fetch(symbol: str, from_date: str, to_date: str):
        return list(spy_candles if symbol == "SPY" else good_candles)

    # Force a BUY signal so the portfolio is invested and has non-zero returns.
    def fake_compute(symbol, as_of, lookback, candles_all):
        return main.SignalResponse(
            symbol=symbol,
            as_of=as_of,
            lookback=lookback,
            effective_lookback=lookback,
            regime="TRENDING",
            thresholds={"buy": 0.1, "sell": -0.1},
            score=0.5,
            signal="BUY",
            confidence=0.8,
            features={},
        )

    monkeypatch.setattr(main, "massive_fetch_daily_bars_cached", fake_fetch)
    monkeypatch.setattr(main, "compute_signal_for_symbol_from_candles", fake_compute)

    req = main.PortfolioBacktestRequest(
        symbols=["AAA", "BBB"],
        from_date="2023-01-01",
        to_date="2023-12-31",
        include_equity_curve=False,
    )
    resp = main.backtest_portfolio(req)

    assert resp.beta is not None
    assert resp.beta > 0.0  # portfolio and SPY both trend up → positive beta


def test_portfolio_backtest_beta_none_when_spy_fails(monkeypatch):
    good_candles = _make_candles()

    def fake_fetch(symbol: str, from_date: str, to_date: str):
        if symbol == "SPY":
            raise HTTPException(status_code=429, detail="rate limited")
        return list(good_candles)

    monkeypatch.setattr(main, "massive_fetch_daily_bars_cached", fake_fetch)

    req = main.PortfolioBacktestRequest(
        symbols=["AAA", "BBB"],
        from_date="2023-01-01",
        to_date="2023-12-31",
        include_equity_curve=False,
    )
    resp = main.backtest_portfolio(req)
    assert resp.beta is None


def test_portfolio_backtest_signals_precomputed(monkeypatch):
    # Verify the signal function is called once per (symbol, rebalance_date)
    # rather than once per (symbol, trading_day).
    call_count = {"n": 0}
    good_candles = _make_candles()

    def fake_fetch(symbol: str, from_date: str, to_date: str):
        return list(good_candles)

    original_compute = main.compute_signal_for_symbol_from_candles

    def counting_compute(symbol, as_of, lookback, candles_all):
        call_count["n"] += 1
        return original_compute(symbol, as_of, lookback, candles_all)

    monkeypatch.setattr(main, "massive_fetch_daily_bars_cached", fake_fetch)
    monkeypatch.setattr(main, "compute_signal_for_symbol_from_candles", counting_compute)

    req = main.PortfolioBacktestRequest(
        symbols=["AAA", "BBB"],
        from_date="2023-06-01",
        to_date="2023-12-31",
        rebalance_every=5,
        include_equity_curve=False,
    )
    main.backtest_portfolio(req)

    # Trading days in the range ≈ 130 → rebalance events ≈ 26 per symbol
    # Total calls = 2 symbols × ~26 rebalance dates (pre-computed, not per trading day)
    assert call_count["n"] < 130, (
        f"Expected pre-computed calls (<130), got {call_count['n']}"
    )
