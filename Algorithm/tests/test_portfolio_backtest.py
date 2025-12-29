import datetime as dt

from fastapi import HTTPException

from api import main


def _make_candles(num_days: int = 1200):
    start = dt.date(2020, 1, 1)
    candles = []
    for i in range(num_days):
        day = start + dt.timedelta(days=i)
        candles.append(main.Candle(timestamp=day.isoformat(), close=100.0 + i * 0.1))
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
