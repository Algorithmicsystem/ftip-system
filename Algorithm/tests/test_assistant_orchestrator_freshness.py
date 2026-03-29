import datetime as dt

import pytest
from fastapi import HTTPException

from api.assistant import orchestrator


class _DummyCursor:
    def __init__(self):
        self.executed = 0

    def execute(self, *_args, **_kwargs):
        self.executed += 1


class _DummyConn:
    def commit(self):
        return None


class _ConnCtx:
    def __init__(self, cursor: _DummyCursor):
        self.cursor = cursor

    def __enter__(self):
        return _DummyConn(), self.cursor

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.mark.anyio
async def test_ensure_freshness_hydrates_from_prosperity_when_market_bars_empty(monkeypatch):
    symbol = "AAPL"
    today = dt.datetime.now(dt.timezone.utc).date()
    stale_day = today - dt.timedelta(days=10)

    state = {"hydrated": False}

    def fake_fetchone(query, params):
        if "FROM market_bars_daily" in query:
            if state["hydrated"]:
                return (stale_day, dt.datetime.now(dt.timezone.utc))
            return (None, None)
        if "FROM prosperity_daily_bars" in query:
            return (stale_day, dt.datetime.now(dt.timezone.utc))
        if "FROM news_raw" in query:
            return (None, None)
        if "FROM sentiment_daily" in query:
            return (None, None)
        return None

    def fake_fetchall(query, params):
        if "FROM prosperity_daily_bars" in query:
            return [
                (symbol, stale_day, 1.0, 1.2, 0.9, 1.1, 1000, "massive"),
            ]
        return []

    cursor = _DummyCursor()

    def fake_with_connection():
        state["hydrated"] = True
        return _ConnCtx(cursor)

    async def fake_ingest(_req):
        return {
            "status": "failed",
            "symbols_ok": [],
            "symbols_failed": [{"symbol": symbol, "reason_code": "NO_DATA"}],
        }
    
    async def fake_ok(_req):
        return {"status": "ok"}

    monkeypatch.setattr("api.assistant.orchestrator.db.safe_fetchone", fake_fetchone)
    monkeypatch.setattr("api.assistant.orchestrator.db.safe_fetchall", fake_fetchall)
    monkeypatch.setattr("api.assistant.orchestrator.db.with_connection", fake_with_connection)
    monkeypatch.setattr("api.assistant.orchestrator.db.db_enabled", lambda: True)
    monkeypatch.setattr("api.assistant.orchestrator.db.db_read_enabled", lambda: True)
    monkeypatch.setattr("api.assistant.orchestrator.db.db_write_enabled", lambda: True)
    monkeypatch.setattr(
        "api.assistant.orchestrator.market_data_job.ingest_bars_daily", fake_ingest
    )
    monkeypatch.setattr("api.assistant.orchestrator.market_data_job.ingest_news", fake_ok)
    monkeypatch.setattr(
        "api.assistant.orchestrator.market_data_job.compute_sentiment", fake_ok
    )

    freshness = await orchestrator.ensure_freshness(symbol)

    assert freshness["as_of_date"] == stale_day
    assert cursor.executed == 1


@pytest.mark.anyio
async def test_ensure_freshness_raises_404_when_no_market_or_prosperity_bars(monkeypatch):
    def fake_fetchone(query, params):
        if "FROM market_bars_daily" in query:
            return (None, None)
        if "FROM prosperity_daily_bars" in query:
            return (None, None)
        if "FROM news_raw" in query:
            return (None, None)
        if "FROM sentiment_daily" in query:
            return (None, None)
        return None

    monkeypatch.setattr("api.assistant.orchestrator.db.safe_fetchone", fake_fetchone)
    monkeypatch.setattr("api.assistant.orchestrator.db.safe_fetchall", lambda *_: [])
    monkeypatch.setattr("api.assistant.orchestrator.db.db_enabled", lambda: True)
    monkeypatch.setattr("api.assistant.orchestrator.db.db_read_enabled", lambda: True)
    monkeypatch.setattr("api.assistant.orchestrator.db.db_write_enabled", lambda: True)

    async def fake_ingest(_req):
        return {"status": "failed", "symbols_ok": [], "symbols_failed": []}
    
    async def fake_ok(_req):
        return {"status": "ok"}

    monkeypatch.setattr(
        "api.assistant.orchestrator.market_data_job.ingest_bars_daily", fake_ingest
    )
    monkeypatch.setattr("api.assistant.orchestrator.market_data_job.ingest_news", fake_ok)
    monkeypatch.setattr(
        "api.assistant.orchestrator.market_data_job.compute_sentiment", fake_ok
    )

    with pytest.raises(HTTPException) as excinfo:
        await orchestrator.ensure_freshness("AAPL")

    assert excinfo.value.status_code == 404
    assert "no market bars available" in excinfo.value.detail
