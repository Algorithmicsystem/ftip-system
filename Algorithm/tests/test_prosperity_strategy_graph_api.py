import datetime as dt
from typing import List

import pytest
from fastapi.testclient import TestClient

from api import db, migrations
from api.main import Candle, app
from api.prosperity import strategy_graph_db


def _disable_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "db_enabled", lambda: False)
    monkeypatch.setattr(db, "db_write_enabled", lambda: False)
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FTIP_DB_ENABLED", "0")
    monkeypatch.setenv("FTIP_DB_WRITE_ENABLED", "0")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "0")
    monkeypatch.setattr(migrations, "ensure_schema", lambda: [])
    with TestClient(app) as client:
        yield client


def test_strategy_graph_run_without_db(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    _disable_db(monkeypatch)

    monkeypatch.setattr(
        strategy_graph_db, "upsert_strategy_rows", lambda rows: len(list(rows))
    )
    monkeypatch.setattr(strategy_graph_db, "upsert_ensemble_row", lambda row: None)

    sample = [
        Candle(
            timestamp=(dt.date(2024, 1, 1) + dt.timedelta(days=i)).isoformat(),
            close=100 + i,
        )
        for i in range(60)
    ]

    def fake_fetch(symbol: str, from_date: dt.date, as_of_date: dt.date):
        return []

    def fake_massive(sym: str, start: str, end: str):
        return sample

    monkeypatch.setattr(
        "api.prosperity.strategy_graph.query.fetch_bars", fake_fetch, raising=False
    )
    monkeypatch.setattr(
        "api.main.massive_fetch_daily_bars", fake_massive, raising=False
    )

    payload = {
        "symbols": ["AAPL"],
        "from_date": "2024-01-01",
        "to_date": "2024-02-10",
        "as_of_date": "2024-02-10",
        "lookback": 30,
        "persist": False,
    }

    res = client.post("/prosperity/strategy_graph/run", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in {"ok", "partial"}
    assert body["result"]["symbols_ok"] == ["AAPL"]
    assert set(body["result"]["rows_written"].keys()) == {"strategies", "ensembles"}


def test_strategy_graph_latest_endpoints(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    monkeypatch.setattr(db, "db_enabled", lambda: True)
    monkeypatch.setattr(db, "db_write_enabled", lambda: True)
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)

    sample_ensemble = {"symbol": "AAPL", "as_of_date": "2024-01-10", "lookback": 30}
    sample_strategies: List[dict] = [
        {"symbol": "AAPL", "as_of_date": "2024-01-10", "lookback": 30}
    ]
    monkeypatch.setattr(
        "api.prosperity.strategy_graph.latest_ensemble", lambda s, lb: sample_ensemble
    )
    monkeypatch.setattr(
        "api.prosperity.strategy_graph.latest_strategies",
        lambda s, lb: sample_strategies,
    )

    res_e = client.get(
        "/prosperity/strategy_graph/latest/ensemble",
        params={"symbol": "AAPL", "lookback": 30},
    )
    res_s = client.get(
        "/prosperity/strategy_graph/latest/strategies",
        params={"symbol": "AAPL", "lookback": 30},
    )

    assert res_e.status_code == 200
    assert res_s.status_code == 200
    assert res_e.json()["data"]["symbol"] == "AAPL"
    assert res_s.json()["data"][0]["symbol"] == "AAPL"
