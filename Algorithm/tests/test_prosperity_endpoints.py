import datetime as dt
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from api import db
from api import migrations
from api.prosperity import ingest, query
from api.main import app


def _enable_db_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "db_enabled", lambda: True)
    monkeypatch.setattr(db, "db_write_enabled", lambda: True)
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    # Avoid touching a real database in unit tests
    monkeypatch.setattr(db, "ensure_schema", lambda: None)
    monkeypatch.setattr(migrations.runner, "apply_migrations", lambda: None)
    with TestClient(app) as client:
        yield client


def test_prosperity_health(client: TestClient):
    res = client.get("/prosperity/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"


def test_latest_signal_missing_returns_404(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    _enable_db_flags(monkeypatch)
    monkeypatch.setattr(query, "latest_signal", lambda symbol, lookback: None)

    res = client.get("/prosperity/latest/signal", params={"symbol": "ZZZ_MISSING", "lookback": 5})
    assert res.status_code == 404
    body = res.json()
    assert "trace_id" in body
    assert body["error"]["type"] == "http_error"


def test_snapshot_run_and_latest(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    _enable_db_flags(monkeypatch)

    symbols: List[str] = ["AAPL", "MSFT"]
    features_store: Dict[str, Dict[str, str]] = {}
    signals_store: Dict[str, Dict[str, str]] = {}

    monkeypatch.setattr(ingest, "upsert_universe", lambda syms: (len(syms), syms))
    monkeypatch.setattr(ingest, "ingest_bars", lambda *args, **kwargs: {"inserted": 1, "updated": 0})

    def fake_features(symbol: str, as_of_date: dt.date, lookback: int):
        payload = {
            "symbol": symbol,
            "as_of_date": as_of_date.isoformat(),
            "lookback": lookback,
            "stored": True,
            "features": {"mom_5": 0.1},
            "regime": "TEST",
        }
        features_store[symbol] = payload
        return payload

    def fake_signal(symbol: str, as_of_date: dt.date, lookback: int):
        payload = {
            "symbol": symbol,
            "as_of_date": as_of_date.isoformat(),
            "lookback": lookback,
            "score_mode": "stacked",
            "score": 0.5,
            "signal": "BUY",
            "regime": "TEST",
            "thresholds": {},
            "confidence": 0.7,
        }
        signals_store[symbol] = payload
        return payload

    monkeypatch.setattr(ingest, "compute_and_store_features", fake_features)
    monkeypatch.setattr(ingest, "compute_and_store_signal", fake_signal)
    monkeypatch.setattr(query, "latest_features", lambda sym, lb: features_store.get(sym))
    monkeypatch.setattr(query, "latest_signal", lambda sym, lb: signals_store.get(sym))

    payload = {
        "symbols": symbols,
        "from_date": "2024-01-01",
        "to_date": "2024-01-05",
        "as_of_date": "2024-01-05",
        "lookback": 5,
        "concurrency": 10,
    }
    res = client.post("/prosperity/snapshot/run", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["result"]["rows_written"] == {"signals": 2, "features": 2}
    assert set(body["result"]["symbols_ok"]) == set(symbols)
    assert body["requested"]["concurrency"] == 5  # clamped
    assert body["trace_id"]

    sig_res = client.get("/prosperity/latest/signal", params={"symbol": "AAPL", "lookback": 5})
    assert sig_res.status_code == 200
    assert sig_res.json()["signal"] == "BUY"

    feat_res = client.get("/prosperity/latest/features", params={"symbol": "AAPL", "lookback": 5})
    assert feat_res.status_code == 200
    assert feat_res.json()["regime"] == "TEST"
