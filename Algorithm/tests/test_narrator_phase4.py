import datetime as dt

import pytest
from fastapi.testclient import TestClient

from api import db, migrations, security
from api.main import app
from api.prosperity import query
from ftip.narrator import client as narrator_client


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in [
        "FTIP_API_KEY",
        "FTIP_API_KEYS",
        "FTIP_API_KEY_PRIMARY",
        "FTIP_DB_ENABLED",
        "FTIP_DB_READ_ENABLED",
        "FTIP_DB_WRITE_ENABLED",
        "OPENAI_API_KEY",
        "OpenAI_ftip-system",
    ]:
        monkeypatch.delenv(key, raising=False)
    security.reset_auth_cache()
    yield
    for key in [
        "FTIP_API_KEY",
        "FTIP_API_KEYS",
        "FTIP_API_KEY_PRIMARY",
        "FTIP_DB_ENABLED",
        "FTIP_DB_READ_ENABLED",
        "FTIP_DB_WRITE_ENABLED",
        "OPENAI_API_KEY",
        "OpenAI_ftip-system",
    ]:
        monkeypatch.delenv(key, raising=False)
    security.reset_auth_cache()


def _prepare_env(monkeypatch):
    monkeypatch.setenv("FTIP_API_KEY", "test-key")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_WRITE_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    security.reset_auth_cache()


def test_strategy_graph_requires_auth(monkeypatch):
    _prepare_env(monkeypatch)
    monkeypatch.setattr(db, "db_enabled", lambda: False)
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    client = TestClient(app)
    resp = client.post(
        "/narrator/explain-strategy-graph",
        json={"symbol": "AAPL", "lookback": 252, "days": 30, "to_date": "2024-12-31"},
    )
    assert resp.status_code == 401


def test_strategy_graph_explanation(monkeypatch):
    _prepare_env(monkeypatch)
    monkeypatch.setattr(db, "db_enabled", lambda: True)
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    history = [
        {
            "as_of": "2024-12-29",
            "signal": "HOLD",
            "score": 0.1,
            "regime": "NEUTRAL",
            "confidence": 0.5,
        },
        {
            "as_of": "2024-12-30",
            "signal": "BUY",
            "score": 0.6,
            "regime": "TRENDING",
            "confidence": 0.7,
        },
        {
            "as_of": "2024-12-31",
            "signal": "BUY",
            "score": 0.65,
            "regime": "TRENDING",
            "confidence": 0.8,
        },
    ]
    monkeypatch.setattr(query, "signal_history", lambda *_, **__: history)
    monkeypatch.setattr(
        narrator_client, "complete_chat", lambda *_, **__: ("Graph summary", "gpt", {})
    )

    client = TestClient(app)
    resp = client.post(
        "/narrator/explain-strategy-graph",
        headers={"X-FTIP-API-Key": "test-key"},
        json={"symbol": "AAPL", "lookback": 252, "days": 30, "to_date": "2024-12-31"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "AAPL"
    assert "Graph summary" in body["explanation"]
    assert body["graph"]["nodes"]
    assert body["graph"]["edges"]
    assert body["window"]["to"] == dt.date(2024, 12, 31).isoformat()


def test_diagnose_reports_checks(monkeypatch):
    _prepare_env(monkeypatch)
    monkeypatch.setattr(db, "db_enabled", lambda: True)
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db, "db_write_enabled", lambda: True)
    monkeypatch.setattr(migrations, "ensure_schema", lambda: [])
    monkeypatch.setattr(
        db, "safe_fetchall", lambda *_args, **_kwargs: [("001", dt.datetime.utcnow())]
    )
    monkeypatch.setattr(query, "latest_signal", lambda *_, **__: {"signal": "BUY"})

    client = TestClient(app)
    resp = client.post("/narrator/diagnose", headers={"X-FTIP-API-Key": "test-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    check_names = {item["name"] for item in body["checks"]}
    assert {
        "auth",
        "database",
        "migrations",
        "latest_signal",
        "openai_api_key",
    }.issubset(check_names)
    auth_check = next(item for item in body["checks"] if item["name"] == "auth")
    assert auth_check["details"]["auth_enabled"] is True
