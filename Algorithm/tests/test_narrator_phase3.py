from typing import Dict

import pytest
from fastapi.testclient import TestClient

from api import security
from api.main import app
from ftip.narrator import prompts


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    for key in [
        "FTIP_API_KEY",
        "FTIP_DB_ENABLED",
        "FTIP_DB_READ_ENABLED",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    security.reset_auth_cache()
    yield
    for key in [
        "FTIP_API_KEY",
        "FTIP_DB_ENABLED",
        "FTIP_DB_READ_ENABLED",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    security.reset_auth_cache()


def _auth_header(key: str) -> Dict[str, str]:
    return {"X-FTIP-API-Key": key}


def test_prompt_builder_safe_mode():
    context = prompts.build_context_packet(
        question="Why?",
        symbols=[{"symbol": "AAPL", "signal": {"signal": "BUY"}}],
        strategy_graph={"ensemble": {"final_signal": "BUY"}},
        meta={"as_of_date": "2024-12-31", "lookback": 252},
    )
    messages = prompts.build_ask_prompt("What is happening?", context)
    system_message = messages[0]["content"]
    combined = " ".join([msg["content"] for msg in messages])

    assert "FTIP Narrator" in system_message
    assert "financial advice" in combined.lower()
    assert "OPENAI_API_KEY" not in combined


def test_narrator_requires_key(monkeypatch):
    monkeypatch.setenv("FTIP_API_KEY", "demo")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    client = TestClient(app)

    resp = client.get("/narrator/health")
    assert resp.status_code == 401

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    resp_auth = client.get("/narrator/health", headers=_auth_header("demo"))
    assert resp_auth.status_code == 200
    assert resp_auth.json()["has_api_key"] is True


def test_narrator_missing_openai_key(monkeypatch):
    monkeypatch.setenv("FTIP_API_KEY", "demo")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    client = TestClient(app)

    monkeypatch.setattr("api.narrator.routes._build_context", lambda req: {})
    payload = {
        "question": "Hello?",
        "symbols": ["AAPL"],
        "as_of_date": "2024-12-31",
        "lookback": 252,
        "days": 5,
    }
    resp = client.post("/narrator/ask", json=payload, headers=_auth_header("demo"))
    assert resp.status_code == 503
    assert resp.json().get("trace_id")


def test_narrator_endpoints_with_mocked_llm(monkeypatch):
    monkeypatch.setenv("FTIP_API_KEY", "demo")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    monkeypatch.setattr(
        "api.narrator.routes.query.signal_as_of", lambda *_, **__: {"signal": "BUY"}
    )
    monkeypatch.setattr(
        "api.narrator.routes.query.features_as_of",
        lambda *_, **__: {"features": {"mom": 1.0}},
    )
    monkeypatch.setattr("api.narrator.routes.query.signal_history", lambda *_, **__: [])
    monkeypatch.setattr(
        "api.narrator.routes.strategy_graph_db.ensemble_as_of",
        lambda sym, lookback, as_of_date: {
            "symbol": sym,
            "as_of_date": as_of_date.isoformat(),
            "final_signal": "BUY",
        },
    )
    monkeypatch.setattr(
        "api.narrator.routes.strategy_graph_db.strategies_as_of", lambda *_, **__: []
    )
    monkeypatch.setattr(
        "api.narrator.routes.narrator_client.complete_chat",
        lambda *_, **__: (
            "Mocked answer",
            "gpt-mock",
            {"prompt_tokens": 1, "completion_tokens": 1},
        ),
    )

    client = TestClient(app)

    ask_payload = {
        "question": "What is the latest view?",
        "symbols": ["AAPL", "MSFT"],
        "as_of_date": "2024-12-31",
        "lookback": 252,
        "days": 30,
    }
    ask_resp = client.post(
        "/narrator/ask", json=ask_payload, headers=_auth_header("demo")
    )
    assert ask_resp.status_code == 200
    ask_body = ask_resp.json()
    assert ask_body["answer"].startswith("Mocked answer")
    assert ask_body["context_used"]["symbols"] == ["AAPL", "MSFT"]

    explain_payload = {
        "symbol": "AAPL",
        "as_of_date": "2024-12-31",
        "signal": {
            "action": "BUY",
            "confidence": 0.7,
            "reason_codes": ["TREND_UP"],
            "stop_loss": 90.0,
        },
        "features": {},
        "quality": {
            "sentiment_ok": True,
            "intraday_ok": False,
            "fundamentals_ok": True,
        },
        "bars": {},
        "sentiment": {"headline_count": 2},
    }
    explain_resp = client.post(
        "/narrator/explain-signal", json=explain_payload, headers=_auth_header("demo")
    )
    assert explain_resp.status_code == 200
    explain_body = explain_resp.json()
    assert explain_body["symbol"] == "AAPL"
    assert "BUY signal" in explain_body["explanation"]
    assert explain_body["trace_id"]
