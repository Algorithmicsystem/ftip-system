import datetime as dt
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api import security
from api.llm import prompts, routes
from api.main import app


@pytest.fixture(autouse=True)
def clear_openai_env(monkeypatch):
    monkeypatch.delenv("OpenAI_ftip-system", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    security._API_KEYS = None
    yield
    monkeypatch.delenv("OpenAI_ftip-system", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    security._API_KEYS = None


def test_narrator_signal_missing_key(monkeypatch):
    client = TestClient(app)
    payload = {
        "symbol": "AAPL",
        "as_of": "2024-01-31",
        "lookback": 10,
    }
    resp = client.post("/narrator/signal", json=payload)
    assert resp.status_code == 503
    assert resp.json().get("trace_id")


def test_narrator_signal_uses_disclaimer(monkeypatch):
    monkeypatch.setenv("OpenAI_ftip-system", "test-key")

    def fake_complete_chat(messages, **kwargs):
        return "Mocked narration", "model", {"prompt_tokens": 1, "completion_tokens": 1}

    def fake_resolver(symbol: str, as_of: dt.date, lookback: int, **_: Any) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "as_of": as_of.isoformat(),
            "lookback": lookback,
            "signal": "BUY",
            "score_mode": "base",
            "score": 0.5,
            "regime": "TRENDING",
            "confidence": 0.7,
            "thresholds": {"buy": 0.2, "sell": -0.2},
            "features": {"mom_21": 0.1},
            "history": [],
        }

    monkeypatch.setattr(routes, "_resolve_signal", fake_resolver)
    monkeypatch.setattr(routes.client, "complete_chat", fake_complete_chat)

    client = TestClient(app)
    resp = client.post(
        "/narrator/signal",
        json={"symbol": "AAPL", "as_of": "2024-02-01", "lookback": 20},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["disclaimer"] == prompts.DISCLAIMER
    assert data["narrative"] == "Mocked narration"


def test_narrator_ask_returns_citations(monkeypatch):
    monkeypatch.setenv("OpenAI_ftip-system", "test-key")

    def fake_complete_chat(messages, **kwargs):
        return "Answer", "model", {"prompt_tokens": 1, "completion_tokens": 1}

    def fake_resolver(symbol: str, as_of: dt.date, lookback: int, **_: Any) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "as_of": as_of.isoformat(),
            "lookback": lookback,
            "signal": "HOLD",
            "score_mode": "base",
            "score": 0.1,
            "regime": "CHOPPY",
            "confidence": 0.4,
            "thresholds": {"buy": 0.2, "sell": -0.2},
            "features": {"mom_21": 0.01, "last_close": 100.0, "volatility_ann": 0.2, "rsi14": 50.0},
            "history": [],
        }

    monkeypatch.setattr(routes, "_resolve_signal", fake_resolver)
    monkeypatch.setattr(routes.client, "complete_chat", fake_complete_chat)

    client = TestClient(app)
    resp = client.post(
        "/narrator/ask/legacy",
        json={
            "question": "Why is AAPL a BUY?",
            "symbols": ["AAPL"],
            "as_of": "2024-02-02",
            "lookback": 10,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["disclaimer"] == prompts.DISCLAIMER
    assert data["citations"]
    assert data["citations"][0]["symbol"] == "AAPL"
