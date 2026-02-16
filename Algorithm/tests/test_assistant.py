from typing import Any

from fastapi.testclient import TestClient

from api.assistant import service
from api.assistant.storage import AssistantStorage
from api.main import app


class DummySignal:
    def __init__(self, symbol: str = "AAPL"):
        self.symbol = symbol

    def model_dump(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": "2024-01-01",
            "lookback": 10,
            "signal": "BUY",
            "score": 1.0,
            "confidence": 0.8,
            "thresholds": {"buy": 0.5},
            "notes": ["test"],
        }


class DummyBacktest:
    def model_dump(self) -> dict[str, Any]:
        return {
            "total_return": 0.1,
            "sharpe": 1.0,
            "max_drawdown": -0.05,
            "volatility": 0.2,
            "lookback": 252,
        }


def test_chat_returns_503_when_disabled(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post("/assistant/chat", json={"message": "hi"})
    assert resp.status_code == 503


def test_chat_missing_api_key(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post("/assistant/chat", json={"message": "hi"})
    assert resp.status_code == 500
    assert "LLM API key not configured" in resp.json()["error"]["message"]


def test_storage_memory_roundtrip():
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session(metadata={"foo": "bar"})
    store.add_message(session_id, "user", "hello")
    store.upsert_session_metadata(session_id, {"title": "Test"})

    session = store.get_session(session_id)
    assert session is not None
    assert session["metadata"].get("foo") == "bar"
    assert session["metadata"].get("title") == "Test"

    messages = store.get_messages(session_id)
    assert len(messages) == 1
    assert messages[0]["content"] == "hello"


def test_explain_signal_mocked(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    monkeypatch.setattr(
        service,
        "_safe_completion",
        lambda messages: (
            "mocked reply",
            "model",
            {"prompt_tokens": 1, "completion_tokens": 1},
        ),
    )
    result = service.explain_signal(
        {"symbol": "AAPL", "as_of": "2024-01-01", "lookback": 10},
        signal_fetcher=lambda symbol, as_of, lookback: DummySignal(symbol),
        store=AssistantStorage(use_memory=True),
    )
    assert result["reply"] == "mocked reply"


def test_explain_backtest_mocked(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    monkeypatch.setattr(
        service,
        "_safe_completion",
        lambda messages: (
            "backtest reply",
            "model",
            {"prompt_tokens": 1, "completion_tokens": 1},
        ),
    )
    result = service.explain_backtest(
        {
            "symbols": ["AAPL"],
            "from_date": "2023-01-01",
            "to_date": "2023-12-31",
            "lookback": 252,
            "rebalance_every": 21,
            "trading_cost_bps": 10.0,
            "slippage_bps": 5.0,
            "max_weight": None,
            "min_trade_delta": 0.0005,
            "max_turnover_per_rebalance": 0.25,
            "allow_shorts": False,
        },
        backtest_runner=lambda req: DummyBacktest(),
        store=AssistantStorage(use_memory=True),
    )
    assert result["reply"] == "backtest reply"


def test_providers_health_is_registered_and_in_openapi() -> None:
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    r = client.get("/providers/health")
    assert r.status_code == 200
    data = r.json()
    assert "providers" in data
    for k in ("openai", "massive", "finnhub", "fred", "secedgar"):
        assert k in data["providers"]
    openapi = client.get("/openapi.json").json()
    assert "/providers/health" in openapi.get("paths", {})
