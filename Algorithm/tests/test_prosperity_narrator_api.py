import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in [
        "FTIP_LLM_ENABLED",
        "FTIP_DB_ENABLED",
        "FTIP_DB_READ_ENABLED",
        "OpenAI_ftip-system",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield
    for key in [
        "FTIP_LLM_ENABLED",
        "FTIP_DB_ENABLED",
        "FTIP_DB_READ_ENABLED",
        "OpenAI_ftip-system",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_health_reports_flags(monkeypatch):
    client = TestClient(app)
    resp = client.get("/prosperity/narrator/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_enabled"] is False
    assert body["has_api_key"] is False
    assert body["trace_id"]


def test_explain_not_found_when_missing(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    import api.prosperity.narrator as narrator

    monkeypatch.setattr(narrator, "ensemble_as_of", lambda *_, **__: None)

    client = TestClient(app)
    resp = client.get(
        "/prosperity/narrator/explain",
        params={"symbol": "AAPL", "as_of_date": "2024-01-31", "lookback": 252},
    )
    assert resp.status_code == 404
    assert resp.json().get("trace_id")


def test_explain_ok_when_context_present(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    import api.prosperity.narrator as narrator

    ensemble = {
        "symbol": "AAPL",
        "as_of_date": "2024-01-31",
        "lookback": 252,
        "regime": "TRENDING",
        "final_signal": "BUY",
        "final_score": 0.6,
        "final_confidence": 0.7,
        "thresholds": {"buy": 0.2, "sell": -0.2},
        "risk_overlay_applied": False,
        "strategies_used": [{"strategy_id": "mom", "weight": 0.7}],
    }
    strategies = [
        {
            "strategy_id": "mom",
            "signal": "BUY",
            "confidence": 0.7,
            "normalized_score": 0.6,
            "rationale": ["momentum positive"],
            "as_of_date": "2024-01-31",
        }
    ]

    monkeypatch.setattr(narrator, "ensemble_as_of", lambda *_, **__: ensemble)
    monkeypatch.setattr(narrator, "strategies_as_of", lambda *_, **__: strategies)
    monkeypatch.setattr(
        narrator.query, "features_as_of", lambda *_, **__: {"features": {"mom": 1.0}}
    )
    monkeypatch.setattr(
        narrator.query,
        "signal_as_of",
        lambda *_, **__: {"signal": "BUY", "score": 0.5, "as_of": "2024-01-31"},
    )
    monkeypatch.setattr(
        narrator.llm_client, "complete_chat", lambda *_, **__: ("Narration", "gpt", {})
    )

    client = TestClient(app)
    resp = client.get(
        "/prosperity/narrator/explain",
        params={"symbol": "AAPL", "as_of_date": "2024-01-31", "lookback": 252},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "AAPL"
    assert body["narration"] == "Narration"
    assert body["grounding"]["ensemble_fields"]["final_signal"] == "BUY"
    assert body["top_strategies"]
    assert body["trace_id"]
