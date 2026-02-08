import datetime as dt

from fastapi.testclient import TestClient

from api.main import app


def test_assistant_analyze_returns_schema(monkeypatch):
    from api.assistant import orchestrator

    async def _fake_freshness(symbol: str, refresh: bool = True):
        return {
            "as_of_date": dt.date(2024, 1, 2),
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "bars_updated_at": "2024-01-02T00:00:00Z",
            "news_updated_at": "2024-01-02T00:00:00Z",
            "sentiment_updated_at": "2024-01-02T00:00:00Z",
            "warnings": [],
        }

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(orchestrator, "ensure_freshness", _fake_freshness)
    monkeypatch.setattr(orchestrator, "run_features", _noop)
    monkeypatch.setattr(orchestrator, "run_signals", _noop)
    monkeypatch.setattr(
        orchestrator,
        "fetch_signal",
        lambda *_args, **_kwargs: {
            "action": "BUY",
            "score": 0.7,
            "confidence": 0.6,
            "entry_low": 100,
            "entry_high": 110,
            "stop_loss": 90,
            "take_profit_1": 130,
            "take_profit_2": 150,
            "horizon_days": 10,
            "reason_codes": ["MOMO_UP"],
            "reason_details": {"MOMO_UP": "Momentum rising"},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_key_features",
        lambda *_args, **_kwargs: {"ret_5d": 0.12, "vol_21d": 0.3},
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_quality",
        lambda *_args, **_kwargs: {
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "warnings": [],
        },
    )

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/analyze",
            json={"symbol": "NVDA", "horizon": "swing", "risk_mode": "balanced"},
        )
        assert resp.status_code == 200
        data = resp.json()

    assert set(data.keys()) == {
        "symbol",
        "as_of_date",
        "signal",
        "key_features",
        "quality",
        "evidence",
    }
    assert data["signal"]["action"] == "BUY"


def test_assistant_top_picks_schema(monkeypatch):
    from api.assistant import orchestrator

    monkeypatch.setattr(
        orchestrator,
        "fetch_top_picks",
        lambda limit: (
            dt.date(2024, 1, 2),
            [
                {
                    "symbol": "NVDA",
                    "direction": "long",
                    "score": 0.7,
                    "confidence": 0.6,
                    "reason_codes": ["MOMO_UP"],
                }
            ],
        ),
    )
    monkeypatch.setattr(
        orchestrator, "universe_coverage", lambda *_args, **_kwargs: 0.95
    )

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/top-picks",
            json={
                "universe": "sp500",
                "horizon": "swing",
                "risk_mode": "balanced",
                "limit": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

    assert "picks" in data
    assert data["picks"][0]["symbol"] == "NVDA"
