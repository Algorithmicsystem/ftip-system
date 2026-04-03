import datetime as dt
import math
from decimal import Decimal

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


def test_assistant_analyze_sanitizes_non_finite_numeric_values(monkeypatch):
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
            "score": Decimal("0.7"),
            "confidence": Decimal("Infinity"),
            "entry_low": 100.0,
            "entry_high": 110.0,
            "stop_loss": 90.0,
            "take_profit_1": 130.0,
            "take_profit_2": 150.0,
            "horizon_days": 10,
            "reason_codes": ["MOMO_UP"],
            "reason_details": {"MOMO_UP": "Momentum rising"},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_key_features",
        lambda *_args, **_kwargs: {
            "ret_5d": Decimal("-Infinity"),
            "finite_decimal": Decimal("1.25"),
            "vol_21d": Decimal("0.3"),
            "nested": {
                "feature": float("nan"),
                "decimal_feature": Decimal("NaN"),
                "items": [Decimal("1.5"), float("-inf"), {"z": Decimal("Infinity")}],
            },
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_quality",
        lambda *_args, **_kwargs: {
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "risk": {"drawdown": float("inf"), "sharpe": Decimal("1.2")},
            "trace": [0.1, math.nan, {"z": Decimal("NaN"), "w": math.inf}],
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
    assert data["signal"]["score"] == 0.7
    assert data["signal"]["confidence"] is None
    assert data["signal"]["entry_low"] == 100.0
    assert data["key_features"]["ret_5d"] is None
    assert data["key_features"]["vol_21d"] == 0.3
    assert data["key_features"]["finite_decimal"] == 1.25
    assert data["key_features"]["nested"]["feature"] is None
    assert data["key_features"]["nested"]["decimal_feature"] is None
    assert data["key_features"]["nested"]["items"] == [1.5, None, {"z": None}]
    assert data["quality"]["risk"]["drawdown"] is None
    assert data["quality"]["risk"]["sharpe"] == 1.2
    assert data["quality"]["trace"] == [0.1, None, {"z": None, "w": None}]


def test_fetch_signal_falls_back_to_prosperity_row(monkeypatch):
    from api.assistant import orchestrator

    calls = {"count": 0}

    def _fake_fetchone(query, params):
        calls["count"] += 1
        if "FROM signals_daily" in query:
            return None
        if "FROM prosperity_signals_daily" in query:
            assert "as_of = %s" in query
            assert params == ("AAPL", dt.date(2024, 1, 2))
            return ("BUY", 0.81, 0.64)
        raise AssertionError(f"unexpected query: {query}")

    monkeypatch.setattr(orchestrator.db, "safe_fetchone", _fake_fetchone)

    signal = orchestrator.fetch_signal("AAPL", dt.date(2024, 1, 2))

    assert calls["count"] == 2
    assert signal == {
        "action": "BUY",
        "score": 0.81,
        "confidence": 0.64,
        "entry_low": None,
        "entry_high": None,
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "horizon_days": None,
        "reason_codes": [],
        "reason_details": {},
    }


def test_fetch_signal_falls_back_to_prosperity_row_with_as_of_date(monkeypatch):
    from api.assistant import orchestrator

    orchestrator._PROSPERITY_SIGNAL_ASOF_COLUMN = None
    orchestrator._PROSPERITY_SIGNAL_ACTION_COLUMN = None

    def _fake_fetchone(query, params=None):
        if "FROM signals_daily" in query:
            return None
        if "FROM information_schema.columns" in query and "as_of_date" in query:
            return ("as_of_date",)
        if "FROM information_schema.columns" in query and "signal', 'action" in query:
            return ("signal",)
        if "FROM prosperity_signals_daily" in query:
            assert "as_of_date = %s" in query
            return ("SELL", -0.22, 0.51)
        raise AssertionError(f"unexpected query: {query}")

    monkeypatch.setattr(orchestrator.db, "safe_fetchone", _fake_fetchone)

    signal = orchestrator.fetch_signal("AAPL", dt.date(2024, 1, 2))

    assert signal is not None
    assert signal["action"] == "SELL"
    assert signal["score"] == -0.22
    assert signal["confidence"] == 0.51
