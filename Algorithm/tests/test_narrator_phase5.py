import datetime as dt
from typing import Dict

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api import security
from api.llm import routes as narrator_routes
from api.main import app


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
    monkeypatch.setenv("FTIP_API_KEY", "demo")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    security.reset_auth_cache()


def _assert_perf_numbers(perf: Dict[str, float]):
    assert set(perf.keys()) == {"return", "sharpe", "max_drawdown", "turnover"}
    for value in perf.values():
        assert value is not None
        assert isinstance(value, (int, float))


def test_portfolio_no_backtest_returns_defaults(monkeypatch):
    _prepare_env(monkeypatch)
    monkeypatch.setattr(
        narrator_routes.client,
        "complete_chat",
        lambda *_, **__: ("Portfolio summary", "mock-model", {}),
    )

    client = TestClient(app)
    payload = {
        "symbols": ["AAPL", "MSFT"],
        "from_date": dt.date(2024, 1, 1).isoformat(),
        "to_date": dt.date(2024, 6, 30).isoformat(),
        "lookback": 252,
        "include_backtest": False,
    }
    resp = client.post("/narrator/portfolio", json=payload, headers={"X-FTIP-API-Key": "demo"})
    assert resp.status_code == 200
    body = resp.json()
    _assert_perf_numbers(body["performance"])


def test_portfolio_backtest_coerces_missing_stats(monkeypatch):
    _prepare_env(monkeypatch)
    monkeypatch.setattr(
        narrator_routes.client,
        "complete_chat",
        lambda *_, **__: ("Portfolio with backtest", "mock-model", {}),
    )

    class DummyBacktest:
        def __init__(self):
            self.total_return = None
            self.sharpe = 1.5
            self.max_drawdown = None
            self.turnover = "3.0"
            self.audit = None

    monkeypatch.setattr(api_main, "backtest_portfolio", lambda *_args, **_kwargs: DummyBacktest())

    client = TestClient(app)
    payload = {
        "symbols": ["AAPL"],
        "from_date": dt.date(2023, 1, 1).isoformat(),
        "to_date": dt.date(2024, 1, 31).isoformat(),
        "lookback": 252,
        "include_backtest": True,
    }
    resp = client.post("/narrator/portfolio", json=payload, headers={"X-FTIP-API-Key": "demo"})
    assert resp.status_code == 200
    body = resp.json()
    _assert_perf_numbers(body["performance"])
    assert body["performance"]["sharpe"] == 1.5
    assert body["performance"]["turnover"] == 3.0
