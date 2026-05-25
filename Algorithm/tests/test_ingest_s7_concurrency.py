"""Session 7: httpx migration and ingest_bars_bulk concurrency."""
from __future__ import annotations

import datetime as dt
import threading
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from api.prosperity import ingest


# ---------------------------------------------------------------------------
# ingest_bars_bulk: concurrency behavior
# ---------------------------------------------------------------------------

def _make_fake_ingest(results: Dict[str, Any], barrier: threading.Barrier | None = None):
    """Return an ingest_bars stub that records peak concurrency."""

    lock = threading.Lock()
    state = {"active": 0, "peak": 0}

    def _fake(sym, from_date, to_date, *, force_refresh=False):
        with lock:
            state["active"] += 1
            if state["active"] > state["peak"]:
                state["peak"] = state["active"]
        if barrier is not None:
            barrier.wait()
        with lock:
            state["active"] -= 1
        outcome = results.get(sym)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome or {}

    return _fake, state


def test_ingest_bars_bulk_respects_concurrency_limit():
    n = 6
    concurrency = 3
    barrier = threading.Barrier(concurrency)
    fake, state = _make_fake_ingest({sym: {} for sym in [f"SYM{i}" for i in range(n)]}, barrier)

    with patch.object(ingest, "ingest_bars", fake):
        result = ingest.ingest_bars_bulk(
            [f"SYM{i}" for i in range(n)],
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 31),
            concurrency=concurrency,
        )

    assert result["ok"] == n
    assert result["errors"] == {}
    assert state["peak"] == concurrency


def test_ingest_bars_bulk_collects_errors_without_stopping():
    symbols = ["AAPL", "FAIL1", "MSFT", "FAIL2"]
    outcomes = {
        "AAPL": {},
        "FAIL1": RuntimeError("provider down"),
        "MSFT": {},
        "FAIL2": ValueError("bad symbol"),
    }
    fake, _ = _make_fake_ingest(outcomes)

    with patch.object(ingest, "ingest_bars", fake):
        result = ingest.ingest_bars_bulk(
            symbols,
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 31),
            concurrency=2,
        )

    assert result["ok"] == 2
    assert set(result["errors"].keys()) == {"FAIL1", "FAIL2"}
    assert "provider down" in result["errors"]["FAIL1"]


def test_ingest_bars_bulk_single_symbol_runs_without_error():
    fake, _ = _make_fake_ingest({"AAPL": {}})

    with patch.object(ingest, "ingest_bars", fake):
        result = ingest.ingest_bars_bulk(
            ["AAPL"],
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 31),
        )

    assert result["ok"] == 1
    assert result["errors"] == {}


def test_ingest_bars_bulk_empty_symbols_returns_zero():
    result = ingest.ingest_bars_bulk([], dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert result["ok"] == 0
    assert result["errors"] == {}


def test_ingest_bars_bulk_passes_force_refresh():
    called_with: Dict[str, Any] = {}

    def _fake(sym, from_date, to_date, *, force_refresh=False):
        called_with[sym] = force_refresh

    with patch.object(ingest, "ingest_bars", _fake):
        ingest.ingest_bars_bulk(
            ["AAPL"],
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 31),
            force_refresh=True,
        )

    assert called_with["AAPL"] is True


# ---------------------------------------------------------------------------
# httpx migration: data providers import httpx not requests
# ---------------------------------------------------------------------------

def test_providers_use_httpx_not_requests():
    import importlib, types
    provider_modules = [
        "api.data_providers.alphavantage",
        "api.data_providers.bars",
        "api.data_providers.finnhub",
        "api.data_providers.fred",
        "api.data_providers.gnews",
        "api.data_providers.newsapi",
        "api.data_providers.gdelt",
        "api.data_providers.world_bank",
        "api.data_providers.sec_edgar",
        "api.data_providers.news",
    ]
    for mod_name in provider_modules:
        mod = importlib.import_module(mod_name)
        module_vars = vars(mod)
        assert "httpx" in module_vars or any(
            getattr(v, "__name__", None) == "httpx"
            for v in module_vars.values()
            if isinstance(v, types.ModuleType)
        ), f"{mod_name} should import httpx"
        assert "requests" not in module_vars or not isinstance(
            module_vars.get("requests"), types.ModuleType
        ), f"{mod_name} still has 'requests' module in namespace"


def test_httpx_get_called_not_requests(monkeypatch):
    """alphavantage._request uses httpx.get (smoke-check the substitution)."""
    import httpx
    from api.data_providers import alphavantage
    from api.data_providers.errors import ProviderUnavailable

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"GlobalQuote": {"05. price": "150.0"}}

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    with patch.object(httpx, "get", return_value=mock_response) as mock_get:
        result = alphavantage._request({"function": "GLOBAL_QUOTE", "symbol": "AAPL"})
        mock_get.assert_called_once()
        assert result == mock_response.json.return_value
