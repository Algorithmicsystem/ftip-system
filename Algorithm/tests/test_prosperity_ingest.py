import datetime as dt
import os
from typing import Any, List

import pytest

from api.main import Candle, SignalResponse
from api.prosperity import ingest


def _sample_rows(n: int = 40, *, start: dt.date | None = None) -> List[tuple[dt.date, float, float]]:
    base = start or dt.date(2024, 1, 1)
    return [(base + dt.timedelta(days=i), 100.0 + i, 1_000.0 + i) for i in range(n)]


def test_compute_and_store_signal_passes_all_candles(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    as_of_date = dt.date(2024, 2, 15)
    rows = _sample_rows(n=60, start=dt.date(2023, 12, 15))
    lookback = 50

    monkeypatch.setattr("api.prosperity.ingest.db.safe_fetchall", lambda *args, **kwargs: rows)

    def _fake_execute(sql: str, params: Any) -> None:
        captured["execute_params"] = params

    monkeypatch.setattr("api.prosperity.ingest.db.safe_execute", _fake_execute)

    def _fake_compute(symbol: str, as_of: str, lookback_val: int, candles_all: List[Candle]):
        captured["compute_args"] = (symbol, as_of, lookback_val, candles_all)
        return SignalResponse(
            symbol=symbol,
            as_of=as_of,
            lookback=lookback_val,
            effective_lookback=min(lookback_val, len(candles_all)),
            regime="TRENDING",
            thresholds={"buy": 1.0, "sell": -1.0},
            score=0.5,
            signal="HOLD",
            confidence=0.5,
            features={"foo": 1.0},
        )

    monkeypatch.setattr("api.main.compute_signal_for_symbol_from_candles", _fake_compute)

    result = ingest.compute_and_store_signal("aapl", as_of_date, lookback)

    assert result["symbol"] == "AAPL"
    assert "compute_args" in captured
    symbol_arg, as_of_arg, lookback_arg, candles_arg = captured["compute_args"]
    assert symbol_arg == "AAPL"
    assert as_of_arg == as_of_date.isoformat()
    assert lookback_arg == lookback
    assert len(candles_arg) == len(rows)
    assert all(isinstance(c, Candle) for c in candles_arg)
    assert max(c.timestamp for c in candles_arg) <= as_of_arg
    assert captured["execute_params"][1] == as_of_date
@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
def test_latest_signal_endpoint_returns_inserted_row(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    os.environ.setdefault("FTIP_DB_WRITE_ENABLED", "1")
    os.environ.setdefault("FTIP_DB_READ_ENABLED", "1")

    from api import db, migrations
    from api.main import app
    from fastapi.testclient import TestClient

    migrations.ensure_schema()
    db.ensure_schema()

    symbol = f"TST{dt.datetime.utcnow().timestamp():.0f}"
    as_of = dt.date(2024, 2, 20)
    lookback = 252

    db.safe_execute(
        """
        INSERT INTO prosperity_signals_daily(
            symbol, as_of, lookback, score, signal, thresholds, regime, confidence, notes, features, meta
        ) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
        ON CONFLICT(symbol, as_of, lookback) DO NOTHING
        """,
        (
            symbol,
            as_of,
            lookback,
            1.23,
            "BUY",
            "{}",
            "TRENDING",
            0.9,
            "[]",
            "{}",
            "{}",
        ),
    )

    client = TestClient(app)
    resp = client.get(f"/prosperity/latest/signal?symbol={symbol}&lookback={lookback}")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["symbol"] == symbol
