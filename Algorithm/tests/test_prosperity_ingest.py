import datetime as dt
import os
from typing import Any, List, Optional, Tuple

import pytest

from api.main import Candle, SignalResponse
from api.prosperity import ingest


def _sample_rows(
    n: int = 40, *, start: Optional[dt.date] = None
) -> List[Tuple[dt.date, float, float]]:
    base = start or dt.date(2024, 1, 1)
    return [(base + dt.timedelta(days=i), 100.0 + i, 1_000.0 + i) for i in range(n)]


def test_compute_and_store_signal_passes_all_candles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    as_of_date = dt.date(2024, 2, 15)
    rows = _sample_rows(n=60, start=dt.date(2023, 12, 15))
    lookback = 50

    monkeypatch.setattr(
        "api.prosperity.ingest.db.safe_fetchall", lambda *args, **kwargs: rows
    )

    def _fake_execute(sql: str, params: Any) -> None:
        captured["execute_params"] = params

    monkeypatch.setattr("api.prosperity.ingest.db.safe_execute", _fake_execute)

    def _fake_compute(
        symbol: str, as_of: str, lookback_val: int, candles_all: List[Candle]
    ):
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

    monkeypatch.setattr(
        "api.main.compute_signal_for_symbol_from_candles", _fake_compute
    )

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
    params = captured["execute_params"]
    assert params[1] == as_of_date
    assert params[3] == "stacked"  # inferred fallback mode
    assert params[5] == params[4]


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
def test_compute_and_store_signal_persists_score_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    os.environ.setdefault("FTIP_DB_WRITE_ENABLED", "1")
    os.environ.setdefault("FTIP_DB_READ_ENABLED", "1")

    from api import db, migrations
    from api.main import _score_mode

    migrations.ensure_schema()
    db.ensure_schema()

    symbol = f"TST{dt.datetime.utcnow().timestamp():.0f}"[:12]
    as_of = dt.date.today()
    lookback = 30

    db.safe_execute("DELETE FROM prosperity_signals_daily WHERE symbol=%s", (symbol,))
    db.safe_execute("DELETE FROM prosperity_daily_bars WHERE symbol=%s", (symbol,))

    sample_rows = _sample_rows(
        n=lookback + 5, start=as_of - dt.timedelta(days=lookback + 10)
    )
    for day, close, volume in sample_rows:
        db.safe_execute(
            """
            INSERT INTO prosperity_daily_bars(symbol, date, close, volume, source)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT(symbol, date) DO UPDATE SET close=EXCLUDED.close, volume=EXCLUDED.volume
            """,
            (symbol, day, close, volume, "test"),
        )

    def _fake_signal(*_args, **_kwargs):
        return SignalResponse(
            symbol=symbol,
            as_of=as_of.isoformat(),
            lookback=lookback,
            effective_lookback=lookback,
            regime="TRENDING",
            thresholds={"buy": 1.0, "sell": -1.0},
            score=1.5,
            base_score=None,
            stacked_score=2.5,
            signal="BUY",
            confidence=0.8,
            features={"foo": 1.0},
            notes=["Score mode: STACKED"],
            calibration_meta={"score_mode": "stacked", "base_score": 1.5},
        )

    monkeypatch.setattr("api.main.compute_signal_for_symbol_from_candles", _fake_signal)
    monkeypatch.setattr("api.main._score_mode", lambda: _score_mode() or "stacked")

    ingest.compute_and_store_signal(symbol, as_of, lookback)

    row = db.safe_fetchone(
        "SELECT score_mode, base_score, stacked_score FROM prosperity_signals_daily WHERE symbol=%s AND as_of=%s AND lookback=%s",
        (symbol, as_of, lookback),
    )
    assert row is not None
    assert row[0] == "stacked"
    assert row[1] == 1.5
    assert row[2] == 2.5


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
def test_latest_signal_endpoint_returns_inserted_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            symbol, as_of, lookback, score_mode, score, base_score, stacked_score, signal, thresholds, regime, confidence, notes, features, calibration_meta, meta, signal_hash
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s)
        ON CONFLICT(symbol, as_of, lookback) DO NOTHING
        """,
        (
            symbol,
            as_of,
            lookback,
            "stacked",
            1.23,
            1.23,
            2.0,
            "BUY",
            "{}",
            "TRENDING",
            0.9,
            "[]",
            "{}",
            "{}",
            None,
            "hash1",
        ),
    )

    client = TestClient(app)
    resp = client.get(f"/prosperity/latest/signal?symbol={symbol}&lookback={lookback}")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["symbol"] == symbol
