import os
import datetime as dt
import uuid

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from api.main import app
from api import db
from api.main import Candle, compute_signal_for_symbol_from_candles

pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def test_run_snapshot_route_registered() -> None:
    client = TestClient(app)
    paths = [route.path for route in client.app.routes]
    assert "/db/run_snapshot" in paths


def _sample_candles(n: int = 120) -> list[Candle]:
    today = dt.date.today()
    candles: list[Candle] = []
    for i in range(n):
        candles.append(
            Candle(
                timestamp=(today - dt.timedelta(days=n - i)).isoformat(),
                close=100 + i * 0.5,
                volume=1_000 + i,
            )
        )
    return candles


def _patched_signal(monkeypatch: pytest.MonkeyPatch, candles: list[Candle]) -> None:
    def _fake_compute(symbol: str, as_of: str, lookback: int):
        return compute_signal_for_symbol_from_candles(symbol, as_of, lookback, candles)

    monkeypatch.setattr("api.main.compute_signal_for_symbol", _fake_compute)


def test_db_save_signal_persists_json_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    db.ensure_schema()

    candles = _sample_candles()
    as_of = candles[-1].timestamp
    lookback = 60
    symbol = f"TST{uuid.uuid4().hex[:5]}"
    _patched_signal(monkeypatch, candles)

    client = TestClient(app)
    resp = client.post(
        "/db/save_signal", json={"symbol": symbol, "as_of": as_of, "lookback": lookback}
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["id"] is not None

    row = db.fetch1(
        """
        SELECT thresholds, features, notes, calibration_meta, raw_signal_payload
        FROM signals
        WHERE id=%s
        """,
        (payload["id"],),
    )

    assert isinstance(row[0], dict)
    assert isinstance(row[1], dict)
    assert isinstance(row[2], list)
    assert row[3] is None or isinstance(row[3], dict)
    assert isinstance(row[4], dict)
    assert row[4].get("symbol") == symbol.upper()


def test_db_run_snapshot_handles_symbol_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    db.ensure_schema()

    candles = _sample_candles()
    as_of = candles[-1].timestamp
    lookback = 45
    good_symbol = f"OK{uuid.uuid4().hex[:4]}"
    bad_symbol = f"BAD{uuid.uuid4().hex[:4]}"

    def _fake_compute(symbol: str, as_of_date: str, lookback_val: int):
        if symbol.upper() == bad_symbol.upper():
            raise HTTPException(status_code=500, detail="boom")
        return compute_signal_for_symbol_from_candles(
            symbol, as_of_date, lookback_val, candles
        )

    monkeypatch.setattr("api.main.compute_signal_for_symbol", _fake_compute)

    db.upsert_universe([good_symbol, bad_symbol], source="test_case")

    client = TestClient(app)
    resp = client.post(
        "/db/run_snapshot",
        json={"as_of": as_of, "lookback": lookback, "active_only": True, "limit": 10},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["saved_count"] >= 1
    assert data["error_count"] == 1
    assert bad_symbol in data["errors"]

    saved_rows = db.fetchall(
        "SELECT symbol, raw_signal_payload FROM signals WHERE as_of=%s AND lookback=%s AND symbol=%s",
        (as_of, lookback, good_symbol.upper()),
    )
    assert len(saved_rows) >= 1
    assert isinstance(saved_rows[0][1], dict)
