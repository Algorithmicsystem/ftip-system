import datetime as dt
from types import SimpleNamespace

from typing import Any

import api.db as db
from api.prosperity import ingest


class DummyCandle:
    def __init__(self, timestamp: str, close: float, volume: float | None = None):
        self.timestamp = timestamp
        self.close = close
        self.volume = volume


def test_apply_migrations_safe_when_disabled(monkeypatch):
    monkeypatch.setattr(db.config, "db_enabled", lambda: False)
    # Should no-op without raising
    db.apply_migrations()


def test_hash_helper_stable():
    payload = {"a": 1, "b": 2}
    h1 = ingest._hash_dict(payload)
    h2 = ingest._hash_dict(payload)
    assert h1 == h2


def test_compute_features_uses_as_of(monkeypatch):
    calls: dict[str, Any] = {}
    recorded_sql: list[str] = []

    def fake_fetch(sql, params):
        # return three rows; as_of should clamp to the last entry
        return [
            (dt.date(2024, 1, 1), 1.0, 10.0),
            (dt.date(2024, 1, 2), 2.0, 11.0),
            (dt.date(2024, 1, 3), 3.0, 12.0),
        ]

    monkeypatch.setattr(db, "safe_fetchall", fake_fetch)

    def fake_compute(candles):
        calls["candles"] = candles
        return {"mom_5": 0.0, "mom_21": 0.0, "mom_63": 0.0, "trend_sma20_50": 0.0, "volatility_ann": 0.0, "rsi14": 50.0, "volume_z20": 0.0, "last_close": candles[-1].close}

    def fake_regime(_):
        return "TEST"

    monkeypatch.setattr("api.main.compute_features", fake_compute)
    monkeypatch.setattr("api.main.detect_regime", fake_regime)

    def _record_sql(sql: str, params=None):
        recorded_sql.append(sql)

    monkeypatch.setattr(db, "safe_execute", _record_sql)

    res = ingest.compute_and_store_features("AAPL", dt.date(2024, 1, 3), 2)
    assert res["stored"] is True
    assert calls["candles"][-1].timestamp == "2024-01-03"
    assert any("as_of" in stmt for stmt in recorded_sql)


def test_ingest_identifies_missing(monkeypatch):
    recorded = []

    def fake_fetch(sql, params):
        return []

    def fake_execute(sql, params=None):
        recorded.append((sql, params))

    def fake_massive(symbol, f, t):
        return [DummyCandle("2024-01-01", 10.0, 100), DummyCandle("2024-01-02", 11.0, 110)]

    monkeypatch.setattr(db, "safe_fetchall", fake_fetch)
    monkeypatch.setattr(db, "safe_execute", fake_execute)
    monkeypatch.setattr("api.main.massive_fetch_daily_bars", fake_massive)

    out = ingest.ingest_bars("AAPL", dt.date(2024, 1, 1), dt.date(2024, 1, 2))
    assert out["inserted"] == 2
    assert len(recorded) >= 1
