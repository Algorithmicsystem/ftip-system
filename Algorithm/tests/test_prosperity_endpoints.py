import datetime as dt
import os
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from api import db
from api import migrations
from api.prosperity import ingest, query
from api.main import SignalResponse, app


def _enable_db_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "db_enabled", lambda: True)
    monkeypatch.setattr(db, "db_write_enabled", lambda: True)
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)


def _weekday_bars(end_date: dt.date, count: int) -> List[Dict[str, float]]:
    bars: List[Dict[str, float]] = []
    day = end_date
    close_price = 100.0
    while len(bars) < count:
        if day.weekday() < 5:
            bars.append(
                {
                    "date": day.isoformat(),
                    "close": close_price,
                    "volume": 1_000 + len(bars),
                }
            )
            close_price += 1.0
        day -= dt.timedelta(days=1)
    return list(reversed(bars))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    # Avoid touching a real database in unit tests
    monkeypatch.setattr(db, "ensure_schema", lambda: None)
    monkeypatch.setattr(migrations, "ensure_schema", lambda: [])
    with TestClient(app) as client:
        yield client


def test_prosperity_health(client: TestClient):
    res = client.get("/prosperity/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"


def test_latest_signal_missing_returns_404(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    _enable_db_flags(monkeypatch)
    monkeypatch.setattr(query, "latest_signal", lambda symbol, lookback: None)

    res = client.get(
        "/prosperity/latest/signal", params={"symbol": "ZZZ_MISSING", "lookback": 5}
    )
    assert res.status_code == 404
    body = res.json()
    assert "trace_id" in body
    assert body["error"]["type"] == "http_error"


def test_snapshot_run_and_latest(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    _enable_db_flags(monkeypatch)

    symbols: List[str] = ["AAPL", "MSFT"]
    features_store: Dict[str, Dict[str, str]] = {}
    signals_store: Dict[str, Dict[str, str]] = {}

    monkeypatch.setattr(ingest, "upsert_universe", lambda syms: (len(syms), syms))

    bars = _weekday_bars(dt.date(2024, 1, 5), 260)

    monkeypatch.setattr(
        "api.main.compute_signal_for_symbol_from_candles",
        lambda *args, **kwargs: SignalResponse(
            symbol="AAPL",
            as_of="2024-01-05",
            lookback=5,
            effective_lookback=5,
            regime="TRENDING",
            thresholds={"buy": 0.1, "sell": -0.1},
            score=0.5,
            base_score=0.5,
            stacked_score=0.5,
            signal="BUY",
            confidence=0.7,
            features={"mom_21": 0.1},
            notes=["Score mode: STACKED"],
            calibration_meta={"score_mode": "stacked", "base_score": 0.5},
        ),
    )

    def fake_persist(
        symbol: str,
        as_of_date: dt.date,
        lookback: int,
        feats,
        feature_meta,
        signal_payload,
        **_kwargs,
    ):
        meta = {"regime": "TEST"}
        features_store[symbol] = {
            "symbol": symbol,
            "as_of": as_of_date.isoformat(),
            "lookback": lookback,
            "features": feats,
            "meta": meta,
        }
        signals_store[symbol] = {
            "symbol": symbol,
            "as_of": as_of_date.isoformat(),
            "lookback": lookback,
            "score": signal_payload["signal_dict"].get("score"),
            "signal": signal_payload["signal_dict"].get("signal"),
            "thresholds": signal_payload["thresholds"],
            "regime": signal_payload["regime"],
            "confidence": signal_payload["confidence"],
            "notes": signal_payload["notes"],
            "meta": signal_payload["meta"],
        }
        return {"features": 1, "signals": 1, "strategies": 0, "ensembles": 0}

    monkeypatch.setattr(
        ingest, "ingest_bars", lambda *args, **kwargs: {"inserted": 1, "updated": 0}
    )
    monkeypatch.setattr(query, "fetch_bars", lambda *args, **kwargs: bars)
    monkeypatch.setattr("api.prosperity.routes._persist_symbol_outputs", fake_persist)
    monkeypatch.setattr(
        "api.prosperity.routes._log_symbol_coverage", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        query, "latest_features", lambda sym, lb: features_store.get(sym)
    )
    monkeypatch.setattr(query, "latest_signal", lambda sym, lb: signals_store.get(sym))

    payload = {
        "symbols": symbols,
        "from_date": "2023-01-01",
        "to_date": "2024-01-05",
        "as_of_date": "2024-01-05",
        "lookback": 5,
        "concurrency": 10,
    }
    res = client.post("/prosperity/snapshot/run", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["result"]["rows_written"] == {"signals": 2, "features": 2}
    assert set(body["result"]["symbols_ok"]) == set(symbols)
    assert body["requested"]["concurrency"] == 5  # clamped
    assert body["trace_id"]

    sig_res = client.get(
        "/prosperity/latest/signal", params={"symbol": "AAPL", "lookback": 5}
    )
    assert sig_res.status_code == 200
    assert sig_res.json()["signal"] == "BUY"

    feat_res = client.get(
        "/prosperity/latest/features", params={"symbol": "AAPL", "lookback": 5}
    )
    assert feat_res.status_code == 200
    assert feat_res.json()["meta"]["regime"] == "TEST"


def test_snapshot_run_with_insufficient_bars(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    _enable_db_flags(monkeypatch)

    monkeypatch.setattr(ingest, "upsert_universe", lambda syms: (len(syms), syms))
    monkeypatch.setattr(ingest, "ingest_bars", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "api.prosperity.routes._persist_symbol_outputs", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        "api.prosperity.routes._log_symbol_coverage", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        query,
        "fetch_bars",
        lambda *args, **kwargs: [
            {"date": "2024-01-02", "close": 100.0, "volume": 1_000},
        ],
    )

    payload = {
        "symbols": ["SHORT"],
        "from_date": "2024-01-01",
        "to_date": "2024-01-05",
        "as_of_date": "2024-01-05",
        "lookback": 5,
        "concurrency": 3,
    }
    res = client.post("/prosperity/snapshot/run", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "partial"
    failure = body["result"]["symbols_failed"][0]
    assert failure["reason_code"] == "INSUFFICIENT_BARS"
    assert "required=252" in failure["reason_detail"]
    assert "returned=1" in failure["reason_detail"]
    assert "window=" in failure["reason_detail"]


def test_snapshot_run_uses_latest_trading_day(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    _enable_db_flags(monkeypatch)

    monkeypatch.setattr(ingest, "upsert_universe", lambda syms: (len(syms), syms))
    monkeypatch.setattr(
        ingest, "ingest_bars", lambda *args, **kwargs: {"inserted": 1, "updated": 0}
    )
    monkeypatch.setattr(
        "api.prosperity.routes._log_symbol_coverage", lambda *args, **kwargs: None
    )

    friday = dt.date(2024, 1, 5)
    sunday = dt.date(2024, 1, 7)
    bars = _weekday_bars(friday, 260)

    monkeypatch.setattr(query, "fetch_bars", lambda *args, **kwargs: bars)
    monkeypatch.setattr(
        "api.main.compute_signal_for_symbol_from_candles",
        lambda *args, **kwargs: SignalResponse(
            symbol="AAPL",
            as_of=friday.isoformat(),
            lookback=5,
            effective_lookback=5,
            regime="TRENDING",
            thresholds={"buy": 0.1, "sell": -0.1},
            score=0.5,
            base_score=0.5,
            stacked_score=0.5,
            signal="BUY",
            confidence=0.7,
            features={"mom_21": 0.1},
            notes=["Score mode: STACKED"],
            calibration_meta={"score_mode": "stacked", "base_score": 0.5},
        ),
    )

    captured_as_of: Dict[str, dt.date] = {}

    def fake_persist(symbol: str, as_of_date: dt.date, *args, **kwargs):
        captured_as_of[symbol] = as_of_date
        return {"features": 1, "signals": 1, "strategies": 0, "ensembles": 0}

    monkeypatch.setattr("api.prosperity.routes._persist_symbol_outputs", fake_persist)

    payload = {
        "symbols": ["AAPL"],
        "from_date": "2023-01-01",
        "to_date": sunday.isoformat(),
        "as_of_date": sunday.isoformat(),
        "lookback": 5,
        "concurrency": 3,
    }
    res = client.post("/prosperity/snapshot/run", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert captured_as_of["AAPL"] == friday


def test_latest_endpoints_return_404(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
):
    _enable_db_flags(monkeypatch)
    monkeypatch.setattr(query, "latest_signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(query, "latest_features", lambda *_args, **_kwargs: None)

    sig_res = client.get(
        "/prosperity/latest/signal", params={"symbol": "MSFT", "lookback": 10}
    )
    feat_res = client.get(
        "/prosperity/latest/features", params={"symbol": "MSFT", "lookback": 10}
    )

    assert sig_res.status_code == 404
    assert feat_res.status_code == 404


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
def test_snapshot_run_round_trip_with_db(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    os.environ.setdefault("FTIP_DB_WRITE_ENABLED", "1")
    os.environ.setdefault("FTIP_DB_READ_ENABLED", "1")

    from api import db, migrations
    from api.main import app

    migrations.ensure_schema()
    db.ensure_schema()

    symbol = f"APX{dt.datetime.utcnow().timestamp():.0f}"[:8]
    as_of = dt.date(2024, 2, 5)
    lookback = 10
    start_date = as_of - dt.timedelta(days=300)

    db.safe_execute("DELETE FROM prosperity_daily_bars WHERE symbol=%s", (symbol,))
    db.safe_execute("DELETE FROM prosperity_signals_daily WHERE symbol=%s", (symbol,))
    db.safe_execute("DELETE FROM prosperity_features_daily WHERE symbol=%s", (symbol,))

    def _fake_bars(sym: str, from_date: dt.date, to_date: dt.date, **_kwargs):
        for day in [
            from_date + dt.timedelta(days=i)
            for i in range((to_date - from_date).days + 1)
        ]:
            db.safe_execute(
                """
                INSERT INTO prosperity_daily_bars(symbol, date, close, volume, source)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT(symbol, date) DO UPDATE SET close=EXCLUDED.close, volume=EXCLUDED.volume
                """,
                (sym, day, 100.0, 10_000, "test"),
            )
        return {"inserted": 1, "updated": 0}

    def _fake_features(sym: str, as_of_date: dt.date, lookback_val: int):
        db.safe_execute(
            """
            INSERT INTO prosperity_features_daily(symbol, as_of, lookback, features, meta)
            VALUES (%s,%s,%s,%s::jsonb,%s::jsonb)
            ON CONFLICT(symbol, as_of, lookback) DO UPDATE SET features=EXCLUDED.features, meta=EXCLUDED.meta
            """,
            (sym, as_of_date, lookback_val, "{}", "{}"),
        )
        return {
            "symbol": sym,
            "as_of": as_of_date.isoformat(),
            "lookback": lookback_val,
            "features": {},
            "regime": None,
            "stored": True,
            "meta": {},
        }

    def _fake_signal(sym: str, as_of_date: dt.date, lookback_val: int):
        return ingest.compute_and_store_signal(sym, as_of_date, lookback_val)

    monkeypatch.setattr(ingest, "ingest_bars", _fake_bars)
    monkeypatch.setattr(ingest, "compute_and_store_features", _fake_features)
    monkeypatch.setattr(
        "api.main.compute_signal_for_symbol_from_candles",
        lambda *args, **kwargs: SignalResponse(
            symbol=kwargs.get("symbol") or args[0],
            as_of=kwargs.get("as_of") or args[1],
            lookback=kwargs.get("lookback_val") or args[2],
            effective_lookback=lookback,
            regime="TRENDING",
            thresholds={"buy": 1.0, "sell": -1.0},
            score=0.4,
            base_score=0.4,
            stacked_score=0.4,
            signal="BUY",
            confidence=0.6,
            features={"foo": 1.0},
            notes=["Score mode: STACKED"],
            calibration_meta={"score_mode": "stacked", "base_score": 0.4},
        ),
    )

    with TestClient(app) as client:
        res = client.post(
            "/prosperity/snapshot/run",
            json={
                "symbols": [symbol],
                "from_date": start_date.isoformat(),
                "to_date": as_of.isoformat(),
                "as_of_date": as_of.isoformat(),
                "lookback": lookback,
                "concurrency": 2,
            },
        )

        assert res.status_code == 200
        sig_res = client.get(
            "/prosperity/latest/signal", params={"symbol": symbol, "lookback": lookback}
        )
        assert sig_res.status_code == 200
        assert sig_res.json().get("score_mode") == "stacked"
