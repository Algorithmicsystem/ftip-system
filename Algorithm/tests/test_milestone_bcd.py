import datetime as dt
import os
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from api import migrations, security
from api.feature_engine import compute_daily_features
from api.main import app
from api.signal_engine import compute_daily_signal


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
def test_new_migrations_apply_and_tables_exist():
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    migrations.ensure_schema()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name IN (
                    'market_symbols', 'market_bars_daily', 'market_bars_intraday',
                    'fundamentals_quarterly', 'news_raw', 'sentiment_daily', 'quality_daily',
                    'features_daily', 'features_intraday', 'signals_daily', 'signals_intraday'
                )
                """
            )
            tables = {row[0] for row in cur.fetchall()}

    assert "market_symbols" in tables
    assert "signals_daily" in tables


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
def test_unique_indexes_for_upserts():
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    migrations.ensure_schema()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT conname, conrelid::regclass::text
                FROM pg_constraint
                WHERE contype = 'u'
                """
            )
            uniques = {(row[0], row[1]) for row in cur.fetchall()}

    assert any("market_bars_daily" in row[1] for row in uniques)
    assert any("signals_daily" in row[1] for row in uniques)


def test_feature_computation_expected_columns():
    base_date = dt.date(2024, 1, 1)
    bars = []
    for i in range(70):
        bars.append(
            {
                "symbol": "AAPL",
                "as_of_date": base_date + dt.timedelta(days=i),
                "open": 100 + i,
                "high": 101 + i,
                "low": 99 + i,
                "close": 100 + i,
                "volume": 1_000_000 + i,
            }
        )
    features = compute_daily_features(bars, sentiment_score=0.1, sentiment_mean=0.05)
    for key in [
        "ret_1d",
        "ret_5d",
        "ret_21d",
        "vol_21d",
        "vol_63d",
        "atr_14",
        "atr_pct",
        "trend_slope_21d",
        "trend_r2_21d",
        "trend_slope_63d",
        "trend_r2_63d",
        "mom_vol_adj_21d",
        "maxdd_63d",
        "dollar_vol_21d",
        "sentiment_score",
        "sentiment_surprise",
        "regime_label",
        "regime_strength",
    ]:
        assert key in features


def test_signal_computation_bounds():
    features = {
        "trend_slope_63d": 0.02,
        "mom_vol_adj_21d": 0.5,
        "sentiment_score": 0.2,
        "maxdd_63d": -0.1,
        "atr_pct": 0.02,
        "vol_63d": 0.4,
    }
    signal = compute_daily_signal(features, quality_score=80, latest_close=100.0)
    assert signal["action"] in {"BUY", "SELL", "HOLD"}
    assert signal["reason_codes"]
    assert 0.0 <= signal["confidence"] <= 1.0


def test_endpoints_require_auth_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FTIP_API_KEY", "demo-key")
    security.reset_auth_cache()

    with TestClient(app) as client:
        res_jobs = client.post("/jobs/data/bars-daily", json={"from_date": "2024-01-01", "to_date": "2024-01-02"})
        assert res_jobs.status_code == 401
        res_signals = client.get("/signals/latest", params={"symbol": "AAPL"})
        assert res_signals.status_code == 401


def test_milestoneB_verify_script_is_posix():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "milestoneB_verify.sh"
    assert script_path.exists()
    content = script_path.read_text()
    assert content.startswith("#!/usr/bin/env sh")
    assert "python3" in content
    assert "mapfile" not in content
    assert "readarray" not in content
