"""Tests for api.alpha.calibration_store and the DB-first calibration loading."""
from __future__ import annotations

import json
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_PAYLOAD = {
    "created_at_utc": "2024-01-01T00:00:00Z",
    "symbol": "AAPL",
    "train_range": {"from_date": "2023-01-01", "to_date": "2023-12-31"},
    "optimize_horizon": 21,
    "thresholds_by_regime": {
        "TRENDING": {"buy": 0.20, "sell": -0.20},
        "CHOPPY": {"buy": 0.30, "sell": -0.30},
    },
    "diagnostics": {},
}


# ---------------------------------------------------------------------------
# calibration_store unit tests (DB disabled path)
# ---------------------------------------------------------------------------


def test_upsert_calibration_db_disabled_returns_false(monkeypatch):
    monkeypatch.setenv("FTIP_DB_ENABLED", "0")
    from api.alpha.calibration_store import upsert_calibration

    ok = upsert_calibration("AAPL", _SAMPLE_PAYLOAD, optimize_horizon=21)
    assert ok is False


def test_upsert_calibration_empty_symbol_returns_false(monkeypatch):
    monkeypatch.setenv("FTIP_DB_ENABLED", "0")
    from api.alpha.calibration_store import upsert_calibration

    assert upsert_calibration("", _SAMPLE_PAYLOAD) is False
    assert upsert_calibration("  ", _SAMPLE_PAYLOAD) is False


def test_load_calibration_from_db_db_disabled(monkeypatch):
    monkeypatch.setenv("FTIP_DB_ENABLED", "0")
    from api.alpha.calibration_store import load_calibration_from_db

    loaded, cal = load_calibration_from_db("AAPL")
    assert not loaded
    assert cal is None


def test_load_calibration_from_db_empty_symbol(monkeypatch):
    monkeypatch.setenv("FTIP_DB_ENABLED", "0")
    from api.alpha.calibration_store import load_calibration_from_db

    loaded, cal = load_calibration_from_db("")
    assert not loaded
    assert cal is None


def test_upsert_calibration_db_execute_called(monkeypatch):
    """Verify upsert calls safe_execute when DB is enabled."""
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")

    calls = []

    def fake_safe_execute(sql, params):
        calls.append(params)

    with patch("api.alpha.calibration_store.db.db_read_enabled", return_value=True), \
         patch("api.alpha.calibration_store.db.safe_execute", side_effect=fake_safe_execute):
        from importlib import reload
        import api.alpha.calibration_store as cs
        reload(cs)
        ok = cs.upsert_calibration("AAPL", _SAMPLE_PAYLOAD, optimize_horizon=21)

    assert ok is True
    assert len(calls) == 1
    # First param is the symbol, normalised to uppercase
    assert calls[0][0] == "AAPL"


def test_load_calibration_from_db_returns_payload(monkeypatch):
    """load_calibration_from_db returns (True, payload) when DB has a row."""
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")

    with patch("api.alpha.calibration_store.db.db_read_enabled", return_value=True), \
         patch("api.alpha.calibration_store.db.safe_fetchone", return_value=(_SAMPLE_PAYLOAD,)):
        from importlib import reload
        import api.alpha.calibration_store as cs
        reload(cs)
        loaded, cal = cs.load_calibration_from_db("AAPL")

    assert loaded is True
    assert cal == _SAMPLE_PAYLOAD


def test_load_calibration_from_db_no_row(monkeypatch):
    """load_calibration_from_db returns (False, None) when DB has no row."""
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")

    with patch("api.alpha.calibration_store.db.db_read_enabled", return_value=True), \
         patch("api.alpha.calibration_store.db.safe_fetchone", return_value=None):
        from importlib import reload
        import api.alpha.calibration_store as cs
        reload(cs)
        loaded, cal = cs.load_calibration_from_db("AAPL")

    assert not loaded
    assert cal is None


def test_load_calibration_from_db_db_exception(monkeypatch):
    """load_calibration_from_db returns (False, None) when DB raises."""
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")

    with patch("api.alpha.calibration_store.db.db_read_enabled", return_value=True), \
         patch("api.alpha.calibration_store.db.safe_fetchone", side_effect=RuntimeError("db down")):
        from importlib import reload
        import api.alpha.calibration_store as cs
        reload(cs)
        loaded, cal = cs.load_calibration_from_db("AAPL")

    assert not loaded
    assert cal is None


# ---------------------------------------------------------------------------
# _payload_hash determinism
# ---------------------------------------------------------------------------


def test_payload_hash_is_deterministic():
    from importlib import reload
    import api.alpha.calibration_store as cs
    reload(cs)

    h1 = cs._payload_hash(_SAMPLE_PAYLOAD)
    h2 = cs._payload_hash(_SAMPLE_PAYLOAD)
    assert h1 == h2
    assert len(h1) == 16  # truncated hex


def test_payload_hash_differs_for_different_payloads():
    from importlib import reload
    import api.alpha.calibration_store as cs
    reload(cs)

    other = dict(_SAMPLE_PAYLOAD, optimize_horizon=42)
    assert cs._payload_hash(_SAMPLE_PAYLOAD) != cs._payload_hash(other)


# ---------------------------------------------------------------------------
# main._load_calibration_for_symbol: DB-first priority
# ---------------------------------------------------------------------------


def test_main_load_calibration_prefers_db_over_env(monkeypatch):
    """DB row should take precedence over FTIP_CALIBRATION_JSON env var."""
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv(
        "FTIP_CALIBRATION_JSON",
        json.dumps({"thresholds_by_regime": {"TRENDING": {"buy": 0.99, "sell": -0.99}}}),
    )

    with patch("api.main.db.db_read_enabled", return_value=True), \
         patch(
             "api.alpha.calibration_store.db.safe_fetchone",
             return_value=(_SAMPLE_PAYLOAD,),
         ):
        import api.main as main_mod
        loaded, cal = main_mod._load_calibration_for_symbol("AAPL")

    assert loaded is True
    # DB payload should win — its buy threshold is 0.20, not 0.99
    assert cal is not None
    tr = (cal.get("thresholds_by_regime") or {}).get("TRENDING") or {}
    assert float(tr.get("buy", 99)) != 0.99


def test_main_load_calibration_falls_back_to_env_when_db_empty(monkeypatch):
    """Falls back to env var when DB has no row for the symbol."""
    env_cal = {"thresholds_by_regime": {"TRENDING": {"buy": 0.25, "sell": -0.25}}}
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_CALIBRATION_JSON", json.dumps(env_cal))

    with patch("api.main.db.db_read_enabled", return_value=True), \
         patch("api.alpha.calibration_store.db.safe_fetchone", return_value=None):
        import api.main as main_mod
        loaded, cal = main_mod._load_calibration_for_symbol("MSFT")

    assert loaded is True
    assert cal == env_cal


def test_main_load_calibration_no_db_no_env(monkeypatch):
    monkeypatch.setenv("FTIP_DB_ENABLED", "0")
    monkeypatch.delenv("FTIP_CALIBRATION_JSON", raising=False)
    monkeypatch.delenv("FTIP_CALIBRATION_JSON_MAP", raising=False)

    import api.main as main_mod
    loaded, cal = main_mod._load_calibration_for_symbol("NVDA")
    assert not loaded
    assert cal is None
