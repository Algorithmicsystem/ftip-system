import pytest

from api import lifecycle


def test_startup_requires_read_write_when_db_required(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("api.db.db_enabled", lambda: True)
    monkeypatch.setattr("api.config.db_required", lambda: True)
    monkeypatch.setattr("api.db.db_write_enabled", lambda: False)
    monkeypatch.setattr("api.db.db_read_enabled", lambda: True)
    monkeypatch.setattr("api.config.env", lambda name, default=None: "postgresql://example" if name == "DATABASE_URL" else default)

    with pytest.raises(RuntimeError, match="FTIP_DB_WRITE_ENABLED=1 and FTIP_DB_READ_ENABLED=1"):
        lifecycle.startup()


def test_startup_requires_bootstrap_when_migrations_auto_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("api.db.db_enabled", lambda: True)
    monkeypatch.setattr("api.config.db_required", lambda: True)
    monkeypatch.setattr("api.db.db_write_enabled", lambda: True)
    monkeypatch.setattr("api.db.db_read_enabled", lambda: True)
    monkeypatch.setattr("api.config.migrations_auto", lambda: False)
    monkeypatch.setattr("api.config.env", lambda name, default=None: "postgresql://example" if name == "DATABASE_URL" else default)
    monkeypatch.setattr("api.db.safe_fetchone", lambda sql, params=None: (1,) if sql == "SELECT 1" else (None,))

    with pytest.raises(RuntimeError, match="run POST /prosperity/bootstrap"):
        lifecycle.startup()
