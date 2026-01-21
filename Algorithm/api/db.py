from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterable, List, Optional, Sequence

import psycopg
from psycopg import OperationalError
from psycopg_pool import ConnectionPool

from api import config

logger = logging.getLogger(__name__)

_POOL: Optional[ConnectionPool] = None


class DBError(Exception):
    """Raised when a database operation fails in a controlled, user-facing way."""

    def __init__(self, message: str, *, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def db_enabled() -> bool:
    return config.db_enabled()


def db_write_enabled() -> bool:
    return config.db_write_enabled()


def db_read_enabled() -> bool:
    return config.db_read_enabled()


def _db_url() -> str:
    url = config.env("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is required when FTIP_DB_ENABLED=1")
    return url


# ---------------------------------------------------------------------------
# Pool + helpers
# ---------------------------------------------------------------------------

def get_pool() -> ConnectionPool:
    global _POOL
    if _POOL is not None:
        return _POOL

    if not db_enabled():
        raise RuntimeError("Database is disabled (set FTIP_DB_ENABLED=1 to enable)")

    try:
        max_size = int(config.env("FTIP_DB_POOL_MAX", "5") or "5")
    except Exception:
        max_size = 5

    _POOL = ConnectionPool(
        conninfo=_db_url(),
        min_size=1,
        max_size=max_size,
        kwargs={"connect_timeout": 5},
        open=True,
    )
    return _POOL


@contextmanager
def with_connection():
    pool = get_pool()
    max_retries = config.env_int("FTIP_DB_MAX_RETRIES", 2)
    attempt = 0
    backoff = 0.25

    while True:
        try:
            with pool.connection(timeout=10) as conn:
                with conn.cursor() as cur:
                    set_statement_timeout(cur, config.env_int("FTIP_DB_STATEMENT_TIMEOUT_MS", 10_000))
                    yield conn, cur
                    return
        except OperationalError as exc:  # pragma: no cover - exercised in integration
            attempt += 1
            if attempt > max_retries:
                raise DBError(f"database unavailable: {exc}") from exc
            time.sleep(backoff)
            backoff = min(backoff * 2, 2.0)


def _clamp_timeout_ms(timeout_ms: int, *, default: int = 5000) -> int:
    try:
        parsed = int(timeout_ms)
    except Exception:
        parsed = default
    return max(1000, min(parsed, 120_000))


def set_statement_timeout(cur: Any, timeout_ms: int, *, local: bool = True) -> int:
    ms = _clamp_timeout_ms(timeout_ms)
    scope = "LOCAL " if local else ""
    cur.execute(f"SET {scope}statement_timeout TO {ms}")
    return ms


# ---------------------------------------------------------------------------
# Safe execution helpers
# ---------------------------------------------------------------------------

def safe_execute(sql: str, params: Sequence[Any] | None = None) -> None:
    try:
        with with_connection() as (conn, cur):
            cur.execute(sql, params or ())
            conn.commit()
    except DBError:
        raise
    except psycopg.Error as exc:
        raise DBError(f"database error during execute: {exc}") from exc


def safe_fetchall(sql: str, params: Sequence[Any] | None = None) -> List[Sequence[Any]]:
    try:
        with with_connection() as (_conn, cur):
            cur.execute(sql, params or ())
            return list(cur.fetchall())
    except DBError:
        raise
    except psycopg.Error as exc:
        raise DBError(f"database error during fetchall: {exc}") from exc


def safe_fetchone(sql: str, params: Sequence[Any] | None = None) -> Optional[Sequence[Any]]:
    try:
        with with_connection() as (_conn, cur):
            cur.execute(sql, params or ())
            return cur.fetchone()
    except DBError:
        raise
    except psycopg.Error as exc:
        raise DBError(f"database error during fetchone: {exc}") from exc


# ---------------------------------------------------------------------------
# Backwards-compatible helpers (used across codebase)
# ---------------------------------------------------------------------------

def exec1(sql: str, params: Sequence[Any] | None = None) -> Optional[Sequence[Any]]:
    try:
        with with_connection() as (conn, cur):
            cur.execute(sql, params or ())
            row = None
            try:
                row = cur.fetchone()
            except psycopg.ProgrammingError:
                row = None
            conn.commit()
            return row
    except DBError:
        raise
    except psycopg.Error as exc:
        raise DBError(f"database error during exec1: {exc}") from exc


def fetch1(sql: str, params: Sequence[Any] | None = None) -> Optional[Sequence[Any]]:
    return safe_fetchone(sql, params)


def fetchall(sql: str, params: Sequence[Any] | None = None) -> Iterable[Sequence[Any]]:
    return safe_fetchall(sql, params)


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def apply_migrations() -> None:
    from api import migrations

    migrations.ensure_schema()


# ---------------------------------------------------------------------------
# Simple schema bootstrap (legacy) to keep compatibility
# ---------------------------------------------------------------------------

def ensure_schema() -> None:
    if not db_enabled():
        return
    if not config.env("DATABASE_URL"):
        if config.db_required():
            raise RuntimeError("DATABASE_URL is required when FTIP_DB_ENABLED=1")
        logger.warning("DATABASE_URL missing; skipping schema bootstrap")
        return
    from api import migrations

    migrations.ensure_schema()

    statements = [
        """
        CREATE TABLE IF NOT EXISTS signals (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            as_of DATE NOT NULL,
            lookback INT NOT NULL,
            regime TEXT NOT NULL,
            score DOUBLE PRECISION NOT NULL,
            signal TEXT NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            thresholds JSONB NOT NULL,
            features JSONB NOT NULL,
            notes JSONB NOT NULL,
            score_mode TEXT,
            base_score DOUBLE PRECISION,
            stacked_score DOUBLE PRECISION,
            stacked_meta JSONB,
            calibration_loaded BOOLEAN,
            calibration_meta JSONB,
            raw_signal_payload JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(symbol, as_of, lookback, score_mode)
        )
        """,
        # Ensure JSONB columns remain JSONB even if pre-existing
        """ALTER TABLE IF EXISTS signals ALTER COLUMN thresholds TYPE JSONB USING thresholds::jsonb""",
        """ALTER TABLE IF EXISTS signals ALTER COLUMN features TYPE JSONB USING features::jsonb""",
        """ALTER TABLE IF EXISTS signals ALTER COLUMN notes TYPE JSONB USING notes::jsonb""",
        """ALTER TABLE IF EXISTS signals ALTER COLUMN calibration_meta TYPE JSONB USING calibration_meta::jsonb""",
        """ALTER TABLE IF EXISTS signals ADD COLUMN IF NOT EXISTS raw_signal_payload JSONB""",
        """
        CREATE TABLE IF NOT EXISTS signal_run (
            id BIGSERIAL PRIMARY KEY,
            as_of_date DATE NOT NULL,
            lookback INT NOT NULL,
            score_mode TEXT NOT NULL,
            version_sha TEXT,
            env JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS signal_observation (
            run_id BIGINT NOT NULL REFERENCES signal_run(id) ON DELETE CASCADE,
            symbol TEXT NOT NULL,
            regime TEXT,
            signal TEXT,
            score DOUBLE PRECISION,
            confidence DOUBLE PRECISION,
            thresholds JSONB,
            features JSONB,
            notes JSONB,
            calibration_meta JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (run_id, symbol)
        )
        """,
        # Ensure JSONB integrity for signal observations
        """ALTER TABLE IF EXISTS signal_observation ALTER COLUMN thresholds TYPE JSONB USING thresholds::jsonb""",
        """ALTER TABLE IF EXISTS signal_observation ALTER COLUMN features TYPE JSONB USING features::jsonb""",
        """ALTER TABLE IF EXISTS signal_observation ALTER COLUMN notes TYPE JSONB USING notes::jsonb""",
        """
        CREATE TABLE IF NOT EXISTS portfolio_backtests (
            id BIGSERIAL PRIMARY KEY,
            symbols JSONB NOT NULL,
            from_date DATE NOT NULL,
            to_date DATE NOT NULL,
            lookback INT NOT NULL,
            rebalance_every INT NOT NULL,
            trading_cost_bps DOUBLE PRECISION NOT NULL,
            slippage_bps DOUBLE PRECISION NOT NULL,
            max_weight DOUBLE PRECISION,
            min_trade_delta DOUBLE PRECISION,
            max_turnover_per_rebalance DOUBLE PRECISION,
            allow_shorts BOOLEAN NOT NULL DEFAULT FALSE,
            total_return DOUBLE PRECISION NOT NULL,
            annual_return DOUBLE PRECISION NOT NULL,
            sharpe DOUBLE PRECISION NOT NULL,
            max_drawdown DOUBLE PRECISION NOT NULL,
            volatility DOUBLE PRECISION NOT NULL,
            turnover DOUBLE PRECISION NOT NULL,
            audit JSONB,
            equity_curve JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        """CREATE INDEX IF NOT EXISTS idx_signals_symbol_asof ON signals(symbol, as_of)""",
        """CREATE INDEX IF NOT EXISTS idx_portfolio_backtests_range ON portfolio_backtests(from_date, to_date)""",
        """
        CREATE TABLE IF NOT EXISTS prosperity_backtest_runs (
            run_id UUID PRIMARY KEY,
            kind TEXT NOT NULL,
            request JSONB NOT NULL,
            response JSONB NOT NULL,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            status TEXT NOT NULL,
            error TEXT,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS prosperity_calibrations (
            symbol TEXT NOT NULL,
            created_at_utc TEXT,
            payload JSONB NOT NULL,
            optimize_horizon INT,
            train_range JSONB,
            calibration_hash TEXT,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (symbol, calibration_hash)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS prosperity_events (
            id UUID PRIMARY KEY,
            event_type TEXT,
            symbol TEXT,
            payload JSONB,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS prosperity_snapshots (
            id BIGSERIAL PRIMARY KEY,
            started_at TIMESTAMPTZ DEFAULT now(),
            finished_at TIMESTAMPTZ,
            status TEXT NOT NULL DEFAULT 'pending',
            request JSONB,
            result JSONB,
            error TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS prosperity_schema_meta (
            version INT NOT NULL DEFAULT 1,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            details TEXT,
            PRIMARY KEY (version)
        )
        """,
    ]

    for stmt in statements:
        safe_execute(stmt)


__all__ = [
    "db_enabled",
    "db_write_enabled",
    "db_read_enabled",
    "get_pool",
    "with_connection",
    "safe_execute",
    "safe_fetchall",
    "safe_fetchone",
    "apply_migrations",
    "ensure_schema",
    "exec1",
    "fetch1",
    "fetchall",
    "DBError",
]
