import os
from contextlib import contextmanager
from typing import Any, Iterable, Optional, Sequence

import psycopg
from psycopg_pool import ConnectionPool


_POOL: Optional[ConnectionPool] = None


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def db_enabled() -> bool:
    return (_env("FTIP_DB_ENABLED", "0") or "0") == "1"


def _db_url() -> str:
    url = _env("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is required when FTIP_DB_ENABLED=1")
    return url


def get_pool() -> ConnectionPool:
    global _POOL
    if _POOL is not None:
        return _POOL

    if not db_enabled():
        raise RuntimeError("Database is disabled (set FTIP_DB_ENABLED=1 to enable)")

    try:
        max_size = int(_env("FTIP_DB_POOL_MAX", "5") or "5")
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
def _with_cursor(timeout_ms: int = 5000):
    pool = get_pool()
    with pool.connection(timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout TO %s", (timeout_ms,))
            yield conn, cur


def exec1(sql: str, params: Sequence[Any] | None = None) -> Optional[Sequence[Any]]:
    with _with_cursor() as (conn, cur):
        cur.execute(sql, params or ())
        row = None
        try:
            row = cur.fetchone()
        except psycopg.ProgrammingError:
            row = None
        conn.commit()
        return row


def fetch1(sql: str, params: Sequence[Any] | None = None) -> Optional[Sequence[Any]]:
    with _with_cursor() as (_conn, cur):
        cur.execute(sql, params or ())
        return cur.fetchone()


def fetchall(sql: str, params: Sequence[Any] | None = None) -> Iterable[Sequence[Any]]:
    with _with_cursor() as (_conn, cur):
        cur.execute(sql, params or ())
        return cur.fetchall()


def ensure_schema() -> None:
    if not db_enabled():
        return

    statements = [
        """
        CREATE TABLE IF NOT EXISTS prosperity_universe (
            symbol TEXT PRIMARY KEY,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            source TEXT,
            added_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
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
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(symbol, as_of, lookback, score_mode)
        )
        """,
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
    ]

    pool = get_pool()
    with pool.connection(timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout TO %s", (5000,))
            for stmt in statements:
                cur.execute(stmt)
            conn.commit()

