from contextlib import contextmanager
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import psycopg
from psycopg_pool import ConnectionPool

from api import config


_POOL: Optional[ConnectionPool] = None


def db_enabled() -> bool:
    return config.db_enabled()


def _db_url() -> str:
    url = config.env("DATABASE_URL")
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


def _clamp_timeout_ms(timeout_ms: int, *, default: int = 5000) -> int:
    try:
        parsed = int(timeout_ms)
    except Exception:
        parsed = default

    return max(1000, min(parsed, 120_000))


def set_statement_timeout(cur: Any, timeout_ms: int, *, local: bool = True) -> int:
    """Set (local) statement timeout using an inline, validated integer."""

    ms = _clamp_timeout_ms(timeout_ms)
    scope = "LOCAL " if local else ""
    cur.execute(f"SET {scope}statement_timeout TO {ms}")
    return ms


@contextmanager
def _with_cursor(timeout_ms: int = 5000):
    pool = get_pool()
    with pool.connection(timeout=10) as conn:
        with conn.cursor() as cur:
            set_statement_timeout(cur, timeout_ms)
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
        """ALTER TABLE IF EXISTS signal_observation ALTER COLUMN calibration_meta TYPE JSONB USING calibration_meta::jsonb""",
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
            set_statement_timeout(cur, 5000)
            for stmt in statements:
                cur.execute(stmt)
            conn.commit()


def _normalize_symbols(symbols: Iterable[str]) -> List[str]:
    syms: List[str] = []
    for s in symbols:
        norm = (s or "").strip().upper()
        if norm:
            syms.append(norm)
    return syms


def upsert_universe(symbols: Iterable[str], source: Optional[str] = None) -> Tuple[int, int]:
    if not db_enabled():
        raise RuntimeError("Database is disabled (set FTIP_DB_ENABLED=1 to enable)")

    symbols_list = list(symbols or [])
    received = len(symbols_list)
    syms = _normalize_symbols(symbols_list)
    if len(syms) == 0:
        raise ValueError("symbols list is empty")
    if len(syms) > 1000:
        raise ValueError("symbols list exceeds maximum of 1000")

    src = (source or "manual").strip() or "manual"

    pool = get_pool()
    with pool.connection(timeout=10) as conn:
        with conn.cursor() as cur:
            set_statement_timeout(cur, 5000)
            params = [(sym, True, src) for sym in syms]
            cur.executemany(
                """
                INSERT INTO prosperity_universe(symbol, active, source)
                VALUES (%s, %s, %s)
                ON CONFLICT (symbol)
                DO UPDATE SET active=EXCLUDED.active, source=EXCLUDED.source
                """,
                params,
            )
            conn.commit()

    return received, len(syms)


def get_universe(active_only: bool = True, limit: int = 1000) -> List[str]:
    if not db_enabled():
        raise RuntimeError("Database is disabled (set FTIP_DB_ENABLED=1 to enable)")

    try:
        limit_val = max(1, min(int(limit), 5000))
    except Exception:
        limit_val = 1000

    pool = get_pool()
    with pool.connection(timeout=10) as conn:
        with conn.cursor() as cur:
            set_statement_timeout(cur, 5000)
            if active_only:
                cur.execute(
                    """
                    SELECT symbol
                    FROM prosperity_universe
                    WHERE active = TRUE
                    ORDER BY symbol ASC
                    LIMIT %s
                    """,
                    (limit_val,),
                )
            else:
                cur.execute(
                    """
                    SELECT symbol
                    FROM prosperity_universe
                    ORDER BY symbol ASC
                    LIMIT %s
                    """,
                    (limit_val,),
                )
            rows = cur.fetchall() or []
    return [r[0] for r in rows]

