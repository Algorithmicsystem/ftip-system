from __future__ import annotations

import logging
import os
import time
import datetime as dt
import hashlib
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
                    set_statement_timeout(
                        cur, config.env_int("FTIP_DB_STATEMENT_TIMEOUT_MS", 10_000)
                    )
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


def safe_execute(sql: str, params: Optional[Sequence[Any]] = None) -> None:
    try:
        with with_connection() as (conn, cur):
            cur.execute(sql, params or ())
            conn.commit()
    except DBError:
        raise
    except psycopg.Error as exc:
        raise DBError(f"database error during execute: {exc}") from exc


def safe_fetchall(
    sql: str, params: Optional[Sequence[Any]] = None
) -> List[Sequence[Any]]:
    try:
        with with_connection() as (_conn, cur):
            cur.execute(sql, params or ())
            return list(cur.fetchall())
    except DBError:
        raise
    except psycopg.Error as exc:
        raise DBError(f"database error during fetchall: {exc}") from exc


def safe_fetchone(
    sql: str, params: Optional[Sequence[Any]] = None
) -> Optional[Sequence[Any]]:
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


def exec1(sql: str, params: Optional[Sequence[Any]] = None) -> Optional[Sequence[Any]]:
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


def fetch1(sql: str, params: Optional[Sequence[Any]] = None) -> Optional[Sequence[Any]]:
    return safe_fetchone(sql, params)


def fetchall(
    sql: str, params: Optional[Sequence[Any]] = None
) -> Iterable[Sequence[Any]]:
    return safe_fetchall(sql, params)


# ---------------------------------------------------------------------------
# Versioned reality helpers (Phase 1)
# ---------------------------------------------------------------------------


def _code_sha() -> str:
    return (
        os.getenv("RAILWAY_GIT_COMMIT_SHA")
        or os.getenv("GIT_COMMIT_SHA")
        or os.getenv("COMMIT_SHA")
        or "unknown"
    )


def record_data_version(
    source_name: str, source_snapshot_hash: str, notes: str = ""
) -> Dict[str, Any]:
    row = exec1(
        """
        INSERT INTO data_versions (source_name, source_snapshot_hash, code_sha, notes)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (source_name, source_snapshot_hash, code_sha)
        DO UPDATE SET notes = EXCLUDED.notes
        RETURNING id, source_name, source_snapshot_hash, code_sha, notes, created_at
        """,
        (source_name.strip(), source_snapshot_hash.strip(), _code_sha(), notes or ""),
    )
    if not row:
        raise DBError("failed to record data version", status_code=500)
    return {
        "id": int(row[0]),
        "source_name": str(row[1]),
        "source_snapshot_hash": str(row[2]),
        "code_sha": str(row[3]),
        "notes": str(row[4] or ""),
        "created_at": row[5],
    }


def get_prices_daily(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    as_of_ts: dt.datetime,
    adjusted: bool = False,
) -> List[Dict[str, Any]]:
    del adjusted
    rows = safe_fetchall(
        """
        SELECT p.symbol, p.date, p.open, p.high, p.low, p.close, p.volume, p.currency,
               p.as_of_ts, p.data_version_id
        FROM prices_daily_versioned p
        JOIN (
            SELECT symbol, date, MAX(as_of_ts) AS latest_as_of_ts
            FROM prices_daily_versioned
            WHERE symbol = %s
              AND date BETWEEN %s AND %s
              AND as_of_ts <= %s
            GROUP BY symbol, date
        ) latest
          ON p.symbol = latest.symbol
         AND p.date = latest.date
         AND p.as_of_ts = latest.latest_as_of_ts
        ORDER BY p.date ASC
        """,
        (symbol, start_date, end_date, as_of_ts),
    )
    return [
        {
            "symbol": row[0],
            "date": row[1],
            "open": row[2],
            "high": row[3],
            "low": row[4],
            "close": row[5],
            "volume": row[6],
            "currency": row[7],
            "as_of_ts": row[8],
            "data_version_id": row[9],
        }
        for row in rows
    ]


def get_latest_fundamentals(
    symbol: str, as_of_ts: dt.datetime, metric_keys: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    normalized_keys = [k.strip() for k in (metric_keys or []) if k and k.strip()]
    if normalized_keys:
        sql = """
            SELECT f.symbol, f.metric_key, f.metric_value, f.period_end, f.published_ts,
                   f.as_of_ts, f.data_version_id
            FROM fundamentals_pit f
            JOIN (
                SELECT metric_key, MAX(published_ts) AS latest_published_ts
                FROM fundamentals_pit
                WHERE symbol = %s
                  AND as_of_ts <= %s
                  AND published_ts <= %s
                  AND metric_key = ANY(%s)
                GROUP BY metric_key
            ) latest
              ON f.metric_key = latest.metric_key
             AND f.published_ts = latest.latest_published_ts
            WHERE f.symbol = %s
              AND f.as_of_ts <= %s
              AND f.published_ts <= %s
            ORDER BY f.metric_key ASC
        """
        params = (
            symbol,
            as_of_ts,
            as_of_ts,
            normalized_keys,
            symbol,
            as_of_ts,
            as_of_ts,
        )
    else:
        sql = """
            SELECT f.symbol, f.metric_key, f.metric_value, f.period_end, f.published_ts,
                   f.as_of_ts, f.data_version_id
            FROM fundamentals_pit f
            JOIN (
                SELECT metric_key, MAX(published_ts) AS latest_published_ts
                FROM fundamentals_pit
                WHERE symbol = %s
                  AND as_of_ts <= %s
                  AND published_ts <= %s
                GROUP BY metric_key
            ) latest
              ON f.metric_key = latest.metric_key
             AND f.published_ts = latest.latest_published_ts
            WHERE f.symbol = %s
              AND f.as_of_ts <= %s
              AND f.published_ts <= %s
            ORDER BY f.metric_key ASC
        """
        params = (symbol, as_of_ts, as_of_ts, symbol, as_of_ts, as_of_ts)

    rows = safe_fetchall(sql, params)
    return [
        {
            "symbol": row[0],
            "metric_key": row[1],
            "metric_value": row[2],
            "period_end": row[3],
            "published_ts": row[4],
            "as_of_ts": row[5],
            "data_version_id": row[6],
        }
        for row in rows
    ]


def get_news(
    symbol: str,
    as_of_ts: dt.datetime,
    start_ts: Optional[dt.datetime] = None,
    end_ts: Optional[dt.datetime] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    bounded_limit = max(1, min(int(limit), 1000))
    rows = safe_fetchall(
        """
        SELECT symbol, published_ts, as_of_ts, source, credibility, headline, full_text,
               data_version_id
        FROM news_items
        WHERE symbol = %s
          AND as_of_ts <= %s
          AND published_ts <= %s
          AND (%s::timestamptz IS NULL OR published_ts >= %s::timestamptz)
          AND (%s::timestamptz IS NULL OR published_ts <= %s::timestamptz)
        ORDER BY published_ts DESC, id DESC
        LIMIT %s
        """,
        (symbol, as_of_ts, as_of_ts, start_ts, start_ts, end_ts, end_ts, bounded_limit),
    )
    return [
        {
            "symbol": row[0],
            "published_ts": row[1],
            "as_of_ts": row[2],
            "source": row[3],
            "credibility": row[4],
            "headline": row[5],
            "full_text": row[6],
            "data_version_id": row[7],
        }
        for row in rows
    ]


def get_universe_pit(
    as_of_ts: dt.datetime, universe_name: str = "default"
) -> List[str]:
    rows = safe_fetchall(
        """
        SELECT DISTINCT symbol
        FROM universe_membership
        WHERE universe_name = %s
          AND start_ts <= %s
          AND (end_ts IS NULL OR end_ts > %s)
        ORDER BY symbol ASC
        """,
        (universe_name, as_of_ts, as_of_ts),
    )
    return [str(row[0]) for row in rows]


def headline_hash(value: str) -> str:
    return hashlib.sha256((value or "").strip().encode("utf-8")).hexdigest()


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


def upsert_universe(
    symbols: Sequence[str], source: Optional[str] = None
) -> Tuple[int, int]:
    cleaned = [str(s).strip() for s in symbols if s and str(s).strip()]
    received = len(cleaned)
    if not cleaned:
        return 0, 0
    unique = sorted(set(cleaned))
    for sym in unique:
        safe_execute(
            """
            INSERT INTO prosperity_universe(symbol, active)
            VALUES (%s, TRUE)
            ON CONFLICT(symbol) DO UPDATE SET active=EXCLUDED.active, updated_at=now()
            """,
            (sym,),
        )
    return received, len(unique)


def get_universe(*, active_only: bool = True, limit: Optional[int] = None) -> List[str]:
    where = "WHERE active = TRUE" if active_only else ""
    limit_sql = "LIMIT %s" if limit else ""
    params: List[Any] = []
    if limit:
        params.append(int(limit))
    rows = safe_fetchall(
        f"""
        SELECT symbol
        FROM prosperity_universe
        {where}
        ORDER BY updated_at DESC, symbol ASC
        {limit_sql}
        """,
        tuple(params) if params else None,
    )
    return [row[0] for row in rows]


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
    "get_universe",
    "upsert_universe",
]
