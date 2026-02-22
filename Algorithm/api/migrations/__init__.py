from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, List, Sequence

from api import config, db

logger = logging.getLogger(__name__)

Migration = Callable[[Any], None]


def _rename_column_if_exists(cur: Any, table: str, old: str, new: str) -> None:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name=%s AND column_name=%s
        """,
        (table, old),
    )
    if cur.fetchone():
        cur.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")


def _ensure_column(cur: Any, table: str, column: str, definition: str) -> None:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name=%s AND column_name=%s
        """,
        (table, column),
    )
    if not cur.fetchone():
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def _ensure_primary_key(cur: Any, table: str, columns: Sequence[str]) -> None:
    cur.execute(
        """
        SELECT constraint_name
        FROM information_schema.table_constraints
        WHERE table_name=%s AND constraint_type='PRIMARY KEY'
        """,
        (table,),
    )
    existing = cur.fetchone()
    if existing:
        cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT {existing[0]}")
    cols = ", ".join(columns)
    cur.execute(f"ALTER TABLE {table} ADD CONSTRAINT {table}_pkey PRIMARY KEY ({cols})")


def _migration_prosperity_core(cur: Any) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prosperity_universe (
            symbol TEXT PRIMARY KEY,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    _ensure_column(
        cur,
        "prosperity_universe",
        "updated_at",
        "IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
    )
    cur.execute("ALTER TABLE prosperity_universe ALTER COLUMN active SET DEFAULT TRUE")
    cur.execute("ALTER TABLE prosperity_universe ALTER COLUMN active SET NOT NULL")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prosperity_daily_bars (
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            adj_close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            source TEXT,
            raw JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol, date)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prosperity_features_daily (
            symbol TEXT NOT NULL,
            as_of DATE NOT NULL,
            lookback INT NOT NULL,
            features JSONB NOT NULL,
            meta JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol, as_of, lookback)
        )
        """
    )
    _rename_column_if_exists(cur, "prosperity_features_daily", "as_of_date", "as_of")
    _ensure_column(
        cur,
        "prosperity_features_daily",
        "as_of",
        "IF NOT EXISTS as_of DATE NOT NULL DEFAULT CURRENT_DATE",
    )
    _ensure_column(
        cur,
        "prosperity_features_daily",
        "lookback",
        "IF NOT EXISTS lookback INT NOT NULL",
    )
    _ensure_column(
        cur,
        "prosperity_features_daily",
        "features",
        "IF NOT EXISTS features JSONB NOT NULL DEFAULT '{}'::jsonb",
    )
    _ensure_column(cur, "prosperity_features_daily", "meta", "IF NOT EXISTS meta JSONB")
    _ensure_column(
        cur,
        "prosperity_features_daily",
        "created_at",
        "IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now()",
    )
    _ensure_column(
        cur,
        "prosperity_features_daily",
        "updated_at",
        "IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
    )
    cur.execute(
        "ALTER TABLE prosperity_features_daily ALTER COLUMN features SET NOT NULL"
    )
    cur.execute("ALTER TABLE prosperity_features_daily ALTER COLUMN as_of DROP DEFAULT")
    _ensure_primary_key(
        cur, "prosperity_features_daily", ("symbol", "as_of", "lookback")
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prosperity_signals_daily (
            symbol TEXT NOT NULL,
            as_of DATE NOT NULL,
            lookback INT NOT NULL,
            score_mode TEXT NOT NULL DEFAULT 'single',
            score DOUBLE PRECISION NOT NULL,
            signal TEXT NOT NULL,
            thresholds JSONB NOT NULL,
            regime TEXT,
            confidence DOUBLE PRECISION,
            notes JSONB,
            features JSONB,
            meta JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol, as_of, lookback, score_mode)
        )
        """
    )
    _rename_column_if_exists(cur, "prosperity_signals_daily", "as_of_date", "as_of")
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "as_of",
        "IF NOT EXISTS as_of DATE NOT NULL DEFAULT CURRENT_DATE",
    )
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "lookback",
        "IF NOT EXISTS lookback INT NOT NULL",
    )
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "score_mode",
        "IF NOT EXISTS score_mode TEXT NOT NULL DEFAULT 'single'",
    )
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "score",
        "IF NOT EXISTS score DOUBLE PRECISION NOT NULL DEFAULT 0",
    )
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "base_score",
        "IF NOT EXISTS base_score DOUBLE PRECISION",
    )
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "stacked_score",
        "IF NOT EXISTS stacked_score DOUBLE PRECISION",
    )
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "signal",
        "IF NOT EXISTS signal TEXT NOT NULL DEFAULT 'HOLD'",
    )
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "thresholds",
        "IF NOT EXISTS thresholds JSONB NOT NULL DEFAULT '{}'::jsonb",
    )
    _ensure_column(
        cur, "prosperity_signals_daily", "regime", "IF NOT EXISTS regime TEXT"
    )
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "confidence",
        "IF NOT EXISTS confidence DOUBLE PRECISION",
    )
    _ensure_column(
        cur, "prosperity_signals_daily", "notes", "IF NOT EXISTS notes JSONB"
    )
    _ensure_column(
        cur, "prosperity_signals_daily", "features", "IF NOT EXISTS features JSONB"
    )
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "calibration_meta",
        "IF NOT EXISTS calibration_meta JSONB",
    )
    _ensure_column(
        cur, "prosperity_signals_daily", "signal_hash", "IF NOT EXISTS signal_hash TEXT"
    )
    _ensure_column(cur, "prosperity_signals_daily", "meta", "IF NOT EXISTS meta JSONB")
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "created_at",
        "IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now()",
    )
    _ensure_column(
        cur,
        "prosperity_signals_daily",
        "updated_at",
        "IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
    )
    cur.execute(
        "ALTER TABLE prosperity_signals_daily ALTER COLUMN thresholds SET NOT NULL"
    )
    cur.execute("ALTER TABLE prosperity_signals_daily ALTER COLUMN score SET NOT NULL")
    cur.execute("ALTER TABLE prosperity_signals_daily ALTER COLUMN signal SET NOT NULL")
    cur.execute("ALTER TABLE prosperity_signals_daily ALTER COLUMN as_of DROP DEFAULT")
    _ensure_primary_key(
        cur, "prosperity_signals_daily", ("symbol", "as_of", "lookback", "score_mode")
    )

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_prosperity_universe_active ON prosperity_universe(active)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_prosperity_features_symbol_asof ON prosperity_features_daily(symbol, as_of DESC)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_prosperity_signals_symbol_asof ON prosperity_signals_daily(symbol, as_of DESC)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_prosperity_bars_symbol_date ON prosperity_daily_bars(symbol, date DESC)"
    )


def _migration_assistant(cur: Any) -> None:
    sql_path = Path(__file__).with_name("001_assistant.sql")
    cur.execute(sql_path.read_text())


def _migration_assistant_indexes(cur: Any) -> None:
    sql_path = Path(__file__).with_name("002_assistant_indexes.sql")
    cur.execute(sql_path.read_text())


def _migration_strategy_graph(cur: Any) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prosperity_strategy_signals_daily (
            symbol TEXT NOT NULL,
            as_of_date DATE NOT NULL,
            lookback INT NOT NULL,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            regime TEXT,
            raw_score DOUBLE PRECISION,
            normalized_score DOUBLE PRECISION,
            signal TEXT,
            confidence DOUBLE PRECISION,
            rationale JSONB,
            feature_contributions JSONB,
            meta JSONB,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (symbol, as_of_date, lookback, strategy_id, strategy_version)
        )
        """
    )
    _ensure_column(
        cur,
        "prosperity_strategy_signals_daily",
        "created_at",
        "IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now()",
    )
    _ensure_column(
        cur,
        "prosperity_strategy_signals_daily",
        "updated_at",
        "IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now()",
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prosperity_ensemble_signals_daily (
            symbol TEXT NOT NULL,
            as_of_date DATE NOT NULL,
            lookback INT NOT NULL,
            regime TEXT,
            ensemble_method TEXT,
            final_signal TEXT,
            final_score DOUBLE PRECISION,
            final_confidence DOUBLE PRECISION,
            thresholds JSONB,
            risk_overlay_applied BOOLEAN,
            strategies_used JSONB,
            audit JSONB,
            hashes JSONB,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (symbol, as_of_date, lookback)
        )
        """
    )
    _ensure_column(
        cur,
        "prosperity_ensemble_signals_daily",
        "created_at",
        "IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now()",
    )
    _ensure_column(
        cur,
        "prosperity_ensemble_signals_daily",
        "updated_at",
        "IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now()",
    )


def _migration_job_metadata(cur: Any) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ftip_job_locks (
            job_name TEXT PRIMARY KEY,
            locked_until TIMESTAMPTZ NOT NULL,
            lock_owner TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    _ensure_column(
        cur,
        "ftip_job_locks",
        "locked_until",
        "IF NOT EXISTS locked_until TIMESTAMPTZ NOT NULL",
    )
    _ensure_column(
        cur,
        "ftip_job_locks",
        "lock_owner",
        "IF NOT EXISTS lock_owner TEXT NOT NULL DEFAULT 'unknown'",
    )
    _ensure_column(
        cur,
        "ftip_job_locks",
        "updated_at",
        "IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ftip_job_runs (
            run_id UUID PRIMARY KEY,
            job_name TEXT,
            as_of_date DATE,
            started_at TIMESTAMPTZ DEFAULT now(),
            finished_at TIMESTAMPTZ,
            status TEXT,
            requested JSONB DEFAULT '{}'::jsonb,
            result JSONB,
            error TEXT,
            lock_owner TEXT,
            lock_acquired_at TIMESTAMPTZ,
            lock_expires_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    _ensure_column(cur, "ftip_job_runs", "job_name", "IF NOT EXISTS job_name TEXT")
    _ensure_column(cur, "ftip_job_runs", "as_of_date", "IF NOT EXISTS as_of_date DATE")
    _ensure_column(
        cur,
        "ftip_job_runs",
        "started_at",
        "IF NOT EXISTS started_at TIMESTAMPTZ DEFAULT now()",
    )
    _ensure_column(
        cur, "ftip_job_runs", "finished_at", "IF NOT EXISTS finished_at TIMESTAMPTZ"
    )
    _ensure_column(cur, "ftip_job_runs", "status", "IF NOT EXISTS status TEXT")
    _ensure_column(
        cur,
        "ftip_job_runs",
        "requested",
        "IF NOT EXISTS requested JSONB DEFAULT '{}'::jsonb",
    )
    _ensure_column(cur, "ftip_job_runs", "result", "IF NOT EXISTS result JSONB")
    _ensure_column(cur, "ftip_job_runs", "error", "IF NOT EXISTS error TEXT")
    _ensure_column(cur, "ftip_job_runs", "lock_owner", "IF NOT EXISTS lock_owner TEXT")
    _ensure_column(
        cur,
        "ftip_job_runs",
        "lock_acquired_at",
        "IF NOT EXISTS lock_acquired_at TIMESTAMPTZ",
    )
    _ensure_column(
        cur,
        "ftip_job_runs",
        "lock_expires_at",
        "IF NOT EXISTS lock_expires_at TIMESTAMPTZ",
    )
    _ensure_column(
        cur,
        "ftip_job_runs",
        "created_at",
        "IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now()",
    )
    _ensure_column(
        cur,
        "ftip_job_runs",
        "updated_at",
        "IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now()",
    )


def _migration_job_lock_owner(cur: Any) -> None:
    """Ensure job run tables include lock ownership metadata."""

    sql_path = Path(__file__).with_name("004_job_lock_owner.sql")
    cur.execute(sql_path.read_text())


def _migration_ftip_job_runs_columns(cur: Any) -> None:
    """Ensure ftip_job_runs includes all columns used by job code."""

    sql_path = Path(__file__).with_name("005_ftip_job_runs_columns.sql")
    cur.execute(sql_path.read_text())


def _migration_ftip_job_runs_lock_cleanup(cur: Any) -> None:
    """Remove legacy unique indexes and ensure non-unique active run index."""

    sql_path = Path(__file__).with_name("006_ftip_job_runs_lock_cleanup.sql")
    cur.execute(sql_path.read_text())


def _migration_prosperity_symbol_coverage(cur: Any) -> None:
    """Add coverage logging table for per-symbol run results."""

    sql_path = Path(__file__).with_name("007_prosperity_symbol_coverage.sql")
    cur.execute(sql_path.read_text())


def _migration_market_symbols(cur: Any) -> None:
    sql_path = Path(__file__).with_name("008_market_symbols.sql")
    cur.execute(sql_path.read_text())


def _migration_market_bars_daily(cur: Any) -> None:
    sql_path = Path(__file__).with_name("009_market_bars_daily.sql")
    cur.execute(sql_path.read_text())


def _migration_market_bars_intraday(cur: Any) -> None:
    sql_path = Path(__file__).with_name("010_market_bars_intraday.sql")
    cur.execute(sql_path.read_text())


def _migration_fundamentals_quarterly(cur: Any) -> None:
    sql_path = Path(__file__).with_name("011_fundamentals_quarterly.sql")
    cur.execute(sql_path.read_text())


def _migration_news_raw(cur: Any) -> None:
    sql_path = Path(__file__).with_name("012_news_raw.sql")
    cur.execute(sql_path.read_text())


def _migration_sentiment_daily(cur: Any) -> None:
    sql_path = Path(__file__).with_name("013_sentiment_daily.sql")
    cur.execute(sql_path.read_text())


def _migration_quality_daily(cur: Any) -> None:
    sql_path = Path(__file__).with_name("014_quality_daily.sql")
    cur.execute(sql_path.read_text())


def _migration_features_daily(cur: Any) -> None:
    sql_path = Path(__file__).with_name("015_features_daily.sql")
    cur.execute(sql_path.read_text())


def _migration_features_intraday(cur: Any) -> None:
    sql_path = Path(__file__).with_name("016_features_intraday.sql")
    cur.execute(sql_path.read_text())


def _migration_signals_daily(cur: Any) -> None:
    sql_path = Path(__file__).with_name("017_signals_daily.sql")
    cur.execute(sql_path.read_text())


def _migration_signals_intraday(cur: Any) -> None:
    sql_path = Path(__file__).with_name("018_signals_intraday.sql")
    cur.execute(sql_path.read_text())


def _migration_fix_prosperity_signals_score_mode_pk(cur: Any) -> None:
    sql_path = Path(__file__).with_name("019_fix_prosperity_signals_score_mode_pk.sql")
    cur.execute(sql_path.read_text())


def _migration_backtest_tables(cur: Any) -> None:
    sql_path = Path(__file__).with_name("020_backtest_tables.sql")
    cur.execute(sql_path.read_text())


def _migration_prosperity_signals_unique_index(cur: Any) -> None:
    sql_path = Path(__file__).with_name("021_prosperity_signals_unique_index.sql")
    cur.execute(sql_path.read_text())


MIGRATIONS: List[tuple[str, Migration]] = [
    ("001_assistant", _migration_assistant),
    ("001_prosperity_core", _migration_prosperity_core),
    ("002_assistant_indexes", _migration_assistant_indexes),
    ("002_strategy_graph", _migration_strategy_graph),
    ("003_job_metadata", _migration_job_metadata),
    ("004_job_lock_owner", _migration_job_lock_owner),
    ("005_ftip_job_runs_columns", _migration_ftip_job_runs_columns),
    ("006_ftip_job_runs_lock_cleanup", _migration_ftip_job_runs_lock_cleanup),
    ("007_prosperity_symbol_coverage", _migration_prosperity_symbol_coverage),
    ("008_market_symbols", _migration_market_symbols),
    ("009_market_bars_daily", _migration_market_bars_daily),
    ("010_market_bars_intraday", _migration_market_bars_intraday),
    ("011_fundamentals_quarterly", _migration_fundamentals_quarterly),
    ("012_news_raw", _migration_news_raw),
    ("013_sentiment_daily", _migration_sentiment_daily),
    ("014_quality_daily", _migration_quality_daily),
    ("015_features_daily", _migration_features_daily),
    ("016_features_intraday", _migration_features_intraday),
    ("017_signals_daily", _migration_signals_daily),
    ("018_signals_intraday", _migration_signals_intraday),
    (
        "019_fix_prosperity_signals_score_mode_pk",
        _migration_fix_prosperity_signals_score_mode_pk,
    ),
    ("020_backtest_tables", _migration_backtest_tables),
    ("021_prosperity_signals_unique_index", _migration_prosperity_signals_unique_index),
]


def ensure_schema() -> List[str]:
    if not db.db_enabled():
        return []
    if not config.env("DATABASE_URL"):
        if config.db_required():
            raise RuntimeError("DATABASE_URL is required when FTIP_DB_ENABLED=1")
        logger.warning("[migrations] DATABASE_URL missing; skipping migrations")
        return []

    pool = db.get_pool()
    applied: List[str] = []

    # Ensure bookkeeping table exists and is committed before we try to read from it
    with pool.connection(timeout=10) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    with pool.connection(timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(87123)")
            locked = cur.fetchone()[0]
            if not locked:
                logger.info(
                    "[migrations] another instance is applying migrations; skipping"
                )
                return applied

            try:
                cur.execute("SELECT version FROM schema_migrations")
                existing = {row[0] for row in cur.fetchall()}
                for version, migration in MIGRATIONS:
                    if version in existing:
                        continue
                    logger.info("[migrations] applying %s", version)
                    try:
                        migration(cur)
                        cur.execute(
                            "INSERT INTO schema_migrations(version) VALUES (%s)",
                            (version,),
                        )
                        conn.commit()
                        applied.append(version)
                    except Exception:
                        conn.rollback()
                        logger.exception("[migrations] failed applying %s", version)
                        raise

                _verify_job_run_schema(cur)
                conn.commit()
            finally:
                try:
                    cur.execute("SELECT pg_advisory_unlock(87123)")
                    conn.commit()
                except Exception:
                    logger.exception("[migrations] failed to release advisory lock")

    return applied


def _verify_job_run_schema(cur: Any) -> None:
    required_columns = {
        "run_id",
        "job_name",
        "as_of_date",
        "started_at",
        "finished_at",
        "status",
        "requested",
        "result",
        "error",
        "lock_owner",
        "lock_acquired_at",
        "lock_expires_at",
        "created_at",
        "updated_at",
    }

    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'ftip_job_runs'
        """
    )
    existing = {row[0] for row in cur.fetchall()}
    missing = sorted(required_columns - existing)
    if missing:
        raise RuntimeError(
            "ftip_job_runs schema is missing required columns: " + ", ".join(missing)
        )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ftip_job_runs_active
        ON ftip_job_runs(job_name)
        WHERE finished_at IS NULL
        """
    )

    cur.execute(
        """
        SELECT
            i.indisunique,
            pg_get_expr(i.indpred, i.indrelid),
            pg_get_indexdef(i.indexrelid)
        FROM pg_class t
        JOIN pg_index i ON t.oid = i.indrelid
        JOIN pg_class idx ON idx.oid = i.indexrelid
        WHERE t.relname = 'ftip_job_runs'
          AND idx.relname = 'idx_ftip_job_runs_active'
    """
    )
    index_row = cur.fetchone()
    if not index_row:
        raise RuntimeError("database_schema_missing: idx_ftip_job_runs_active")

    is_unique, predicate, index_def = index_row
    predicate = predicate or ""
    index_def = index_def or ""
    if (
        is_unique
        or "finished_at IS NULL" not in predicate
        or "(job_name)" not in index_def
    ):
        raise RuntimeError(
            "idx_ftip_job_runs_active must be a non-unique partial index on job_name where finished_at IS NULL"
        )


__all__ = [
    "ensure_schema",
    "MIGRATIONS",
]
