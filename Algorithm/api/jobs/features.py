from __future__ import annotations

import datetime as dt
import socket
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from psycopg.types.json import Json

from api import config, db, security
from api.data_providers import canonical_symbol
from api.feature_engine import compute_daily_features, compute_intraday_features

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)


class FeaturesDailyRequest(BaseModel):
    as_of_date: dt.date
    lookback_days: int = Field(400, ge=30, le=2000)
    symbols: Optional[List[str]] = None


class FeaturesIntradayRequest(BaseModel):
    start_ts: dt.datetime
    end_ts: dt.datetime
    timeframe: str = "5m"
    symbols: Optional[List[str]] = None


def _require_db_enabled(write: bool = False, read: bool = False) -> None:
    if not db.db_enabled():
        raise HTTPException(status_code=503, detail="database disabled")
    if write and not db.db_write_enabled():
        raise HTTPException(status_code=503, detail="database writes disabled")
    if read and not db.db_read_enabled():
        raise HTTPException(status_code=503, detail="database reads disabled")


def _lock_ttl_seconds() -> int:
    ttl = config.env_int("FTIP_JOB_LOCK_TTL_SECONDS", 1200)
    return max(ttl, 1)


def _lock_window_seconds() -> int:
    window = config.env_int("FTIP_JOB_LOCK_WINDOW_SEC", 120)
    return max(window, 0)


def _job_lock_owner() -> str:
    return config.env("FTIP_JOB_LOCK_OWNER") or socket.gethostname() or "ftip-job-runner"


def _cleanup_stale_job_runs(cur, job_name: str, ttl_seconds: int) -> None:
    cur.execute(
        """
        UPDATE ftip_job_runs
        SET status = 'FAILED',
            error = 'stale lock cleared',
            finished_at = now(),
            updated_at = now()
        WHERE job_name = %s
          AND finished_at IS NULL
          AND started_at < now() - (%s || ' seconds')::interval
        """,
        (job_name, ttl_seconds),
    )


def _acquire_job_lock(
    run_id: str,
    job_name: str,
    as_of_date: dt.date,
    requested: Dict[str, object],
    ttl_seconds: int,
    lock_owner: str,
) -> tuple[bool, Dict[str, Optional[str]]]:
    with db.with_connection() as (conn, cur):
        _cleanup_stale_job_runs(cur, job_name, ttl_seconds)
        lock_window_seconds = _lock_window_seconds()

        cur.execute(
            """
            SELECT run_id, started_at, lock_owner, lock_acquired_at, lock_expires_at
            FROM ftip_job_runs
            WHERE job_name = %s
              AND (finished_at IS NULL OR finished_at > now() - (%s || ' seconds')::interval)
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """,
            (job_name, lock_window_seconds),
        )
        existing_locked = cur.fetchone()
        if existing_locked:
            conn.commit()
            existing_run_id, started_at, owner, lock_acquired_at, lock_expires_at = existing_locked
            return False, {
                "run_id": str(existing_run_id),
                "started_at": started_at.isoformat() if started_at else None,
                "lock_owner": owner,
                "lock_acquired_at": lock_acquired_at.isoformat() if lock_acquired_at else None,
                "lock_expires_at": lock_expires_at.isoformat() if lock_expires_at else None,
            }

        cur.execute(
            """
            SELECT run_id, started_at, lock_owner, lock_acquired_at, lock_expires_at
            FROM ftip_job_runs
            WHERE job_name = %s
              AND (finished_at IS NULL OR finished_at > now() - (%s || ' seconds')::interval)
            LIMIT 1
            """,
            (job_name, lock_window_seconds),
        )
        existing_pending = cur.fetchone()
        if existing_pending:
            conn.commit()
            existing_run_id, started_at, owner, lock_acquired_at, lock_expires_at = existing_pending
            return False, {
                "run_id": str(existing_run_id),
                "started_at": started_at.isoformat() if started_at else None,
                "lock_owner": owner,
                "lock_acquired_at": lock_acquired_at.isoformat() if lock_acquired_at else None,
                "lock_expires_at": lock_expires_at.isoformat() if lock_expires_at else None,
            }

        cur.execute(
            """
            INSERT INTO ftip_job_runs (
                run_id,
                job_name,
                as_of_date,
                started_at,
                requested,
                lock_owner,
                lock_acquired_at,
                lock_expires_at,
                status,
                created_at,
                updated_at,
                finished_at
            )
            VALUES (
                %s, %s, %s, now(), %s, %s, now(), now() + (%s || ' seconds')::interval,
                'RUNNING', now(), now(), NULL
            )
            """,
            (run_id, job_name, as_of_date, Json(requested), lock_owner, ttl_seconds),
        )
        conn.commit()
        return True, {"run_id": run_id}


def _update_job_run(run_id: str, status: str, result: Dict[str, object], error: Optional[str]) -> None:
    with db.with_connection() as (conn, cur):
        cur.execute(
            """
            UPDATE ftip_job_runs
            SET finished_at = now(), status = %s, result = %s, error = %s, updated_at = now()
            WHERE run_id = %s
            """,
            (status, Json(result), error, run_id),
        )
        conn.commit()


def _log_symbol_coverage(run_id: str, job_name: str, as_of_date: dt.date, symbol: str, status: str, reason_code: Optional[str] = None, reason_detail: Optional[str] = None) -> None:
    db.safe_execute(
        """
        INSERT INTO prosperity_symbol_coverage(
            run_id, job_name, as_of_date, symbol, status, reason_code, reason_detail
        ) VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        (run_id, job_name, as_of_date, symbol, status, reason_code, reason_detail),
    )


def _load_active_symbols() -> List[str]:
    rows = db.safe_fetchall("SELECT symbol FROM market_symbols WHERE is_active = TRUE")
    return [row[0] for row in rows]


def _update_quality_score(symbol: str, as_of_date: dt.date) -> None:
    row = db.safe_fetchone(
        """
        SELECT bars_ok, fundamentals_ok, sentiment_ok, intraday_ok
        FROM quality_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )
    if not row:
        return
    bars_ok, fundamentals_ok, sentiment_ok, intraday_ok = row
    score = 0
    score += 25 if bars_ok else 0
    score += 25 if fundamentals_ok else 0
    score += 25 if sentiment_ok else 0
    score += 25 if intraday_ok else 0
    db.safe_execute(
        """
        UPDATE quality_daily
        SET quality_score = %s, updated_at = now()
        WHERE symbol = %s AND as_of_date = %s
        """,
        (score, symbol, as_of_date),
    )


@router.post("/features/daily")
async def compute_features_daily(req: FeaturesDailyRequest):
    _require_db_enabled(write=True, read=True)
    as_of_date = req.as_of_date
    lookback_start = as_of_date - dt.timedelta(days=req.lookback_days)

    symbols = [canonical_symbol(sym) for sym in (req.symbols or _load_active_symbols())]
    if not symbols:
        raise HTTPException(status_code=400, detail="no symbols available")

    run_id = uuid.uuid4().hex
    job_name = "features.daily"
    acquired, lock_info = _acquire_job_lock(run_id, job_name, as_of_date, req.model_dump(), _lock_ttl_seconds(), _job_lock_owner())
    if not acquired:
        return JSONResponse(status_code=409, content={"error": "locked", **lock_info})

    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []
    rows_written = 0

    for symbol in symbols:
        try:
            bars = db.safe_fetchall(
                """
                SELECT as_of_date, open, high, low, close, volume
                FROM market_bars_daily
                WHERE symbol = %s AND as_of_date >= %s AND as_of_date <= %s
                ORDER BY as_of_date
                """,
                (symbol, lookback_start, as_of_date),
            )
            bar_rows = [
                {
                    "symbol": symbol,
                    "as_of_date": row[0],
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                }
                for row in bars
            ]
            if not bar_rows:
                raise ValueError("no bars available")

            sentiment_row = db.safe_fetchone(
                """
                SELECT sentiment_score, sentiment_mean
                FROM sentiment_daily
                WHERE symbol = %s AND as_of_date = %s
                """,
                (symbol, as_of_date),
            )
            sentiment_score = sentiment_row[0] if sentiment_row else None
            sentiment_mean = sentiment_row[1] if sentiment_row else None

            features = compute_daily_features(bar_rows, sentiment_score=sentiment_score, sentiment_mean=sentiment_mean)
            if not features:
                raise ValueError("feature computation failed")

            db.safe_execute(
                """
                INSERT INTO features_daily(
                    symbol, as_of_date, ret_1d, ret_5d, ret_21d, vol_21d, vol_63d,
                    atr_14, atr_pct, trend_slope_21d, trend_r2_21d, trend_slope_63d, trend_r2_63d,
                    mom_vol_adj_21d, maxdd_63d, dollar_vol_21d, sentiment_score, sentiment_surprise,
                    regime_label, regime_strength, feature_version, computed_at
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,now()
                )
                ON CONFLICT (symbol, as_of_date)
                DO UPDATE SET
                    ret_1d = EXCLUDED.ret_1d,
                    ret_5d = EXCLUDED.ret_5d,
                    ret_21d = EXCLUDED.ret_21d,
                    vol_21d = EXCLUDED.vol_21d,
                    vol_63d = EXCLUDED.vol_63d,
                    atr_14 = EXCLUDED.atr_14,
                    atr_pct = EXCLUDED.atr_pct,
                    trend_slope_21d = EXCLUDED.trend_slope_21d,
                    trend_r2_21d = EXCLUDED.trend_r2_21d,
                    trend_slope_63d = EXCLUDED.trend_slope_63d,
                    trend_r2_63d = EXCLUDED.trend_r2_63d,
                    mom_vol_adj_21d = EXCLUDED.mom_vol_adj_21d,
                    maxdd_63d = EXCLUDED.maxdd_63d,
                    dollar_vol_21d = EXCLUDED.dollar_vol_21d,
                    sentiment_score = EXCLUDED.sentiment_score,
                    sentiment_surprise = EXCLUDED.sentiment_surprise,
                    regime_label = EXCLUDED.regime_label,
                    regime_strength = EXCLUDED.regime_strength,
                    feature_version = 1,
                    computed_at = now()
                """,
                (
                    symbol,
                    as_of_date,
                    features.get("ret_1d"),
                    features.get("ret_5d"),
                    features.get("ret_21d"),
                    features.get("vol_21d"),
                    features.get("vol_63d"),
                    features.get("atr_14"),
                    features.get("atr_pct"),
                    features.get("trend_slope_21d"),
                    features.get("trend_r2_21d"),
                    features.get("trend_slope_63d"),
                    features.get("trend_r2_63d"),
                    features.get("mom_vol_adj_21d"),
                    features.get("maxdd_63d"),
                    features.get("dollar_vol_21d"),
                    features.get("sentiment_score"),
                    features.get("sentiment_surprise"),
                    features.get("regime_label"),
                    features.get("regime_strength"),
                ),
            )
            rows_written += 1
            _update_quality_score(symbol, as_of_date)
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "OK")
            symbols_ok.append(symbol)
        except Exception as exc:
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "FAILED", reason_code="FEATURE_ERROR", reason_detail=str(exc))
            symbols_failed.append({"symbol": symbol, "reason": str(exc), "reason_code": "FEATURE_ERROR"})

    status = "ok" if not symbols_failed else "partial" if symbols_ok else "failed"
    payload = {
        "status": status,
        "run_id": run_id,
        "symbols_ok": symbols_ok,
        "symbols_failed": symbols_failed,
        "rows_written": {"features_daily": rows_written},
        "timings": {},
    }
    _update_job_run(run_id, status.upper(), payload, None if status == "ok" else "partial failures")
    return payload


@router.post("/features/intraday")
async def compute_features_intraday(req: FeaturesIntradayRequest):
    _require_db_enabled(write=True, read=True)
    if req.start_ts >= req.end_ts:
        raise HTTPException(status_code=400, detail="start_ts must be before end_ts")

    as_of_date = req.end_ts.astimezone(dt.timezone.utc).date()
    symbols = [canonical_symbol(sym) for sym in (req.symbols or _load_active_symbols())]
    if not symbols:
        raise HTTPException(status_code=400, detail="no symbols available")

    run_id = uuid.uuid4().hex
    job_name = "features.intraday"
    acquired, lock_info = _acquire_job_lock(run_id, job_name, as_of_date, req.model_dump(), _lock_ttl_seconds(), _job_lock_owner())
    if not acquired:
        return JSONResponse(status_code=409, content={"error": "locked", **lock_info})

    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []
    rows_written = 0

    for symbol in symbols:
        try:
            bars = db.safe_fetchall(
                """
                SELECT ts, open, high, low, close, volume
                FROM market_bars_intraday
                WHERE symbol = %s AND ts >= %s AND ts <= %s AND timeframe = %s
                ORDER BY ts
                """,
                (symbol, req.start_ts, req.end_ts, req.timeframe),
            )
            bar_rows = [
                {
                    "symbol": symbol,
                    "ts": row[0],
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                }
                for row in bars
            ]
            if not bar_rows:
                raise ValueError("no intraday bars available")

            features_rows = compute_intraday_features(bar_rows, req.timeframe)
            with db.with_connection() as (conn, cur):
                for row in features_rows:
                    cur.execute(
                        """
                        INSERT INTO features_intraday(
                            symbol, ts, timeframe, ret_1bar, vol_n, trend_slope_n, feature_version, computed_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,1,now())
                        ON CONFLICT (symbol, ts, timeframe)
                        DO UPDATE SET
                            ret_1bar = EXCLUDED.ret_1bar,
                            vol_n = EXCLUDED.vol_n,
                            trend_slope_n = EXCLUDED.trend_slope_n,
                            feature_version = 1,
                            computed_at = now()
                        """,
                        (
                            row["symbol"],
                            row["ts"],
                            row["timeframe"],
                            row.get("ret_1bar"),
                            row.get("vol_n"),
                            row.get("trend_slope_n"),
                        ),
                    )
                    rows_written += 1
                conn.commit()
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "OK")
            symbols_ok.append(symbol)
        except Exception as exc:
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "FAILED", reason_code="FEATURE_ERROR", reason_detail=str(exc))
            symbols_failed.append({"symbol": symbol, "reason": str(exc), "reason_code": "FEATURE_ERROR"})

    status = "ok" if not symbols_failed else "partial" if symbols_ok else "failed"
    payload = {
        "status": status,
        "run_id": run_id,
        "symbols_ok": symbols_ok,
        "symbols_failed": symbols_failed,
        "rows_written": {"features_intraday": rows_written},
        "timings": {},
    }
    _update_job_run(run_id, status.upper(), payload, None if status == "ok" else "partial failures")
    return payload
