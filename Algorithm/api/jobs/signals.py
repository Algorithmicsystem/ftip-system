from __future__ import annotations

import datetime as dt
import socket
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from psycopg.types.json import Json

from api import config, db, security
from api.data_providers import canonical_symbol
from api.signal_engine import compute_daily_signal

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)


class SignalsDailyRequest(BaseModel):
    as_of_date: dt.date
    symbols: Optional[List[str]] = None


class SignalsIntradayRequest(BaseModel):
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


@router.post("/signals/daily")
async def compute_signals_daily(req: SignalsDailyRequest):
    _require_db_enabled(write=True, read=True)
    as_of_date = req.as_of_date
    symbols = [canonical_symbol(sym) for sym in (req.symbols or _load_active_symbols())]
    if not symbols:
        raise HTTPException(status_code=400, detail="no symbols available")

    run_id = uuid.uuid4().hex
    job_name = "signals.daily"
    acquired, lock_info = _acquire_job_lock(run_id, job_name, as_of_date, req.model_dump(), _lock_ttl_seconds(), _job_lock_owner())
    if not acquired:
        return JSONResponse(status_code=409, content={"error": "locked", **lock_info})

    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []
    rows_written = 0

    for symbol in symbols:
        try:
            feat_row = db.safe_fetchone(
                """
                SELECT ret_1d, ret_5d, ret_21d, vol_21d, vol_63d, atr_14, atr_pct,
                       trend_slope_21d, trend_r2_21d, trend_slope_63d, trend_r2_63d,
                       mom_vol_adj_21d, maxdd_63d, dollar_vol_21d, sentiment_score, sentiment_surprise,
                       regime_label, regime_strength
                FROM features_daily
                WHERE symbol = %s AND as_of_date = %s
                """,
                (symbol, as_of_date),
            )
            if not feat_row:
                raise ValueError("missing features")
            feature_keys = [
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
            ]
            features = dict(zip(feature_keys, feat_row))

            quality_row = db.safe_fetchone(
                """
                SELECT quality_score
                FROM quality_daily
                WHERE symbol = %s AND as_of_date = %s
                """,
                (symbol, as_of_date),
            )
            quality_score = int(quality_row[0]) if quality_row else 0

            close_row = db.safe_fetchone(
                """
                SELECT close
                FROM market_bars_daily
                WHERE symbol = %s AND as_of_date = %s
                """,
                (symbol, as_of_date),
            )
            latest_close = float(close_row[0]) if close_row and close_row[0] is not None else None

            signal = compute_daily_signal(features, quality_score, latest_close)
            db.safe_execute(
                """
                INSERT INTO signals_daily(
                    symbol, as_of_date, action, score, confidence, entry_low, entry_high, stop_loss,
                    take_profit_1, take_profit_2, horizon_days, reason_codes, reason_details,
                    signal_version, computed_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,now())
                ON CONFLICT (symbol, as_of_date)
                DO UPDATE SET
                    action = EXCLUDED.action,
                    score = EXCLUDED.score,
                    confidence = EXCLUDED.confidence,
                    entry_low = EXCLUDED.entry_low,
                    entry_high = EXCLUDED.entry_high,
                    stop_loss = EXCLUDED.stop_loss,
                    take_profit_1 = EXCLUDED.take_profit_1,
                    take_profit_2 = EXCLUDED.take_profit_2,
                    horizon_days = EXCLUDED.horizon_days,
                    reason_codes = EXCLUDED.reason_codes,
                    reason_details = EXCLUDED.reason_details,
                    signal_version = 1,
                    computed_at = now()
                """,
                (
                    symbol,
                    as_of_date,
                    signal["action"],
                    signal["score"],
                    signal["confidence"],
                    signal.get("entry_low"),
                    signal.get("entry_high"),
                    signal.get("stop_loss"),
                    signal.get("take_profit_1"),
                    signal.get("take_profit_2"),
                    21,
                    Json(signal.get("reason_codes") or []),
                    Json({}),
                ),
            )
            rows_written += 1
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "OK")
            symbols_ok.append(symbol)
        except Exception as exc:
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "FAILED", reason_code="SIGNAL_ERROR", reason_detail=str(exc))
            symbols_failed.append({"symbol": symbol, "reason": str(exc), "reason_code": "SIGNAL_ERROR"})

    status = "ok" if not symbols_failed else "partial" if symbols_ok else "failed"
    payload = {
        "status": status,
        "run_id": run_id,
        "symbols_ok": symbols_ok,
        "symbols_failed": symbols_failed,
        "rows_written": {"signals_daily": rows_written},
        "timings": {},
    }
    _update_job_run(run_id, status.upper(), payload, None if status == "ok" else "partial failures")
    return payload


@router.post("/signals/intraday")
async def compute_signals_intraday(req: SignalsIntradayRequest):
    _require_db_enabled(write=True, read=True)
    if req.start_ts >= req.end_ts:
        raise HTTPException(status_code=400, detail="start_ts must be before end_ts")

    as_of_date = req.end_ts.astimezone(dt.timezone.utc).date()
    symbols = [canonical_symbol(sym) for sym in (req.symbols or _load_active_symbols())]
    if not symbols:
        raise HTTPException(status_code=400, detail="no symbols available")

    run_id = uuid.uuid4().hex
    job_name = "signals.intraday"
    acquired, lock_info = _acquire_job_lock(run_id, job_name, as_of_date, req.model_dump(), _lock_ttl_seconds(), _job_lock_owner())
    if not acquired:
        return JSONResponse(status_code=409, content={"error": "locked", **lock_info})

    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []
    rows_written = 0

    for symbol in symbols:
        try:
            feat_row = db.safe_fetchone(
                """
                SELECT ret_1bar
                FROM features_intraday
                WHERE symbol = %s AND ts <= %s AND timeframe = %s
                ORDER BY ts DESC
                LIMIT 1
                """,
                (symbol, req.end_ts, req.timeframe),
            )
            if not feat_row:
                raise ValueError("missing intraday features")
            score = float(feat_row[0] or 0.0)
            action = "HOLD"
            if score > 0.002:
                action = "BUY"
            elif score < -0.002:
                action = "SELL"
            confidence = min(1.0, abs(score) * 10)

            db.safe_execute(
                """
                INSERT INTO signals_intraday(
                    symbol, ts, timeframe, action, score, confidence, reason_codes, signal_version, computed_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,1,now())
                ON CONFLICT (symbol, ts, timeframe)
                DO UPDATE SET
                    action = EXCLUDED.action,
                    score = EXCLUDED.score,
                    confidence = EXCLUDED.confidence,
                    reason_codes = EXCLUDED.reason_codes,
                    signal_version = 1,
                    computed_at = now()
                """,
                (
                    symbol,
                    req.end_ts,
                    req.timeframe,
                    action,
                    score,
                    confidence,
                    Json(["INTRADAY_MOMENTUM"]),
                ),
            )
            rows_written += 1
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "OK")
            symbols_ok.append(symbol)
        except Exception as exc:
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "FAILED", reason_code="SIGNAL_ERROR", reason_detail=str(exc))
            symbols_failed.append({"symbol": symbol, "reason": str(exc), "reason_code": "SIGNAL_ERROR"})

    status = "ok" if not symbols_failed else "partial" if symbols_ok else "failed"
    payload = {
        "status": status,
        "run_id": run_id,
        "symbols_ok": symbols_ok,
        "symbols_failed": symbols_failed,
        "rows_written": {"signals_intraday": rows_written},
        "timings": {},
    }
    _update_job_run(run_id, status.upper(), payload, None if status == "ok" else "partial failures")
    return payload
