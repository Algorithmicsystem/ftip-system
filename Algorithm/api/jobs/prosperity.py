from __future__ import annotations

import datetime as dt
import logging
import socket
import uuid
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from psycopg.errors import UndefinedColumn
from psycopg.types.json import Json

from api import config, db, security
from api.prosperity.models import SnapshotRunRequest
from api.prosperity.routes import _require_db_enabled, snapshot_run

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

logger = logging.getLogger(__name__)

JOB_NAME = "prosperity_daily_snapshot"

DEFAULT_UNIVERSE = "AAPL,MSFT,NVDA,AMZN,TSLA,GOOGL,META,JPM,XOM,BRK.B"
CORE5_UNIVERSE = "AAPL,MSFT,NVDA,AMZN,TSLA"
SP500_SAMPLE_UNIVERSE = (
    "AAPL,MSFT,NVDA,AMZN,TSLA,GOOGL,META,JPM,XOM,BRK.B,JNJ,UNH,V,PG,HD"
)

SYMBOLS_MODES = {"core10", "core5", "sp500_sample"}


class CronSnapshotParams(BaseModel):
    as_of_date: Optional[dt.date] = None
    lookback_days: int = Field(365, ge=1)
    symbols_mode: str = Field("core10")
    concurrency: int = Field(3, ge=1)


def _utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def _parse_symbols(universe: Optional[str]) -> List[str]:
    symbols = (universe or DEFAULT_UNIVERSE).split(",")
    return sorted({s.strip().upper() for s in symbols if s and s.strip()})


def _retention_days() -> Optional[int]:
    raw = config.env("FTIP_RETENTION_DAYS")
    if raw is None:
        return None
    try:
        days = int(raw)
        return max(days, 0)
    except Exception:
        return None


def _lock_ttl_seconds() -> int:
    ttl = config.env_int("FTIP_JOB_LOCK_TTL_SECONDS", 1200)
    return max(ttl, 1)


def _lock_window_seconds() -> int:
    window = config.env_int("FTIP_JOB_LOCK_WINDOW_SEC", 120)
    return max(window, 0)


def _job_lock_owner() -> str:
    return (
        config.env("FTIP_JOB_LOCK_OWNER") or socket.gethostname() or "ftip-job-runner"
    )


def _delete_older_than(table: str, cutoff: dt.date) -> int:
    check = db.safe_fetchone("SELECT to_regclass(%s)", (table,))
    if not check or check[0] is None:
        return 0
    rows = db.safe_fetchall(
        f"DELETE FROM {table} WHERE as_of < %s RETURNING 1", (cutoff,)
    )
    deleted = len(rows)
    if deleted:
        logger.info(
            "[prosperity.retention] deleted rows",
            extra={"table": table, "deleted": deleted},
        )
    return deleted


def cleanup_retention(as_of_date: dt.date, retention_days: int) -> Dict[str, int]:
    cutoff = as_of_date - dt.timedelta(days=retention_days)
    deleted: Dict[str, int] = {}
    for table in (
        "prosperity_features_daily",
        "prosperity_signals_daily",
        "prosperity_strategy_graph_daily",
    ):
        deleted[table] = _delete_older_than(table, cutoff)
    return deleted


def _cleanup_stale_job_runs(cur, job_name: str, ttl_seconds: int) -> List[str]:
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
        RETURNING run_id
        """,
        (job_name, ttl_seconds),
    )
    rows = cur.fetchall() or []
    cleared = [str(row[0]) for row in rows]
    if cleared:
        logger.warning(
            "jobs.prosperity.daily_snapshot.stale_lock_cleared",
            extra={"job_name": job_name, "run_ids": cleared},
        )
    return cleared


def _acquire_job_lock(
    run_id: str,
    job_name: str,
    as_of_date: dt.date,
    requested: Dict[str, object],
    ttl_seconds: int,
    lock_owner: str,
) -> Tuple[bool, Dict[str, str]]:
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
            existing_run_id, started_at, owner, lock_acquired_at, lock_expires_at = (
                existing_locked
            )
            return False, {
                "run_id": str(existing_run_id),
                "started_at": started_at.isoformat() if started_at else None,
                "lock_owner": owner,
                "lock_acquired_at": (
                    lock_acquired_at.isoformat() if lock_acquired_at else None
                ),
                "lock_expires_at": (
                    lock_expires_at.isoformat() if lock_expires_at else None
                ),
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
            existing_run_id, started_at, owner, lock_acquired_at, lock_expires_at = (
                existing_pending
            )
            return False, {
                "run_id": str(existing_run_id),
                "started_at": started_at.isoformat() if started_at else None,
                "lock_owner": owner,
                "lock_acquired_at": (
                    lock_acquired_at.isoformat() if lock_acquired_at else None
                ),
                "lock_expires_at": (
                    lock_expires_at.isoformat() if lock_expires_at else None
                ),
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
                %s,
                %s,
                %s,
                now(),
                %s::jsonb,
                %s,
                now(),
                now() + (%s || ' seconds')::interval,
                'IN_PROGRESS',
                now(),
                now(),
                NULL
            )
            RETURNING run_id, started_at, lock_owner, lock_acquired_at, lock_expires_at
            """,
            (
                run_id,
                job_name,
                as_of_date,
                Json(requested),
                lock_owner,
                ttl_seconds,
            ),
        )
        inserted_row = cur.fetchone()
        conn.commit()
        if inserted_row:
            (
                inserted_run_id,
                started_at,
                owner,
                lock_acquired_at,
                lock_expires_at,
            ) = inserted_row
            return True, {
                "run_id": str(inserted_run_id),
                "started_at": started_at.isoformat() if started_at else None,
                "lock_owner": owner,
                "lock_acquired_at": (
                    lock_acquired_at.isoformat() if lock_acquired_at else None
                ),
                "lock_expires_at": (
                    lock_expires_at.isoformat() if lock_expires_at else None
                ),
            }

        return False, {
            "run_id": None,
            "started_at": None,
            "lock_owner": None,
            "lock_acquired_at": None,
            "lock_expires_at": None,
        }


def _release_job_lock(_run_id: str, _job_name: str) -> None:
    """Legacy stub retained for backwards compatibility."""

    return None


def _insert_job_run(*_args, **_kwargs) -> None:
    """Legacy stub retained for backwards compatibility."""

    return None


def _update_job_run(
    run_id: str,
    *,
    status: str,
    result: Optional[Dict[str, object]] = None,
    error: Optional[str] = None,
) -> None:
    with db.with_connection() as (conn, cur):
        cur.execute(
            """
            UPDATE ftip_job_runs
            SET finished_at = now(), status = %s, result = %s, error = %s, updated_at = now()
            WHERE run_id = %s
            """,
            (status, Json(result or {}), error, run_id),
        )
        conn.commit()


def _fetch_last_job_run(job_name: str) -> Optional[Dict[str, object]]:
    with db.with_connection() as (conn, cur):
        cur.execute(
            """
            SELECT run_id, job_name, as_of_date, started_at, finished_at, status, requested, result, error, lock_owner
            FROM ftip_job_runs
            WHERE job_name = %s
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (job_name,),
        )
        row = cur.fetchone()
        if not row:
            return None
        (
            run_id,
            jname,
            as_of_date,
            started_at,
            finished_at,
            status,
            requested,
            result,
            error,
            lock_owner,
        ) = row
        return {
            "run_id": str(run_id),
            "job_name": jname,
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "started_at": started_at.isoformat() if started_at else None,
            "finished_at": finished_at.isoformat() if finished_at else None,
            "status": status,
            "requested": requested,
            "result": result,
            "error": error,
            "lock_owner": lock_owner,
        }


def _result_status(result_payload: Dict[str, object]) -> str:
    failed = result_payload.get("symbols_failed") or []
    if failed:
        return "PARTIAL"
    return "SUCCESS"


def _symbols_for_mode(mode: str) -> List[str]:
    if mode == "core5":
        return _parse_symbols(CORE5_UNIVERSE)
    if mode == "sp500_sample":
        return _parse_symbols(SP500_SAMPLE_UNIVERSE)
    return _parse_symbols(DEFAULT_UNIVERSE)


def _build_snapshot_request(
    *,
    as_of_date: dt.date,
    lookback_days: int,
    symbols: List[str],
    lookback: int,
    concurrency: int,
) -> SnapshotRunRequest:
    to_date = as_of_date
    from_date = to_date - dt.timedelta(days=lookback_days)
    return SnapshotRunRequest(
        symbols=symbols,
        from_date=from_date,
        to_date=to_date,
        as_of_date=as_of_date,
        lookback=lookback,
        concurrency=concurrency,
        compute_strategy_graph=True,
    )


async def _run_daily_snapshot(
    request: Request,
    *,
    as_of_date: dt.date,
    lookback_days: int,
    symbols: List[str],
    concurrency: int,
) -> JSONResponse | Dict[str, object]:
    _require_db_enabled(write=True, read=True)

    run_id = str(uuid.uuid4())
    ttl_seconds = _lock_ttl_seconds()
    lock_owner = _job_lock_owner()
    lookback = config.env_int("FTIP_LOOKBACK", 252)

    req = _build_snapshot_request(
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        symbols=symbols,
        lookback=lookback,
        concurrency=concurrency,
    )

    requested_payload = {
        "symbols": symbols,
        "from_date": req.from_date.isoformat(),
        "to_date": req.to_date.isoformat(),
        "as_of_date": req.as_of_date.isoformat(),
        "lookback": lookback,
        "concurrency": concurrency,
    }

    run_recorded = False
    status = "FAILED"
    result_record: Dict[str, object] = {}
    error_message: Optional[str] = None

    try:
        acquired, lock_info = _acquire_job_lock(
            run_id,
            JOB_NAME,
            as_of_date,
            requested_payload,
            ttl_seconds,
            lock_owner,
        )
        if not acquired:
            logger.info(
                "daily_snapshot.lock_conflict",
                extra={
                    "run_id": lock_info.get("run_id"),
                    "as_of_date": as_of_date.isoformat(),
                    "lock_owner": lock_info.get("lock_owner"),
                },
            )
            return JSONResponse(
                status_code=409, content={"error": "locked", **lock_info}
            )

        run_recorded = True

        logger.info(
            "daily_snapshot.lock_acquired",
            extra={
                "run_id": run_id,
                "as_of_date": as_of_date.isoformat(),
                "lock_owner": lock_owner,
            },
        )
        logger.info(
            "daily_snapshot.start",
            extra={
                "run_id": run_id,
                "as_of_date": as_of_date.isoformat(),
                "lock_owner": lock_owner,
                "started_at": lock_info.get("started_at"),
            },
        )

        result = await snapshot_run(
            req, request, run_id=run_id, job_name=JOB_NAME, lock_owner=lock_owner
        )
        result_payload = result.get("result", {}) if isinstance(result, dict) else {}

        retention_info: Dict[str, int] = {}
        retention_days = _retention_days()
        if retention_days:
            retention_info = cleanup_retention(as_of_date, retention_days)

        summary = {
            "symbols_ok": result_payload.get("symbols_ok", []),
            "symbols_failed": result_payload.get("symbols_failed", []),
            "rows_written": result_payload.get("rows_written", {}),
            "timings": result.get("timings", {}),
        }

        if retention_info:
            summary["retention_deleted"] = retention_info

        status = _result_status(summary)
        response_body = {
            "status": result.get("status"),
            "as_of_date": as_of_date.isoformat(),
            "from_date": req.from_date.isoformat(),
            "to_date": req.to_date.isoformat(),
            "run_id": run_id,
            **summary,
        }
        result_record = {"response": response_body, "raw_result": result_payload}
        return response_body
    except UndefinedColumn as exc:
        error_message = str(exc)
        logger.exception(
            "jobs.prosperity.daily_snapshot.error", extra={"run_id": run_id}
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "database_schema_missing",
                "detail": error_message,
            },
        )
    except Exception as exc:
        error_message = str(exc)
        logger.exception(
            "jobs.prosperity.daily_snapshot.error", extra={"run_id": run_id}
        )
        return JSONResponse(
            status_code=500,
            content={"error": "unexpected_error", "detail": error_message},
        )
    finally:
        try:
            if run_recorded:
                _update_job_run(
                    run_id, status=status, result=result_record, error=error_message
                )
        except Exception:
            logger.warning(
                "jobs.prosperity.daily_snapshot.run_update_failed",
                extra={"run_id": run_id},
            )
        finally:
            if run_recorded:
                logger.info(
                    "daily_snapshot.end",
                    extra={
                        "run_id": run_id,
                        "as_of_date": as_of_date.isoformat(),
                        "lock_owner": lock_owner,
                        "status": status,
                    },
                )


@router.post("/prosperity/daily-snapshot")
async def prosperity_daily_snapshot(request: Request):
    today = _utc_today()
    as_of_date = today - dt.timedelta(days=1)
    lookback_days = config.env_int("FTIP_SNAPSHOT_WINDOW_DAYS", 365)
    concurrency = config.env_int("FTIP_SNAPSHOT_CONCURRENCY", 3)
    symbols = _parse_symbols(config.env("FTIP_UNIVERSE"))
    return await _run_daily_snapshot(
        request,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        symbols=symbols,
        concurrency=concurrency,
    )


@router.post("/prosperity/daily-snapshot/cron")
async def prosperity_daily_snapshot_cron(request: Request):
    body: Dict[str, object] = {}
    try:
        body = await request.json()
    except Exception:
        body = {}

    payload: Dict[str, object] = dict(request.query_params)
    payload.update(body if isinstance(body, dict) else {})

    try:
        params = CronSnapshotParams.model_validate(payload)
    except ValidationError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "detail": exc.errors()},
        )

    symbols_mode = (params.symbols_mode or "core10").strip()
    if symbols_mode not in SYMBOLS_MODES:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_symbols_mode", "allowed": sorted(SYMBOLS_MODES)},
        )

    as_of_date = params.as_of_date or _utc_today()
    return await _run_daily_snapshot(
        request,
        as_of_date=as_of_date,
        lookback_days=params.lookback_days,
        symbols=_symbols_for_mode(symbols_mode),
        concurrency=params.concurrency,
    )


@router.get("/prosperity/daily-snapshot/status")
async def prosperity_daily_snapshot_status():
    _require_db_enabled(read=True)
    last_run = _fetch_last_job_run(JOB_NAME)
    return last_run or {}


@router.get("/prosperity/daily-snapshot/summary")
async def prosperity_daily_snapshot_summary():
    _require_db_enabled(read=True)
    last_run = _fetch_last_job_run(JOB_NAME)
    if not last_run:
        return {
            "last_run": None,
            "last_as_of_date": None,
            "ok_count": 0,
            "failed_count": 0,
            "last_failures_sample": [],
            "coverage": None,
        }

    result = last_run.get("result") or {}
    response_payload = result.get("response") or {}
    symbols_ok = response_payload.get("symbols_ok") or []
    symbols_failed = response_payload.get("symbols_failed") or []

    last_as_of = last_run.get("as_of_date")
    coverage = None
    if last_as_of:
        try:
            coverage = _coverage_response(as_of_date=dt.date.fromisoformat(last_as_of))
        except Exception:
            coverage = None

    return {
        "last_run": {
            "run_id": last_run.get("run_id"),
            "started_at": last_run.get("started_at"),
            "ended_at": last_run.get("finished_at"),
            "status": last_run.get("status"),
        },
        "last_as_of_date": last_as_of,
        "ok_count": len(symbols_ok),
        "failed_count": len(symbols_failed),
        "last_failures_sample": symbols_failed[:5],
        "coverage": coverage,
    }


def _coverage_response(*, as_of_date: dt.date | None = None, run_id: str | None = None):
    filters = ["job_name=%s"]
    params: List[object] = [JOB_NAME]
    if as_of_date:
        filters.append("as_of_date=%s")
        params.append(as_of_date)
    if run_id:
        filters.append("run_id=%s")
        params.append(run_id)
    where_clause = " AND ".join(filters)
    params_tuple = tuple(params)

    attempted_row = db.safe_fetchone(
        f"SELECT COUNT(*) FROM prosperity_symbol_coverage WHERE {where_clause}",
        params_tuple,
    )
    attempted = int(attempted_row[0]) if attempted_row else 0

    grouped = db.safe_fetchall(
        f"SELECT status, reason_code, COUNT(*) FROM prosperity_symbol_coverage WHERE {where_clause} GROUP BY status, reason_code",
        params_tuple,
    )
    status_counts = {"OK": 0, "FAILED": 0, "SKIPPED": 0}
    by_reason: Dict[str, int] = {}
    for status, reason_code, count in grouped:
        if status in status_counts:
            status_counts[status] += int(count)
        reason_key = reason_code or "UNKNOWN"
        by_reason[reason_key] = by_reason.get(reason_key, 0) + int(count)

    failed_rows = db.safe_fetchall(
        f"""
        SELECT symbol, reason_code, reason_detail, bars_required, bars_returned
        FROM prosperity_symbol_coverage
        WHERE {where_clause} AND status='FAILED'
        ORDER BY symbol ASC
        """,
        params_tuple,
    )
    failed_symbols = []
    for symbol, reason_code, reason_detail, bars_required, bars_returned in failed_rows:
        detail = reason_detail or reason_code or "UNKNOWN"
        failed_symbols.append(
            {
                "symbol": symbol,
                "reason": detail,
                "reason_code": reason_code,
                "reason_detail": detail,
                "bars_required": bars_required,
                "bars_returned": bars_returned,
            }
        )

    payload: Dict[str, object] = {
        "attempted": attempted,
        "ok": status_counts.get("OK", 0),
        "failed": status_counts.get("FAILED", 0),
        "skipped": status_counts.get("SKIPPED", 0),
        "by_reason_code": by_reason,
        "failed_symbols": failed_symbols,
    }
    if as_of_date:
        payload["as_of_date"] = as_of_date.isoformat()
    if run_id:
        payload["run_id"] = run_id
    return payload


@router.get("/prosperity/daily-snapshot/coverage")
async def prosperity_daily_snapshot_coverage(as_of_date: dt.date):
    _require_db_enabled(read=True)
    return _coverage_response(as_of_date=as_of_date)


@router.get("/prosperity/daily-snapshot/runs/{run_id}/coverage")
async def prosperity_daily_snapshot_run_coverage(run_id: str):
    _require_db_enabled(read=True)
    return _coverage_response(run_id=run_id)
