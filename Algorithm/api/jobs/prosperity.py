from __future__ import annotations

import datetime as dt
import logging
import socket
import uuid
from typing import Dict, List, Optional, Tuple

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

DEFAULT_UNIVERSE = (
    "AAPL,MSFT,NVDA,AMZN,TSLA,GOOGL,META,JPM,XOM,BRK.B"
)


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


def _job_lock_owner() -> str:
    return (
        config.env("FTIP_JOB_LOCK_OWNER")
        or socket.gethostname()
        or "ftip-job-runner"
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
          AND status = 'IN_PROGRESS'
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

        cur.execute(
            """
            SELECT run_id, started_at, lock_owner
            FROM ftip_job_runs
            WHERE job_name = %s AND status = 'IN_PROGRESS'
            FOR UPDATE
            """,
            (job_name,),
        )
        existing = cur.fetchone()
        if existing:
            existing_run_id, started_at, owner = existing
            conn.commit()
            return False, {
                "run_id": str(existing_run_id),
                "started_at": started_at.isoformat() if started_at else None,
                "lock_owner": owner,
            }

        cur.execute(
            """
            INSERT INTO ftip_job_runs (
                run_id,
                job_name,
                as_of_date,
                started_at,
                status,
                requested,
                lock_owner,
                updated_at
            )
            VALUES (%s, %s, %s, now(), 'IN_PROGRESS', %s, %s, now())
            ON CONFLICT ON CONSTRAINT ftip_job_runs_active_idx DO NOTHING
            RETURNING started_at
            """,
            (run_id, job_name, as_of_date, Json(requested), lock_owner),
        )
        inserted = cur.fetchone()
        if not inserted:
            cur.execute(
                """
                SELECT run_id, started_at, lock_owner
                FROM ftip_job_runs
                WHERE job_name = %s AND status = 'IN_PROGRESS'
                """,
                (job_name,),
            )
            existing_run_id, started_at, owner = cur.fetchone() or (None, None, None)
            conn.commit()
            return False, {
                "run_id": str(existing_run_id) if existing_run_id else None,
                "started_at": started_at.isoformat() if started_at else None,
                "lock_owner": owner,
            }

        started_at = inserted[0]
        conn.commit()

        return True, {
            "run_id": run_id,
            "started_at": started_at.isoformat() if started_at else None,
            "lock_owner": lock_owner,
        }


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


@router.post("/prosperity/daily-snapshot")
async def prosperity_daily_snapshot(request: Request):
    _require_db_enabled(write=True, read=True)

    run_id = str(uuid.uuid4())
    ttl_seconds = _lock_ttl_seconds()
    lock_owner = _job_lock_owner()

    today = _utc_today()
    as_of_date = today - dt.timedelta(days=1)
    to_date = as_of_date
    window_days = config.env_int("FTIP_SNAPSHOT_WINDOW_DAYS", 365)
    from_date = to_date - dt.timedelta(days=window_days)

    lookback = config.env_int("FTIP_LOOKBACK", 252)
    concurrency = config.env_int("FTIP_SNAPSHOT_CONCURRENCY", 3)
    symbols = _parse_symbols(config.env("FTIP_UNIVERSE"))

    req = SnapshotRunRequest(
        symbols=symbols,
        from_date=from_date,
        to_date=to_date,
        as_of_date=as_of_date,
        lookback=lookback,
        concurrency=concurrency,
        compute_strategy_graph=True,
    )

    requested_payload = {
        "symbols": symbols,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "as_of_date": as_of_date.isoformat(),
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
                "jobs.prosperity.daily_snapshot.lock_conflict",
                extra={"job_name": JOB_NAME, **lock_info},
            )
            return JSONResponse(status_code=409, content={"error": "locked", **lock_info})

        run_recorded = True

        logger.info(
            "jobs.prosperity.daily_snapshot.start",
            extra={
                "run_id": run_id,
                "job_name": JOB_NAME,
                "lock_owner": lock_owner,
                "started_at": lock_info.get("started_at"),
            },
        )

        result = await snapshot_run(req, request)
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
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "run_id": run_id,
            **summary,
        }
        result_record = {"response": response_body, "raw_result": result_payload}
        return response_body
    except UndefinedColumn as exc:
        error_message = str(exc)
        logger.exception("jobs.prosperity.daily_snapshot.error", extra={"run_id": run_id})
        return JSONResponse(
            status_code=500,
            content={
                "error": "database_schema_missing",
                "detail": error_message,
            },
        )
    except Exception as exc:
        error_message = str(exc)
        logger.exception("jobs.prosperity.daily_snapshot.error", extra={"run_id": run_id})
        return JSONResponse(
            status_code=500,
            content={"error": "unexpected_error", "detail": error_message},
        )
    finally:
        try:
            if run_recorded:
                _update_job_run(run_id, status=status, result=result_record, error=error_message)
        except Exception:
            logger.warning("jobs.prosperity.daily_snapshot.run_update_failed", extra={"run_id": run_id})
        finally:
            if run_recorded:
                logger.info(
                    "jobs.prosperity.daily_snapshot.lock_released",
                    extra={"run_id": run_id, "job_name": JOB_NAME, "status": status},
                )


@router.get("/prosperity/daily-snapshot/status")
async def prosperity_daily_snapshot_status():
    _require_db_enabled(read=True)
    last_run = _fetch_last_job_run(JOB_NAME)
    return last_run or {}
