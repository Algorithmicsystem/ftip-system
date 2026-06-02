from __future__ import annotations

import datetime as dt
import logging
import socket
import uuid
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from psycopg.errors import UndefinedColumn
from psycopg.types.json import Json
from api.errors import simple_error

from api import config, db, security
from api.prosperity.constants import FEATURE_VERSION
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
LARGE30_UNIVERSE = (
    "AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,BRK.B,JPM,JNJ,V,UNH,XOM,WMT,PG,"
    "MA,HD,CVX,MRK,LLY,ABBV,PEP,KO,AVGO,COST,MCD,TMO,ACN,DHR,NEE"
)

SYMBOLS_MODES = {"core10", "core5", "sp500_sample", "large30"}
RETENTION_TABLE_DATE_COLUMNS = {
    "prosperity_features_daily": "as_of",
    "prosperity_signals_daily": "as_of",
    "prosperity_strategy_signals_daily": "as_of_date",
    "prosperity_ensemble_signals_daily": "as_of_date",
}
RETENTION_DATE_COLUMN_CANDIDATES = (
    "as_of",
    "as_of_date",
    "date",
    "created_at",
)


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
    date_column = _resolve_retention_date_column(table)
    if not date_column:
        raise ValueError(f"no retention date column available for {table}")
    rows = db.safe_fetchall(
        f"DELETE FROM {table} WHERE {date_column} < %s RETURNING 1", (cutoff,)
    )
    deleted = len(rows)
    if deleted:
        logger.info(
            "[prosperity.retention] deleted rows",
            extra={"table": table, "date_column": date_column, "deleted": deleted},
        )
    return deleted


def _retention_table_columns(table: str) -> List[str]:
    rows = db.safe_fetchall(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table,),
    )
    return [str(row[0]) for row in rows if row and row[0]]


def _resolve_retention_date_column(table: str) -> Optional[str]:
    columns = set(_retention_table_columns(table))
    if not columns:
        return None
    mapped = RETENTION_TABLE_DATE_COLUMNS.get(table)
    if mapped and mapped in columns:
        return mapped
    for candidate in RETENTION_DATE_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def cleanup_retention(as_of_date: dt.date, retention_days: int) -> Dict[str, int]:
    return cleanup_retention_report(as_of_date, retention_days)["deleted"]


def cleanup_retention_report(
    as_of_date: dt.date, retention_days: int
) -> Dict[str, object]:
    cutoff = as_of_date - dt.timedelta(days=retention_days)
    deleted: Dict[str, int] = {}
    skipped: Dict[str, str] = {}
    warnings: List[str] = []
    for table in (
        "prosperity_features_daily",
        "prosperity_signals_daily",
        "prosperity_strategy_signals_daily",
        "prosperity_ensemble_signals_daily",
    ):
        try:
            deleted[table] = _delete_older_than(table, cutoff)
        except ValueError as exc:
            warning = (
                f"Skipped retention cleanup for {table}: {exc}. "
                f"Supported date columns are {', '.join(RETENTION_DATE_COLUMN_CANDIDATES)}."
            )
            skipped[table] = str(exc)
            warnings.append(warning)
            logger.warning(
                "jobs.prosperity.retention.skipped",
                extra={"table": table, "warning": warning},
            )
        except Exception as exc:
            warning = f"Skipped retention cleanup for {table}: {exc}"
            skipped[table] = str(exc)
            warnings.append(warning)
            logger.warning(
                "jobs.prosperity.retention.failed",
                extra={"table": table, "warning": warning},
            )
    return {"deleted": deleted, "skipped": skipped, "warnings": warnings}


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
        response_payload = {}
        if isinstance(result, dict):
            response_payload = result.get("response") or {}
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
            "dataset_fingerprint": response_payload.get("dataset_fingerprint"),
            "feature_version": response_payload.get("feature_version")
            or FEATURE_VERSION,
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
    if mode == "large30":
        return _parse_symbols(LARGE30_UNIVERSE)
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
) -> Union[JSONResponse, Dict[str, object]]:
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
            return simple_error("locked", "job is already in progress", status_code=409, extra=lock_info)

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

        retention_info: Dict[str, object] = {}
        retention_days = _retention_days()
        if retention_days:
            retention_info = cleanup_retention_report(as_of_date, retention_days)

        summary = {
            "symbols_ok": result_payload.get("symbols_ok", []),
            "symbols_failed": result_payload.get("symbols_failed", []),
            "symbols_degraded": result_payload.get("symbols_degraded", []),
            "summary_stats": result_payload.get("summary_stats", {}),
            "failure_report": result_payload.get("failure_report", {}),
            "failure_summary": result_payload.get("failure_summary", {}),
            "provider_usage_summary": result_payload.get("provider_usage_summary", {}),
            "provider_failure_summary": result_payload.get(
                "provider_failure_summary", {}
            ),
            "provider_run_suppression_summary": result_payload.get(
                "provider_run_suppression_summary", {}
            ),
            "rows_written": result_payload.get("rows_written", {}),
            "strategy_graph_rows": result_payload.get("strategy_graph_rows", {}),
            "timings": result.get("timings", {}),
            "dataset_fingerprint": result.get("dataset_fingerprint"),
            "feature_version": result.get("feature_version"),
        }

        if retention_info:
            summary["retention_deleted"] = retention_info.get("deleted", {})
            if retention_info.get("skipped"):
                summary["retention_skipped"] = retention_info.get("skipped")
            if retention_info.get("warnings"):
                summary["retention_warnings"] = retention_info.get("warnings")

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
    lookback_days = config.env_int("FTIP_SNAPSHOT_WINDOW_DAYS", 420)
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


def _coverage_response(
    *, as_of_date: Optional[dt.date] = None, run_id: Optional[str] = None
):
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


# =============================================================================
# Nightly recalibration job
# =============================================================================

RECALIB_JOB_NAME = "prosperity_nightly_recalibrate"


class RecalibrateParams(BaseModel):
    symbols_mode: str = Field("core10")
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    lookback: int = Field(252, ge=60)
    optimize_horizon: int = Field(21, ge=5)
    min_trades_per_side: int = Field(5, ge=1)
    horizons: Optional[List[int]] = None


def _recalib_date_range(
    from_date: Optional[str], to_date: Optional[str]
) -> Tuple[str, str]:
    today = _utc_today()
    td = dt.date.fromisoformat(to_date) if to_date else today - dt.timedelta(days=1)
    fd = dt.date.fromisoformat(from_date) if from_date else td - dt.timedelta(days=365)
    return fd.isoformat(), td.isoformat()


def _run_recalibrate_symbol(
    symbol: str,
    from_date: str,
    to_date: str,
    lookback: int,
    optimize_horizon: int,
    min_trades_per_side: int,
    horizons: List[int],
) -> Dict[str, object]:
    # Late imports to avoid circular dependency (api/main.py imports this module).
    from api.main import walk_forward_table, calibrate_thresholds
    from api.alpha.calibration_store import upsert_calibration

    try:
        rows = walk_forward_table(symbol, from_date, to_date, lookback, horizons)
        if not rows:
            return {"symbol": symbol, "status": "SKIPPED", "reason": "no_rows"}

        cal = calibrate_thresholds(rows, optimize_horizon, min_trades_per_side)
        payload = {
            "created_at_utc": cal["created_at_utc"],
            "symbol": symbol.upper(),
            "train_range": {"from_date": from_date, "to_date": to_date},
            "optimize_horizon": cal["optimize_horizon"],
            "thresholds_by_regime": cal["thresholds_by_regime"],
            "diagnostics": cal["diagnostics"],
        }
        ok = upsert_calibration(
            symbol,
            payload,
            optimize_horizon=int(optimize_horizon),
            train_range={"from_date": from_date, "to_date": to_date},
        )
        return {
            "symbol": symbol,
            "status": "OK" if ok else "WRITE_FAILED",
            "row_count": len(rows),
            "regimes": list(cal["thresholds_by_regime"].keys()),
        }
    except Exception as exc:
        return {"symbol": symbol, "status": "error", "error": "computation_error", "detail": str(exc)}


@router.post("/prosperity/nightly-recalibrate")
async def prosperity_nightly_recalibrate(request: Request):
    _require_db_enabled(write=True, read=True)

    body: Dict[str, object] = {}
    try:
        body = await request.json()
    except Exception:
        body = {}

    payload: Dict[str, object] = dict(request.query_params)
    payload.update(body if isinstance(body, dict) else {})

    try:
        params = RecalibrateParams.model_validate(payload)
    except ValidationError as exc:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "detail": exc.errors()},
        )

    symbols_mode = (params.symbols_mode or "core10").strip()
    if symbols_mode not in SYMBOLS_MODES:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_symbols_mode", "allowed": sorted(SYMBOLS_MODES)},
        )

    symbols = _symbols_for_mode(symbols_mode)
    from_date, to_date = _recalib_date_range(params.from_date, params.to_date)
    horizons = params.horizons or [params.optimize_horizon]
    if params.optimize_horizon not in horizons:
        horizons = sorted(set(horizons) | {params.optimize_horizon})

    run_id = str(uuid.uuid4())
    ttl_seconds = _lock_ttl_seconds()
    lock_owner = _job_lock_owner()
    as_of_date = dt.date.fromisoformat(to_date)

    requested_payload: Dict[str, object] = {
        "symbols_mode": symbols_mode,
        "symbols": symbols,
        "from_date": from_date,
        "to_date": to_date,
        "lookback": params.lookback,
        "optimize_horizon": params.optimize_horizon,
    }

    run_recorded = False
    status = "FAILED"
    result_record: Dict[str, object] = {}
    error_message: Optional[str] = None

    try:
        acquired, lock_info = _acquire_job_lock(
            run_id, RECALIB_JOB_NAME, as_of_date, requested_payload, ttl_seconds, lock_owner
        )
        if not acquired:
            return simple_error("locked", "job is already in progress", status_code=409, extra=lock_info)

        run_recorded = True
        results = []
        for sym in symbols:
            r = _run_recalibrate_symbol(
                sym,
                from_date,
                to_date,
                params.lookback,
                params.optimize_horizon,
                params.min_trades_per_side,
                horizons,
            )
            results.append(r)

        ok_count = sum(1 for r in results if r.get("status") == "OK")
        error_count = sum(1 for r in results if r.get("status") == "ERROR")
        status = "SUCCESS" if error_count == 0 else "PARTIAL"
        result_record = {
            "ok_count": ok_count,
            "error_count": error_count,
            "symbol_results": results,
        }
        return {
            "run_id": run_id,
            "status": status,
            "from_date": from_date,
            "to_date": to_date,
            "symbols": symbols,
            "ok_count": ok_count,
            "error_count": error_count,
            "symbol_results": results,
        }
    except Exception as exc:
        error_message = str(exc)
        logger.exception("jobs.prosperity.nightly_recalibrate.error", extra={"run_id": run_id})
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"error": "unexpected_error", "detail": error_message},
        )
    finally:
        try:
            if run_recorded:
                _update_job_run(run_id, status=status, result=result_record, error=error_message)
        except Exception:
            logger.warning(
                "jobs.prosperity.nightly_recalibrate.run_update_failed",
                extra={"run_id": run_id},
            )
