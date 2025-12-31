from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Request

from api import config, db, security
from api.prosperity.models import SnapshotRunRequest
from api.prosperity.routes import _require_db_enabled, snapshot_run

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

logger = logging.getLogger(__name__)

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


@router.post("/prosperity/daily-snapshot")
async def prosperity_daily_snapshot(request: Request):
    _require_db_enabled(write=True, read=True)

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

    return {
        "status": result.get("status"),
        "as_of_date": as_of_date.isoformat(),
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        **summary,
    }
