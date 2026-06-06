"""Phase 24.5: Outcome fill — backfills forward returns into axiom_scores_daily.outcome_payload."""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends

from api import db, security

router = APIRouter(
    prefix="/jobs/outcome-fill",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)
logger = logging.getLogger(__name__)

_HORIZON_DAYS = 21
_BUY_DAU_THRESHOLD = 65.0
_SELL_DAU_THRESHOLD = 40.0


def run_outcome_fill(
    as_of_date: Optional[dt.date] = None,
    horizon_days: int = _HORIZON_DAYS,
) -> Dict[str, Any]:
    """Fill forward returns for matured axiom_scores rows.

    A row is "matured" when as_of_date + horizon_days <= today and outcome_payload is empty.
    """
    if not db.db_enabled():
        return {"filled": 0, "skipped": 0, "errors": 0, "status": "db_disabled"}

    today = dt.date.today()
    cutoff = today - dt.timedelta(days=horizon_days)
    target_date = as_of_date or cutoff

    filled = 0
    skipped = 0
    errors = 0

    try:
        rows = db.safe_fetchall(
            """
            SELECT symbol, as_of_date,
                   (payload->>'deployable_alpha_utility')::numeric AS dau
              FROM axiom_scores_daily
             WHERE as_of_date <= %s
               AND (outcome_payload IS NULL OR outcome_payload = '{}'::jsonb)
             ORDER BY as_of_date DESC
             LIMIT 500
            """,
            (cutoff,),
        ) or []
    except Exception as exc:
        logger.warning("outcome_fill.fetch_rows_failed error=%s", exc)
        return {"filled": 0, "skipped": 0, "errors": 1, "status": "fetch_failed"}

    for row in rows:
        symbol = str(row[0])
        score_date = row[1]
        dau = float(row[2]) if row[2] is not None else 50.0

        try:
            exit_date = score_date + dt.timedelta(days=int(horizon_days * 365.0 / 252.0) + 1)

            price_rows = db.safe_fetchall(
                """
                SELECT b1.close AS entry_close,
                       b2.exit_close
                  FROM market_bars_daily b1
                  JOIN LATERAL (
                      SELECT close AS exit_close
                        FROM market_bars_daily
                       WHERE symbol = b1.symbol
                         AND as_of_date >= %s
                         AND close IS NOT NULL
                       ORDER BY as_of_date ASC
                       LIMIT 1
                  ) b2 ON true
                 WHERE b1.symbol = %s
                   AND b1.as_of_date = %s
                   AND b1.close IS NOT NULL
                """,
                (exit_date, symbol, score_date),
            ) or []

            if not price_rows or not price_rows[0][0] or not price_rows[0][1]:
                skipped += 1
                continue

            entry_close = float(price_rows[0][0])
            exit_close = float(price_rows[0][1])

            if entry_close <= 0:
                skipped += 1
                continue

            fwd_return = (exit_close / entry_close) - 1.0

            if dau >= _BUY_DAU_THRESHOLD:
                hit = fwd_return > 0
            elif dau <= _SELL_DAU_THRESHOLD:
                hit = fwd_return < 0
            else:
                hit = None

            outcome = {
                "forward_return_21d": round(fwd_return, 6),
                "hit": hit,
                "filled_at": today.isoformat(),
            }

            db.safe_execute(
                """
                UPDATE axiom_scores_daily
                   SET outcome_payload = %s::jsonb
                 WHERE symbol = %s AND as_of_date = %s
                """,
                (json.dumps(outcome), symbol, score_date),
            )
            filled += 1

        except Exception as exc:
            logger.debug("outcome_fill.symbol_failed symbol=%s error=%s", symbol, exc)
            errors += 1

    return {
        "filled": filled,
        "skipped": skipped,
        "errors": errors,
        "status": "ok",
        "target_cutoff": cutoff.isoformat(),
    }


@router.post("/run")
def outcome_fill_run(
    as_of_date: Optional[str] = None,
    horizon_days: int = _HORIZON_DAYS,
) -> Dict[str, Any]:
    aod = dt.date.fromisoformat(as_of_date) if as_of_date else None
    return run_outcome_fill(as_of_date=aod, horizon_days=horizon_days)
