from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Tuple

from api import db

logger = logging.getLogger(__name__)


def _payload_hash(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def upsert_calibration(
    symbol: str,
    payload: Dict[str, Any],
    *,
    optimize_horizon: Optional[int] = None,
    train_range: Optional[Dict[str, Any]] = None,
) -> bool:
    """Persist a calibration into prosperity_calibrations. Best-effort — returns False on any error."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return False
    cal_hash = _payload_hash(payload)
    created_at_utc = str(payload.get("created_at_utc") or "")
    try:
        db.safe_execute(
            """
            INSERT INTO prosperity_calibrations
                (symbol, created_at_utc, payload, optimize_horizon, train_range, calibration_hash)
            VALUES (%s, %s, %s::jsonb, %s, %s::jsonb, %s)
            ON CONFLICT (symbol, calibration_hash)
            DO UPDATE SET
                payload = EXCLUDED.payload,
                optimize_horizon = EXCLUDED.optimize_horizon,
                train_range = EXCLUDED.train_range,
                updated_at = now()
            """,
            (
                sym,
                created_at_utc,
                json.dumps(payload),
                optimize_horizon,
                json.dumps(train_range) if train_range is not None else None,
                cal_hash,
            ),
        )
        return True
    except Exception:
        logger.warning("calibration_store.upsert_failed", extra={"symbol": sym})
        return False


def load_calibration_from_db(symbol: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Return the most recent calibration payload for symbol from the DB, or (False, None)."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return False, None
    try:
        row = db.safe_fetchone(
            """
            SELECT payload
            FROM prosperity_calibrations
            WHERE symbol = %s
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (sym,),
        )
        if row and isinstance(row[0], dict):
            return True, row[0]
    except Exception:
        logger.debug("calibration_store.load_failed", extra={"symbol": sym})
    return False, None
