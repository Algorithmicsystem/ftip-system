"""Phase 28: Regime Transition Detection.

Detects day-over-day dominant regime shifts from axiom_scores_daily,
stores events in regime_transitions, and exposes history via ops router.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _dominant_regime(as_of_date: dt.date) -> Optional[str]:
    if not db.db_read_enabled():
        return None
    row = db.safe_fetchone(
        """
        SELECT payload->>'regime_label' AS regime, COUNT(*) AS cnt
        FROM axiom_scores_daily
        WHERE as_of_date = %s
          AND payload->>'regime_label' IS NOT NULL
        GROUP BY 1
        ORDER BY cnt DESC
        LIMIT 1
        """,
        (as_of_date,),
    )
    return str(row[0]) if row and row[0] else None


def detect_regime_transition(
    as_of_date: dt.date,
    *,
    prior_date: Optional[dt.date] = None,
) -> Optional[Dict[str, Any]]:
    """Return a transition dict if dominant regime changed, else None."""
    if prior_date is None:
        prior_date = as_of_date - dt.timedelta(days=1)

    today_regime = _dominant_regime(as_of_date)
    prev_regime  = _dominant_regime(prior_date)

    if not today_regime or not prev_regime:
        return None
    if today_regime == prev_regime:
        return None

    breadth_row = db.safe_fetchone(
        "SELECT breadth_state FROM market_breadth_daily WHERE as_of_date = %s",
        (as_of_date,),
    ) if db.db_read_enabled() else None
    ic_row = db.safe_fetchone(
        """
        SELECT ic_state FROM signal_ic_daily
        WHERE score_field = 'composite' AND horizon_label = '21d' AND as_of_date <= %s
        ORDER BY as_of_date DESC LIMIT 1
        """,
        (as_of_date,),
    ) if db.db_read_enabled() else None
    sym_row = db.safe_fetchone(
        "SELECT COUNT(*) FROM axiom_scores_daily WHERE as_of_date = %s",
        (as_of_date,),
    ) if db.db_read_enabled() else None

    return {
        "transition_id": str(uuid.uuid4()),
        "as_of_date": as_of_date,
        "from_regime": prev_regime,
        "to_regime": today_regime,
        "symbol_count": int(sym_row[0]) if sym_row else 0,
        "breadth_state": str(breadth_row[0]) if breadth_row and breadth_row[0] else None,
        "ic_state": str(ic_row[0]) if ic_row and ic_row[0] else None,
        "meta": {},
    }


def store_regime_transition(transition: Dict[str, Any]) -> bool:
    if not db.db_write_enabled():
        return False
    try:
        db.safe_execute(
            """
            INSERT INTO regime_transitions
                (transition_id, as_of_date, from_regime, to_regime,
                 symbol_count, breadth_state, ic_state, meta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (transition_id) DO NOTHING
            """,
            (
                transition["transition_id"],
                transition["as_of_date"],
                transition["from_regime"],
                transition["to_regime"],
                transition["symbol_count"],
                transition["breadth_state"],
                transition["ic_state"],
                json.dumps(transition["meta"]),
            ),
        )
        logger.info(
            "regime_transition.stored date=%s %s → %s",
            transition["as_of_date"], transition["from_regime"], transition["to_regime"],
        )
        return True
    except Exception as exc:
        logger.warning("regime_transition.store_failed error=%s", exc)
        return False


def load_regime_transitions(
    *,
    limit: int = 20,
    since: Optional[dt.date] = None,
) -> List[Dict[str, Any]]:
    if not db.db_read_enabled():
        return []
    params: list = []
    where_clause = ""
    if since:
        where_clause = "WHERE as_of_date >= %s"
        params.append(since)
    params.append(limit)
    rows = db.safe_fetchall(
        f"""
        SELECT transition_id, as_of_date, from_regime, to_regime,
               symbol_count, breadth_state, ic_state, detected_at
        FROM regime_transitions
        {where_clause}
        ORDER BY as_of_date DESC
        LIMIT %s
        """,
        tuple(params),
    )
    return [
        {
            "transition_id": str(r[0]),
            "as_of_date": r[1].isoformat() if hasattr(r[1], "isoformat") else str(r[1]),
            "from_regime": str(r[2]),
            "to_regime": str(r[3]),
            "symbol_count": int(r[4] or 0),
            "breadth_state": str(r[5]) if r[5] else None,
            "ic_state": str(r[6]) if r[6] else None,
            "detected_at": r[7].isoformat() if hasattr(r[7], "isoformat") else str(r[7]),
        }
        for r in (rows or [])
    ]
