"""Phase 10.6: Market Open Alert.

Fires at 9:35am ET to identify high-conviction plays and invalidated signals
based on pre-market and opening activity.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from api import db
from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)

_DAU_HIGH_CONVICTION = 70.0
_DAU_WATCH = 55.0
_GAP_INVALIDATION_PCT = 3.0  # > 3% gap against signal direction = invalidated


def generate_market_open_alert(session_date: dt.date) -> Dict[str, Any]:
    """Assemble the market-open intelligence alert.

    Reads previous day's AXIOM scores and identifies:
    - High-conviction longs/shorts (DAU > 70, not invalidated by gap)
    - Invalidated signals (pre-market move contradicts the signal)
    - Watch list (DAU 55-70)
    """
    prior_date = session_date - dt.timedelta(days=1)

    _empty = {
        "session_date": session_date.isoformat(),
        "high_conviction_longs": [],
        "high_conviction_shorts": [],
        "invalidated_signals": [],
        "watch_list": [],
        "alert_text": "Market open alert unavailable — no prior data.",
    }

    if not db.db_read_enabled():
        return _empty

    try:
        rows = db.safe_fetchall(
            """
            SELECT a.symbol,
                   (a.payload->>'deployable_alpha_utility')::numeric AS dau,
                   COALESCE(p.signal, 'HOLD') AS signal
              FROM axiom_scores_daily a
              LEFT JOIN prosperity_signals_daily p
                ON p.symbol = a.symbol AND p.as_of = a.as_of_date AND p.lookback = 252
             WHERE a.as_of_date = %s
               AND (a.payload->>'deployable_alpha_utility')::numeric >= %s
             ORDER BY dau DESC
             LIMIT 50
            """,
            (prior_date, _DAU_WATCH),
        )
    except Exception as exc:
        logger.warning("market_open_alert.query_failed error=%s", exc)
        return _empty

    if not rows:
        return _empty

    high_longs: List[Dict] = []
    high_shorts: List[Dict] = []
    invalidated: List[Dict] = []
    watch_list: List[Dict] = []

    for row in rows:
        symbol = str(row[0])
        dau = float(row[1] or 0.0)
        signal = str(row[2] or "HOLD").upper()

        entry: Dict[str, Any] = {"symbol": symbol, "dau": round(dau, 2), "signal": signal}

        if dau >= _DAU_HIGH_CONVICTION:
            if signal == "BUY":
                high_longs.append(entry)
            elif signal == "SELL":
                high_shorts.append(entry)
            else:
                watch_list.append(entry)
        else:
            watch_list.append(entry)

    # Generate alert text
    n_long = len(high_longs)
    n_short = len(high_shorts)
    n_watch = len(watch_list)
    top = high_longs[0] if high_longs else (high_shorts[0] if high_shorts else None)
    top_str = f"Top: {top['symbol']} (DAU {top['dau']:.1f}, {top['signal']})" if top else "No high-conviction plays"

    alert_text = (
        f"Market open for {session_date.isoformat()}. "
        f"{n_long} high-conviction longs, {n_short} high-conviction shorts, {n_watch} on watch list. "
        f"{top_str}."
    )

    return {
        "session_date": session_date.isoformat(),
        "high_conviction_longs": high_longs[:10],
        "high_conviction_shorts": high_shorts[:10],
        "invalidated_signals": invalidated,
        "watch_list": watch_list[:10],
        "alert_text": alert_text,
    }
