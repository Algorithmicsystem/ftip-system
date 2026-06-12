"""Writes batch-fetched bars to prosperity_daily_bars.

Handles upsert logic, raw JSON serialisation, and the prosperity_universe
FK parent requirement.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, List

from api import db

logger = logging.getLogger(__name__)


def ensure_prosperity_universe(symbols: List[str]) -> None:
    """Ensure all symbols exist in prosperity_universe (FK parent for bars)."""
    if not db.db_enabled():
        return
    for sym in symbols:
        try:
            db.safe_execute(
                """
                INSERT INTO prosperity_universe (symbol, active)
                VALUES (%s, TRUE)
                ON CONFLICT (symbol) DO NOTHING
                """,
                (sym,),
            )
        except Exception as exc:
            logger.debug("ensure_prosperity_universe failed sym=%s err=%s", sym, exc)


def write_bars_to_db(
    bars_by_symbol: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, int]:
    """Write batch-fetched bars to prosperity_daily_bars.

    Returns {symbol: rows_written}.
    """
    if not db.db_enabled():
        return {}

    results: Dict[str, int] = {}

    for symbol, bars in bars_by_symbol.items():
        if not bars:
            results[symbol] = 0
            continue

        written = 0
        for bar in bars:
            try:
                raw_json = _bar_to_raw_json(bar)
                db.safe_execute(
                    """
                    INSERT INTO prosperity_daily_bars
                        (symbol, date, open, high, low, close,
                         adj_close, volume, source, raw)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (symbol, date) DO UPDATE SET
                        open       = EXCLUDED.open,
                        high       = EXCLUDED.high,
                        low        = EXCLUDED.low,
                        close      = EXCLUDED.close,
                        adj_close  = EXCLUDED.adj_close,
                        volume     = EXCLUDED.volume,
                        source     = EXCLUDED.source,
                        raw        = EXCLUDED.raw,
                        updated_at = now()
                    """,
                    (
                        bar["symbol"],
                        bar["date"],
                        bar["open"],
                        bar["high"],
                        bar["low"],
                        bar["close"],
                        bar["adj_close"],
                        bar["volume"],
                        bar["source"],
                        raw_json,
                    ),
                )
                written += 1
            except Exception as exc:
                logger.debug(
                    "bars_writer.insert_failed sym=%s date=%s err=%s",
                    symbol, bar.get("date"), exc,
                )
        results[symbol] = written

    total_ok = sum(1 for v in results.values() if v > 0)
    total_rows = sum(results.values())
    logger.info(
        "bars_writer.done symbols_ok=%d total_rows=%d",
        total_ok, total_rows,
    )
    return results


def _bar_to_raw_json(bar: dict) -> str:
    safe: Dict[str, Any] = {}
    for k, v in bar.items():
        if k == "source":
            continue
        if isinstance(v, (dt.date, dt.datetime)):
            safe[k] = v.isoformat()
        elif isinstance(v, float):
            safe[k] = v
        elif isinstance(v, int):
            safe[k] = v
        elif v is None:
            safe[k] = None
        else:
            safe[k] = str(v)
    return json.dumps(safe)
