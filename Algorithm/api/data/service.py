from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Sequence

from api import db


def canonicalize_symbol(symbol: str) -> str:
    cleaned = (symbol or "").strip().upper()
    if not cleaned:
        raise ValueError("symbol is required")
    return cleaned


def record_data_version(
    source_name: str, source_snapshot_hash: str, notes: str = ""
) -> Dict[str, Any]:
    return db.record_data_version(source_name, source_snapshot_hash, notes)


def upsert_symbols(items: Sequence[Dict[str, Any]]) -> int:
    count = 0
    with db.with_connection() as (conn, cur):
        for item in items:
            symbol = canonicalize_symbol(str(item.get("symbol") or ""))
            country = item.get("country")
            exchange = item.get("exchange")
            currency = item.get("currency")
            cur.execute(
                """
                INSERT INTO symbols (symbol, country, exchange, currency)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (symbol)
                DO UPDATE SET
                    country = EXCLUDED.country,
                    exchange = EXCLUDED.exchange,
                    currency = EXCLUDED.currency,
                    updated_at = now()
                """,
                (symbol, country, exchange, currency),
            )
            count += 1
        conn.commit()
    return count


def set_universe(
    universe_name: str,
    symbols: Sequence[str],
    start_ts: Optional[dt.datetime] = None,
    end_ts: Optional[dt.datetime] = None,
    data_version_id: Optional[int] = None,
) -> int:
    if not universe_name.strip():
        raise ValueError("universe_name is required")

    current_start = start_ts or dt.datetime.now(dt.timezone.utc)
    cleaned_symbols = [canonicalize_symbol(symbol) for symbol in symbols]
    upsert_symbols([{"symbol": symbol} for symbol in cleaned_symbols])

    count = 0
    with db.with_connection() as (conn, cur):
        for symbol in cleaned_symbols:
            cur.execute(
                """
                INSERT INTO universe_membership
                    (universe_name, symbol, start_ts, end_ts, data_version_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (universe_name, symbol, start_ts)
                DO UPDATE SET
                    end_ts = EXCLUDED.end_ts,
                    data_version_id = EXCLUDED.data_version_id
                """,
                (universe_name, symbol, current_start, end_ts, data_version_id),
            )
            count += 1
        conn.commit()
    return count


def ingest_prices_daily(data_version_id: int, items: Sequence[Dict[str, Any]]) -> int:
    count = 0
    with db.with_connection() as (conn, cur):
        for item in items:
            symbol = canonicalize_symbol(str(item.get("symbol") or ""))
            as_of_ts = item.get("as_of_ts") or dt.datetime.now(dt.timezone.utc)
            cur.execute(
                """
                INSERT INTO prices_daily_versioned
                    (data_version_id, symbol, date, as_of_ts, open, high, low, close, volume, currency)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (data_version_id, symbol, date)
                DO UPDATE SET
                    as_of_ts = EXCLUDED.as_of_ts,
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    currency = EXCLUDED.currency
                """,
                (
                    data_version_id,
                    symbol,
                    item.get("date"),
                    as_of_ts,
                    item.get("open"),
                    item.get("high"),
                    item.get("low"),
                    item.get("close"),
                    item.get("volume"),
                    item.get("currency"),
                ),
            )
            count += 1
        conn.commit()
    return count


def query_prices_daily(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    as_of_ts: dt.datetime,
    adjusted: bool = False,
) -> List[Dict[str, Any]]:
    return db.get_prices_daily(
        canonicalize_symbol(symbol),
        start_date,
        end_date,
        as_of_ts,
        adjusted=adjusted,
    )


def ingest_corp_actions(data_version_id: int, items: Sequence[Dict[str, Any]]) -> int:
    count = 0
    with db.with_connection() as (conn, cur):
        for item in items:
            symbol = canonicalize_symbol(str(item.get("symbol") or ""))
            as_of_ts = item.get("as_of_ts") or item.get("announced_ts")
            cur.execute(
                """
                INSERT INTO corp_actions_versioned
                    (data_version_id, symbol, action_type, effective_date, factor, value, announced_ts, as_of_ts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (data_version_id, symbol, action_type, effective_date, announced_ts)
                DO UPDATE SET
                    factor = EXCLUDED.factor,
                    value = EXCLUDED.value,
                    as_of_ts = EXCLUDED.as_of_ts
                """,
                (
                    data_version_id,
                    symbol,
                    item.get("action_type"),
                    item.get("effective_date"),
                    item.get("factor"),
                    item.get("value"),
                    item.get("announced_ts"),
                    as_of_ts,
                ),
            )
            count += 1
        conn.commit()
    return count


def ingest_fundamentals(data_version_id: int, items: Sequence[Dict[str, Any]]) -> int:
    count = 0
    with db.with_connection() as (conn, cur):
        for item in items:
            symbol = canonicalize_symbol(str(item.get("symbol") or ""))
            published_ts = item.get("published_ts")
            as_of_ts = item.get("as_of_ts") or published_ts
            cur.execute(
                """
                INSERT INTO fundamentals_pit
                    (data_version_id, symbol, metric_key, metric_value, period_end, published_ts, as_of_ts)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (data_version_id, symbol, metric_key, period_end, published_ts)
                DO UPDATE SET
                    metric_value = EXCLUDED.metric_value,
                    as_of_ts = EXCLUDED.as_of_ts
                """,
                (
                    data_version_id,
                    symbol,
                    item.get("metric_key"),
                    item.get("metric_value"),
                    item.get("period_end"),
                    published_ts,
                    as_of_ts,
                ),
            )
            count += 1
        conn.commit()
    return count


def query_latest_fundamentals(
    symbol: str, as_of_ts: dt.datetime, metric_keys: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    return db.get_latest_fundamentals(
        canonicalize_symbol(symbol), as_of_ts, metric_keys
    )


def ingest_news(data_version_id: int, items: Sequence[Dict[str, Any]]) -> int:
    count = 0
    with db.with_connection() as (conn, cur):
        for item in items:
            symbol = canonicalize_symbol(str(item.get("symbol") or ""))
            headline = str(item.get("headline") or "").strip()
            as_of_ts = item.get("as_of_ts") or item.get("published_ts")
            cur.execute(
                """
                INSERT INTO news_items
                    (data_version_id, symbol, published_ts, as_of_ts, source, credibility, headline, headline_hash, full_text)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (data_version_id, symbol, published_ts, source, headline_hash)
                DO UPDATE SET
                    credibility = EXCLUDED.credibility,
                    full_text = EXCLUDED.full_text,
                    as_of_ts = EXCLUDED.as_of_ts
                """,
                (
                    data_version_id,
                    symbol,
                    item.get("published_ts"),
                    as_of_ts,
                    item.get("source"),
                    item.get("credibility"),
                    headline,
                    db.headline_hash(headline),
                    item.get("full_text"),
                ),
            )
            count += 1
        conn.commit()
    return count


def query_news(
    symbol: str,
    as_of_ts: dt.datetime,
    start_ts: Optional[dt.datetime] = None,
    end_ts: Optional[dt.datetime] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    return db.get_news(
        canonicalize_symbol(symbol),
        as_of_ts,
        start_ts=start_ts,
        end_ts=end_ts,
        limit=limit,
    )


def query_universe(as_of_ts: dt.datetime, universe_name: str = "default") -> List[str]:
    return db.get_universe_pit(as_of_ts, universe_name=universe_name)
