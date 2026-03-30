from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from api import db
from api.data_providers import canonical_symbol
from api.jobs import features as features_job
from api.jobs import market_data as market_data_job
from api.jobs import signals as signals_job

_PROSPERITY_SIGNAL_ASOF_COLUMN: Optional[str] = None
_PROSPERITY_SIGNAL_ACTION_COLUMN: Optional[str] = None


def _require_db_enabled(write: bool = False, read: bool = False) -> None:
    if not db.db_enabled():
        raise HTTPException(status_code=503, detail="database disabled")
    if write and not db.db_write_enabled():
        raise HTTPException(status_code=503, detail="database writes disabled")
    if read and not db.db_read_enabled():
        raise HTTPException(status_code=503, detail="database reads disabled")


def _latest_bar_info(symbol: str) -> Tuple[Optional[dt.date], Optional[dt.datetime]]:
    row = db.safe_fetchone(
        """
        SELECT MAX(as_of_date), MAX(ingested_at)
        FROM market_bars_daily
        WHERE symbol = %s
        """,
        (symbol,),
    )
    if not row:
        return None, None
    return row[0], row[1]


def _latest_prosperity_bar_info(
    symbol: str,
) -> Tuple[Optional[dt.date], Optional[dt.datetime]]:
    try:
        row = db.safe_fetchone(
            """
            SELECT MAX(date), MAX(updated_at)
            FROM prosperity_daily_bars
            WHERE symbol = %s
            """,
            (symbol,),
        )
    except Exception:
        return None, None
    if not row:
        return None, None
    return row[0], row[1]


def _hydrate_market_bars_from_prosperity(symbol: str, from_date: dt.date) -> int:
    rows = db.safe_fetchall(
        """
        SELECT symbol, date, open, high, low, close, volume, source
        FROM prosperity_daily_bars
        WHERE symbol = %s AND date >= %s
        ORDER BY date ASC
        """,
        (symbol, from_date),
    )
    if not rows:
        return 0

    inserted = 0
    with db.with_connection() as (conn, cur):
        for row in rows:
            cur.execute(
                """
                INSERT INTO market_bars_daily(
                    symbol, as_of_date, open, high, low, close, volume, source, ingested_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (symbol, as_of_date)
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    source = EXCLUDED.source,
                    ingested_at = now()
                """,
                (
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7] or "prosperity_daily_bars",
                ),
            )
            inserted += 1
        conn.commit()
    return inserted


def _latest_news_info(
    symbol: str,
) -> Tuple[Optional[dt.datetime], Optional[dt.datetime]]:
    row = db.safe_fetchone(
        """
        SELECT MAX(published_at), MAX(ingested_at)
        FROM news_raw
        WHERE symbol = %s
        """,
        (symbol,),
    )
    if not row:
        return None, None
    return row[0], row[1]


def _latest_sentiment_info(
    symbol: str,
) -> Tuple[Optional[dt.date], Optional[dt.datetime]]:
    row = db.safe_fetchone(
        """
        SELECT MAX(as_of_date), MAX(computed_at)
        FROM sentiment_daily
        WHERE symbol = %s
        """,
        (symbol,),
    )
    if not row:
        return None, None
    return row[0], row[1]


async def ensure_freshness(symbol: str, *, refresh: bool = True) -> Dict[str, Any]:
    _require_db_enabled(read=True, write=refresh)
    symbol = canonical_symbol(symbol)

    today = dt.datetime.now(dt.timezone.utc).date()
    bars_date, bars_updated = _latest_bar_info(symbol)
    news_date, news_updated = _latest_news_info(symbol)
    sentiment_date, sentiment_updated = _latest_sentiment_info(symbol)

    warnings: List[str] = []

    if refresh:
        if bars_date is None or (today - bars_date).days > 3:
            request = market_data_job.DailyBarsRequest(
                as_of_date=today,
                from_date=today - dt.timedelta(days=7),
                to_date=today,
                symbols=[symbol],
            )
            resp = await market_data_job.ingest_bars_daily(request)
            if isinstance(resp, JSONResponse) and resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code, detail=resp.body.decode()
                )
            bars_date, bars_updated = _latest_bar_info(symbol)
            if bars_date is None:
                fallback_bars_date, _ = _latest_prosperity_bar_info(symbol)
                if fallback_bars_date is not None:
                    # Use the prosperity recency anchor (not "today") so hydration
                    # still works when the only available prosperity history is stale.
                    # If we anchor on today-30 and prosperity has no rows in that window,
                    # market_bars_daily stays empty and /assistant/analyze incorrectly 404s.
                    _hydrate_market_bars_from_prosperity(
                        symbol, fallback_bars_date - dt.timedelta(days=30)
                    )
                    bars_date, bars_updated = _latest_bar_info(symbol)

        if news_date is None or (today - news_date.date()).days > 5:
            request = market_data_job.NewsRequest(
                from_ts=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10),
                to_ts=dt.datetime.now(dt.timezone.utc),
                symbols=[symbol],
            )
            resp = await market_data_job.ingest_news(request)
            if isinstance(resp, JSONResponse) and resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code, detail=resp.body.decode()
                )
            news_date, news_updated = _latest_news_info(symbol)

        if sentiment_date is None or (today - sentiment_date).days > 3:
            request = market_data_job.SentimentRequest(
                as_of_date=today,
                symbols=[symbol],
            )
            resp = await market_data_job.compute_sentiment(request)
            if isinstance(resp, JSONResponse) and resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code, detail=resp.body.decode()
                )
            sentiment_date, sentiment_updated = _latest_sentiment_info(symbol)

    if bars_date is None:
        raise HTTPException(
            status_code=404, detail="no market bars available for symbol"
        )

    if news_date is None:
        warnings.append("news data missing")
    if sentiment_date is None:
        warnings.append("sentiment data missing")

    return {
        "as_of_date": bars_date,
        "bars_ok": bars_date is not None and (today - bars_date).days <= 3,
        "news_ok": (
            news_date is not None and (today - news_date.date()).days <= 7
            if news_date
            else False
        ),
        "sentiment_ok": sentiment_date is not None
        and (today - sentiment_date).days <= 5,
        "bars_updated_at": bars_updated.isoformat() if bars_updated else None,
        "news_updated_at": news_updated.isoformat() if news_updated else None,
        "sentiment_updated_at": (
            sentiment_updated.isoformat() if sentiment_updated else None
        ),
        "warnings": warnings,
    }


async def run_features(symbol: str, as_of_date: dt.date) -> None:
    _require_db_enabled(write=True, read=True)
    request = features_job.FeaturesDailyRequest(as_of_date=as_of_date, symbols=[symbol])
    resp = await features_job.compute_features_daily(request)
    if isinstance(resp, JSONResponse) and resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.body.decode())


async def run_signals(symbol: str, as_of_date: dt.date) -> None:
    _require_db_enabled(write=True, read=True)
    request = signals_job.SignalsDailyRequest(as_of_date=as_of_date, symbols=[symbol])
    resp = await signals_job.compute_signals_daily(request)
    if isinstance(resp, JSONResponse) and resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.body.decode())


def _prosperity_signal_asof_column() -> str:
    global _PROSPERITY_SIGNAL_ASOF_COLUMN
    if _PROSPERITY_SIGNAL_ASOF_COLUMN:
        return _PROSPERITY_SIGNAL_ASOF_COLUMN
    try:
        row = db.safe_fetchone(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'prosperity_signals_daily'
              AND column_name IN ('as_of', 'as_of_date')
            ORDER BY CASE WHEN column_name = 'as_of' THEN 1 ELSE 2 END
            LIMIT 1
            """
        )
    except Exception:
        row = None
    _PROSPERITY_SIGNAL_ASOF_COLUMN = (
        str(row[0]) if row and row[0] in {"as_of", "as_of_date"} else "as_of"
    )
    return _PROSPERITY_SIGNAL_ASOF_COLUMN


def _prosperity_signal_action_column() -> str:
    global _PROSPERITY_SIGNAL_ACTION_COLUMN
    if _PROSPERITY_SIGNAL_ACTION_COLUMN:
        return _PROSPERITY_SIGNAL_ACTION_COLUMN
    try:
        row = db.safe_fetchone(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'prosperity_signals_daily'
              AND column_name IN ('signal', 'action')
            ORDER BY CASE WHEN column_name = 'signal' THEN 1 ELSE 2 END
            LIMIT 1
            """
        )
    except Exception:
        row = None
    _PROSPERITY_SIGNAL_ACTION_COLUMN = (
        str(row[0]) if row and row[0] in {"signal", "action"} else "signal"
    )
    return _PROSPERITY_SIGNAL_ACTION_COLUMN


def fetch_signal(symbol: str, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT action, score, confidence, entry_low, entry_high, stop_loss,
               take_profit_1, take_profit_2, horizon_days, reason_codes, reason_details
        FROM signals_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )
    if not row:
        as_of_column = _prosperity_signal_asof_column()
        action_column = _prosperity_signal_action_column()
        prosperity_row = db.safe_fetchone(
            f"""
            SELECT {action_column}, score, confidence
            FROM prosperity_signals_daily
            WHERE symbol = %s AND {as_of_column} = %s
            ORDER BY updated_at DESC NULLS LAST
            LIMIT 1
            """,
            (symbol, as_of_date),
        )
        if not prosperity_row:
            return None
        return {
            "action": prosperity_row[0],
            "score": prosperity_row[1],
            "confidence": prosperity_row[2],
            "entry_low": None,
            "entry_high": None,
            "stop_loss": None,
            "take_profit_1": None,
            "take_profit_2": None,
            "horizon_days": None,
            "reason_codes": [],
            "reason_details": {},
        }
    return {
        "action": row[0],
        "score": row[1],
        "confidence": row[2],
        "entry_low": row[3],
        "entry_high": row[4],
        "stop_loss": row[5],
        "take_profit_1": row[6],
        "take_profit_2": row[7],
        "horizon_days": row[8],
        "reason_codes": row[9] or [],
        "reason_details": row[10] or {},
    }


def fetch_key_features(symbol: str, as_of_date: dt.date) -> Dict[str, Any]:
    row = db.safe_fetchone(
        """
        SELECT ret_1d, ret_5d, ret_21d, vol_21d, vol_63d, atr_pct,
               trend_slope_21d, trend_slope_63d, mom_vol_adj_21d,
               sentiment_score, sentiment_surprise, regime_label, regime_strength
        FROM features_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )
    if not row:
        return {}
    return {
        "ret_1d": row[0],
        "ret_5d": row[1],
        "ret_21d": row[2],
        "vol_21d": row[3],
        "vol_63d": row[4],
        "atr_pct": row[5],
        "trend_slope_21d": row[6],
        "trend_slope_63d": row[7],
        "mom_vol_adj_21d": row[8],
        "sentiment_score": row[9],
        "sentiment_surprise": row[10],
        "regime_label": row[11],
        "regime_strength": row[12],
    }


def fetch_quality(
    symbol: str, as_of_date: dt.date, freshness: Dict[str, Any]
) -> Dict[str, Any]:
    row = db.safe_fetchone(
        """
        SELECT bars_ok, fundamentals_ok, sentiment_ok, intraday_ok, missingness, anomaly_flags, quality_score
        FROM quality_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )
    return {
        "bars_ok": row[0] if row else None,
        "fundamentals_ok": row[1] if row else None,
        "sentiment_ok": row[2] if row else None,
        "intraday_ok": row[3] if row else None,
        "missingness": row[4] if row else None,
        "anomaly_flags": row[5] if row else None,
        "quality_score": row[6] if row else None,
        "news_ok": freshness.get("news_ok"),
        "bars_updated_at": freshness.get("bars_updated_at"),
        "news_updated_at": freshness.get("news_updated_at"),
        "sentiment_updated_at": freshness.get("sentiment_updated_at"),
        "warnings": freshness.get("warnings") or [],
    }


def fetch_top_picks(limit: int) -> Tuple[Optional[dt.date], List[Dict[str, Any]]]:
    latest_row = db.safe_fetchone("SELECT MAX(as_of_date) FROM signals_daily")
    as_of_date = latest_row[0] if latest_row else None
    if not as_of_date:
        return None, []
    rows = db.safe_fetchall(
        """
        SELECT symbol, action, score, confidence, reason_codes
        FROM signals_daily
        WHERE as_of_date = %s
        ORDER BY ABS(score) DESC
        LIMIT %s
        """,
        (as_of_date, limit),
    )
    picks: List[Dict[str, Any]] = []
    for row in rows:
        direction = (
            "long"
            if (row[1] or "").upper() == "BUY"
            else "short"
            if (row[1] or "").upper() == "SELL"
            else "hold"
        )
        picks.append(
            {
                "symbol": row[0],
                "direction": direction,
                "score": row[2],
                "confidence": row[3],
                "reason_codes": row[4] or [],
            }
        )
    return as_of_date, picks


def universe_coverage(as_of_date: Optional[dt.date]) -> float:
    if not as_of_date:
        return 0.0
    counts = db.safe_fetchone(
        """
        SELECT
            (SELECT COUNT(*) FROM signals_daily WHERE as_of_date = %s),
            (SELECT COUNT(*) FROM market_symbols WHERE is_active = TRUE)
        """,
        (as_of_date,),
    )
    if not counts:
        return 0.0
    signal_count, total = counts
    if not total:
        return 0.0
    return float(signal_count) / float(total)


def normalize_symbol(symbol: str) -> str:
    cleaned = canonical_symbol(symbol)
    if not cleaned:
        raise HTTPException(status_code=400, detail="symbol required")
    return cleaned
