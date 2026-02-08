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
        return None
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
