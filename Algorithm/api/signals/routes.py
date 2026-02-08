from __future__ import annotations

import datetime as dt
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from api import db, security
from api.data_providers import canonical_symbol

router = APIRouter(
    prefix="/signals",
    tags=["signals"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)


def _require_db_enabled(read: bool = True) -> None:
    if not db.db_enabled():
        raise HTTPException(status_code=503, detail="database disabled")
    if read and not db.db_read_enabled():
        raise HTTPException(status_code=503, detail="database reads disabled")


@router.get("/latest")
async def latest_signal(symbol: str):
    _require_db_enabled(read=True)
    symbol = canonical_symbol(symbol)
    row = db.safe_fetchone(
        """
        SELECT as_of_date, action, score, confidence, entry_low, entry_high, stop_loss,
               take_profit_1, take_profit_2, horizon_days, reason_codes, reason_details
        FROM signals_daily
        WHERE symbol = %s
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (symbol,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="signal not found")
    return {
        "symbol": symbol,
        "as_of_date": row[0],
        "action": row[1],
        "score": row[2],
        "confidence": row[3],
        "entry_low": row[4],
        "entry_high": row[5],
        "stop_loss": row[6],
        "take_profit_1": row[7],
        "take_profit_2": row[8],
        "horizon_days": row[9],
        "reason_codes": row[10],
        "reason_details": row[11],
    }


@router.get("/top")
async def top_picks(
    mode: str = Query("buy"),
    limit: int = Query(10, ge=1, le=50),
    country: str = Query("ALL"),
):
    _require_db_enabled(read=True)
    mode_upper = mode.upper()
    if mode_upper not in {"BUY", "SELL"}:
        raise HTTPException(status_code=400, detail="mode must be buy or sell")
    country_upper = country.upper()
    if country_upper not in {"US", "CA", "ALL"}:
        raise HTTPException(status_code=400, detail="country must be US, CA, or ALL")

    where_country = "" if country_upper == "ALL" else "AND ms.country = %s"
    params: List[object] = [mode_upper]
    if country_upper != "ALL":
        params.append(country_upper)

    rows = db.safe_fetchall(
        f"""
        WITH latest AS (
            SELECT MAX(as_of_date) AS as_of_date FROM signals_daily
        )
        SELECT sd.symbol, sd.as_of_date, sd.action, sd.score, sd.confidence
        FROM signals_daily sd
        JOIN latest l ON sd.as_of_date = l.as_of_date
        LEFT JOIN market_symbols ms ON ms.symbol = sd.symbol
        WHERE sd.action = %s {where_country}
        ORDER BY sd.score {'DESC' if mode_upper == 'BUY' else 'ASC'}
        LIMIT %s
        """,
        tuple(params + [limit]),
    )

    if not rows:
        rows = db.safe_fetchall(
            f"""
            WITH latest AS (
                SELECT MAX(as_of_date) AS as_of_date FROM signals_daily
            )
            SELECT sd.symbol, sd.as_of_date, sd.action, sd.score, sd.confidence
            FROM signals_daily sd
            JOIN latest l ON sd.as_of_date = l.as_of_date
            LEFT JOIN market_symbols ms ON ms.symbol = sd.symbol
            WHERE 1=1 {where_country}
            ORDER BY sd.score {'DESC' if mode_upper == 'BUY' else 'ASC'}
            LIMIT %s
            """,
            tuple(params[1:] + [limit]) if country_upper != "ALL" else (limit,),
        )

    return [
        {
            "symbol": row[0],
            "as_of_date": row[1],
            "action": row[2],
            "score": row[3],
            "confidence": row[4],
        }
        for row in rows
    ]


@router.get("/evidence")
async def signal_evidence(symbol: str, as_of_date: dt.date):
    _require_db_enabled(read=True)
    symbol = canonical_symbol(symbol)

    signal_row = db.safe_fetchone(
        """
        SELECT action, score, confidence, entry_low, entry_high, stop_loss,
               take_profit_1, take_profit_2, horizon_days, reason_codes, reason_details
        FROM signals_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )
    if not signal_row:
        raise HTTPException(status_code=404, detail="signal not found")

    feature_row = db.safe_fetchone(
        """
        SELECT ret_1d, ret_5d, ret_21d, vol_21d, vol_63d, atr_14, atr_pct,
               trend_slope_21d, trend_r2_21d, trend_slope_63d, trend_r2_63d,
               mom_vol_adj_21d, maxdd_63d, dollar_vol_21d, sentiment_score, sentiment_surprise,
               regime_label, regime_strength
        FROM features_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )

    quality_row = db.safe_fetchone(
        """
        SELECT bars_ok, fundamentals_ok, sentiment_ok, intraday_ok, missingness, anomaly_flags, quality_score
        FROM quality_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )

    bar_row = db.safe_fetchone(
        """
        SELECT open, high, low, close, volume
        FROM market_bars_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )

    sentiment_row = db.safe_fetchone(
        """
        SELECT headline_count, sentiment_score
        FROM sentiment_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )

    return {
        "symbol": symbol,
        "as_of_date": as_of_date,
        "signal": {
            "action": signal_row[0],
            "score": signal_row[1],
            "confidence": signal_row[2],
            "entry_low": signal_row[3],
            "entry_high": signal_row[4],
            "stop_loss": signal_row[5],
            "take_profit_1": signal_row[6],
            "take_profit_2": signal_row[7],
            "horizon_days": signal_row[8],
            "reason_codes": signal_row[9],
            "reason_details": signal_row[10],
        },
        "features": {
            "ret_1d": feature_row[0] if feature_row else None,
            "ret_5d": feature_row[1] if feature_row else None,
            "ret_21d": feature_row[2] if feature_row else None,
            "vol_21d": feature_row[3] if feature_row else None,
            "vol_63d": feature_row[4] if feature_row else None,
            "atr_14": feature_row[5] if feature_row else None,
            "atr_pct": feature_row[6] if feature_row else None,
            "trend_slope_21d": feature_row[7] if feature_row else None,
            "trend_r2_21d": feature_row[8] if feature_row else None,
            "trend_slope_63d": feature_row[9] if feature_row else None,
            "trend_r2_63d": feature_row[10] if feature_row else None,
            "mom_vol_adj_21d": feature_row[11] if feature_row else None,
            "maxdd_63d": feature_row[12] if feature_row else None,
            "dollar_vol_21d": feature_row[13] if feature_row else None,
            "sentiment_score": feature_row[14] if feature_row else None,
            "sentiment_surprise": feature_row[15] if feature_row else None,
            "regime_label": feature_row[16] if feature_row else None,
            "regime_strength": feature_row[17] if feature_row else None,
        },
        "quality": {
            "bars_ok": quality_row[0] if quality_row else None,
            "fundamentals_ok": quality_row[1] if quality_row else None,
            "sentiment_ok": quality_row[2] if quality_row else None,
            "intraday_ok": quality_row[3] if quality_row else None,
            "missingness": quality_row[4] if quality_row else None,
            "anomaly_flags": quality_row[5] if quality_row else None,
            "quality_score": quality_row[6] if quality_row else None,
        },
        "bars": {
            "open": bar_row[0] if bar_row else None,
            "high": bar_row[1] if bar_row else None,
            "low": bar_row[2] if bar_row else None,
            "close": bar_row[3] if bar_row else None,
            "volume": bar_row[4] if bar_row else None,
        },
        "sentiment": {
            "headline_count": sentiment_row[0] if sentiment_row else None,
            "sentiment_score": sentiment_row[1] if sentiment_row else None,
        },
    }
