from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _to_frame(bars: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    if df.empty:
        return df
    df = df.sort_values("as_of_date")
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    return df


def _trend_metrics(series: pd.Series) -> tuple[Optional[float], Optional[float]]:
    if series is None or len(series) < 3:
        return None, None
    y = np.log(series.astype(float).values)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return float(slope), float(r2) if r2 is not None else None


def _max_drawdown(series: pd.Series) -> Optional[float]:
    if series is None or len(series) < 2:
        return None
    values = series.astype(float).values
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return float(drawdown.min())


def compute_daily_features(
    bars: List[Dict[str, object]],
    *,
    sentiment_score: Optional[float] = None,
    sentiment_mean: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    df = _to_frame(bars)
    if df.empty:
        return {}
    close = df["close"].astype(float)
    returns = close.pct_change()

    def _ret(n: int) -> Optional[float]:
        if len(close) <= n:
            return None
        return float(close.iloc[-1] / close.iloc[-(n + 1)] - 1)

    ret_1d = _ret(1)
    ret_5d = _ret(5)
    ret_21d = _ret(21)
    vol_21d = (
        float(returns.tail(21).std() * np.sqrt(252)) if len(returns) >= 21 else None
    )
    vol_63d = (
        float(returns.tail(63).std() * np.sqrt(252)) if len(returns) >= 63 else None
    )

    tr = pd.DataFrame(
        {
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": close,
        }
    )
    tr["prev_close"] = tr["close"].shift(1)
    tr["tr"] = (tr["high"] - tr["low"]).abs()
    tr["tr"] = np.maximum(tr["tr"], (tr["high"] - tr["prev_close"]).abs())
    tr["tr"] = np.maximum(tr["tr"], (tr["low"] - tr["prev_close"]).abs())
    atr_14 = float(tr["tr"].tail(14).mean()) if len(tr) >= 14 else None
    atr_pct = float(atr_14 / close.iloc[-1]) if atr_14 and close.iloc[-1] else None

    trend_slope_21d, trend_r2_21d = _trend_metrics(close.tail(21))
    trend_slope_63d, trend_r2_63d = _trend_metrics(close.tail(63))

    mom_vol_adj_21d = None
    if ret_21d is not None and vol_21d:
        mom_vol_adj_21d = float(ret_21d / vol_21d) if vol_21d else None

    maxdd_63d = _max_drawdown(close.tail(63))
    dollar_vol_21d = None
    if "volume" in df:
        dollar_vol_21d = float((df["volume"].astype(float) * close).tail(21).mean())

    sentiment_surprise = None
    if sentiment_score is not None and sentiment_mean is not None:
        sentiment_surprise = float(sentiment_score - sentiment_mean)

    regime_label = None
    regime_strength = None
    if trend_slope_63d is not None and trend_r2_63d is not None:
        if abs(trend_slope_63d) > 0 and trend_r2_63d >= 0.4:
            regime_label = "trend"
        else:
            regime_label = "range"
        regime_strength = float(abs(trend_slope_63d) * (trend_r2_63d or 0))

    return {
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_21d": ret_21d,
        "vol_21d": vol_21d,
        "vol_63d": vol_63d,
        "atr_14": atr_14,
        "atr_pct": atr_pct,
        "trend_slope_21d": trend_slope_21d,
        "trend_r2_21d": trend_r2_21d,
        "trend_slope_63d": trend_slope_63d,
        "trend_r2_63d": trend_r2_63d,
        "mom_vol_adj_21d": mom_vol_adj_21d,
        "maxdd_63d": maxdd_63d,
        "dollar_vol_21d": dollar_vol_21d,
        "sentiment_score": sentiment_score,
        "sentiment_surprise": sentiment_surprise,
        "regime_label": regime_label,
        "regime_strength": regime_strength,
    }


def compute_intraday_features(
    bars: List[Dict[str, object]],
    timeframe: str,
) -> List[Dict[str, object]]:
    if not bars:
        return []
    df = pd.DataFrame(bars)
    df = df.sort_values("ts")
    df["close"] = df["close"].astype(float)
    df["ret_1bar"] = df["close"].pct_change()
    df["vol_n"] = df["ret_1bar"].rolling(5).std()
    df["trend_slope_n"] = (
        df["close"]
        .rolling(5)
        .apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
        )
    )

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "symbol": row["symbol"],
                "ts": row["ts"],
                "timeframe": timeframe,
                "ret_1bar": (
                    float(row["ret_1bar"]) if pd.notna(row["ret_1bar"]) else None
                ),
                "vol_n": float(row["vol_n"]) if pd.notna(row["vol_n"]) else None,
                "trend_slope_n": (
                    float(row["trend_slope_n"])
                    if pd.notna(row["trend_slope_n"])
                    else None
                ),
            }
        )
    return rows
