from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from api.alpha import build_canonical_features
from api.research import build_research_snapshot_from_bars


def compute_daily_features(
    bars: List[Dict[str, object]],
    *,
    sentiment_score: Optional[float] = None,
    sentiment_mean: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    if not bars:
        return {}
    latest_date = max(str(row.get("as_of_date") or row.get("date") or "") for row in bars if row)
    if not latest_date:
        return {}

    enriched_rows: List[Dict[str, object]] = []
    for row in bars:
        item = dict(row)
        item.setdefault("date", item.get("as_of_date"))
        enriched_rows.append(item)
    source_hint = next(
        (
            str(item.get("source"))
            for item in enriched_rows
            if item.get("source") is not None
        ),
        "provided_market_bars",
    )

    snapshot = build_research_snapshot_from_bars(
        str((bars[-1] or {}).get("symbol") or "UNKNOWN"),
        pd.to_datetime(latest_date).date(),
        252,
        enriched_rows,
        source_hint=source_hint,
    )
    if sentiment_score is not None or sentiment_mean is not None:
        snapshot["sentiment_history"] = [
            {
                "as_of_date": latest_date,
                "sentiment_score": sentiment_score,
                "sentiment_mean": sentiment_mean,
            }
        ]
    payload = build_canonical_features(snapshot)
    return payload.get("features") or {}


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
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=False,
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
