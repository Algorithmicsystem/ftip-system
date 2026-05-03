from __future__ import annotations

import json
import hashlib
import math
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .depth_layers import (
    build_breadth_depth,
    build_cross_asset_depth,
    build_event_depth,
    build_liquidity_depth,
    build_stress_depth,
)

CANONICAL_FEATURE_VERSION = "phase9_canonical_features_v1"
FEATURE_SCHEMA_VERSION = 3


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_int(value: Any) -> Optional[int]:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.fmean(clean))


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return None
    return float(statistics.pstdev(clean))


def _ret(closes: Sequence[float], periods: int) -> Optional[float]:
    if len(closes) <= periods:
        return None
    base = closes[-(periods + 1)]
    if base == 0:
        return None
    return float(closes[-1] / base - 1.0)


def _returns(closes: Sequence[float]) -> List[float]:
    output: List[float] = []
    for idx in range(1, len(closes)):
        prev = closes[idx - 1]
        cur = closes[idx]
        if prev == 0:
            continue
        output.append(float(cur / prev - 1.0))
    return output


def _realized_vol(closes: Sequence[float], window: int) -> Optional[float]:
    rets = _returns(closes)
    if len(rets) < window:
        return None
    sigma = _std(rets[-window:])
    if sigma is None:
        return None
    return float(sigma * math.sqrt(252.0))


def _trend_metrics(closes: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(closes) < 3:
        return None, None
    y_values = [math.log(value) for value in closes if value > 0]
    if len(y_values) < 3:
        return None, None
    x_values = list(range(len(y_values)))
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator == 0:
        return None, None
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    predicted = [slope * x + intercept for x in x_values]
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(y_values, predicted))
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    r2 = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(slope), float(r2) if r2 is not None else None


def _max_drawdown(closes: Sequence[float]) -> Optional[float]:
    if len(closes) < 2:
        return None
    peak = closes[0]
    worst = 0.0
    for close in closes:
        peak = max(peak, close)
        if peak == 0:
            continue
        worst = min(worst, (close - peak) / peak)
    return float(worst)


def _sma(closes: Sequence[float], window: int) -> Optional[float]:
    if len(closes) < window:
        return None
    return float(sum(closes[-window:]) / window)


def _rsi(closes: Sequence[float], window: int = 14) -> Optional[float]:
    if len(closes) < window + 1:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for idx in range(len(closes) - window, len(closes)):
        delta = closes[idx] - closes[idx - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))
    avg_gain = _mean(gains) or 0.0
    avg_loss = _mean(losses) or 0.0
    if avg_loss == 0:
        return 70.0
    rs = avg_gain / avg_loss
    return float(max(0.0, min(100.0, 100.0 - (100.0 / (1.0 + rs)))))


def _zscore_last(values: Sequence[float], window: int) -> Optional[float]:
    if len(values) < window or window < 2:
        return None
    windowed = list(values[-window:])
    mu = statistics.fmean(windowed)
    sigma = _std(windowed)
    if sigma in (None, 0):
        return 0.0
    return float((windowed[-1] - mu) / sigma)


def _atr(rows: Sequence[Dict[str, Any]], window: int = 14) -> Optional[float]:
    if len(rows) < max(window, 2):
        return None
    true_ranges: List[float] = []
    for idx in range(1, len(rows)):
        current = rows[idx]
        previous = rows[idx - 1]
        high = _safe_float(current.get("high"))
        low = _safe_float(current.get("low"))
        close_prev = _safe_float(previous.get("close"))
        close_cur = _safe_float(current.get("close"))
        if close_cur is None:
            continue
        if high is None:
            high = close_cur
        if low is None:
            low = close_cur
        tr = abs(high - low)
        if close_prev is not None:
            tr = max(tr, abs(high - close_prev), abs(low - close_prev))
        true_ranges.append(float(tr))
    if len(true_ranges) < window:
        return None
    return _mean(true_ranges[-window:])


def classify_signal_regime(features: Dict[str, Any]) -> Tuple[str, Optional[float], str]:
    vol_ann = _safe_float(features.get("volatility_ann")) or _safe_float(features.get("vol_63d")) or 0.0
    trend = _safe_float(features.get("trend_sma20_50")) or 0.0
    trend_r2 = _safe_float(features.get("trend_r2_63d")) or 0.0
    regime_strength = abs(trend) * max(trend_r2, 0.25)
    if vol_ann >= 0.45:
        return "high_vol", float(regime_strength), "HIGH_VOL"
    if abs(trend) >= 0.05 or trend_r2 >= 0.4:
        return "trend", float(regime_strength), "TRENDING"
    return "choppy", float(regime_strength), "CHOPPY"


def build_canonical_features(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    price_rows = list(snapshot.get("price_bars") or [])
    closes = [
        float(row["close"])
        for row in price_rows
        if _safe_float(row.get("close")) is not None
    ]
    volumes = [float(row.get("volume") or 0.0) for row in price_rows]
    if not closes:
        return {
            "feature_version": CANONICAL_FEATURE_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "snapshot_id": snapshot.get("snapshot_id"),
            "snapshot_version": snapshot.get("snapshot_version"),
            "requested_lookback": snapshot.get("requested_lookback"),
            "effective_lookback": 0,
            "features": {},
            "meta": {
                "coverage_status": "unavailable",
                "feature_hash": _hash_payload({"snapshot_id": snapshot.get("snapshot_id"), "features": {}}),
            },
        }

    sentiment_history = list(snapshot.get("sentiment_history") or [])
    latest_sentiment = sentiment_history[-1] if sentiment_history else {}
    sentiment_score = _safe_float(latest_sentiment.get("sentiment_score"))
    sentiment_mean = _safe_float(latest_sentiment.get("sentiment_mean"))
    sentiment_surprise = None
    if sentiment_score is not None and sentiment_mean is not None:
        sentiment_surprise = float(sentiment_score - sentiment_mean)

    atr_14 = _atr(price_rows, 14)
    atr_pct = None
    if atr_14 is not None and closes[-1] != 0:
        atr_pct = float(atr_14 / closes[-1])

    trend_slope_21d, trend_r2_21d = _trend_metrics(closes[-21:])
    trend_slope_63d, trend_r2_63d = _trend_metrics(closes[-63:])
    mom_5 = _ret(closes, 5)
    mom_21 = _ret(closes, 21)
    mom_63 = _ret(closes, 63)
    mom_126 = _ret(closes, 126)
    mom_252 = _ret(closes, 252)
    ret_1d = _ret(closes, 1)
    ret_3d = _ret(closes, 3)
    ret_5d = _ret(closes, 5)
    ret_10d = _ret(closes, 10)
    ret_21d = _ret(closes, 21)
    ret_63d = _ret(closes, 63)
    ret_126d = _ret(closes, 126)
    ret_252d = _ret(closes, 252)
    vol_21d = _realized_vol(closes, 21)
    vol_63d = _realized_vol(closes, 63)
    vol_126d = _realized_vol(closes, 126)
    volatility_ann = _realized_vol(closes, max(2, min(len(closes) - 1, 63)))
    mom_vol_adj_21d = None
    if ret_21d is not None and vol_21d not in (None, 0):
        mom_vol_adj_21d = float(ret_21d / float(vol_21d))
    maxdd_63d = _max_drawdown(closes[-63:])
    maxdd_252d = _max_drawdown(closes[-252:])
    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    trend_sma20_50 = None
    if sma20 is not None and sma50 not in (None, 0):
        trend_sma20_50 = float(sma20 / sma50 - 1.0)
    rsi14 = _rsi(closes, 14)
    volume_z20 = _zscore_last(volumes, 20)
    dollar_vol_21d = None
    if len(price_rows) >= 21:
        trailing = price_rows[-21:]
        dollars = []
        for row in trailing:
            close = _safe_float(row.get("close"))
            volume = _safe_float(row.get("volume"))
            if close is None or volume is None:
                continue
            dollars.append(close * volume)
        if dollars:
            dollar_vol_21d = float(sum(dollars) / len(dollars))

    features: Dict[str, Any] = {
        "ret_1d": ret_1d,
        "ret_3d": ret_3d,
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "ret_21d": ret_21d,
        "ret_63d": ret_63d,
        "ret_126d": ret_126d,
        "ret_252d": ret_252d,
        "vol_21d": vol_21d,
        "vol_63d": vol_63d,
        "vol_126d": vol_126d,
        "atr_14": atr_14,
        "atr_pct": atr_pct,
        "trend_slope_21d": trend_slope_21d,
        "trend_r2_21d": trend_r2_21d,
        "trend_slope_63d": trend_slope_63d,
        "trend_r2_63d": trend_r2_63d,
        "mom_vol_adj_21d": mom_vol_adj_21d,
        "maxdd_63d": maxdd_63d,
        "maxdd_252d": maxdd_252d,
        "dollar_vol_21d": dollar_vol_21d,
        "sentiment_score": sentiment_score,
        "sentiment_surprise": sentiment_surprise,
        "mom_5": mom_5,
        "mom_21": mom_21,
        "mom_63": mom_63,
        "mom_126": mom_126,
        "mom_252": mom_252,
        "trend_sma20_50": trend_sma20_50,
        "volatility_ann": volatility_ann,
        "rsi14": rsi14,
        "volume_z20": volume_z20,
        "last_close": closes[-1],
    }
    regime_label, regime_strength, signal_regime = classify_signal_regime(features)
    features["regime_label"] = regime_label
    features["regime_strength"] = regime_strength
    features["signal_regime"] = signal_regime

    event_depth = build_event_depth(snapshot, features)
    liquidity_depth = build_liquidity_depth(snapshot, features, event_depth)
    breadth_depth = build_breadth_depth(snapshot, features)
    cross_asset_depth = build_cross_asset_depth(snapshot, features, breadth_depth)
    stress_depth = build_stress_depth(
        snapshot,
        features,
        event_depth,
        liquidity_depth,
        breadth_depth,
        cross_asset_depth,
    )
    features.update(event_depth)
    features.update(liquidity_depth)
    features.update(breadth_depth)
    features.update(cross_asset_depth)
    features.update(stress_depth)

    feature_hash = _hash_payload(
        {
            "snapshot_id": snapshot.get("snapshot_id"),
            "requested_lookback": snapshot.get("requested_lookback"),
            "features": features,
        }
    )
    coverage_status = (
        "available"
        if len(price_rows) >= 63
        else "partial"
        if len(price_rows) >= 30
        else "insufficient_history"
    )
    return {
        "feature_version": CANONICAL_FEATURE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "snapshot_id": snapshot.get("snapshot_id"),
        "snapshot_version": snapshot.get("snapshot_version"),
        "requested_lookback": snapshot.get("requested_lookback"),
        "effective_lookback": len(price_rows),
        "features": features,
        "meta": {
            "coverage_status": coverage_status,
            "available_history_bars": len(price_rows),
            "feature_hash": feature_hash,
            "price_source": (snapshot.get("provenance") or {}).get("market_bars_source"),
            "event_source": (snapshot.get("provenance") or {}).get("event_source"),
            "breadth_source": (snapshot.get("provenance") or {}).get("breadth_source"),
        },
    }


def features_daily_row(feature_payload: Dict[str, Any]) -> Dict[str, Any]:
    features = dict(feature_payload.get("features") or {})
    return {
        "ret_1d": features.get("ret_1d"),
        "ret_5d": features.get("ret_5d"),
        "ret_21d": features.get("ret_21d"),
        "vol_21d": features.get("vol_21d"),
        "vol_63d": features.get("vol_63d"),
        "atr_14": features.get("atr_14"),
        "atr_pct": features.get("atr_pct"),
        "trend_slope_21d": features.get("trend_slope_21d"),
        "trend_r2_21d": features.get("trend_r2_21d"),
        "trend_slope_63d": features.get("trend_slope_63d"),
        "trend_r2_63d": features.get("trend_r2_63d"),
        "mom_vol_adj_21d": features.get("mom_vol_adj_21d"),
        "maxdd_63d": features.get("maxdd_63d"),
        "dollar_vol_21d": features.get("dollar_vol_21d"),
        "sentiment_score": features.get("sentiment_score"),
        "sentiment_surprise": features.get("sentiment_surprise"),
        "regime_label": features.get("regime_label"),
        "regime_strength": features.get("regime_strength"),
        "feature_version": FEATURE_SCHEMA_VERSION,
        "effective_lookback": feature_payload.get("effective_lookback"),
        "snapshot_id": feature_payload.get("snapshot_id"),
        "snapshot_version": feature_payload.get("snapshot_version"),
        "canonical_features": features,
        "feature_meta": {
            **(feature_payload.get("meta") or {}),
            "feature_version": feature_payload.get("feature_version"),
            "feature_schema_version": feature_payload.get("feature_schema_version"),
        },
    }
