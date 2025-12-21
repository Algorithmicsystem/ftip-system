from __future__ import annotations

import time
import os
import math
import json
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# App + environment helpers
# ------------------------------------------------------------------------------

APP_NAME = "FTIP System API"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _git_sha() -> str:
    return (
        _env("RAILWAY_GIT_COMMIT_SHA")
        or _env("GIT_COMMIT_SHA")
        or _env("COMMIT_SHA")
        or "unknown"
    )


def _railway_env() -> str:
    return _env("RAILWAY_ENVIRONMENT", "unknown") or "unknown"


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------

class Candle(BaseModel):
    timestamp: str  # ISO date "YYYY-MM-DD" (daily)
    close: float
    volume: Optional[float] = None

    # optional extra features (supported for future use)
    fundamental: Optional[float] = None
    sentiment: Optional[float] = None
    crowd: Optional[float] = None


class BacktestRequest(BaseModel):
    # Either provide `data` OR provide (symbol + date range) to fetch from Massive (Polygon legacy)
    data: Optional[List[Candle]] = None

    symbol: Optional[str] = None
    from_date: Optional[str] = None  # YYYY-MM-DD
    to_date: Optional[str] = None    # YYYY-MM-DD

    include_equity_curve: bool = False
    include_returns: bool = False


class BacktestSummary(BaseModel):
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    volatility: float

    equity_curve: Optional[Dict[str, float]] = None
    returns: Optional[Dict[str, float]] = None


class SignalResponse(BaseModel):
    symbol: str
    as_of: str
    lookback: int
    effective_lookback: int
    regime: str
    thresholds: Dict[str, float]
    score: float
    signal: str
    confidence: float
    features: Dict[str, float]
    notes: List[str] = Field(default_factory=list)

    # calibration diagnostics
    calibration_loaded: bool = False
    calibration_meta: Optional[Dict[str, Any]] = None


class SignalsRequest(BaseModel):
    symbols: List[str]
    as_of: str
    lookback: int = 252


class WalkForwardRequest(BaseModel):
    symbol: str
    from_date: str
    to_date: str
    lookback: int = 252
    horizons: List[int] = Field(default_factory=lambda: [5, 21, 63])


class CalibrateRequest(BaseModel):
    symbol: str
    from_date: str
    to_date: str
    lookback: int = 252
    horizons: List[int] = Field(default_factory=lambda: [5, 21, 63])
    optimize_horizon: int = 21
    min_trades_per_side: int = 15  # require enough BUY/SELL signals to be meaningful


class PortfolioRequest(BaseModel):
    symbols: List[str]
    as_of: str
    lookback: int = 252

    # Optional knobs (can also be overridden via env vars)
    max_weight: Optional[float] = None          # cap per position
    hold_multiplier: Optional[float] = None     # HOLD gets smaller weight than BUY
    min_confidence: Optional[float] = None      # ignore weak signals
    allow_shorts: bool = False                  # default False (signals only; no short weights)


class PortfolioBacktestRequest(BaseModel):
    symbols: List[str]
    from_date: str  # YYYY-MM-DD
    to_date: str    # YYYY-MM-DD
    lookback: int = 252

    # Rebalance + costs
    rebalance_every: int = 5  # trading days (5 ≈ weekly)
    trading_cost_bps: float = 5.0  # 5 bps = 0.05% per $ traded (round trip approx)
    slippage_bps: float = 0.0

    # Portfolio knobs (same behavior as /portfolio_signals)
    max_weight: Optional[float] = None
    hold_multiplier: Optional[float] = None
    min_confidence: Optional[float] = None
    allow_shorts: bool = False

    include_equity_curve: bool = True

    # Turnover controls (realism)
    min_trade_delta: float = 0.01          # deadband: ignore changes < 1%
    max_turnover_per_rebalance: float = 0.20  # cap turnover at 20% per rebalance


class PortfolioBacktestResponse(BaseModel):
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    volatility: float
    turnover: float

    equity_curve: Optional[Dict[str, float]] = None


# ------------------------------------------------------------------------------
# Massive (Polygon legacy) REST API client
# ------------------------------------------------------------------------------

MARKETDATA_BASE_URL = (
    _env("MASSIVE_BASE_URL")
    or _env("POLYGON_BASE_URL")
    or "https://api.massive.com"
)

MARKETDATA_API_KEY = _env("MASSIVE_API_KEY") or _env("POLYGON_API_KEY")
MARKETDATA_API_SECRET = _env("MASSIVE_API_SECRET") or _env("POLYGON_API_SECRET")  # optional


def _require_marketdata_key() -> str:
    if not MARKETDATA_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="MASSIVE_API_KEY (or POLYGON_API_KEY) is not set on the server. Add it in Railway Variables.",
        )
    return MARKETDATA_API_KEY


def massive_fetch_daily_bars(symbol: str, from_date: str, to_date: str) -> List[Candle]:
    api_key = _require_marketdata_key()
    sym = symbol.upper().strip()

    url = f"{MARKETDATA_BASE_URL}/v2/aggs/ticker/{sym}/range/1/day/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    # Retry/backoff knobs (env overridable)
    max_retries = int(_env("FTIP_MD_MAX_RETRIES", "4") or "4")
    base_sleep = float(_env("FTIP_MD_RETRY_BASE_SLEEP", "2.0") or "2.0")  # seconds

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
        except Exception as e:
            last_err = f"Market data request failed: {e}"
            # small backoff for network errors too
            time.sleep(base_sleep * (2 ** attempt))
            continue

        # If rate limited, backoff + retry
        if resp.status_code == 429:
            last_err = f"Market data rate-limited 429: {resp.text}"
            sleep_s = base_sleep * (2 ** attempt)
            time.sleep(sleep_s)
            continue

        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Market data error {resp.status_code}: {resp.text}")

        js = resp.json()
        results = js.get("results") or []
        if not results:
            raise HTTPException(status_code=404, detail=f"No bars returned for {sym} in range.")

        candles: List[Candle] = []
        for r in results:
            t_ms = r.get("t")
            c = r.get("c")
            v = r.get("v")
            if t_ms is None or c is None:
                continue
            day = dt.datetime.utcfromtimestamp(t_ms / 1000.0).date().isoformat()
            candles.append(Candle(timestamp=day, close=float(c), volume=float(v) if v is not None else None))

        if not candles:
            raise HTTPException(status_code=404, detail=f"Returned empty/invalid candles for {sym}.")
        return candles

    # If we exhausted retries
    raise HTTPException(status_code=502, detail=last_err or "Market data failed after retries.")


# ------------------------------------------------------------------------------
# Massive Flat Files (S3-compatible) config (keys come from env vars)
# ------------------------------------------------------------------------------

MASSIVE_S3_ENDPOINT = _env("MASSIVE_S3_ENDPOINT", "https://files.massive.com") or "https://files.massive.com"
MASSIVE_S3_ACCESS_KEY_ID = _env("MASSIVE_S3_ACCESS_KEY_ID")
MASSIVE_S3_SECRET_ACCESS_KEY = _env("MASSIVE_S3_SECRET_ACCESS_KEY")


# ------------------------------------------------------------------------------
# Math helpers
# ------------------------------------------------------------------------------

def _pct_change(series: List[float]) -> List[float]:
    out = [0.0]
    for i in range(1, len(series)):
        prev = series[i - 1]
        cur = series[i]
        if prev == 0:
            out.append(0.0)
        else:
            out.append((cur / prev) - 1.0)
    return out


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def _max_drawdown(equity: List[float]) -> float:
    peak = equity[0] if equity else 1.0
    mdd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak != 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _clean_small(x: float, eps: float = 1e-12) -> float:
    return 0.0 if abs(x) < eps else float(x)

def _apply_deadband(old_w: Dict[str, float], new_w: Dict[str, float], min_delta: float) -> Dict[str, float]:
    """Zero-out tiny trades: if |new-old| < min_delta, keep old."""
    out: Dict[str, float] = {}
    syms = set(old_w.keys()) | set(new_w.keys())
    for s in syms:
        ow = float(old_w.get(s, 0.0))
        nw = float(new_w.get(s, 0.0))
        if abs(nw - ow) < float(min_delta):
            if abs(ow) > 0:
                out[s] = ow
        else:
            if abs(nw) > 0:
                out[s] = nw
    return out

def _cap_turnover(old_w: Dict[str, float], target_w: Dict[str, float], max_turnover: float) -> Dict[str, float]:
    """
    Scale deltas so sum(|delta|) <= max_turnover.
    Turnover here is sum abs(delta weights) across assets.
    """
    if max_turnover <= 0:
        return old_w.copy()

    syms = set(old_w.keys()) | set(target_w.keys())
    deltas: Dict[str, float] = {}
    turnover = 0.0
    for s in syms:
        d = float(target_w.get(s, 0.0)) - float(old_w.get(s, 0.0))
        deltas[s] = d
        turnover += abs(d)

    if turnover <= float(max_turnover) or turnover <= 1e-15:
        return target_w

    scale = float(max_turnover) / turnover
    out: Dict[str, float] = {}
    for s in syms:
        ow = float(old_w.get(s, 0.0))
        nw = ow + deltas[s] * scale
        if abs(nw) > 0:
            out[s] = float(nw)
    return out

def _normalize_long_only(weights: Dict[str, float], max_weight: float) -> Tuple[Dict[str, float], float]:
    """
    Long-only normalization + cap. Returns (weights, cash).
    Ensures sum(weights) <= 1 and each weight <= max_weight.
    """
    # keep only positive (long-only)
    w = {s: float(v) for s, v in weights.items() if float(v) > 0.0}

    if not w:
        return {}, 1.0

    # normalize to 1 first
    s0 = sum(w.values())
    if s0 <= 0:
        return {}, 1.0
    for s in list(w.keys()):
        w[s] = w[s] / s0

    # apply cap
    max_w = float(_clamp(float(max_weight), 0.01, 1.0))
    w = {s: min(v, max_w) for s, v in w.items()}

    # after capping, weights sum may be < 1 => keep cash
    s1 = sum(w.values())
    if s1 <= 0:
        return {}, 1.0

    # rescale capped weights to their own sum (so we don't artificially shrink)
    # BUT do NOT force them to 1.0, because cap may intentionally leave cash.
    # Keep as-is: cash = 1 - s1.
    cash = float(_clamp(1.0 - s1, 0.0, 1.0))

    # clean tiny floats
    w = {s: _clean_small(v) for s, v in w.items() if abs(v) > 1e-12}
    cash = _clean_small(cash)

    return w, cash


# ------------------------------------------------------------------------------
# Backtest engine (simple buy-and-hold placeholder)
# ------------------------------------------------------------------------------

def run_backtest_core(candles: List[Candle], include_equity: bool, include_returns: bool) -> BacktestSummary:
    closes = [c.close for c in candles]
    dates = [c.timestamp for c in candles]

    rets = _pct_change(closes)  # daily returns
    equity = [1.0]
    for r in rets[1:]:
        equity.append(equity[-1] * (1.0 + r))

    total_return = equity[-1] - 1.0

    n = max(1, len(rets) - 1)
    ann_return = (1.0 + total_return) ** (252.0 / n) - 1.0 if n > 0 else 0.0

    vol = _std(rets[1:]) * math.sqrt(252.0) if len(rets) > 2 else 0.0
    sharpe = (ann_return / vol) if vol > 0 else 0.0
    mdd = _max_drawdown(equity)

    out = BacktestSummary(
        total_return=float(total_return),
        annual_return=float(ann_return),
        sharpe=float(sharpe),
        max_drawdown=float(mdd),
        volatility=float(vol),
        equity_curve={d: float(equity[i]) for i, d in enumerate(dates)} if include_equity else None,
        returns={d: float(rets[i]) for i, d in enumerate(dates)} if include_returns else None,
    )
    return out


# ------------------------------------------------------------------------------
# Feature engine + regime detection + thresholds
# ------------------------------------------------------------------------------

def _sma(values: List[float], window: int) -> float:
    if window <= 0 or len(values) < window:
        return float("nan")
    return _mean(values[-window:])


def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(len(closes) - period, len(closes)):
        chg = closes[i] - closes[i - 1]
        if chg >= 0:
            gains.append(chg)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-chg)
    avg_gain = _mean(gains)
    avg_loss = _mean(losses)
    if avg_loss == 0:
        return 70.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(_clamp(rsi, 0.0, 100.0))


def _zscore_last(values: List[float], window: int) -> float:
    if len(values) < window or window < 2:
        return 0.0
    w = values[-window:]
    mu = _mean(w)
    sd = _std(w)
    if sd == 0:
        return 0.0
    return float((w[-1] - mu) / sd)


def compute_features(candles: List[Candle]) -> Dict[str, float]:
    closes = [c.close for c in candles]
    vols = [float(c.volume or 0.0) for c in candles]

    def mom(k: int) -> float:
        if len(closes) < k + 1:
            return 0.0
        return float(closes[-1] / closes[-1 - k] - 1.0)

    r = _pct_change(closes)
    vol_ann = _std(r[1:]) * math.sqrt(252.0) if len(r) > 2 else 0.0

    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    trend = 0.0
    if not (math.isnan(sma20) or math.isnan(sma50)) and sma50 != 0:
        trend = float((sma20 / sma50) - 1.0)

    feat = {
        "mom_5": mom(5),
        "mom_21": mom(21),
        "mom_63": mom(63),
        "trend_sma20_50": trend,
        "volatility_ann": float(vol_ann),
        "rsi14": float(_rsi(closes, 14)),
        "volume_z20": float(_zscore_last(vols, 20)),
        "last_close": float(closes[-1]),
    }
    return feat


def detect_regime(features: Dict[str, float]) -> str:
    vol = features.get("volatility_ann", 0.0)
    trend = abs(features.get("trend_sma20_50", 0.0))

    if vol >= 0.45:
        return "HIGH_VOL"
    if trend >= 0.05:
        return "TRENDING"
    return "CHOPPY"


def score_from_features(features: Dict[str, float]) -> Tuple[float, List[str]]:
    notes: List[str] = []

    rsi = features["rsi14"]
    rsi_sig = _clamp((rsi - 50.0) / 25.0, -1.0, 1.0)

    mom = 0.4 * features["mom_21"] + 0.6 * features["mom_63"]
    mom_sig = _clamp(mom / 0.25, -1.0, 1.0)

    trend_sig = _clamp(features["trend_sma20_50"] / 0.10, -1.0, 1.0)
    if features["trend_sma20_50"] > 0:
        notes.append("Short-term trend (SMA20 vs SMA50) is positive.")
    elif features["trend_sma20_50"] < 0:
        notes.append("Short-term trend (SMA20 vs SMA50) is negative.")

    volz = features["volume_z20"]
    vol_sig = _clamp(volz / 3.0, -1.0, 1.0)

    vola = features["volatility_ann"]
    vola_pen = _clamp((vola - 0.25) / 0.50, 0.0, 0.5)  # 0..0.5

    raw = 0.45 * mom_sig + 0.30 * trend_sig + 0.20 * rsi_sig + 0.05 * vol_sig
    score = _clamp(raw * (1.0 - vola_pen), -1.0, 1.0)

    if vola >= 0.45:
        notes.append("Volatility is elevated; treat signal with caution.")
    return float(score), notes


# ------------------------------------------------------------------------------
# Calibration (optional) via env var
# ------------------------------------------------------------------------------

def _load_calibration() -> Tuple[bool, Optional[Dict[str, Any]]]:
    s = _env("FTIP_CALIBRATION_JSON")
    if not s:
        return False, None
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return False, None
        return True, obj
    except Exception:
        return False, None


def _thresholds_for_regime(regime: str, calibration: Optional[Dict[str, Any]]) -> Dict[str, float]:
    defaults = {
        "TRENDING": {"buy": 0.2, "sell": -0.2},
        "CHOPPY": {"buy": 0.3, "sell": -0.3},
        "HIGH_VOL": {"buy": 0.45, "sell": -0.45},
    }
    if not calibration:
        return defaults.get(regime, {"buy": 0.3, "sell": -0.3})

    thr_by_reg = calibration.get("thresholds_by_regime") or {}
    if regime in thr_by_reg and isinstance(thr_by_reg[regime], dict):
        b = float(thr_by_reg[regime].get("buy", defaults.get(regime, {"buy": 0.3})["buy"]))
        s = float(thr_by_reg[regime].get("sell", defaults.get(regime, {"sell": -0.3})["sell"]))
        return {"buy": b, "sell": s}

    return defaults.get(regime, {"buy": 0.3, "sell": -0.3})


# ------------------------------------------------------------------------------
# Signal computation (single symbol)
# ------------------------------------------------------------------------------

def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _filter_upto(candles: List[Candle], as_of: str) -> List[Candle]:
    cutoff = _parse_date(as_of)
    out = []
    for c in candles:
        d = _parse_date(c.timestamp)
        if d <= cutoff:
            out.append(c)
    return out


def compute_signal_for_symbol(symbol: str, as_of: str, lookback: int) -> SignalResponse:
    as_of_d = _parse_date(as_of)
    from_guess = (as_of_d - dt.timedelta(days=900)).isoformat()
    candles_all = massive_fetch_daily_bars(symbol, from_guess, as_of)

    candles_upto = _filter_upto(candles_all, as_of)
    if len(candles_upto) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data to compute signal. Need at least 30 bars <= {as_of}, got {len(candles_upto)}.",
        )

    effective = min(lookback, len(candles_upto))
    candles = candles_upto[-effective:]

    features = compute_features(candles)
    regime = detect_regime(features)

    score, notes = score_from_features(features)

    calibration_loaded, calibration = _load_calibration()
    thresholds = _thresholds_for_regime(regime, calibration)

    sig = "HOLD"
    if score >= thresholds["buy"]:
        sig = "BUY"
    elif score <= thresholds["sell"]:
        sig = "SELL"

    conf = abs(score)
    if regime == "HIGH_VOL":
        conf *= 0.65
        notes.append("Confidence reduced due to HIGH_VOL regime.")
    if effective < lookback:
        notes.insert(
            0,
            f"Requested lookback={lookback}, but only {effective} bars available by {as_of}. Using effective_lookback={effective}.",
        )

    if regime == "TRENDING":
        notes.append("Regime: TRENDING.")
    elif regime == "CHOPPY":
        notes.append("Regime: CHOPPY.")
    else:
        notes.append("Regime: HIGH_VOL.")

    if calibration_loaded:
        notes.append("Using calibrated thresholds from FTIP_CALIBRATION_JSON.")

    meta = None
    if calibration_loaded and calibration:
        meta = {
            "optimize_horizon": calibration.get("optimize_horizon"),
            "created_at_utc": calibration.get("created_at_utc"),
            "symbol": calibration.get("symbol"),
            "train_range": calibration.get("train_range"),
        }

    return SignalResponse(
        symbol=symbol.upper(),
        as_of=as_of,
        lookback=lookback,
        effective_lookback=effective,
        regime=regime,
        thresholds=thresholds,
        score=float(score),
        signal=sig,
        confidence=float(conf),
        features={k: float(v) for k, v in features.items()},
        notes=notes,
        calibration_loaded=bool(calibration_loaded),
        calibration_meta=meta,
    )


def compute_signal_for_symbol_from_candles(symbol: str, as_of: str, lookback: int, candles_all: List[Candle]) -> SignalResponse:
    candles_upto = _filter_upto(candles_all, as_of)

    if len(candles_upto) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data to compute signal. Need at least 30 bars <= {as_of}, got {len(candles_upto)}.",
        )

    effective = min(lookback, len(candles_upto))
    candles = candles_upto[-effective:]

    features = compute_features(candles)
    regime = detect_regime(features)

    score, notes = score_from_features(features)

    calibration_loaded, calibration = _load_calibration()
    thresholds = _thresholds_for_regime(regime, calibration)

    sig = "HOLD"
    if score >= thresholds["buy"]:
        sig = "BUY"
    elif score <= thresholds["sell"]:
        sig = "SELL"

    conf = abs(score)
    if regime == "HIGH_VOL":
        conf *= 0.65
        notes.append("Confidence reduced due to HIGH_VOL regime.")

    if effective < lookback:
        notes.insert(0, f"Requested lookback={lookback}, but only {effective} bars available by {as_of}. Using effective_lookback={effective}.")

    if regime == "TRENDING":
        notes.append("Regime: TRENDING.")
    elif regime == "CHOPPY":
        notes.append("Regime: CHOPPY.")
    else:
        notes.append("Regime: HIGH_VOL.")

    if calibration_loaded:
        notes.append("Using calibrated thresholds from FTIP_CALIBRATION_JSON.")

    meta = None
    if calibration_loaded and calibration:
        meta = {
            "optimize_horizon": calibration.get("optimize_horizon"),
            "created_at_utc": calibration.get("created_at_utc"),
            "symbol": calibration.get("symbol"),
            "train_range": calibration.get("train_range"),
        }

    return SignalResponse(
        symbol=symbol.upper(),
        as_of=as_of,
        lookback=lookback,
        effective_lookback=effective,
        regime=regime,
        thresholds=thresholds,
        score=float(score),
        signal=sig,
        confidence=float(conf),
        features={k: float(v) for k, v in features.items()},
        notes=notes,
        calibration_loaded=bool(calibration_loaded),
        calibration_meta=meta,
    )


# ------------------------------------------------------------------------------
# Walk-forward + Calibration endpoints (training helper)
# ------------------------------------------------------------------------------

def _forward_return(closes: List[float], idx: int, horizon: int) -> Optional[float]:
    j = idx + horizon
    if j >= len(closes):
        return None
    if closes[idx] == 0:
        return None
    return float(closes[j] / closes[idx] - 1.0)


def walk_forward_table(symbol: str, from_date: str, to_date: str, lookback: int, horizons: List[int]) -> List[Dict[str, Any]]:
    candles_all = massive_fetch_daily_bars(symbol, from_date, to_date)

    closes = [c.close for c in candles_all]
    dates = [c.timestamp for c in candles_all]

    rows: List[Dict[str, Any]] = []
    for i in range(len(candles_all)):
        start = max(0, i - lookback + 1)
        window = candles_all[start:i + 1]
        if len(window) < 30:
            continue

        feats = compute_features(window)
        regime = detect_regime(feats)
        score, _notes = score_from_features(feats)

        cal_loaded, cal = _load_calibration()
        thr = _thresholds_for_regime(regime, cal)

        sig = "HOLD"
        if score >= thr["buy"]:
            sig = "BUY"
        elif score <= thr["sell"]:
            sig = "SELL"

        row = {
            "date": dates[i],
            "signal": sig,
            "score": float(score),
            "confidence": float(abs(score) * (0.65 if regime == "HIGH_VOL" else 1.0)),
            "regime": regime,
        }
        for h in horizons:
            fr = _forward_return(closes, i, h)
            row[f"fwd_ret_{h}"] = fr
        rows.append(row)

    return rows


def calibrate_thresholds(rows: List[Dict[str, Any]], optimize_horizon: int, min_trades_per_side: int) -> Dict[str, Any]:
    created = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    regimes = sorted(set(r["regime"] for r in rows))
    thresholds_by_regime: Dict[str, Dict[str, float]] = {}
    diagnostics: Dict[str, Any] = {}

    buys = [0.15, 0.20, 0.25, 0.30, 0.35, 0.45]
    sells = [-0.15, -0.20, -0.25, -0.30, -0.35, -0.45]

    key = f"fwd_ret_{optimize_horizon}"

    for reg in regimes:
        rrows = [r for r in rows if r["regime"] == reg and r.get(key) is not None]
        if not rrows:
            continue

        best = None
        best_metric = None

        for b in buys:
            for s in sells:
                if s >= 0:
                    continue
                buysig = [r for r in rrows if r["score"] >= b]
                sellsig = [r for r in rrows if r["score"] <= s]

                if len(buysig) < min_trades_per_side or len(sellsig) < min_trades_per_side:
                    continue

                buy_ret = _mean([float(r[key]) for r in buysig])
                sell_ret = _mean([float(r[key]) for r in sellsig])

                metric = buy_ret - sell_ret
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best = {"buy": float(b), "sell": float(s), "metric": float(metric), "buy_n": len(buysig), "sell_n": len(sellsig)}

        if best is None:
            if reg == "TRENDING":
                thresholds_by_regime[reg] = {"buy": 0.2, "sell": -0.2}
            elif reg == "CHOPPY":
                thresholds_by_regime[reg] = {"buy": 0.3, "sell": -0.3}
            else:
                thresholds_by_regime[reg] = {"buy": 0.45, "sell": -0.45}

            diagnostics[reg] = {"note": "No valid threshold pair met min_trades constraints; using defaults.", "row_count": len(rrows)}
        else:
            thresholds_by_regime[reg] = {"buy": best["buy"], "sell": best["sell"]}
            diagnostics[reg] = {"best": best, "row_count": len(rrows)}

    return {
        "created_at_utc": created,
        "optimize_horizon": int(optimize_horizon),
        "thresholds_by_regime": thresholds_by_regime,
        "diagnostics": diagnostics,
    }


# ------------------------------------------------------------------------------
# Portfolio construction layer (FIXED cap redistribution)
# ------------------------------------------------------------------------------

def _portfolio_knobs(req: PortfolioRequest) -> Dict[str, float]:
    max_w = req.max_weight if req.max_weight is not None else _env_float("FTIP_PORTFOLIO_MAX_WEIGHT", 0.30)
    hold_mult = req.hold_multiplier if req.hold_multiplier is not None else _env_float("FTIP_PORTFOLIO_HOLD_MULT", 0.35)
    min_conf = req.min_confidence if req.min_confidence is not None else _env_float("FTIP_PORTFOLIO_MIN_CONF", 0.10)

    max_w = _clamp(float(max_w), 0.01, 1.0)
    hold_mult = _clamp(float(hold_mult), 0.0, 1.0)
    min_conf = _clamp(float(min_conf), 0.0, 1.0)
    return {"max_weight": max_w, "hold_multiplier": hold_mult, "min_confidence": min_conf}


def _raw_weight_from_signal(sig: Dict[str, Any], hold_mult: float, min_conf: float, allow_shorts: bool) -> float:
    signal = (sig.get("signal") or "HOLD").upper()
    conf = float(sig.get("confidence") or 0.0)
    vola = float(
        sig.get("features", {}).get("volatility_ann")
        or sig.get("features", {}).get("volatility")
        or sig.get("volatility_ann")
        or 0.0
    )

    if conf < min_conf:
        return 0.0

    vol_adj = 1.0 / (max(1e-6, vola))

    if signal == "BUY":
        return conf * vol_adj
    if signal == "HOLD":
        return conf * vol_adj * hold_mult
    if signal == "SELL":
        return (-conf * vol_adj) if allow_shorts else 0.0
    return 0.0

def _daily_returns_from_closes(dates: List[str], closes: List[float]) -> Dict[str, float]:
    # returns keyed by date (date has return from prev close -> this close)
    out: Dict[str, float] = {}
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev == 0:
            out[dates[i]] = 0.0
        else:
            out[dates[i]] = float(cur / prev - 1.0)
    return out


def _portfolio_weights_for_date(req: PortfolioBacktestRequest, as_of: str) -> Tuple[Dict[str, float], float]:
    try:
        preq = PortfolioRequest(
            symbols=req.symbols,
            as_of=as_of,
            lookback=req.lookback,
            max_weight=req.max_weight,
            hold_multiplier=req.hold_multiplier,
            min_confidence=req.min_confidence,
            allow_shorts=req.allow_shorts,
        )

        knobs = _portfolio_knobs(preq)
        max_w = knobs["max_weight"]
        hold_mult = knobs["hold_multiplier"]
        min_conf = knobs["min_confidence"]

        per: Dict[str, Any] = {}
        for sym in preq.symbols:
            s = (sym or "").strip().upper()
            if not s:
                continue
            out = compute_signal_for_symbol(s, preq.as_of, preq.lookback).model_dump(exclude_none=True)
            per[s] = out

        raw: Dict[str, float] = {}
        for s, sig in per.items():
            rw = _raw_weight_from_signal(sig, hold_mult=hold_mult, min_conf=min_conf, allow_shorts=preq.allow_shorts)
            if rw != 0.0:
                raw[s] = float(rw)

        if not raw:
            return {}, 1.0

        portfolio: Dict[str, float] = {}
        cash = 1.0

        if preq.allow_shorts:
            gross = sum(abs(v) for v in raw.values()) or 1.0
            for s, v in raw.items():
                w = v / gross
                w = _clamp(w, -max_w, max_w)
                portfolio[s] = float(w)
            net = sum(portfolio.values())
            cash = float(_clamp(1.0 - max(net, 0.0), 0.0, 1.0))
        else:
            long_raw = {s: v for s, v in raw.items() if v > 0}
            total = sum(long_raw.values()) or 1.0
            tmp = {s: min(v / total, max_w) for s, v in long_raw.items()}

            capped_sum = sum(tmp.values())
            if capped_sum > 0:
                scale = min(1.0, capped_sum) / capped_sum
                portfolio = {s: float(v * scale) for s, v in tmp.items()}
            cash = float(_clamp(1.0 - sum(portfolio.values()), 0.0, 1.0))

        return portfolio, cash

    except HTTPException as e:
        # Most common: not enough history early in the backtest period
        if e.status_code in (400, 404):
            return {}, 1.0
        raise


def backtest_portfolio(req: PortfolioBacktestRequest) -> PortfolioBacktestResponse:
    # ------------------------------------------------------------
    # 1) Fetch candles once per symbol (include extra buffer)
    # ------------------------------------------------------------
    start_dt = _parse_date(req.from_date)
    end_dt = _parse_date(req.to_date)

    # Buffer: use max(900 days, lookback*4) to be safe for early days
    warmup_days = max(900, int(req.lookback) * 4)
    buffer_start = (start_dt - dt.timedelta(days=warmup_days)).isoformat()

    candles_by_sym: Dict[str, List[Candle]] = {}
    for sym in req.symbols:
        s = (sym or "").strip().upper()
        if not s:
            continue
        candles_by_sym[s] = massive_fetch_daily_bars(s, buffer_start, req.to_date)

    if not candles_by_sym:
        return PortfolioBacktestResponse(
            total_return=0.0,
            annual_return=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            turnover=0.0,
            equity_curve={} if req.include_equity_curve else None,
        )

    # ------------------------------------------------------------
    # 2) Build common trading calendar within [from_date, to_date]
    # ------------------------------------------------------------
    date_sets: List[set] = []
    for s, cs in candles_by_sym.items():
        ds = {c.timestamp for c in cs if req.from_date <= c.timestamp <= req.to_date}
        if ds:
            date_sets.append(ds)

    if not date_sets:
        return PortfolioBacktestResponse(
            total_return=0.0,
            annual_return=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            turnover=0.0,
            equity_curve={} if req.include_equity_curve else None,
        )

    common_dates = sorted(set.intersection(*date_sets))
    if len(common_dates) < 2:
        return PortfolioBacktestResponse(
            total_return=0.0,
            annual_return=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            turnover=0.0,
            equity_curve={} if req.include_equity_curve else None,
        )

    common_dates_set = set(common_dates)

    # ------------------------------------------------------------
    # 3) Build daily return series per symbol keyed by date
    # ------------------------------------------------------------
    rets_by_sym: Dict[str, Dict[str, float]] = {}
    closes_by_sym_by_date: Dict[str, Dict[str, float]] = {}

    for s, cs in candles_by_sym.items():
        pairs = [(c.timestamp, c.close) for c in cs if c.timestamp in common_dates_set]
        pairs.sort(key=lambda x: x[0])
        if len(pairs) < 2:
            continue

        ds2 = [p[0] for p in pairs]
        closes2 = [p[1] for p in pairs]

        rets_by_sym[s] = _daily_returns_from_closes(ds2, closes2)
        closes_by_sym_by_date[s] = {d: float(c) for d, c in zip(ds2, closes2)}

    # ------------------------------------------------------------
    # 4) Helper: compute per-symbol signal using CACHED candles
    #    (no API calls, no rate-limits)
    # ------------------------------------------------------------
    def _compute_signal_from_cached(symbol: str, as_of: str, lookback: int) -> Optional[Dict[str, Any]]:
        # Use pre-fetched candles, filter up to as_of
        cs_all = candles_by_sym.get(symbol, [])
        if not cs_all:
            return None

        # Keep candles <= as_of
        cutoff = _parse_date(as_of)
        cs_upto: List[Candle] = []
        for c in cs_all:
            try:
                d = _parse_date(c.timestamp)
            except Exception:
                continue
            if d <= cutoff:
                cs_upto.append(c)

        # If not enough history yet, return None (we'll stay in cash)
        if len(cs_upto) < 30:
            return None

        eff = min(int(lookback), len(cs_upto))
        window = cs_upto[-eff:]

        feats = compute_features(window)
        regime = detect_regime(feats)
        score, notes = score_from_features(feats)

        cal_loaded, cal = _load_calibration()
        thr = _thresholds_for_regime(regime, cal)

        sig = "HOLD"
        if score >= thr["buy"]:
            sig = "BUY"
        elif score <= thr["sell"]:
            sig = "SELL"

        conf = abs(score)
        if regime == "HIGH_VOL":
            conf *= 0.65
            notes.append("Confidence reduced due to HIGH_VOL regime.")

        # Minimal dict shape compatible with _raw_weight_from_signal()
        return {
            "symbol": symbol,
            "as_of": as_of,
            "lookback": int(lookback),
            "effective_lookback": int(eff),
            "regime": regime,
            "thresholds": thr,
            "score": float(score),
            "signal": sig,
            "confidence": float(conf),
            "features": {k: float(v) for k, v in feats.items()},
            "notes": notes,
            "calibration_loaded": bool(cal_loaded),
        }

    # ------------------------------------------------------------
    # 5) Helper: compute portfolio weights for a given as_of date
    #    (same logic as /portfolio_signals, but fully offline)
    # ------------------------------------------------------------
    def _portfolio_weights_for_date_cached(as_of: str) -> Tuple[Dict[str, float], float]:
        # knobs: request overrides env
        max_w = float(req.max_weight) if getattr(req, "max_weight", None) is not None else _env_float("FTIP_PORTFOLIO_MAX_WEIGHT", 0.30)
        hold_mult = float(req.hold_multiplier) if getattr(req, "hold_multiplier", None) is not None else _env_float("FTIP_PORTFOLIO_HOLD_MULT", 0.35)
        min_conf = float(req.min_confidence) if getattr(req, "min_confidence", None) is not None else _env_float("FTIP_PORTFOLIO_MIN_CONF", 0.10)
        allow_shorts = bool(getattr(req, "allow_shorts", False))

        # clamps
        max_w = _clamp(max_w, 0.01, 1.0)
        hold_mult = _clamp(hold_mult, 0.0, 1.0)
        min_conf = _clamp(min_conf, 0.0, 1.0)

        # compute per-symbol signals
        per: Dict[str, Any] = {}
        for s in candles_by_sym.keys():
            sig = _compute_signal_from_cached(s, as_of, int(req.lookback))
            if sig is not None:
                per[s] = sig

        # raw weights
        raw: Dict[str, float] = {}
        for s, sig in per.items():
            rw = _raw_weight_from_signal(sig, hold_mult=hold_mult, min_conf=min_conf, allow_shorts=allow_shorts)
            if rw != 0.0:
                raw[s] = float(rw)

        if not raw:
            return {}, 1.0  # all cash

        portfolio: Dict[str, float] = {}

        if allow_shorts:
            gross = sum(abs(v) for v in raw.values()) or 1.0
            for s, v in raw.items():
                w = v / gross
                w = _clamp(w, -max_w, max_w)
                portfolio[s] = float(w)

            net_long = sum(w for w in portfolio.values() if w > 0)
            cash = float(_clamp(1.0 - net_long, 0.0, 1.0))

            # ensure exact-ish accounting
            # (do not renormalize short portfolios; cash is residual)
            return portfolio, cash

        # long-only
        long_raw = {s: v for s, v in raw.items() if v > 0}
        total = sum(long_raw.values()) or 1.0

        # initial weights with cap
        tmp = {s: min(v / total, max_w) for s, v in long_raw.items()}

        # scale so sum(tmp) <= 1 (keeps cash leftover)
        ssum = sum(tmp.values())
        if ssum > 0:
            scale = min(1.0, ssum) / ssum
            for s, v in tmp.items():
                portfolio[s] = float(v * scale)

        cash = float(_clamp(1.0 - sum(portfolio.values()), 0.0, 1.0))
        return portfolio, cash

    # ------------------------------------------------------------
    # 6) Rebalance loop
    # ------------------------------------------------------------
cost_rate = (float(req.trading_cost_bps) + float(req.slippage_bps)) / 10000.0

equity = 1.0
equity_curve: Dict[str, float] = {}
daily_port_rets: List[float] = []

weights: Dict[str, float] = {}
cash_w = 1.0
total_turnover = 0.0

reb_every = max(1, int(req.rebalance_every))

for i, d in enumerate(common_dates):
    if i == 0:
        equity_curve[d] = equity
        continue

    # Rebalance on schedule
    if (i - 1) % reb_every == 0:
        # 1) target weights from cached signal engine
        target_w, target_cash = _portfolio_weights_for_date_cached(as_of=d)

        # 2) deadband: ignore tiny changes to reduce churn
        min_delta = float(getattr(req, "min_trade_delta", 0.01))
        target_w = _apply_deadband(weights, target_w, min_delta=min_delta)

        # 3) turnover cap: scale changes so sum(|delta|) <= max_turnover_per_rebalance
        max_t = float(getattr(req, "max_turnover_per_rebalance", 0.20))
        capped_w = _cap_turnover(weights, target_w, max_turnover=max_t)

        # 4) normalize + cap again (long-only) and recompute cash
        capped_w, capped_cash = _normalize_long_only(capped_w, max_weight=float(req.max_weight))

        # 5) turnover (assets only) for trading costs
        turnover = 0.0
        syms = set(weights.keys()) | set(capped_w.keys())
        for s in syms:
            turnover += abs(float(capped_w.get(s, 0.0)) - float(weights.get(s, 0.0)))

        # apply costs on turnover
        equity *= (1.0 - turnover * cost_rate)

        total_turnover += turnover
        weights = capped_w
        cash_w = capped_cash  # tracked, not used in return (cash return = 0)

    # Daily portfolio return (cash return = 0)
    pr = 0.0
    for s, w in weights.items():
        pr += float(w) * float(rets_by_sym.get(s, {}).get(d, 0.0))

    equity *= (1.0 + pr)
    equity_curve[d] = equity
    daily_port_rets.append(pr)

    # ------------------------------------------------------------
    # 7) Stats
    # ------------------------------------------------------------
    total_return = equity - 1.0

    n = max(1, len(daily_port_rets))
    ann_return = (1.0 + total_return) ** (252.0 / n) - 1.0 if n > 0 else 0.0

    vol = _std(daily_port_rets) * math.sqrt(252.0) if len(daily_port_rets) > 2 else 0.0
    sharpe = (ann_return / vol) if vol > 0 else 0.0

    mdd = _max_drawdown(list(equity_curve.values()))

    return PortfolioBacktestResponse(
        total_return=float(total_return),
        annual_return=float(ann_return),
        sharpe=float(sharpe),
        max_drawdown=float(mdd),
        volatility=float(vol),
        turnover=float(total_turnover),
        equity_curve=equity_curve if req.include_equity_curve else None,
    )


def _normalize_long_only_with_caps(raw_pos: Dict[str, float], max_w: float) -> Tuple[Dict[str, float], float]:
    """
    Correct long-only allocation:
    - Start from proportional preferences (raw_pos)
    - Apply per-name cap
    - Redistribute leftover to uncapped names until done
    """
    items = {k: float(v) for k, v in raw_pos.items() if v > 0.0}
    if not items:
        return {}, 1.0

    total = sum(items.values())
    if total <= 0:
        return {}, 1.0

    pref = {k: v / total for k, v in items.items()}  # preferences sum to 1

    final: Dict[str, float] = {k: 0.0 for k in pref.keys()}
    active = set(pref.keys())
    remaining = 1.0

    for _ in range(10000):
        if remaining <= 1e-12 or not active:
            break

        denom = sum(pref[k] for k in active)
        if denom <= 0:
            break

        progressed = False
        for k in list(active):
            target = remaining * (pref[k] / denom)
            room = max_w - final[k]
            add = min(target, room)
            if add > 0:
                final[k] += add
                remaining -= add
                progressed = True
            if final[k] >= max_w - 1e-12:
                active.remove(k)

        if not progressed:
            break

    # Filter tiny weights
    final = {k: float(v) for k, v in final.items() if v > 1e-12}
    cash = float(_clamp(1.0 - sum(final.values()), 0.0, 1.0))
    return final, cash


def _normalize_allow_shorts_with_caps(raw: Dict[str, float], max_w: float) -> Tuple[Dict[str, float], float]:
    """
    Shorts allowed:
    - Normalize by gross exposure
    - Cap abs(weight)
    - Cash is based on net long exposure
    """
    if not raw:
        return {}, 1.0

    gross = sum(abs(v) for v in raw.values())
    if gross <= 0:
        return {}, 1.0

    wts: Dict[str, float] = {}
    for k, v in raw.items():
        w = v / gross
        w = _clamp(w, -max_w, max_w)
        if abs(w) > 1e-12:
            wts[k] = float(w)

    net = sum(wts.values())
    cash = float(_clamp(1.0 - max(net, 0.0), 0.0, 1.0))
    return wts, cash


# ------------------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------------------

app = FastAPI(title=APP_NAME, version="1.0.0")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": APP_NAME,
        "status": "ok",
        "endpoints": [
            "/health",
            "/version",
            "/run_all",
            "/run_backtest",
            "/run_scores",
            "/signal",
            "/signals",
            "/portfolio_signals",
            "/walk_forward",
            "/calibrate",
            "/market/massive/bars",
            "/storage/massive/s3_config",
            "/docs",
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/version")
def version() -> Dict[str, Any]:
    return {
        "railway_git_commit_sha": _git_sha(),
        "railway_environment": _railway_env(),
    }


@app.get("/market/massive/bars")
def market_massive_bars(
    symbol: str = Query(..., description="Ticker symbol (e.g. AAPL)"),
    from_date: str = Query(..., description="YYYY-MM-DD"),
    to_date: str = Query(..., description="YYYY-MM-DD"),
) -> Dict[str, Any]:
    candles = massive_fetch_daily_bars(symbol, from_date, to_date)
    return {
        "symbol": symbol.upper(),
        "from_date": from_date,
        "to_date": to_date,
        "data": [c.model_dump() for c in candles],
    }


@app.get("/storage/massive/s3_config")
def massive_s3_config() -> Dict[str, Any]:
    return {
        "s3_endpoint": MASSIVE_S3_ENDPOINT,
        "access_key_set": bool(MASSIVE_S3_ACCESS_KEY_ID),
        "secret_key_set": bool(MASSIVE_S3_SECRET_ACCESS_KEY),
    }


@app.post("/run_backtest")
def run_backtest(req: BacktestRequest) -> Dict[str, Any]:
    if req.data and len(req.data) > 0:
        candles = req.data
    else:
        if not (req.symbol and req.from_date and req.to_date):
            raise HTTPException(
                status_code=400,
                detail="Provide either `data` OR (`symbol`, `from_date`, `to_date`) to fetch from Massive.",
            )
        candles = massive_fetch_daily_bars(req.symbol, req.from_date, req.to_date)

    summary = run_backtest_core(candles, req.include_equity_curve, req.include_returns)
    return {"backtest_summary": summary.model_dump(exclude_none=True)}


@app.get("/signal")
def signal(
    symbol: str = Query(...),
    as_of: str = Query(..., description="YYYY-MM-DD"),
    lookback: int = Query(252, ge=30, le=2000),
) -> Dict[str, Any]:
    out = compute_signal_for_symbol(symbol, as_of, lookback)
    return out.model_dump(exclude_none=True)


@app.post("/signals")
def signals(req: SignalsRequest) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}

    for sym in req.symbols:
        s = (sym or "").strip().upper()
        if not s:
            continue
        try:
            out = compute_signal_for_symbol(s, req.as_of, req.lookback)
            results[s] = out.model_dump(exclude_none=True)
        except HTTPException as e:
            errors[s] = {"status_code": e.status_code, "detail": e.detail}
        except Exception as e:
            errors[s] = {"status_code": 500, "detail": str(e)}

    return {
        "as_of": req.as_of,
        "lookback": req.lookback,
        "count_ok": len(results),
        "count_error": len(errors),
        "results": results,
        "errors": errors,
    }


@app.post("/portfolio_signals")
def portfolio_signals(req: PortfolioRequest) -> Dict[str, Any]:
    knobs = _portfolio_knobs(req)
    max_w = knobs["max_weight"]
    hold_mult = knobs["hold_multiplier"]
    min_conf = knobs["min_confidence"]

    # Step 1: compute per-symbol signals
    per: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}
    for sym in req.symbols:
        s = (sym or "").strip().upper()
        if not s:
            continue
        try:
            out = compute_signal_for_symbol(s, req.as_of, req.lookback).model_dump(exclude_none=True)
            per[s] = out
        except HTTPException as e:
            errors[s] = {"status_code": e.status_code, "detail": e.detail}
        except Exception as e:
            errors[s] = {"status_code": 500, "detail": str(e)}

    # Step 2: raw weights
    raw: Dict[str, float] = {}
    for s, sig in per.items():
        rw = _raw_weight_from_signal(sig, hold_mult=hold_mult, min_conf=min_conf, allow_shorts=req.allow_shorts)
        if rw != 0.0:
            raw[s] = float(rw)

    if not raw:
        return {
            "as_of": req.as_of,
            "lookback": req.lookback,
            "method": "confidence_vol_targeted",
            "knobs": {
                "max_weight": max_w,
                "hold_multiplier": hold_mult,
                "min_confidence": min_conf,
                "allow_shorts": req.allow_shorts,
            },
            "portfolio": {},
            "cash": 1.0,
            "count_ok": len(per),
            "count_error": len(errors),
            "errors": errors,
            "signals": per,
            "meta": {"count_buy": 0, "count_hold": 0, "count_sell": 0},
            "notes": ["No positions met min_confidence / signal criteria. 100% cash."],
        }

    # Step 3: normalize + caps (FIXED)
    if req.allow_shorts:
        portfolio, cash = _normalize_allow_shorts_with_caps(raw, max_w)
    else:
        portfolio, cash = _normalize_long_only_with_caps(raw, max_w)

    count_buy = sum(1 for s in per.values() if (s.get("signal") == "BUY"))
    count_hold = sum(1 for s in per.values() if (s.get("signal") == "HOLD"))
    count_sell = sum(1 for s in per.values() if (s.get("signal") == "SELL"))

    return {
        "as_of": req.as_of,
        "lookback": req.lookback,
        "method": "confidence_vol_targeted",
        "knobs": {
            "max_weight": max_w,
            "hold_multiplier": hold_mult,
            "min_confidence": min_conf,
            "allow_shorts": req.allow_shorts,
        },
        "portfolio": portfolio,
        "cash": cash,
        "count_ok": len(per),
        "count_error": len(errors),
        "errors": errors,
        "signals": per,
        "meta": {"count_buy": count_buy, "count_hold": count_hold, "count_sell": count_sell},
    }


@app.post("/portfolio_backtest")
def portfolio_backtest(req: PortfolioBacktestRequest) -> Dict[str, Any]:
    out = backtest_portfolio(req)
    return out.model_dump(exclude_none=True)


@app.post("/walk_forward")
def walk_forward(req: WalkForwardRequest) -> Dict[str, Any]:
    rows = walk_forward_table(req.symbol, req.from_date, req.to_date, req.lookback, req.horizons)
    return {
        "symbol": req.symbol.upper(),
        "from_date": req.from_date,
        "to_date": req.to_date,
        "lookback": req.lookback,
        "horizons": req.horizons,
        "rows_sample": rows[:50],
        "row_count": len(rows),
    }


@app.post("/calibrate")
def calibrate(req: CalibrateRequest) -> Dict[str, Any]:
    rows = walk_forward_table(req.symbol, req.from_date, req.to_date, req.lookback, req.horizons)
    cal = calibrate_thresholds(rows, req.optimize_horizon, req.min_trades_per_side)

    env_payload = {
        "created_at_utc": cal["created_at_utc"],
        "symbol": req.symbol.upper(),
        "train_range": {
            "from_date": req.from_date,
            "to_date": req.to_date,
            "eval_start": req.from_date,
            "eval_end": req.to_date,
        },
        "optimize_horizon": cal["optimize_horizon"],
        "thresholds_by_regime": cal["thresholds_by_regime"],
        "diagnostics": cal["diagnostics"],
    }
    env_value = json.dumps(env_payload, separators=(",", ":"))

    return {
        "calibration": env_payload,
        "env_var_name": "FTIP_CALIBRATION_JSON",
        "env_var_value": env_value,
        "next_step": "Paste env_var_value into Railway Variable FTIP_CALIBRATION_JSON then redeploy/restart.",
    }


@app.post("/run_scores")
def run_scores(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_scores endpoint placeholder", "received_keys": list(payload.keys())}


@app.post("/run_all")
def run_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_all endpoint placeholder", "received_keys": list(payload.keys())}

