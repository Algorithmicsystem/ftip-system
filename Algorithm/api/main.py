from __future__ import annotations

import os
import math
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel


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


def _parse_date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid date '{s}'. Use YYYY-MM-DD.")


# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------

class Candle(BaseModel):
    timestamp: str  # ISO date "YYYY-MM-DD" (daily)
    close: float
    volume: Optional[float] = None

    # optional extra features (your system supports them)
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


# ------------------------------------------------------------------------------
# Massive (Polygon-compatible) REST API client
# ------------------------------------------------------------------------------

# NOTE:
# - For Polygon, set MARKETDATA_BASE_URL = "https://api.polygon.io"
# - Your code supports MASSIVE_* and POLYGON_* env var names.
MARKETDATA_BASE_URL = (
    _env("MASSIVE_BASE_URL")
    or _env("POLYGON_BASE_URL")
    or "https://api.polygon.io"
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
    """
    Polygon-compatible Aggregates (Daily):
      GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}?adjusted=true&sort=asc&limit=50000&apiKey=...
    """
    api_key = _require_marketdata_key()
    sym = symbol.upper().strip()

    url = f"{MARKETDATA_BASE_URL}/v2/aggs/ticker/{sym}/range/1/day/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=60)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Market data request failed: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Market data error {resp.status_code}: {resp.text}")

    js = resp.json()
    results = js.get("results") or []
    if not results:
        raise HTTPException(status_code=404, detail=f"No bars returned for {sym} in range.")

    candles: List[Candle] = []
    for r in results:
        # Polygon agg fields: t (ms), c (close), v (volume)
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


# ------------------------------------------------------------------------------
# Massive Flat Files (S3-compatible) config (not used yet; just exposed safely)
# ------------------------------------------------------------------------------

MASSIVE_S3_ENDPOINT = _env("MASSIVE_S3_ENDPOINT", "https://files.massive.com") or "https://files.massive.com"
MASSIVE_S3_ACCESS_KEY_ID = _env("MASSIVE_S3_ACCESS_KEY_ID")
MASSIVE_S3_SECRET_ACCESS_KEY = _env("MASSIVE_S3_SECRET_ACCESS_KEY")


# ------------------------------------------------------------------------------
# Helpers: math / stats / indicators
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


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sma(values: List[float], window: int) -> Optional[float]:
    if window <= 0 or len(values) < window:
        return None
    return sum(values[-window:]) / window


def _max_drawdown(equity: List[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak != 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def _rsi(values: List[float], period: int = 14) -> Optional[float]:
    # Classic Wilder RSI
    if period <= 0 or len(values) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        chg = values[i] - values[i - 1]
        gains.append(max(0.0, chg))
        losses.append(max(0.0, -chg))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Continue smoothing
    for i in range(period + 1, len(values)):
        chg = values[i] - values[i - 1]
        gain = max(0.0, chg)
        loss = max(0.0, -chg)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _zscore_last(values: List[float], window: int) -> Optional[float]:
    if window <= 1 or len(values) < window:
        return None
    w = values[-window:]
    mu = _mean(w)
    sd = _std(w)
    if sd == 0:
        return 0.0
    return (values[-1] - mu) / sd


def _safe_momentum(closes: List[float], window: int) -> Optional[float]:
    # momentum over window days: close/close[-window] - 1
    if window <= 0 or len(closes) <= window:
        return None
    base = closes[-(window + 1)]
    if base == 0:
        return None
    return (closes[-1] / base) - 1.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ------------------------------------------------------------------------------
# Backtest core (placeholder buy-and-hold)
# ------------------------------------------------------------------------------

def run_backtest_core(candles: List[Candle], include_equity: bool, include_returns: bool) -> BacktestSummary:
    closes = [c.close for c in candles]
    dates = [c.timestamp for c in candles]

    rets = _pct_change(closes)  # daily returns
    equity = [1.0]
    for r in rets[1:]:
        equity.append(equity[-1] * (1.0 + r))

    total_return = equity[-1] - 1.0
    n = max(1, len(rets) - 1)  # number of return periods
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
# Signal engine: features -> regime -> score -> signal
# ------------------------------------------------------------------------------

def _compute_features(candles: List[Candle]) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []
    closes = [c.close for c in candles]
    vols = [c.volume for c in candles]

    last_close = closes[-1]
    mom_5 = _safe_momentum(closes, 5)
    mom_21 = _safe_momentum(closes, 21)
    mom_63 = _safe_momentum(closes, 63)

    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    trend_sma20_50 = None
    if sma20 is not None and sma50 is not None and sma50 != 0:
        trend_sma20_50 = (sma20 / sma50) - 1.0

    # Volatility from daily returns
    rets = _pct_change(closes)
    vol_ann = _std(rets[1:]) * math.sqrt(252.0) if len(rets) > 2 else None

    rsi14 = _rsi(closes, 14)

    volume_z20 = None
    if all(v is not None for v in vols):
        vol_vals = [float(v) for v in vols if v is not None]
        if len(vol_vals) == len(vols):
            volume_z20 = _zscore_last(vol_vals, 20)

    # Notes
    if trend_sma20_50 is not None:
        if trend_sma20_50 > 0:
            notes.append("Short-term trend (SMA20 vs SMA50) is positive.")
        else:
            notes.append("Short-term trend (SMA20 vs SMA50) is negative.")

    if vol_ann is not None and vol_ann >= 0.45:
        notes.append("Volatility is elevated; treat signal with caution.")

    feats: Dict[str, Any] = {
        "mom_5": mom_5,
        "mom_21": mom_21,
        "mom_63": mom_63,
        "trend_sma20_50": trend_sma20_50,
        "volatility_ann": vol_ann,
        "rsi14": rsi14,
        "volume_z20": volume_z20,
        "last_close": last_close,
    }
    return feats, notes


def _detect_regime(features: Dict[str, Any]) -> Tuple[str, Dict[str, float], List[str]]:
    """
    Simple regime detection:
      - HIGH_VOL if vol >= 0.45
      - TRENDING if abs(trend) >= 0.03 and vol < 0.35
      - else CHOPPY
    """
    notes: List[str] = []
    vol = features.get("volatility_ann")
    trend = features.get("trend_sma20_50")

    regime = "CHOPPY"
    if isinstance(vol, (int, float)) and vol is not None and vol >= 0.45:
        regime = "HIGH_VOL"
        notes.append("Regime: HIGH_VOL (volatility high).")
    elif isinstance(trend, (int, float)) and trend is not None and abs(trend) >= 0.03 and (vol is None or vol < 0.35):
        regime = "TRENDING"
        notes.append("Regime: TRENDING (trend strength strong).")
    else:
        regime = "CHOPPY"
        notes.append("Regime: CHOPPY (trend not strong or mixed).")

    # Dynamic thresholds by regime
    thresholds = {"buy": 0.25, "sell": -0.25}  # baseline
    if regime == "TRENDING":
        thresholds = {"buy": 0.20, "sell": -0.20}
    elif regime == "CHOPPY":
        thresholds = {"buy": 0.35, "sell": -0.35}
    elif regime == "HIGH_VOL":
        thresholds = {"buy": 0.45, "sell": -0.45}

    return regime, thresholds, notes


def _score_signal(features: Dict[str, Any], regime: str) -> Tuple[float, float, List[str]]:
    """
    Produce:
      - score in [-1, 1]
      - confidence in [0, 1]
    """
    notes: List[str] = []

    mom5 = features.get("mom_5")
    mom21 = features.get("mom_21")
    mom63 = features.get("mom_63")
    trend = features.get("trend_sma20_50")
    vol = features.get("volatility_ann")
    rsi = features.get("rsi14")
    volz = features.get("volume_z20")

    # Data quality factor: how many core features are available
    core = [mom21, mom63, trend, vol, rsi]
    available = sum(1 for x in core if isinstance(x, (int, float)) and x is not None)
    data_quality = available / len(core)  # 0..1

    # Normalize / shape terms
    def _tanh_scaled(x: float, scale: float) -> float:
        return math.tanh(x * scale)

    score_raw = 0.0
    w_sum = 0.0

    # Momentum (medium/long are more stable)
    if isinstance(mom21, (int, float)) and mom21 is not None:
        score_raw += 1.2 * _tanh_scaled(float(mom21), 6.0)
        w_sum += 1.2
    if isinstance(mom63, (int, float)) and mom63 is not None:
        score_raw += 1.5 * _tanh_scaled(float(mom63), 4.0)
        w_sum += 1.5
    if isinstance(mom5, (int, float)) and mom5 is not None:
        # short-term is noisier: smaller weight
        score_raw += 0.5 * _tanh_scaled(float(mom5), 8.0)
        w_sum += 0.5

    # Trend
    if isinstance(trend, (int, float)) and trend is not None:
        score_raw += 1.3 * _tanh_scaled(float(trend), 18.0)
        w_sum += 1.3

    # RSI mean-reversion tilt (below 50 slightly bullish, above 50 slightly bearish for entries)
    if isinstance(rsi, (int, float)) and rsi is not None:
        # map RSI 0..100 to centered -1..1 (50 -> 0)
        rsi_center = (50.0 - float(rsi)) / 50.0
        score_raw += 0.6 * _tanh_scaled(rsi_center, 2.5)
        w_sum += 0.6

    # Volume anomaly as confirmation (very light)
    if isinstance(volz, (int, float)) and volz is not None:
        score_raw += 0.2 * _tanh_scaled(float(volz), 0.8)
        w_sum += 0.2

    if w_sum == 0:
        return 0.0, 0.0, ["Insufficient features to compute score."]

    score = score_raw / w_sum  # already ~[-1, 1] due to tanh parts
    score = float(_clamp(score, -1.0, 1.0))

    # Confidence: abs(score) tempered by data quality and regime risk
    conf = abs(score) * data_quality
    if regime == "HIGH_VOL":
        conf *= 0.65
        notes.append("Confidence reduced due to HIGH_VOL regime.")
    elif regime == "CHOPPY":
        conf *= 0.80
        notes.append("Confidence reduced due to CHOPPY regime.")
    else:
        conf *= 1.00

    # Additional penalty if vol is extreme
    if isinstance(vol, (int, float)) and vol is not None and float(vol) > 0.80:
        conf *= 0.75
        notes.append("Confidence reduced due to extremely high volatility.")

    conf = float(_clamp(conf, 0.0, 1.0))
    return score, conf, notes


def _signal_from_score(score: float, thresholds: Dict[str, float]) -> str:
    buy_th = thresholds.get("buy", 0.25)
    sell_th = thresholds.get("sell", -0.25)
    if score >= buy_th:
        return "BUY"
    if score <= sell_th:
        return "SELL"
    return "HOLD"


def _fetch_lookback_candles(symbol: str, as_of: str, lookback: int) -> Tuple[List[Candle], int, List[str]]:
    """
    Fetch candles from market data API ending at as_of.
    We over-fetch by calendar days because trading days are fewer than calendar days.
    If lookback is bigger than what exists, we compute with what's available and return effective_lookback.
    """
    notes: List[str] = []
    sym = symbol.upper().strip()
    as_of_d = _parse_date(as_of)

    # Over-fetch window: rough mapping trading->calendar days (252 trading ~ 365 calendar)
    # Add buffer.
    cal_days = int(max(30, lookback * 2))  # generous buffer
    from_d = as_of_d - dt.timedelta(days=cal_days)
    to_d = as_of_d

    candles = massive_fetch_daily_bars(sym, from_d.isoformat(), to_d.isoformat())

    # keep only <= as_of (defensive)
    candles = [c for c in candles if _parse_date(c.timestamp) <= as_of_d]
    if len(candles) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough bars to compute signal. Need at least 20 bars <= {as_of}, got {len(candles)}.",
        )

    # Use last N bars available up to lookback
    effective = min(lookback, len(candles))
    if effective < lookback:
        notes.append(f"Requested lookback={lookback}, but only {effective} bars available by {as_of}. Using effective_lookback={effective}.")

    candles = candles[-effective:]
    return candles, effective, notes


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
    # Never return secrets; just show whether they are set.
    return {
        "s3_endpoint": MASSIVE_S3_ENDPOINT,
        "access_key_set": bool(MASSIVE_S3_ACCESS_KEY_ID),
        "secret_key_set": bool(MASSIVE_S3_SECRET_ACCESS_KEY),
    }


@app.post("/run_backtest")
def run_backtest(req: BacktestRequest) -> Dict[str, Any]:
    # Path A: user provided candles directly
    if req.data and len(req.data) > 0:
        candles = req.data
    else:
        # Path B: fetch from Massive/Polygon
        if not (req.symbol and req.from_date and req.to_date):
            raise HTTPException(
                status_code=400,
                detail="Provide either `data` OR (`symbol`, `from_date`, `to_date`) to fetch from Massive/Polygon.",
            )
        candles = massive_fetch_daily_bars(req.symbol, req.from_date, req.to_date)

    summary = run_backtest_core(candles, req.include_equity_curve, req.include_returns)
    return {"backtest_summary": summary.model_dump(exclude_none=True)}


@app.get("/signal")
def signal(
    symbol: str = Query(..., description="Ticker symbol (e.g. AAPL)"),
    as_of: str = Query(..., description="YYYY-MM-DD (last date to include)"),
    lookback: int = Query(252, ge=20, le=5000, description="Trading bars to use (auto-adjusts if not available)"),
) -> Dict[str, Any]:
    candles, effective_lookback, fetch_notes = _fetch_lookback_candles(symbol, as_of, lookback)

    features, feat_notes = _compute_features(candles)
    regime, thresholds, regime_notes = _detect_regime(features)
    score, confidence, score_notes = _score_signal(features, regime)
    sig = _signal_from_score(score, thresholds)

    notes = []
    notes.extend(fetch_notes)
    notes.extend(feat_notes)
    notes.extend(regime_notes)
    notes.extend(score_notes)

    # remove None fields from features for cleaner output
    clean_features = {k: v for k, v in features.items() if v is not None}

    return {
        "symbol": symbol.upper(),
        "as_of": as_of,
        "lookback": lookback,
        "effective_lookback": effective_lookback,
        "regime": regime,
        "thresholds": thresholds,
        "score": float(score),
        "signal": sig,
        "confidence": float(confidence),
        "features": clean_features,
        "notes": notes,
    }


@app.post("/run_scores")
def run_scores(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_scores endpoint placeholder", "received_keys": list(payload.keys())}


@app.post("/run_all")
def run_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_all endpoint placeholder", "received_keys": list(payload.keys())}

