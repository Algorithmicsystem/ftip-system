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


def _model_dump(m: BaseModel, *, exclude_none: bool = True) -> Dict[str, Any]:
    """
    Compatibility helper: works on Pydantic v2 (model_dump) and v1 (dict).
    """
    if hasattr(m, "model_dump"):
        return m.model_dump(exclude_none=exclude_none)  # type: ignore[attr-defined]
    return m.dict(exclude_none=exclude_none)  # type: ignore[call-arg]


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


class SignalResponse(BaseModel):
    symbol: str
    as_of: str
    lookback: int

    score: float
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0..1

    features: Dict[str, float]
    notes: List[str]


# ------------------------------------------------------------------------------
# Massive (Polygon legacy) REST API client
# ------------------------------------------------------------------------------

# Massive is the new brand name; keep POLYGON_* envs as backward-compatible fallbacks.
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
    """
    Aggregates (Daily):
      GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}?adjusted=true&sort=asc&limit=50000&apiKey=...
    Works on Massive, and still works on Polygon legacy domains.
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
        # aggregate fields:
        # t = timestamp in ms, c = close, v = volume
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
# Massive Flat Files (S3-compatible) config (keys come from env vars)
# NOTE: You are NOT using these in code yet; this just stores them properly.
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


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
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


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0.0, -0.0) else default


def _parse_date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid date '{s}'. Use YYYY-MM-DD.")


def _date_to_str(d: dt.date) -> str:
    return d.isoformat()


def _slice_as_of(candles: List[Candle], as_of: str) -> List[Candle]:
    """
    Candles are daily ISO strings. Keep <= as_of.
    """
    asof = _parse_date(as_of)
    out: List[Candle] = []
    for c in candles:
        try:
            d = _parse_date(c.timestamp)
        except Exception:
            continue
        if d <= asof:
            out.append(c)
    return out


# ------------------------------------------------------------------------------
# Simple backtest engine (buy-and-hold placeholder)
# ------------------------------------------------------------------------------

def run_backtest_core(candles: List[Candle], include_equity: bool, include_returns: bool) -> BacktestSummary:
    closes = [c.close for c in candles]
    dates = [c.timestamp for c in candles]

    rets = _pct_change(closes)  # daily returns
    equity = [1.0]
    for r in rets[1:]:
        equity.append(equity[-1] * (1.0 + r))

    total_return = equity[-1] - 1.0

    # Annualization (assume ~252 trading days)
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
# Feature engineering + scoring engine (signal-only; no broker execution)
# ------------------------------------------------------------------------------

def _sma(values: List[float], window: int) -> float:
    if window <= 0 or len(values) < window:
        return values[-1] if values else 0.0
    return _mean(values[-window:])


def _rsi_like(returns: List[float], window: int = 14) -> float:
    """
    Simple RSI-style oscillator in range [0,100].
    Uses returns, not price deltas. Good enough as a baseline.
    """
    if len(returns) < window + 1:
        return 50.0
    rs = returns[-window:]
    gains = [r for r in rs if r > 0]
    losses = [-r for r in rs if r < 0]
    avg_gain = _mean(gains) if gains else 0.0
    avg_loss = _mean(losses) if losses else 0.0
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0
    rel = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rel))


def _zscore(x: float, series: List[float]) -> float:
    mu = _mean(series)
    sd = _std(series)
    if sd == 0.0:
        return 0.0
    return (x - mu) / sd


def compute_features(candles: List[Candle]) -> Dict[str, float]:
    closes = [c.close for c in candles]
    vols = [float(c.volume) for c in candles if c.volume is not None]
    returns = _pct_change(closes)

    # Momentum features
    mom_5 = _safe_div(closes[-1], closes[-6], 1.0) - 1.0 if len(closes) >= 6 else 0.0
    mom_21 = _safe_div(closes[-1], closes[-22], 1.0) - 1.0 if len(closes) >= 22 else 0.0
    mom_63 = _safe_div(closes[-1], closes[-64], 1.0) - 1.0 if len(closes) >= 64 else 0.0

    sma_20 = _sma(closes, 20)
    sma_50 = _sma(closes, 50)
    trend_sma = _safe_div(sma_20, sma_50, 1.0) - 1.0 if sma_50 != 0 else 0.0

    # Volatility (annualized)
    vol_ann = _std(returns[1:]) * math.sqrt(252.0) if len(returns) > 2 else 0.0

    # RSI-ish
    rsi14 = _rsi_like(returns, 14)

    # Volume zscore vs trailing (if volume exists)
    vol_z = 0.0
    if len(vols) >= 20:
        last_v = vols[-1]
        trailing = vols[-20:]
        vol_z = _zscore(last_v, trailing)

    return {
        "mom_5": float(mom_5),
        "mom_21": float(mom_21),
        "mom_63": float(mom_63),
        "trend_sma20_50": float(trend_sma),
        "volatility_ann": float(vol_ann),
        "rsi14": float(rsi14),
        "volume_z20": float(vol_z),
        "last_close": float(closes[-1]) if closes else 0.0,
    }


def score_signal(features: Dict[str, float]) -> Tuple[float, str, float, List[str]]:
    """
    Returns: (score, signal, confidence, notes)
    score: roughly [-1, +1]
    """
    notes: List[str] = []

    mom = 0.45 * _clamp(features.get("mom_21", 0.0) * 5.0, -1.0, 1.0)
    trend = 0.30 * _clamp(features.get("trend_sma20_50", 0.0) * 10.0, -1.0, 1.0)

    # RSI: prefer 40-60; penalize extreme overbought/oversold
    rsi = features.get("rsi14", 50.0)
    rsi_centered = (rsi - 50.0) / 50.0  # [-1,+1]
    rsi_term = 0.10 * _clamp(rsi_centered, -1.0, 1.0)

    # Volatility penalty: high vol reduces confidence/score slightly
    vol = features.get("volatility_ann", 0.0)
    vol_penalty = 0.15 * _clamp((vol - 0.25) / 0.50, 0.0, 1.0)  # start penalizing above ~25% ann vol

    # Volume confirmation
    volz = features.get("volume_z20", 0.0)
    vol_confirm = 0.10 * _clamp(volz / 2.0, -1.0, 1.0)

    raw = mom + trend + rsi_term + vol_confirm - vol_penalty
    score = _clamp(raw, -1.0, 1.0)

    # Signal thresholds
    if score >= 0.25:
        signal = "BUY"
    elif score <= -0.25:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Confidence: how far from 0 plus volatility impact
    base_conf = abs(score)
    conf = _clamp(base_conf * (1.0 - _clamp(vol / 1.0, 0.0, 0.7)), 0.0, 1.0)

    # Notes
    if features.get("trend_sma20_50", 0.0) > 0:
        notes.append("Short-term trend (SMA20 vs SMA50) is positive.")
    else:
        notes.append("Short-term trend (SMA20 vs SMA50) is negative or flat.")

    if rsi >= 70:
        notes.append("RSI is high (potentially overbought).")
    elif rsi <= 30:
        notes.append("RSI is low (potentially oversold).")

    if vol >= 0.40:
        notes.append("Volatility is elevated; treat signal with caution.")

    if volz >= 1.0:
        notes.append("Volume is above recent average (confirmation).")
    elif volz <= -1.0:
        notes.append("Volume is below recent average (weak participation).")

    return float(score), signal, float(conf), notes


def fetch_lookback_window(symbol: str, as_of: str, lookback: int) -> List[Candle]:
    """
    Fetch enough calendar days to reliably contain `lookback` trading days.
    """
    if lookback < 20 or lookback > 2000:
        raise HTTPException(status_code=400, detail="lookback must be between 20 and 2000.")

    asof = _parse_date(as_of)

    # Calendar buffer: ~1.6x trading days + 30 days padding
    cal_days = int(lookback * 1.6) + 30
    from_d = asof - dt.timedelta(days=cal_days)

    candles = massive_fetch_daily_bars(symbol, _date_to_str(from_d), _date_to_str(asof))
    candles = _slice_as_of(candles, as_of)

    if len(candles) < lookback:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough data to compute signal. Needed {lookback} bars <= {as_of}, got {len(candles)}.",
        )

    return candles[-lookback:]


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
        "data": [_model_dump(c) for c in candles],
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
        # Path B: fetch from Massive (Polygon legacy)
        if not (req.symbol and req.from_date and req.to_date):
            raise HTTPException(
                status_code=400,
                detail="Provide either `data` OR (`symbol`, `from_date`, `to_date`) to fetch from Massive.",
            )
        candles = massive_fetch_daily_bars(req.symbol, req.from_date, req.to_date)

    summary = run_backtest_core(candles, req.include_equity_curve, req.include_returns)
    return {"backtest_summary": _model_dump(summary)}


@app.get("/signal")
def signal(
    symbol: str = Query(..., description="Ticker symbol (e.g. AAPL)"),
    as_of: str = Query(..., description="YYYY-MM-DD (signal computed using bars up to this date)"),
    lookback: int = Query(252, description="Trading-day lookback window (default 252)"),
) -> Dict[str, Any]:
    window = fetch_lookback_window(symbol, as_of, lookback)
    feats = compute_features(window)
    score, sig, conf, notes = score_signal(feats)

    out = SignalResponse(
        symbol=symbol.upper(),
        as_of=as_of,
        lookback=lookback,
        score=score,
        signal=sig,
        confidence=conf,
        features=feats,
        notes=notes,
    )
    return _model_dump(out)


@app.post("/run_scores")
def run_scores(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_scores endpoint placeholder", "received_keys": list(payload.keys())}


@app.post("/run_all")
def run_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_all endpoint placeholder", "received_keys": list(payload.keys())}

