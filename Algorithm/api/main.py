from __future__ import annotations

import os
import math
import datetime as dt
from typing import Any, Dict, List, Optional

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


# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------

class Candle(BaseModel):
    timestamp: str  # ISO date "YYYY-MM-DD"
    close: float
    volume: Optional[float] = None

    # optional extra features (not used yet in signal core)
    fundamental: Optional[float] = None
    sentiment: Optional[float] = None
    crowd: Optional[float] = None


class BacktestRequest(BaseModel):
    # Either provide `data` OR provide (symbol + date range) to fetch from Polygon
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


class WalkForwardRequest(BaseModel):
    symbol: str
    from_date: str  # YYYY-MM-DD
    to_date: str    # YYYY-MM-DD
    lookback: int = 252
    horizons: List[int] = [5, 21, 63]  # trading bars
    # Optional evaluation window inside [from_date, to_date]
    eval_start: Optional[str] = None   # YYYY-MM-DD
    eval_end: Optional[str] = None     # YYYY-MM-DD


# ------------------------------------------------------------------------------
# Market data (Polygon; "Massive" naming kept for backward compatibility)
# ------------------------------------------------------------------------------

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
    Polygon Aggregates (Daily):
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
# Massive Flat Files (S3-compatible) config (NOT used yet)
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


def _sma(values: List[float], window: int) -> Optional[float]:
    if window <= 0 or len(values) < window:
        return None
    return _mean(values[-window:])


def _rsi(values: List[float], window: int = 14) -> Optional[float]:
    if len(values) < window + 1:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for i in range(len(values) - window, len(values)):
        chg = values[i] - values[i - 1]
        if chg >= 0:
            gains.append(chg)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-chg)
    avg_gain = _mean(gains)
    avg_loss = _mean(losses)
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
    return (w[-1] - mu) / sd


def _momentum(values: List[float], lookback: int) -> Optional[float]:
    if lookback <= 0 or len(values) < lookback + 1:
        return None
    prev = values[-(lookback + 1)]
    cur = values[-1]
    if prev == 0:
        return None
    return (cur / prev) - 1.0


def _annualized_vol(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return _std(returns) * math.sqrt(252.0)


# ------------------------------------------------------------------------------
# Backtest engine (buy-and-hold placeholder; still useful)
# ------------------------------------------------------------------------------

def run_backtest_core(candles: List[Candle], include_equity: bool, include_returns: bool) -> BacktestSummary:
    closes = [c.close for c in candles]
    dates = [c.timestamp for c in candles]

    rets = _pct_change(closes)
    equity = [1.0]
    for r in rets[1:]:
        equity.append(equity[-1] * (1.0 + r))

    total_return = equity[-1] - 1.0
    n = max(1, len(rets) - 1)
    ann_return = (1.0 + total_return) ** (252.0 / n) - 1.0 if n > 0 else 0.0

    vol = _annualized_vol(rets[1:]) if len(rets) > 2 else 0.0
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
# Signal engine (core)
# ------------------------------------------------------------------------------

def _detect_regime(
    closes: List[float],
    returns: List[float],
    vol_ann: float,
    trend_strength: Optional[float],
) -> str:
    if vol_ann >= 0.45:
        return "HIGH_VOL"
    if trend_strength is not None and abs(trend_strength) >= 0.05:
        return "TRENDING"
    return "CHOPPY"


def _thresholds_for_regime(regime: str) -> Dict[str, float]:
    if regime == "HIGH_VOL":
        return {"buy": 0.45, "sell": -0.45}
    if regime == "TRENDING":
        return {"buy": 0.20, "sell": -0.20}
    return {"buy": 0.30, "sell": -0.30}


def compute_signal_core(
    candles: List[Candle],
    as_of: str,
    lookback: int = 252,
) -> Dict[str, Any]:
    notes: List[str] = []

    cutoff = dt.date.fromisoformat(as_of)
    usable = [c for c in candles if dt.date.fromisoformat(c.timestamp) <= cutoff]
    usable.sort(key=lambda c: c.timestamp)

    if len(usable) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data to compute signal. Need at least ~30 bars <= {as_of}, got {len(usable)}.",
        )

    effective_lookback = min(lookback, len(usable))
    if effective_lookback < lookback:
        notes.append(
            f"Requested lookback={lookback}, but only {len(usable)} bars available by {as_of}. "
            f"Using effective_lookback={effective_lookback}."
        )

    window = usable[-effective_lookback:]
    closes = [c.close for c in window]
    vols = [float(c.volume) if c.volume is not None else float("nan") for c in window]
    vol_clean: List[float] = []
    for v in vols:
        vol_clean.append(0.0 if v != v else float(v))  # NaN -> 0

    rets = _pct_change(closes)
    rets_1 = rets[1:]

    mom_5 = _momentum(closes, 5) or 0.0
    mom_21 = _momentum(closes, 21) or 0.0
    mom_63 = _momentum(closes, 63) or 0.0

    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    trend_sma20_50 = 0.0
    if sma20 is not None and sma50 is not None and sma50 != 0:
        trend_sma20_50 = (sma20 / sma50) - 1.0

    vol_ann = _annualized_vol(rets_1)

    rsi14 = _rsi(closes, 14)
    rsi14_val = float(rsi14) if rsi14 is not None else 50.0

    volume_z20 = _zscore_last(vol_clean, 20)
    volume_z20_val = float(volume_z20) if volume_z20 is not None else 0.0

    last_close = float(closes[-1])

    regime = _detect_regime(closes, rets_1, vol_ann, trend_sma20_50)
    thresholds = _thresholds_for_regime(regime)

    rsi_component = (rsi14_val - 50.0) / 50.0
    volz_component = max(-2.0, min(2.0, volume_z20_val)) / 2.0

    score = (
        0.25 * mom_5 +
        0.30 * mom_21 +
        0.35 * mom_63 +
        0.20 * trend_sma20_50 +
        0.10 * rsi_component +
        0.05 * volz_component
    )
    score = max(-1.0, min(1.0, float(score)))

    confidence = abs(score)
    if regime == "HIGH_VOL":
        confidence *= 0.65
        notes.append("Confidence reduced due to HIGH_VOL regime.")

    signal = "HOLD"
    if score >= thresholds["buy"]:
        signal = "BUY"
    elif score <= thresholds["sell"]:
        signal = "SELL"

    if trend_sma20_50 > 0:
        notes.append("Short-term trend (SMA20 vs SMA50) is positive.")
    elif trend_sma20_50 < 0:
        notes.append("Short-term trend (SMA20 vs SMA50) is negative.")

    if vol_ann >= 0.45:
        notes.append("Volatility is elevated; treat signal with caution.")

    if regime == "TRENDING":
        notes.append("Regime: TRENDING (trend strength strong).")
    elif regime == "CHOPPY":
        notes.append("Regime: CHOPPY (trend not strong; signals are less reliable).")
    else:
        notes.append("Regime: HIGH_VOL (volatility high).")

    features = {
        "mom_5": float(mom_5),
        "mom_21": float(mom_21),
        "mom_63": float(mom_63),
        "trend_sma20_50": float(trend_sma20_50),
        "volatility_ann": float(vol_ann),
        "rsi14": float(rsi14_val),
        "volume_z20": float(volume_z20_val),
        "last_close": float(last_close),
    }

    return {
        "as_of": as_of,
        "lookback": int(lookback),
        "effective_lookback": int(effective_lookback),
        "regime": regime,
        "thresholds": thresholds,
        "score": float(score),
        "signal": signal,
        "confidence": float(confidence),
        "features": features,
        "notes": notes,
    }


# ------------------------------------------------------------------------------
# Walk-forward validation core
# ------------------------------------------------------------------------------

def walk_forward_core(
    candles: List[Candle],
    lookback: int,
    horizons: List[int],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    if not candles or len(candles) < 80:
        raise ValueError("Need more candles for walk-forward evaluation (recommend >= ~80).")

    if not horizons:
        raise ValueError("horizons must be non-empty, e.g. [5, 21, 63].")

    horizons = sorted(list(set(int(h) for h in horizons if int(h) > 0)))
    max_h = max(horizons)

    candles = sorted(candles, key=lambda c: c.timestamp)

    sd = dt.date.fromisoformat(start_date) if start_date else dt.date.fromisoformat(candles[0].timestamp)
    ed = dt.date.fromisoformat(end_date) if end_date else dt.date.fromisoformat(candles[-1].timestamp)

    dates = [dt.date.fromisoformat(c.timestamp) for c in candles]

    MIN_BARS_FOR_SIGNAL = 30
    rows: List[Dict[str, Any]] = []

    for i in range(len(candles)):
        day = dates[i]
        if day < sd or day > ed:
            continue

        if (i + 1) < MIN_BARS_FOR_SIGNAL:
            continue

        if i + max_h >= len(candles):
            continue

        as_of = candles[i].timestamp

        try:
            sig = compute_signal_core(candles[: i + 1], as_of=as_of, lookback=lookback)
        except HTTPException:
            continue

        base_close = candles[i].close
        fwd: Dict[str, float] = {}
        for h in horizons:
            fwd_ret = (candles[i + h].close / base_close) - 1.0 if base_close != 0 else 0.0
            fwd[f"fwd_ret_{h}"] = float(fwd_ret)

        rows.append({
            "date": as_of,
            "signal": sig["signal"],
            "score": float(sig["score"]),
            "confidence": float(sig["confidence"]),
            "regime": sig["regime"],
            **fwd,
        })

    if not rows:
        return {
            "rows_sample": [],
            "row_count": 0,
            "summary": {},
            "by_regime": {},
            "by_confidence_bucket": {},
            "meta": {
                "lookback": lookback,
                "horizons": horizons,
                "start_date": sd.isoformat(),
                "end_date": ed.isoformat(),
                "note": "No rows produced (date range too tight or not enough history/future).",
            },
        }

    def _bucket(conf: float) -> str:
        if conf >= 0.75:
            return "0.75-1.00"
        if conf >= 0.50:
            return "0.50-0.75"
        if conf >= 0.25:
            return "0.25-0.50"
        return "0.00-0.25"

    def _init_stat() -> Dict[str, Any]:
        return {
            "count": 0,
            "win_rate": {str(h): 0.0 for h in horizons},
            "avg_return": {str(h): 0.0 for h in horizons},
        }

    def _update(stat: Dict[str, Any], row: Dict[str, Any]) -> None:
        stat["count"] += 1
        for h in horizons:
            r = float(row[f"fwd_ret_{h}"])
            stat["avg_return"][str(h)] += r
            if r > 0:
                stat["win_rate"][str(h)] += 1.0

    def _finalize(stat: Dict[str, Any]) -> Dict[str, Any]:
        n = stat["count"]
        if n <= 0:
            return stat
        for h in horizons:
            stat["avg_return"][str(h)] /= n
            stat["win_rate"][str(h)] /= n
        return stat

    summary: Dict[str, Any] = {"BUY": _init_stat(), "HOLD": _init_stat(), "SELL": _init_stat()}
    by_regime: Dict[str, Any] = {}
    by_bucket: Dict[str, Any] = {}

    for row in rows:
        s = row["signal"]
        _update(summary[s], row)

        rg = row["regime"]
        if rg not in by_regime:
            by_regime[rg] = {"BUY": _init_stat(), "HOLD": _init_stat(), "SELL": _init_stat()}
        _update(by_regime[rg][s], row)

        b = _bucket(float(row["confidence"]))
        if b not in by_bucket:
            by_bucket[b] = {"BUY": _init_stat(), "HOLD": _init_stat(), "SELL": _init_stat()}
        _update(by_bucket[b][s], row)

    summary = {k: _finalize(v) for k, v in summary.items()}
    by_regime = {rg: {k: _finalize(v) for k, v in per.items()} for rg, per in by_regime.items()}
    by_bucket = {b: {k: _finalize(v) for k, v in per.items()} for b, per in by_bucket.items()}

    head_n = 10
    tail_n = 10
    sample_rows = rows[:head_n] + (rows[-tail_n:] if len(rows) > head_n else [])

    return {
        "rows_sample": sample_rows,
        "row_count": len(rows),
        "summary": summary,
        "by_regime": by_regime,
        "by_confidence_bucket": by_bucket,
        "meta": {
            "lookback": lookback,
            "horizons": horizons,
            "start_date": sd.isoformat(),
            "end_date": ed.isoformat(),
        },
    }


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
            "/walk_forward",
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
                detail="Provide either `data` OR (`symbol`, `from_date`, `to_date`) to fetch from Polygon.",
            )
        candles = massive_fetch_daily_bars(req.symbol, req.from_date, req.to_date)

    summary = run_backtest_core(candles, req.include_equity_curve, req.include_returns)
    return {"backtest_summary": summary.model_dump(exclude_none=True)}


@app.get("/signal")
def signal(
    symbol: str = Query(..., description="Ticker symbol (e.g. AAPL)"),
    as_of: str = Query(..., description="YYYY-MM-DD"),
    lookback: int = Query(252, ge=30, le=5000),
) -> Dict[str, Any]:
    as_of_d = dt.date.fromisoformat(as_of)
    from_d = as_of_d - dt.timedelta(days=max(lookback * 3, 365))
    candles = massive_fetch_daily_bars(symbol, from_d.isoformat(), as_of_d.isoformat())

    out = compute_signal_core(candles, as_of=as_of, lookback=lookback)
    out["symbol"] = symbol.upper().strip()
    return out


@app.post("/walk_forward")
def walk_forward(req: WalkForwardRequest) -> Dict[str, Any]:
    sym = req.symbol.upper().strip()

    if req.lookback < 30:
        raise HTTPException(status_code=400, detail="lookback must be >= 30")
    if not req.horizons:
        raise HTTPException(status_code=400, detail="horizons must be non-empty")
    if any(int(h) <= 0 for h in req.horizons):
        raise HTTPException(status_code=400, detail="all horizons must be positive integers")

    # Fetch exactly the requested range from Polygon
    candles = massive_fetch_daily_bars(sym, req.from_date, req.to_date)

    # Optional evaluation sub-window
    start = req.eval_start or req.from_date
    end = req.eval_end or req.to_date

    try:
        result = walk_forward_core(
            candles=candles,
            lookback=int(req.lookback),
            horizons=[int(h) for h in req.horizons],
            start_date=start,
            end_date=end,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result["symbol"] = sym
    return result


@app.post("/run_scores")
def run_scores(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_scores endpoint placeholder", "received_keys": list(payload.keys())}


@app.post("/run_all")
def run_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_all endpoint placeholder", "received_keys": list(payload.keys())}

