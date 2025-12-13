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
# Simple backtest engine (your existing logic can be plugged in here later)
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

def run_backtest_core(candles: List[Candle], include_equity: bool, include_returns: bool) -> BacktestSummary:
    # Placeholder "strategy": buy-and-hold on close-to-close returns
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
    return {"symbol": symbol.upper(), "from_date": from_date, "to_date": to_date, "data": [c.model_dump() for c in candles]}


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
    return {"backtest_summary": summary.model_dump(exclude_none=True)}


@app.post("/run_scores")
def run_scores(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_scores endpoint placeholder", "received_keys": list(payload.keys())}


@app.post("/run_all")
def run_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_all endpoint placeholder", "received_keys": list(payload.keys())}

