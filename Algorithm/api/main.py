from __future__ import annotations

import os
import math
import statistics
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field


# -----------------------------
# App + metadata
# -----------------------------

APP_NAME = "FTIP System API"

# Railway injects these at runtime (often present):
# - RAILWAY_GIT_COMMIT_SHA
# - RAILWAY_ENVIRONMENT
RAILWAY_GIT_SHA = os.getenv("RAILWAY_GIT_COMMIT_SHA", "unknown")
RAILWAY_ENV = os.getenv("RAILWAY_ENVIRONMENT", "unknown")


app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

PUBLIC_ENDPOINTS = [
    "/health",
    "/version",
    "/run_all",
    "/run_backtest",
    "/run_scores",
    "/docs",
]


# -----------------------------
# Models
# -----------------------------

class DataPoint(BaseModel):
    # ISO date or datetime string (e.g. "2024-01-01" or "2024-01-01T00:00:00")
    timestamp: str = Field(..., description="ISO date or datetime")
    close: float = Field(..., description="Close price")
    volume: Optional[float] = Field(None, description="Volume")
    fundamental: Optional[float] = Field(None, description="Fundamental signal")
    sentiment: Optional[float] = Field(None, description="Sentiment signal (-1..1)")
    crowd: Optional[float] = Field(None, description="Crowd signal (-1..1)")


class BacktestRequest(BaseModel):
    data: List[DataPoint]
    include_equity_curve: bool = False
    include_returns: bool = False


class ScoresRequest(BaseModel):
    data: List[DataPoint]


# -----------------------------
# Helpers
# -----------------------------

def _parse_iso_to_date(ts: str) -> str:
    """
    Normalize timestamp to YYYY-MM-DD string.
    Accepts date or datetime ISO strings.
    """
    try:
        # datetime.fromisoformat accepts YYYY-MM-DD too (gives midnight)
        dt = datetime.fromisoformat(ts)
        return dt.date().isoformat()
    except Exception:
        # last-resort: try strict date
        try:
            d = date.fromisoformat(ts)
            return d.isoformat()
        except Exception:
            # keep original if unparsable (but stable key)
            return ts


def _pct_change(series: List[float]) -> List[float]:
    if not series:
        return []
    out = [0.0]
    for i in range(1, len(series)):
        prev = series[i - 1]
        cur = series[i]
        if prev == 0:
            out.append(0.0)
        else:
            out.append((cur / prev) - 1.0)
    return out


def _max_drawdown(equity: List[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak != 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _safe_stdev(x: List[float]) -> float:
    if len(x) < 2:
        return 0.0
    try:
        return statistics.stdev(x)
    except Exception:
        return 0.0


def _safe_mean(x: List[float]) -> float:
    if not x:
        return 0.0
    try:
        return statistics.mean(x)
    except Exception:
        return 0.0


def _compute_backtest_summary(
    closes: List[float],
    dates: List[str],
    include_equity_curve: bool,
    include_returns: bool,
) -> Dict[str, Any]:
    """
    Simple baseline backtest:
    - daily returns derived from close series
    - equity curve = cumulative product of (1 + r)
    - metrics computed from daily returns (assume ~252 trading days)
    """
    rets = _pct_change(closes)
    equity = []
    eq = 1.0
    for r in rets:
        eq *= (1.0 + r)
        equity.append(eq)

    total_return = (equity[-1] - 1.0) if equity else 0.0

    # Annualization (geometric)
    n = max(len(rets) - 1, 1)  # exclude first 0.0 return for count
    years = n / 252.0
    if years <= 0:
        annual_return = 0.0
    else:
        annual_return = (equity[-1] ** (1.0 / years)) - 1.0 if equity else 0.0

    # Volatility + Sharpe (rf=0)
    # Use returns excluding first element (often 0.0)
    rets_eff = rets[1:] if len(rets) > 1 else []
    mu = _safe_mean(rets_eff)
    sigma = _safe_stdev(rets_eff)
    volatility = sigma * math.sqrt(252.0) if sigma > 0 else 0.0
    sharpe = (mu / sigma) * math.sqrt(252.0) if sigma > 0 else 0.0

    max_dd = _max_drawdown(equity)

    summary: Dict[str, Any] = {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "volatility": float(volatility),
    }

    if include_equity_curve:
        summary["equity_curve"] = {dates[i]: float(equity[i]) for i in range(min(len(dates), len(equity)))}

    if include_returns:
        summary["returns"] = {dates[i]: float(rets[i]) for i in range(min(len(dates), len(rets)))}

    return summary


def _compute_scores(points: List[DataPoint]) -> Dict[str, Any]:
    """
    Example scoring layer: combines (fundamental, sentiment, crowd) into a simple score.
    This keeps the endpoint stable; you can replace logic later without breaking the API.
    """
    if not points:
        return {"score": 0.0, "components": {"fundamental": 0.0, "sentiment": 0.0, "crowd": 0.0}}

    # Take latest non-null values
    latest = points[-1]
    f = float(latest.fundamental) if latest.fundamental is not None else 0.0
    s = float(latest.sentiment) if latest.sentiment is not None else 0.0
    c = float(latest.crowd) if latest.crowd is not None else 0.0

    # Normalize fundamental loosely (avoid explode)
    # If you have a defined scale later, replace this.
    f_norm = math.tanh(f / 10.0)  # keeps it in (-1..1)

    # Weighted sum
    score = 0.50 * f_norm + 0.30 * s + 0.20 * c

    return {
        "score": float(score),
        "components": {
            "fundamental": float(f_norm),
            "sentiment": float(s),
            "crowd": float(c),
        },
    }


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": APP_NAME,
        "status": "ok",
        "endpoints": PUBLIC_ENDPOINTS,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> Dict[str, str]:
    return {
        "railway_git_commit_sha": RAILWAY_GIT_SHA,
        "railway_environment": RAILWAY_ENV,
    }


@app.post("/run_scores")
def run_scores(payload: ScoresRequest) -> Dict[str, Any]:
    scores = _compute_scores(payload.data)
    return {"scores": scores}


@app.post("/run_backtest")
def run_backtest(payload: BacktestRequest) -> Dict[str, Any]:
    # Normalize dates + close series
    dates = [_parse_iso_to_date(p.timestamp) for p in payload.data]
    closes = [float(p.close) for p in payload.data]

    summary = _compute_backtest_summary(
        closes=closes,
        dates=dates,
        include_equity_curve=payload.include_equity_curve,
        include_returns=payload.include_returns,
    )

    # IMPORTANT: no "raw" wrapper, and series are only included if flags are True
    return {"backtest_summary": summary}


@app.post("/run_all")
def run_all(payload: BacktestRequest) -> Dict[str, Any]:
    # Reuse the same request structure for convenience
    dates = [_parse_iso_to_date(p.timestamp) for p in payload.data]
    closes = [float(p.close) for p in payload.data]

    scores = _compute_scores(payload.data)
    summary = _compute_backtest_summary(
        closes=closes,
        dates=dates,
        include_equity_curve=payload.include_equity_curve,
        include_returns=payload.include_returns,
    )

    return {
        "scores": scores,
        "backtest_summary": summary,
    }

