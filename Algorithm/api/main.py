from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ftip.system import FTIPSystem


# ----------------------------
# JSON-safety helpers
# ----------------------------
def _clean_value(v: Any) -> Any:
    """
    Make a single value JSON-safe:
    - Convert NaN / +inf / -inf to None
    - Leave normal numbers & strings as-is
    """
    if v is None:
        return None

    if isinstance(v, (bool, int, str)):
        return v

    try:
        f = float(v)
    except (TypeError, ValueError):
        return v

    if math.isfinite(f):
        return f
    return None


def _to_serializable(obj: Any) -> Any:
    """Convert pandas objects & nested structures to JSON-safe plain Python."""
    if isinstance(obj, pd.Series):
        return {
            "index": [str(i) for i in obj.index],
            "values": [_clean_value(v) for v in obj.to_list()],
        }

    if isinstance(obj, pd.DataFrame):
        data = []
        for row in obj.to_numpy().tolist():
            data.append([_clean_value(v) for v in row])
        return {
            "index": [str(i) for i in obj.index],
            "columns": [str(c) for c in obj.columns],
            "data": data,
        }

    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]

    return _clean_value(obj)


def _build_backtest_summary(backtest: Any, include_equity_curve: bool, include_returns: bool) -> Optional[Dict[str, Any]]:
    """
    Create a compact, stable summary block from whatever FTIP returns.
    Your FTIP backtest may be:
      - dict with 'equity_curve' and 'returns'
      - dict with nested stats
      - None
    We normalize into a consistent summary payload.
    """
    if backtest is None:
        return None

    if not isinstance(backtest, dict):
        # Unknown structure; serialize safely
        return {"raw": _to_serializable(backtest)}

    summary: Dict[str, Any] = {}

    # If direct keys exist
    eq = backtest.get("equity_curve")
    rets = backtest.get("returns")

    # Common stats keys we try to extract if present anywhere
    for key in ["total_return", "annual_return", "cagr", "sharpe", "max_drawdown", "volatility", "num_trades", "win_rate"]:
        if key in backtest:
            summary[key] = _clean_value(backtest.get(key))

    # nested stats dict support
    stats = backtest.get("stats")
    if isinstance(stats, dict):
        for key in ["total_return", "annual_return", "cagr", "sharpe", "max_drawdown", "volatility", "num_trades", "win_rate"]:
            if key in stats and key not in summary:
                summary[key] = _clean_value(stats.get(key))

    # include heavy series only if requested
    if include_equity_curve:
        summary["equity_curve"] = _to_serializable(eq) if eq is not None else None
    if include_returns:
        summary["returns"] = _to_serializable(rets) if rets is not None else None

    return summary


# ----------------------------
# Pydantic models
# ----------------------------
class DataPoint(BaseModel):
    timestamp: str
    close: float
    volume: Optional[float] = None
    fundamental: Optional[float] = None
    sentiment: Optional[float] = None
    crowd: Optional[float] = None


class RunAllRequest(BaseModel):
    data: List[DataPoint]
    # payload control flags (important for web)
    include_backtest: bool = Field(default=True)
    include_backtest_summary: bool = Field(default=True)
    include_equity_curve: bool = Field(default=False)  # big JSON, default off
    include_returns: bool = Field(default=False)       # big JSON, default off


class RunBacktestRequest(BaseModel):
    data: List[DataPoint]
    include_equity_curve: bool = Field(default=False)
    include_returns: bool = Field(default=False)


class RunScoresRequest(BaseModel):
    data: List[DataPoint]


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="FTIP System API")
system = FTIPSystem()


@app.get("/")
def root():
    return {
        "name": "FTIP System API",
        "status": "ok",
        "endpoints": ["/health", "/version", "/run_all", "/run_backtest", "/run_scores", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    # Railway sets RAILWAY_GIT_COMMIT_SHA automatically (or similar). We also fallback to env vars.
    return {
        "railway_git_commit_sha": os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_COMMIT_SHA") or "unknown",
        "railway_environment": os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_ENV") or "unknown",
    }


def _df_from_points(points: List[DataPoint]) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump() for p in points])
    df = df.set_index("timestamp")
    return df


@app.post("/run_all")
def run_all(req: RunAllRequest):
    df = _df_from_points(req.data)

    results = system.run_all(df)  # dict

    # Always return the core blocks if present
    payload: Dict[str, Any] = {}

    # Serialize core keys if they exist
    for k in ["features", "labels", "scores", "structural_alpha", "superfactor"]:
        if k in results:
            payload[k] = _to_serializable(results[k])

    # backtest handling
    backtest = results.get("backtest")

    if req.include_backtest:
        payload["backtest"] = _to_serializable(backtest)

    if req.include_backtest_summary:
        payload["backtest_summary"] = _build_backtest_summary(
            backtest,
            include_equity_curve=req.include_equity_curve,
            include_returns=req.include_returns,
        )

    return payload


@app.post("/run_backtest")
def run_backtest(req: RunBacktestRequest):
    """
    Focused endpoint: only backtest summary (optionally include equity_curve/returns).
    """
    df = _df_from_points(req.data)
    results = system.run_all(df)
    backtest = results.get("backtest")

    return {
        "backtest_summary": _build_backtest_summary(
            backtest,
            include_equity_curve=req.include_equity_curve,
            include_returns=req.include_returns,
        )
    }


@app.post("/run_scores")
def run_scores(req: RunScoresRequest):
    """
    Focused endpoint: only scores (smaller response, ideal for frontend calls).
    """
    df = _df_from_points(req.data)
    results = system.run_all(df)

    out: Dict[str, Any] = {}
    if "scores" in results:
        out["scores"] = _to_serializable(results["scores"])
    if "features" in results:
        out["features"] = _to_serializable(results["features"])
    return out

