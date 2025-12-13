from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections.abc import Mapping
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
    if v is None:
        return None

    if isinstance(v, (bool, int, str)):
        return v

    try:
        f = float(v)
    except (TypeError, ValueError):
        return v

    return f if math.isfinite(f) else None


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

    # IMPORTANT: treat any dict-like object as mapping (not only dict)
    if isinstance(obj, Mapping):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]

    return _clean_value(obj)


def _build_backtest_summary(
    backtest: Any,
    include_equity_curve: bool,
    include_returns: bool,
) -> Optional[Dict[str, Any]]:
    """
    Normalizes whatever FTIP returns into a stable summary.
    If equity/returns are not requested, we keep output small.
    """
    if backtest is None:
        return None

    # Accept dict-like objects
    if isinstance(backtest, Mapping):
        bt: Dict[str, Any] = dict(backtest)  # normalize to real dict
    else:
        # Unknown object — safest is to serialize raw
        return {"raw": _to_serializable(backtest)}

    summary: Dict[str, Any] = {}

    # pull common metrics directly if present
    for key in [
        "total_return",
        "annual_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "volatility",
        "num_trades",
        "win_rate",
    ]:
        if key in bt:
            summary[key] = _clean_value(bt.get(key))

    # nested stats support
    stats = bt.get("stats")
    if isinstance(stats, Mapping):
        for key in [
            "total_return",
            "annual_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "volatility",
            "num_trades",
            "win_rate",
        ]:
            if key in stats and key not in summary:
                summary[key] = _clean_value(stats.get(key))

    # heavy series only if requested
    if include_equity_curve and "equity_curve" in bt:
        summary["equity_curve"] = _to_serializable(bt.get("equity_curve"))
    if include_returns and "returns" in bt:
        summary["returns"] = _to_serializable(bt.get("returns"))

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
    # IMPORTANT: keep this list accurate (helps you sanity-check deployments)
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
    return {
        "railway_git_commit_sha": os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_COMMIT_SHA") or "unknown",
        "railway_environment": os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_ENV") or "unknown",
    }


def _df_from_points(points: List[DataPoint]) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump() for p in points])
    return df.set_index("timestamp")


@app.post("/run_all")
def run_all(req: RunAllRequest):
    df = _df_from_points(req.data)
    results = system.run_all(df)

    payload: Dict[str, Any] = {}

    for k in ["features", "labels", "scores", "structural_alpha", "superfactor"]:
        if k in results:
            payload[k] = _to_serializable(results[k])

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
    df = _df_from_points(req.data)
    results = system.run_all(df)

    out: Dict[str, Any] = {}
    if "scores" in results:
        out["scores"] = _to_serializable(results["scores"])
    if "features" in results:
        out["features"] = _to_serializable(results["features"])
    return out

