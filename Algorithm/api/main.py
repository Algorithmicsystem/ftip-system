from typing import Any, Dict, List, Optional
import math
import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ftip.system import FTIPSystem


# ---------------------------
# JSON-safety helpers
# ---------------------------
def _clean_value(v: Any):
    if v is None:
        return None
    if isinstance(v, (int, str, bool)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if math.isfinite(f) else None


def _to_serializable(obj: Any):
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

    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    return _clean_value(obj)


def _backtest_summary(backtest_obj: Any):
    if backtest_obj is None:
        return None

    if isinstance(backtest_obj, dict):
        stats = backtest_obj.get("stats", backtest_obj)
        if isinstance(stats, dict):
            keep: Dict[str, Any] = {}
            for k in ["cagr", "sharpe", "max_drawdown", "volatility", "num_trades", "win_rate"]:
                if k in stats:
                    keep[k] = _to_serializable(stats[k])
            return keep if keep else _to_serializable(backtest_obj)

        return _to_serializable(backtest_obj)

    return _to_serializable(backtest_obj)


# ---------------------------
# API models
# ---------------------------
class DataPoint(BaseModel):
    timestamp: str
    close: float
    volume: Optional[float] = None
    fundamental: Optional[float] = None
    sentiment: Optional[float] = None
    crowd: Optional[float] = None


class RunAllRequest(BaseModel):
    data: List[DataPoint]


# ---------------------------
# App + system
# ---------------------------
app = FastAPI(title="FTIP System API")
system = FTIPSystem()


@app.get("/")
def root():
    return {
        "name": "FTIP System API",
        "status": "ok",
        "endpoints": ["/health", "/version", "/run_all", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    # Railway typically provides this env var; if not, you still get “unknown”
    return {
        "railway_git_commit_sha": os.getenv("RAILWAY_GIT_COMMIT_SHA", "unknown"),
        "railway_environment": os.getenv("RAILWAY_ENVIRONMENT", "unknown"),
    }


@app.post("/run_all")
def run_all(req: RunAllRequest):
    df = pd.DataFrame([row.model_dump() for row in req.data]).set_index("timestamp")

    results = system.run_all(df)

    payload = {k: _to_serializable(v) for k, v in results.items()}
    payload["backtest_summary"] = _backtest_summary(results.get("backtest"))

    return payload

