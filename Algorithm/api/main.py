from typing import Any, Dict, List, Optional
import math

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ftip.system import FTIPSystem


# ---------- Helpers for cleaning / summarizing results ----------

def _sanitize_scalar(val: Any) -> Optional[float]:
    """Make sure floats are JSON-safe (no NaN/inf)."""
    if isinstance(val, (float, int)) and not isinstance(val, bool):
        if math.isfinite(val):
            return float(val)
        return None
    return None


def build_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a compact performance summary from the backtest block.

    This is defensive: if the expected keys aren't there, we just
    return an empty dict instead of crashing.
    """
    summary: Dict[str, Any] = {}

    backtest = result.get("backtest")
    if not isinstance(backtest, dict):
        return summary

    # If there's a nested stats dict, use that, otherwise use backtest itself.
    stats = backtest.get("stats", backtest)
    if not isinstance(stats, dict):
        return summary

    # Common metrics we care about
    for key in ["cagr", "sharpe", "max_drawdown", "volatility", "num_trades", "win_rate"]:
        if key in stats:
            summary[key] = _sanitize_scalar(stats[key])

    return summary


def _clean_value(v: Any) -> Any:
    """
    Make a single value JSON-safe:
    - Convert NaN / +inf / -inf to None
    - Leave normal numbers & strings as-is
    """
    if v is None:
        return None

    if isinstance(v, (str, bool)):
        return v

    # ints / floats
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        if math.isfinite(f):
            return f
        return None

    # Anything else (e.g. objects we don't want to try to cast)
    return v


def _to_serializable(obj: Any) -> Any:
    """Convert pandas objects (and nested structures) to plain Python types for JSON."""
    # Pandas Series -> {index, values}
    if isinstance(obj, pd.Series):
        return {
            "index": [str(i) for i in obj.index],
            "values": [_clean_value(v) for v in obj.to_list()],
        }

    # Pandas DataFrame -> {index, columns, data}
    if isinstance(obj, pd.DataFrame):
        data: List[List[Any]] = []
        for row in obj.to_numpy().tolist():
            data.append([_clean_value(v) for v in row])
        return {
            "index": [str(i) for i in obj.index],
            "columns": [str(c) for c in obj.columns],
            "data": data,
        }

    # Dicts: recurse into values
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    # Lists / tuples: recurse element-wise
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    # Fallback scalar
    return _clean_value(obj)


# ---------- Request models ----------

class DataPoint(BaseModel):
    timestamp: str
    close: float
    volume: Optional[float] = None
    fundamental: Optional[float] = None
    sentiment: Optional[float] = None
    crowd: Optional[float] = None


class RunAllRequest(BaseModel):
    data: List[DataPoint]


# ---------- FastAPI app + system ----------

app = FastAPI(title="FTIP System API")
system = FTIPSystem()


@app.get("/")
def root():
    """
    Simple root endpoint so Railway/health checks get 200 on '/'.
    """
    return {
        "status": "ok",
        "message": "FTIP System API is running. See /docs for interactive UI.",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run_all")
def run_all(req: RunAllRequest) -> Dict[str, Any]:
    # Build DataFrame from request
    df = pd.DataFrame([row.model_dump() for row in req.data])
    df = df.set_index("timestamp")

    # Run the full system pipeline
    raw_results: Dict[str, Any] = system.run_all(df)

    # Convert everything to JSON-safe types
    serializable = {k: _to_serializable(v) for k, v in raw_results.items()}

    # Add a compact performance summary (if backtest stats are present)
    serializable["summary"] = build_summary(raw_results)

    return serializable

