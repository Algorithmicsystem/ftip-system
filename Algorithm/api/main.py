from typing import List, Optional
import math

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ftip.system import FTIPSystem


class DataPoint(BaseModel):
    timestamp: str
    close: float
    volume: Optional[float] = None
    fundamental: Optional[float] = None
    sentiment: Optional[float] = None
    crowd: Optional[float] = None


class RunAllRequest(BaseModel):
    data: List[DataPoint]


app = FastAPI(title="FTIP System API")
system = FTIPSystem()


@app.get("/health")
def health():
    return {"status": "ok"}


def _clean_value(v):
    """
    Make a single value JSON-safe:
    - Convert NaN / +inf / -inf to None
    - Leave normal numbers & strings as-is
    """
    if v is None:
        return None

    # bool is also an int subclass, handle separately if needed
    if isinstance(v, (int, str, bool)):
        return v

    # Try to treat as float
    try:
        f = float(v)
    except (TypeError, ValueError):
        # Not a number, just return as-is
        return v

    if math.isfinite(f):
        return f
    # NaN, +inf, -inf → None
    return None


def _to_serializable(obj):
    """Convert pandas objects to plain Python structures for JSON."""
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
    # Fallback for plain dicts, lists, scalars, etc.
    if isinstance(obj, dict):
        return {k: _clean_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_value(v) for v in obj]
    return _clean_value(obj)


@app.post("/run_all")
def run_all(req: RunAllRequest):
    # Build DataFrame from request
    df = pd.DataFrame([row.model_dump() for row in req.data])
    df = df.set_index("timestamp")

    # Run the full system pipeline
    results = system.run_all(df)  # dict of Series/DataFrames/etc.

    # Convert everything to JSON-safe types
    serializable = {k: _to_serializable(v) for k, v in results.items()}
    return serializable

