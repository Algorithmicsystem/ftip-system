from typing import List, Optional, Dict, Any
import math
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ftip.system import FTIPSystem

# ----------------------------
# Helpers for JSON safety
# ----------------------------

def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, (bool, str)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_serializable(obj):
    if isinstance(obj, pd.Series):
        return {
            "index": [str(i) for i in obj.index],
            "values": [_clean_value(v) for v in obj.tolist()],
        }

    if isinstance(obj, pd.DataFrame):
        return {
            "index": [str(i) for i in obj.index],
            "columns": [str(c) for c in obj.columns],
            "data": [[_clean_value(v) for v in row] for row in obj.to_numpy()],
        }

    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]

    return _clean_value(obj)


# ----------------------------
# API models
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


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="FTIP System API")
system = FTIPSystem()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run_all")
def run_all(req: RunAllRequest):
    df = pd.DataFrame([row.model_dump() for row in req.data])
    df = df.set_index("timestamp")

    results = system.run_all(df)
    return {k: _to_serializable(v) for k, v in results.items()}

