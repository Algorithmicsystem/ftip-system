from __future__ import annotations

import os
import math
import json
import time
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    UniqueConstraint,
    Index,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker


# =============================================================================
# App
# =============================================================================

app = FastAPI(
    title="FTIP System API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Environment / Version
# =============================================================================

def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None else default


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "FTIP System API",
        "status": "ok",
        "endpoints": [
            "/health",
            "/version",
            "/run_all",
            "/run_backtest",
            "/run_scores",
            "/ingest_prices",
            "/prices",
            "/docs",
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/version")
def version() -> Dict[str, Any]:
    # Railway commonly exposes these env vars (depending on setup)
    sha = _env("RAILWAY_GIT_COMMIT_SHA", _env("GIT_COMMIT_SHA", "unknown"))
    env = _env("RAILWAY_ENVIRONMENT", _env("ENVIRONMENT", "unknown"))
    return {
        "railway_git_commit_sha": sha,
        "railway_environment": env,
    }


# =============================================================================
# Database (Postgres on Railway, SQLite locally)
# =============================================================================

DATABASE_URL = _env("DATABASE_URL", "").strip()

if DATABASE_URL:
    # Railway sometimes provides postgres:// which SQLAlchemy expects as postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    DATABASE_URL = "sqlite:///./ftip.db"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class PriceBar(Base):
    __tablename__ = "price_bars"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)  # UTC
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_symbol_timestamp"),
        Index("ix_symbol_timestamp", "symbol", "timestamp"),
    )


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    # Lightweight connectivity check (won't crash if db is slow)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        # Let the app still start; health checks + logs will show DB issues
        pass


# =============================================================================
# Alpaca Market Data (Daily bars)
# =============================================================================

ALPACA_KEY = _env("ALPACA_API_KEY", "").strip()
ALPACA_SECRET = _env("ALPACA_API_SECRET", "").strip()
ALPACA_DATA_BASE = _env("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").strip()


def _alpaca_headers() -> Dict[str, str]:
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Alpaca API keys not configured. Set ALPACA_API_KEY and ALPACA_API_SECRET.",
        )
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }


def _parse_dateish(s: str) -> datetime:
    """
    Accepts 'YYYY-MM-DD' or full ISO timestamps.
    Returns timezone-aware UTC datetime.
    """
    s = s.strip()
    # date only
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        d = date.fromisoformat(s)
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

    # ISO datetime
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_alpaca_daily_bars(symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """
    Fetches 1Day bars from Alpaca v2.
    Returns list of dict bars with keys: t,o,h,l,c,v (as Alpaca returns).
    """
    url = f"{ALPACA_DATA_BASE}/v2/stocks/bars"
    params = {
        "symbols": symbol.upper(),
        "timeframe": "1Day",
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "limit": 10000,
        "adjustment": "raw",
        "feed": "sip",  # if not entitled, Alpaca may fallback/deny. You can change to "iex" if needed.
    }

    r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=60)
    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Alpaca data error ({r.status_code}): {r.text}",
        )

    payload = r.json()
    bars_by_symbol = payload.get("bars", {})
    bars = bars_by_symbol.get(symbol.upper(), []) or []
    return bars


def upsert_bars(symbol: str, bars: List[Dict[str, Any]]) -> Dict[str, int]:
    inserted = 0
    updated = 0

    with SessionLocal() as db:
        for b in bars:
            # Alpaca bar timestamps are RFC3339 like '2024-01-02T00:00:00Z'
            ts = _parse_dateish(b["t"])
            close = float(b["c"])

            row = (
                db.query(PriceBar)
                .filter(PriceBar.symbol == symbol.upper(), PriceBar.timestamp == ts)
                .one_or_none()
            )

            if row is None:
                row = PriceBar(
                    symbol=symbol.upper(),
                    timestamp=ts,
                    open=float(b.get("o")) if b.get("o") is not None else None,
                    high=float(b.get("h")) if b.get("h") is not None else None,
                    low=float(b.get("l")) if b.get("l") is not None else None,
                    close=close,
                    volume=float(b.get("v")) if b.get("v") is not None else None,
                )
                db.add(row)
                inserted += 1
            else:
                row.open = float(b.get("o")) if b.get("o") is not None else row.open
                row.high = float(b.get("h")) if b.get("h") is not None else row.high
                row.low = float(b.get("l")) if b.get("l") is not None else row.low
                row.close = close
                row.volume = float(b.get("v")) if b.get("v") is not None else row.volume
                updated += 1

        db.commit()

    return {"inserted": inserted, "updated": updated}


def load_bars_from_db(symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    with SessionLocal() as db:
        rows = (
            db.query(PriceBar)
            .filter(
                PriceBar.symbol == symbol.upper(),
                PriceBar.timestamp >= start,
                PriceBar.timestamp <= end,
            )
            .order_by(PriceBar.timestamp.asc())
            .all()
        )

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "timestamp": r.timestamp.astimezone(timezone.utc).date().isoformat(),
                "close": float(r.close),
                "volume": float(r.volume) if r.volume is not None else None,
                "open": float(r.open) if r.open is not None else None,
                "high": float(r.high) if r.high is not None else None,
                "low": float(r.low) if r.low is not None else None,
            }
        )
    return out


# =============================================================================
# API Models
# =============================================================================

class PricePoint(BaseModel):
    timestamp: str
    close: float
    volume: Optional[float] = None
    fundamental: Optional[float] = None
    sentiment: Optional[float] = None
    crowd: Optional[float] = None


class IngestRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g., AAPL")
    start: str = Field(..., description="Start date/time (YYYY-MM-DD or ISO)")
    end: str = Field(..., description="End date/time (YYYY-MM-DD or ISO)")
    source: str = Field("alpaca", description="Data source (currently only alpaca)")
    timeframe: str = Field("1Day", description="Timeframe (currently only 1Day)")


class PricesResponse(BaseModel):
    symbol: str
    start: str
    end: str
    count: int
    data: List[Dict[str, Any]]


class BacktestRequest(BaseModel):
    # Option A: provide data directly (your current workflow)
    data: Optional[List[PricePoint]] = None

    # Option B: provide a symbol + date range and we will use stored DB data
    symbol: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None

    include_equity_curve: bool = False
    include_returns: bool = False


# =============================================================================
# Backtest Core (simple, deterministic, no ML yet)
# =============================================================================

def _compute_returns(closes: List[float]) -> List[float]:
    rets = [0.0]
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev == 0:
            rets.append(0.0)
        else:
            rets.append((cur / prev) - 1.0)
    return rets


def _compute_equity_curve(returns: List[float]) -> List[float]:
    eq = [1.0]
    for i in range(1, len(returns)):
        eq.append(eq[-1] * (1.0 + returns[i]))
    return eq


def _max_drawdown(equity: List[float]) -> float:
    peak = -float("inf")
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            dd = (peak - v) / peak
            mdd = max(mdd, dd)
    return float(mdd)


def _annualized_return(total_return: float, periods: int, periods_per_year: int = 252) -> float:
    if periods <= 0:
        return 0.0
    # total_return is (ending/starting - 1)
    ending = 1.0 + total_return
    if ending <= 0:
        return -1.0
    return float(ending ** (periods_per_year / periods) - 1.0)


def _annualized_volatility(returns: List[float], periods_per_year: int = 252) -> float:
    if len(returns) <= 2:
        return 0.0
    # ignore first 0 return
    xs = returns[1:]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return float(math.sqrt(var) * math.sqrt(periods_per_year))


def _sharpe_ratio(returns: List[float], periods_per_year: int = 252, rf: float = 0.0) -> float:
    if len(returns) <= 2:
        return 0.0
    xs = returns[1:]
    mean = sum(xs) / len(xs)
    # excess mean (rf assumed per-period)
    mean_excess = mean - (rf / periods_per_year)
    # sample std
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return float((mean_excess / std) * math.sqrt(periods_per_year))


def run_backtest_series(
    timestamps: List[str],
    closes: List[float],
    include_equity_curve: bool,
    include_returns: bool,
) -> Dict[str, Any]:
    rets = _compute_returns(closes)
    eq = _compute_equity_curve(rets)

    total_return = float(eq[-1] - 1.0)
    annual_return = _annualized_return(total_return, periods=max(1, len(closes) - 1))
    vol = _annualized_volatility(rets)
    sharpe = _sharpe_ratio(rets)
    mdd = _max_drawdown(eq)

    out: Dict[str, Any] = {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "volatility": vol,
    }

    if include_equity_curve:
        out["equity_curve"] = {timestamps[i]: float(eq[i]) for i in range(len(timestamps))}
    if include_returns:
        out["returns"] = {timestamps[i]: float(rets[i]) for i in range(len(timestamps))}

    return out


# =============================================================================
# Endpoints: ingestion + query
# =============================================================================

@app.post("/ingest_prices")
def ingest_prices(req: IngestRequest) -> Dict[str, Any]:
    if req.source.lower() != "alpaca":
        raise HTTPException(status_code=400, detail="Only source='alpaca' is supported right now.")
    if req.timeframe != "1Day":
        raise HTTPException(status_code=400, detail="Only timeframe='1Day' is supported right now.")

    start_dt = _parse_dateish(req.start)
    end_dt = _parse_dateish(req.end)
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end must be after start")

    bars = fetch_alpaca_daily_bars(req.symbol, start_dt, end_dt)
    stats = upsert_bars(req.symbol, bars)

    return {
        "status": "ok",
        "symbol": req.symbol.upper(),
        "start": start_dt.date().isoformat(),
        "end": end_dt.date().isoformat(),
        "fetched": len(bars),
        **stats,
    }


@app.get("/prices", response_model=PricesResponse)
def prices(
    symbol: str = Query(..., description="Ticker symbol, e.g., AAPL"),
    start: str = Query(..., description="Start date/time (YYYY-MM-DD or ISO)"),
    end: str = Query(..., description="End date/time (YYYY-MM-DD or ISO)"),
    auto_ingest_if_missing: bool = Query(True, description="If DB is missing data, fetch from Alpaca and store"),
) -> PricesResponse:
    start_dt = _parse_dateish(start)
    end_dt = _parse_dateish(end)
    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end must be after start")

    data = load_bars_from_db(symbol, start_dt, end_dt)

    # If missing and user allows, ingest and try again
    if auto_ingest_if_missing and len(data) == 0:
        bars = fetch_alpaca_daily_bars(symbol, start_dt, end_dt)
        upsert_bars(symbol, bars)
        data = load_bars_from_db(symbol, start_dt, end_dt)

    return PricesResponse(
        symbol=symbol.upper(),
        start=start_dt.date().isoformat(),
        end=end_dt.date().isoformat(),
        count=len(data),
        data=data,
    )


# =============================================================================
# Endpoints: backtest + scores + run_all
# =============================================================================

@app.post("/run_backtest")
def run_backtest(req: BacktestRequest) -> Dict[str, Any]:
    # Resolve data source:
    series: List[Dict[str, Any]] = []

    if req.data and len(req.data) > 0:
        # Use provided data directly
        # sort by timestamp just in case
        series = sorted([p.model_dump() for p in req.data], key=lambda x: x["timestamp"])
    else:
        # Try symbol mode
        if not req.symbol or not req.start or not req.end:
            raise HTTPException(
                status_code=400,
                detail="Provide either data=[...] OR (symbol, start, end).",
            )

        start_dt = _parse_dateish(req.start)
        end_dt = _parse_dateish(req.end)

        # Load from DB, auto-ingest if empty
        series = load_bars_from_db(req.symbol, start_dt, end_dt)
        if len(series) == 0:
            bars = fetch_alpaca_daily_bars(req.symbol, start_dt, end_dt)
            upsert_bars(req.symbol, bars)
            series = load_bars_from_db(req.symbol, start_dt, end_dt)

        if len(series) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {req.symbol.upper()} in the requested window.",
            )

    # Validate minimum length
    if len(series) < 2:
        raise HTTPException(status_code=400, detail="Not enough data points (need at least 2).")

    timestamps = [str(x["timestamp"]) for x in series]
    closes = [float(x["close"]) for x in series]

    summary = run_backtest_series(
        timestamps=timestamps,
        closes=closes,
        include_equity_curve=req.include_equity_curve,
        include_returns=req.include_returns,
    )

    return {"backtest_summary": summary}


@app.post("/run_scores")
def run_scores(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder for your scoring pipeline (features -> signals -> scores).
    Kept as an endpoint for compatibility.
    """
    # You can wire this to the “signal engine” in the next layer.
    return {"status": "ok", "message": "run_scores placeholder", "input_keys": sorted(payload.keys())}


@app.post("/run_all")
def run_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience endpoint to run the full pipeline.
    For now: runs backtest if payload matches BacktestRequest shape.
    """
    try:
        req = BacktestRequest(**payload)
        bt = run_backtest(req)
        return {"status": "ok", "backtest": bt["backtest_summary"]}
    except Exception as e:
        # Keep errors readable
        raise HTTPException(status_code=400, detail=f"run_all failed: {str(e)}")

