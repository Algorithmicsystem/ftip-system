from __future__ import annotations

import os
import time
import math
import json
import datetime as dt
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from psycopg.types.json import Json

from api import db, lifecycle, security
from api.db import DBError
from api.assistant.routes import router as assistant_router
from api.backtest.routes import router as backtest_router
from api.llm.routes import router as llm_router
from api.prosperity.routes import router as prosperity_router
from api.narrator.routes import router as narrator_router
from api.ops import metrics_tracker, router as ops_router
from api.jobs.prosperity import router as prosperity_jobs_router
from api.jobs.market_data import router as market_data_jobs_router
from api.jobs.features import router as features_jobs_router
from api.jobs.signals import router as signals_jobs_router
from api.signals.routes import router as signals_router

# =============================================================================
# App + environment helpers
# =============================================================================

APP_NAME = "FTIP System API"
logger = logging.getLogger("ftip.api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


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


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =============================================================================
# Phase 4: Prosperity DB (Railway Postgres) - SAFE feature-flagged persistence
# =============================================================================

# NOTE: To enable DB, set in Railway Variables:
#   DATABASE_URL=postgresql://...
#   FTIP_DB_ENABLED=1
# Optional:
#   FTIP_DB_REQUIRED=1   (fail startup if DB cannot init)

DB_ENABLED = db.db_enabled()
DB_REQUIRED = (_env("FTIP_DB_REQUIRED", "0") or "0") == "1"
RATE_LIMIT_RPM = _env_int("FTIP_RATE_LIMIT_RPM", 60)
rate_limiter = security.RateLimiter(RATE_LIMIT_RPM)


def _db_pool_ready() -> bool:
    if not DB_ENABLED:
        return False
    try:
        db.get_pool()
        return True
    except Exception:
        return False


def db_insert_signal_run(as_of: str, lookback: int, score_mode: str) -> Optional[str]:
    try:
        row = db.exec1(
            """
            INSERT INTO signal_run (as_of_date, lookback, score_mode, version_sha, env)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                as_of,
                int(lookback),
                score_mode,
                _git_sha(),
                json.dumps({"railway_environment": _railway_env()}),
            ),
        )
        if row:
            return str(row[0])
    except Exception:
        return None
    return None


def db_insert_signal_observation(run_id: str, obs: Dict[str, Any]) -> None:
    try:
        db.exec1(
            """
            INSERT INTO signal_observation
              (run_id, symbol, regime, signal, score, confidence, thresholds, features, notes, calibration_meta)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
            ON CONFLICT (run_id, symbol)
            DO UPDATE SET
              regime=EXCLUDED.regime,
              signal=EXCLUDED.signal,
              score=EXCLUDED.score,
              confidence=EXCLUDED.confidence,
              thresholds=EXCLUDED.thresholds,
              features=EXCLUDED.features,
              notes=EXCLUDED.notes,
              calibration_meta=EXCLUDED.calibration_meta
            """,
            (
                run_id,
                (obs.get("symbol") or "").strip().upper(),
                obs.get("regime") or "UNKNOWN",
                obs.get("signal") or "HOLD",
                float(obs.get("score") or 0.0),
                float(obs.get("confidence") or 0.0),
                json.dumps(obs.get("thresholds") or {}),
                json.dumps(obs.get("features") or {}),
                json.dumps(obs.get("notes") or []),
                json.dumps(obs.get("calibration_meta")),
            ),
        )
    except Exception:
        return


def db_insert_portfolio_backtest(req_obj: Any, out_obj: Any) -> None:
    """
    Persist portfolio backtest summary + audit.
    Never raises.
    """
    payload = (
        out_obj.model_dump(exclude_none=True)
        if hasattr(out_obj, "model_dump")
        else dict(out_obj)
    )
    audit = payload.get("audit")

    try:
        db.exec1(
            """
            INSERT INTO portfolio_backtest_run
              (from_date, to_date, lookback, rebalance_every, trading_cost_bps, slippage_bps,
               max_weight, min_trade_delta, max_turnover_per_rebalance, allow_shorts,
               score_mode, symbols, result, audit, version_sha)
            VALUES
              (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s)
            """,
            (
                req_obj.from_date,
                req_obj.to_date,
                int(req_obj.lookback),
                int(req_obj.rebalance_every),
                float(req_obj.trading_cost_bps),
                float(req_obj.slippage_bps),
                float(req_obj.max_weight) if req_obj.max_weight is not None else None,
                float(req_obj.min_trade_delta),
                float(req_obj.max_turnover_per_rebalance),
                bool(req_obj.allow_shorts),
                _score_mode(),
                json.dumps(req_obj.symbols),
                json.dumps(payload),
                json.dumps(audit) if audit is not None else None,
                _git_sha(),
            ),
        )
    except Exception:
        return


def db_insert_calibration_snapshot(symbol: str, payload: Dict[str, Any]) -> None:
    try:
        db.exec1(
            """
            INSERT INTO calibration_snapshot (symbol, payload, source, version_sha)
            VALUES (%s, %s::jsonb, %s, %s)
            """,
            (
                (symbol or "").strip().upper(),
                json.dumps(payload),
                "api_calibrate",
                _git_sha(),
            ),
        )
    except Exception:
        return


SIGNAL_UPSERT_SQL = """
INSERT INTO signals (
  symbol, as_of, lookback, regime, score, signal, confidence, thresholds,
  features, notes, score_mode, base_score, stacked_score, stacked_meta,
  calibration_loaded, calibration_meta, raw_signal_payload
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (symbol, as_of, lookback, score_mode)
DO UPDATE SET
  regime=EXCLUDED.regime,
  score=EXCLUDED.score,
  signal=EXCLUDED.signal,
  confidence=EXCLUDED.confidence,
  thresholds=EXCLUDED.thresholds,
  features=EXCLUDED.features,
  notes=EXCLUDED.notes,
  score_mode=EXCLUDED.score_mode,
  base_score=EXCLUDED.base_score,
  stacked_score=EXCLUDED.stacked_score,
  stacked_meta=EXCLUDED.stacked_meta,
  calibration_loaded=EXCLUDED.calibration_loaded,
  calibration_meta=EXCLUDED.calibration_meta,
  raw_signal_payload=EXCLUDED.raw_signal_payload
RETURNING id, (xmax = 0) AS inserted
"""


def _signal_upsert_params(sig: SignalResponse) -> Tuple[Any, ...]:
    try:
        as_of_date = dt.date.fromisoformat(sig.as_of)
    except Exception:
        as_of_date = sig.as_of

    raw_payload = sig.model_dump(exclude_none=False)

    return (
        (sig.symbol or "").strip().upper(),
        as_of_date,
        int(sig.lookback),
        sig.regime,
        float(sig.score),
        sig.signal,
        float(sig.confidence),
        Json(sig.thresholds or {}),
        Json(sig.features or {}),
        Json(sig.notes or []),
        sig.score_mode,
        sig.base_score,
        sig.stacked_score,
        Json(sig.stacked_meta) if sig.stacked_meta is not None else None,
        sig.calibration_loaded,
        Json(sig.calibration_meta) if sig.calibration_meta is not None else None,
        Json(raw_payload),
    )


def persist_signal_record(sig: SignalResponse) -> Tuple[bool, Optional[int]]:
    row = db.exec1(SIGNAL_UPSERT_SQL, _signal_upsert_params(sig))
    if not row:
        return False, None
    inserted = bool(row[1]) if len(row) > 1 else False
    return inserted, int(row[0])


def persist_portfolio_backtest_record(req_obj: Any, out_obj: Any) -> Optional[int]:
    audit = getattr(out_obj, "audit", None)
    equity_curve = getattr(out_obj, "equity_curve", None)

    row = db.exec1(
        """
        INSERT INTO portfolio_backtests (
            symbols, from_date, to_date, lookback, rebalance_every, trading_cost_bps,
            slippage_bps, max_weight, min_trade_delta, max_turnover_per_rebalance,
            allow_shorts, total_return, annual_return, sharpe, max_drawdown, volatility,
            turnover, audit, equity_curve
        )
        VALUES
        (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id
        """,
        (
            Json(req_obj.symbols),
            req_obj.from_date,
            req_obj.to_date,
            int(req_obj.lookback),
            int(req_obj.rebalance_every),
            float(req_obj.trading_cost_bps),
            float(req_obj.slippage_bps),
            float(req_obj.max_weight) if req_obj.max_weight is not None else None,
            float(req_obj.min_trade_delta),
            float(req_obj.max_turnover_per_rebalance),
            bool(req_obj.allow_shorts),
            float(out_obj.total_return),
            float(out_obj.annual_return),
            float(out_obj.sharpe),
            float(out_obj.max_drawdown),
            float(out_obj.volatility),
            float(out_obj.turnover),
            Json(audit) if audit is not None else None,
            Json(equity_curve) if equity_curve is not None else None,
        ),
    )
    return int(row[0]) if row else None


# =============================================================================
# Calibration loader (single + per-symbol map)
# =============================================================================


def _load_calibration_for_symbol(
    symbol: Optional[str],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Loads calibration thresholds.
    Priority:
      1) FTIP_CALIBRATION_JSON_MAP (per-symbol)
      2) FTIP_CALIBRATION_JSON (single fallback)
    """
    sym = (symbol or "").strip().upper()

    raw_map = _env("FTIP_CALIBRATION_JSON_MAP")
    if raw_map:
        try:
            m = json.loads(raw_map)
            if isinstance(m, dict):
                if sym and sym in m and isinstance(m[sym], dict):
                    return True, m[sym]
                if "DEFAULT" in m and isinstance(m["DEFAULT"], dict):
                    return True, m["DEFAULT"]
        except Exception:
            pass  # invalid JSON -> ignore

    raw_one = _env("FTIP_CALIBRATION_JSON")
    if raw_one:
        try:
            cal = json.loads(raw_one)
            if isinstance(cal, dict):
                return True, cal
        except Exception:
            pass

    return False, None


def _load_calibration_single() -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Legacy single calibration loader (kept for backwards compatibility)."""
    raw = _env("FTIP_CALIBRATION_JSON")
    if not raw:
        return False, None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return True, obj
    except Exception:
        pass
    return False, None


# =============================================================================
# Models
# =============================================================================


class Candle(BaseModel):
    timestamp: str  # ISO date "YYYY-MM-DD" (daily)
    close: float
    volume: Optional[float] = None

    # optional extra features (supported for future use)
    fundamental: Optional[float] = None
    sentiment: Optional[float] = None
    crowd: Optional[float] = None


class BacktestRequest(BaseModel):
    # Either provide `data` OR provide (symbol + date range) to fetch from market data
    data: Optional[List[Candle]] = None

    symbol: Optional[str] = None
    from_date: Optional[str] = None  # YYYY-MM-DD
    to_date: Optional[str] = None  # YYYY-MM-DD

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


class SignalResponse(BaseModel):
    symbol: str
    as_of: str
    lookback: int
    effective_lookback: int
    regime: str
    thresholds: Dict[str, float]
    score: float
    signal: str
    confidence: float
    features: Dict[str, float]
    notes: List[str] = Field(default_factory=list)

    # scoring diagnostics
    score_mode: str = "stacked"  # "base" or "stacked"
    base_score: Optional[float] = None
    stacked_score: Optional[float] = None
    stacked_meta: Optional[Dict[str, Any]] = None

    # calibration diagnostics
    calibration_loaded: bool = False
    calibration_meta: Optional[Dict[str, Any]] = None


class SignalsRequest(BaseModel):
    symbols: List[str]
    as_of: str
    lookback: int = 252


class SaveSignalRequest(BaseModel):
    symbol: str
    as_of: str
    lookback: int = 252


class SaveSignalsRequest(BaseModel):
    symbols: List[str]
    as_of: str
    lookback: int = 252


class RunSnapshotRequest(BaseModel):
    as_of: str
    lookback: int = Field(252, ge=30, le=2000)
    active_only: bool = True
    limit: int = Field(1000, ge=1, le=5000)


class WalkForwardRequest(BaseModel):
    symbol: str
    from_date: str
    to_date: str
    lookback: int = 252
    horizons: List[int] = Field(default_factory=lambda: [5, 21, 63])


class CalibrateRequest(BaseModel):
    symbol: str
    from_date: str
    to_date: str
    lookback: int = 252
    horizons: List[int] = Field(default_factory=lambda: [5, 21, 63])
    optimize_horizon: int = 21
    min_trades_per_side: int = 15


class PortfolioRequest(BaseModel):
    symbols: List[str]
    as_of: str
    lookback: int = 252

    max_weight: Optional[float] = None
    hold_multiplier: Optional[float] = None
    min_confidence: Optional[float] = None
    allow_shorts: bool = False


class PortfolioBacktestRequest(BaseModel):
    symbols: List[str]
    from_date: str
    to_date: str
    lookback: int = 252

    # Rebalance + costs
    rebalance_every: int = 5
    trading_cost_bps: float = 5.0
    slippage_bps: float = 0.0

    # Portfolio knobs
    max_weight: Optional[float] = None
    hold_multiplier: Optional[float] = None
    min_confidence: Optional[float] = None
    allow_shorts: bool = False

    include_equity_curve: bool = True

    # Turnover controls
    min_trade_delta: float = 0.01
    max_turnover_per_rebalance: float = 0.20

    # Audit controls
    audit_no_lookahead: bool = False
    audit_max_rows: int = 50


class PortfolioBacktestResponse(BaseModel):
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    volatility: float
    turnover: float

    equity_curve: Optional[Dict[str, float]] = None
    audit: Optional[Dict[str, Any]] = None


# --- Phase 4: Universe models (Top 1,000, etc.) ---
class UniverseUpsertRequest(BaseModel):
    as_of_date: str
    name: str = "TOP1000_US"
    source: str = "manual_seed"
    symbols: List[str]  # ordered best -> worst


class UniverseDbUpsertRequest(BaseModel):
    symbols: List[str]
    source: Optional[str] = "manual"


# =============================================================================
# Simple in-memory candle cache (reduces 429 + speeds backtests/calibration)
# =============================================================================

_CANDLE_CACHE: Dict[Tuple[str, str, str], Tuple[float, List[Candle]]] = {}
_CANDLE_CACHE_TTL_SEC = 20 * 60  # 20 minutes


def _cache_get(key: Tuple[str, str, str]) -> Optional[List[Candle]]:
    now = time.time()
    item = _CANDLE_CACHE.get(key)
    if not item:
        return None
    ts, val = item
    if (now - ts) > _CANDLE_CACHE_TTL_SEC:
        _CANDLE_CACHE.pop(key, None)
        return None
    return val


def _cache_set(key: Tuple[str, str, str], val: List[Candle]) -> None:
    _CANDLE_CACHE[key] = (time.time(), val)


# =============================================================================
# Market data client (Massive / Polygon legacy)
# =============================================================================

MARKETDATA_BASE_URL = (
    _env("MASSIVE_BASE_URL") or _env("POLYGON_BASE_URL") or "https://api.polygon.io"
)

MARKETDATA_API_KEY = _env("MASSIVE_API_KEY") or _env("POLYGON_API_KEY")


def _require_marketdata_key() -> str:
    if not MARKETDATA_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="MASSIVE_API_KEY (or POLYGON_API_KEY) is not set on the server. Add it in Railway Variables.",
        )
    return MARKETDATA_API_KEY


def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def massive_fetch_daily_bars(symbol: str, from_date: str, to_date: str) -> List[Candle]:
    api_key = _require_marketdata_key()
    sym = (symbol or "").strip().upper()

    url = (
        f"{MARKETDATA_BASE_URL}/v2/aggs/ticker/{sym}/range/1/day/{from_date}/{to_date}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    max_retries = _env_int("FTIP_MD_MAX_RETRIES", 4)
    base_sleep = _env_float("FTIP_MD_RETRY_BASE_SLEEP", 2.0)

    last_err: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
        except Exception as e:
            last_err = f"Market data request failed: {e}"
            time.sleep(base_sleep * (2**attempt))
            continue

        if resp.status_code == 429:
            last_err = f"Market data rate-limited 429: {resp.text}"
            time.sleep(base_sleep * (2**attempt))
            continue

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Market data error {resp.status_code}: {resp.text}",
            )

        js = resp.json()
        results = js.get("results") or []
        if not results:
            raise HTTPException(
                status_code=404, detail=f"No bars returned for {sym} in range."
            )

        candles: List[Candle] = []
        for r in results:
            t_ms = r.get("t")
            c = r.get("c")
            v = r.get("v")
            if t_ms is None or c is None:
                continue
            day = dt.datetime.utcfromtimestamp(t_ms / 1000.0).date().isoformat()
            candles.append(
                Candle(
                    timestamp=day,
                    close=float(c),
                    volume=float(v) if v is not None else None,
                )
            )

        if not candles:
            raise HTTPException(
                status_code=404, detail=f"Returned empty/invalid candles for {sym}."
            )
        return candles

    raise HTTPException(
        status_code=502, detail=last_err or "Market data failed after retries."
    )


def massive_fetch_daily_bars_cached(
    symbol: str, from_date: str, to_date: str
) -> List[Candle]:
    s = (symbol or "").strip().upper()
    key = (s, from_date, to_date)
    hit = _cache_get(key)
    if hit is not None:
        return hit

    # Prefer DB cache when enabled
    try:
        if db.db_enabled() and db.db_read_enabled():
            import datetime as _dt
            from api.prosperity import ingest as prosperity_ingest

            start_d = _dt.date.fromisoformat(from_date)
            end_d = _dt.date.fromisoformat(to_date)
            rows = db.safe_fetchall(
                """
                SELECT date, close, volume FROM prosperity_daily_bars
                WHERE symbol=%s AND date BETWEEN %s AND %s ORDER BY date ASC
                """,
                (s, start_d, end_d),
            )
            have_dates = {r[0] for r in rows}
            missing = []
            cur = start_d
            while cur <= end_d:
                if cur not in have_dates:
                    missing.append(cur)
                cur += _dt.timedelta(days=1)

            if missing and db.db_write_enabled():
                prosperity_ingest.ingest_bars(s, start_d, end_d)
                rows = db.safe_fetchall(
                    "SELECT date, close, volume FROM prosperity_daily_bars WHERE symbol=%s AND date BETWEEN %s AND %s ORDER BY date ASC",
                    (s, start_d, end_d),
                )

            if rows:
                bars = [
                    Candle(
                        timestamp=r[0].isoformat(),
                        close=float(r[1]),
                        volume=float(r[2]) if r[2] is not None else None,
                    )
                    for r in rows
                ]
                _cache_set(key, bars)
                return bars
    except Exception:
        pass

    bars = massive_fetch_daily_bars(s, from_date, to_date)
    _cache_set(key, bars)
    return bars


# =============================================================================
# Math helpers
# =============================================================================


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


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


def _max_drawdown(equity: List[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak != 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


# =============================================================================
# Features + regime detection + scoring
# =============================================================================


def _sma(values: List[float], window: int) -> float:
    if window <= 0 or len(values) < window:
        return float("nan")
    return _mean(values[-window:])


def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains: List[float] = []
    losses: List[float] = []
    for i in range(len(closes) - period, len(closes)):
        chg = closes[i] - closes[i - 1]
        if chg >= 0:
            gains.append(chg)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-chg)
    avg_gain = _mean(gains)
    avg_loss = _mean(losses)
    if avg_loss == 0:
        return 70.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(_clamp(rsi, 0.0, 100.0))


def _zscore_last(values: List[float], window: int) -> float:
    if len(values) < window or window < 2:
        return 0.0
    w = values[-window:]
    mu = _mean(w)
    sd = _std(w)
    if sd == 0:
        return 0.0
    return float((w[-1] - mu) / sd)


def compute_features(candles: List[Candle]) -> Dict[str, float]:
    closes = [c.close for c in candles]
    vols = [float(c.volume or 0.0) for c in candles]

    def mom(k: int) -> float:
        if len(closes) < k + 1:
            return 0.0
        return float(closes[-1] / closes[-1 - k] - 1.0)

    r = _pct_change(closes)
    vol_ann = _std(r[1:]) * math.sqrt(252.0) if len(r) > 2 else 0.0

    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    trend = 0.0
    if not (math.isnan(sma20) or math.isnan(sma50)) and sma50 != 0:
        trend = float((sma20 / sma50) - 1.0)

    return {
        "mom_5": mom(5),
        "mom_21": mom(21),
        "mom_63": mom(63),
        "trend_sma20_50": trend,
        "volatility_ann": float(vol_ann),
        "rsi14": float(_rsi(closes, 14)),
        "volume_z20": float(_zscore_last(vols, 20)),
        "last_close": float(closes[-1]),
    }


def detect_regime(features: Dict[str, float]) -> str:
    vol = float(features.get("volatility_ann", 0.0))
    trend = abs(float(features.get("trend_sma20_50", 0.0)))

    if vol >= 0.45:
        return "HIGH_VOL"
    if trend >= 0.05:
        return "TRENDING"
    return "CHOPPY"


def score_from_features(features: Dict[str, float]) -> Tuple[float, List[str]]:
    notes: List[str] = []

    rsi = float(features.get("rsi14", 50.0))
    rsi_sig = _clamp((rsi - 50.0) / 25.0, -1.0, 1.0)

    mom = 0.4 * float(features.get("mom_21", 0.0)) + 0.6 * float(
        features.get("mom_63", 0.0)
    )
    mom_sig = _clamp(mom / 0.25, -1.0, 1.0)

    trend = float(features.get("trend_sma20_50", 0.0))
    trend_sig = _clamp(trend / 0.10, -1.0, 1.0)
    if trend > 0:
        notes.append("Short-term trend (SMA20 vs SMA50) is positive.")
    elif trend < 0:
        notes.append("Short-term trend (SMA20 vs SMA50) is negative.")

    volz = float(features.get("volume_z20", 0.0))
    vol_sig = _clamp(volz / 3.0, -1.0, 1.0)

    vola = float(features.get("volatility_ann", 0.0))
    vola_pen = _clamp((vola - 0.25) / 0.50, 0.0, 0.5)

    raw = 0.45 * mom_sig + 0.30 * trend_sig + 0.20 * rsi_sig + 0.05 * vol_sig
    score = _clamp(raw * (1.0 - vola_pen), -1.0, 1.0)

    if vola >= 0.45:
        notes.append("Volatility is elevated; treat signal with caution.")
    return float(score), notes


def _env_csv_floats(name: str, default: List[float]) -> List[float]:
    raw = _env(name)
    if not raw:
        return default
    try:
        parts = [p.strip() for p in raw.split(",")]
        vals = [float(p) for p in parts if p != ""]
        if len(vals) != len(default):
            return default
        return vals
    except Exception:
        return default


def _stack_weights_for_regime(regime: str) -> Dict[str, float]:
    w_tr = _env_csv_floats("FTIP_STACK_W_TRENDING", [0.15, 0.35, 0.50])
    w_ch = _env_csv_floats("FTIP_STACK_W_CHOPPY", [0.45, 0.35, 0.20])
    w_hv = _env_csv_floats("FTIP_STACK_W_HIGH_VOL", [0.20, 0.40, 0.40])

    if regime == "TRENDING":
        w = w_tr
    elif regime == "HIGH_VOL":
        w = w_hv
    else:
        w = w_ch

    s = sum(w) if sum(w) != 0 else 1.0
    return {"short": float(w[0] / s), "mid": float(w[1] / s), "long": float(w[2] / s)}


def _score_components_from_features(features: Dict[str, float]) -> Dict[str, float]:
    rsi = float(features.get("rsi14", 50.0))
    rsi_sig = _clamp((rsi - 50.0) / 25.0, -1.0, 1.0)

    mom5 = float(features.get("mom_5", 0.0))
    mom21 = float(features.get("mom_21", 0.0))
    mom63 = float(features.get("mom_63", 0.0))
    trend = float(features.get("trend_sma20_50", 0.0))

    mom5_sig = _clamp(mom5 / 0.10, -1.0, 1.0)
    mom21_sig = _clamp(mom21 / 0.20, -1.0, 1.0)
    mom63_sig = _clamp(mom63 / 0.30, -1.0, 1.0)
    trend_sig = _clamp(trend / 0.10, -1.0, 1.0)

    short = _clamp(0.70 * mom5_sig + 0.30 * rsi_sig, -1.0, 1.0)
    mid = _clamp(0.60 * mom21_sig + 0.40 * trend_sig, -1.0, 1.0)
    long = _clamp(0.55 * mom63_sig + 0.45 * trend_sig, -1.0, 1.0)

    return {"short": float(short), "mid": float(mid), "long": float(long)}


def stacked_score_from_features(
    features: Dict[str, float], regime: str
) -> Tuple[float, Dict[str, Any]]:
    comps = _score_components_from_features(features)
    w = _stack_weights_for_regime(regime)

    raw = (
        w["short"] * comps["short"]
        + w["mid"] * comps["mid"]
        + w["long"] * comps["long"]
    )

    vola = float(features.get("volatility_ann", 0.0))
    vola_pen = _clamp((vola - 0.25) / 0.50, 0.0, 0.5)
    score = _clamp(raw * (1.0 - vola_pen), -1.0, 1.0)

    meta = {
        "stack_weights": w,
        "components": comps,
        "vola_penalty": float(vola_pen),
        "raw_before_vol_penalty": float(raw),
    }
    return float(score), meta


# =============================================================================
# Thresholds
# =============================================================================


def _thresholds_for_regime(
    regime: str, calibration: Optional[Dict[str, Any]]
) -> Dict[str, float]:
    defaults = {
        "TRENDING": {"buy": 0.2, "sell": -0.2},
        "CHOPPY": {"buy": 0.3, "sell": -0.3},
        "HIGH_VOL": {"buy": 0.45, "sell": -0.45},
    }
    if not calibration:
        return defaults.get(regime, {"buy": 0.3, "sell": -0.3})

    thr_by_reg = calibration.get("thresholds_by_regime") or {}
    if regime in thr_by_reg and isinstance(thr_by_reg[regime], dict):
        b = float(
            thr_by_reg[regime].get("buy", defaults.get(regime, {"buy": 0.3})["buy"])
        )
        s = float(
            thr_by_reg[regime].get("sell", defaults.get(regime, {"sell": -0.3})["sell"])
        )
        return {"buy": b, "sell": s}

    return defaults.get(regime, {"buy": 0.3, "sell": -0.3})


def _score_mode() -> str:
    mode = (_env("FTIP_SCORE_MODE", "stacked") or "stacked").strip().lower()
    return mode if mode in ("base", "stacked") else "stacked"


# =============================================================================
# Core backtest engine (simple buy-and-hold placeholder for /run_backtest)
# =============================================================================


def run_backtest_core(
    candles: List[Candle], include_equity: bool, include_returns: bool
) -> BacktestSummary:
    closes = [c.close for c in candles]
    dates = [c.timestamp for c in candles]

    rets = _pct_change(closes)
    equity = [1.0]
    for r in rets[1:]:
        equity.append(equity[-1] * (1.0 + r))

    total_return = equity[-1] - 1.0

    n = max(1, len(rets) - 1)
    ann_return = (1.0 + total_return) ** (252.0 / n) - 1.0 if n > 0 else 0.0

    vol = _std(rets[1:]) * math.sqrt(252.0) if len(rets) > 2 else 0.0
    sharpe = (ann_return / vol) if vol > 0 else 0.0
    mdd = _max_drawdown(equity)

    return BacktestSummary(
        total_return=float(total_return),
        annual_return=float(ann_return),
        sharpe=float(sharpe),
        max_drawdown=float(mdd),
        volatility=float(vol),
        equity_curve=(
            {d: float(equity[i]) for i, d in enumerate(dates)}
            if include_equity
            else None
        ),
        returns=(
            {d: float(rets[i]) for i, d in enumerate(dates)}
            if include_returns
            else None
        ),
    )


# =============================================================================
# Signal computation
# =============================================================================


def _filter_upto(candles: List[Candle], as_of: str) -> List[Candle]:
    cutoff = _parse_date(as_of)
    out: List[Candle] = []
    for c in candles:
        try:
            d = _parse_date(c.timestamp)
        except Exception:
            continue
        if d <= cutoff:
            out.append(c)
    return out


def compute_signal_for_symbol(symbol: str, as_of: str, lookback: int) -> SignalResponse:
    as_of_d = _parse_date(as_of)
    from_guess = (as_of_d - dt.timedelta(days=900)).isoformat()

    candles_all = massive_fetch_daily_bars_cached(symbol, from_guess, as_of)
    candles_upto = _filter_upto(candles_all, as_of)

    if len(candles_upto) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data <= {as_of}. Need 30, got {len(candles_upto)}.",
        )

    effective = min(int(lookback), len(candles_upto))
    window = candles_upto[-effective:]

    feats = compute_features(window)
    regime = detect_regime(feats)

    base_score, notes = score_from_features(feats)
    stack_score, stack_meta = stacked_score_from_features(feats, regime)

    mode = _score_mode()
    if mode == "stacked":
        score = float(stack_score)
        notes.append("Score mode: STACKED (multi-horizon).")
    else:
        score = float(base_score)
        notes.append("Score mode: BASE (legacy).")

    cal_loaded, cal = _load_calibration_for_symbol(symbol)
    thr = _thresholds_for_regime(regime, cal)

    sig = "HOLD"
    if score >= thr["buy"]:
        sig = "BUY"
    elif score <= thr["sell"]:
        sig = "SELL"

    conf = abs(score)
    if regime == "HIGH_VOL":
        conf *= 0.65
        notes.append("Confidence reduced due to HIGH_VOL regime.")

    if effective < lookback:
        notes.insert(
            0,
            f"Requested lookback={lookback}, only {effective} bars available. Using effective_lookback={effective}.",
        )

    if regime == "TRENDING":
        notes.append("Regime: TRENDING.")
    elif regime == "CHOPPY":
        notes.append("Regime: CHOPPY.")
    else:
        notes.append("Regime: HIGH_VOL.")

    if cal_loaded:
        notes.append(
            "Using calibrated thresholds from FTIP_CALIBRATION_JSON_MAP/FTIP_CALIBRATION_JSON."
        )

    meta = None
    if cal_loaded and cal:
        meta = {
            "optimize_horizon": cal.get("optimize_horizon"),
            "created_at_utc": cal.get("created_at_utc"),
            "symbol": cal.get("symbol"),
            "train_range": cal.get("train_range"),
        }

    return SignalResponse(
        symbol=(symbol or "").strip().upper(),
        as_of=as_of,
        lookback=int(lookback),
        effective_lookback=int(effective),
        regime=regime,
        thresholds=thr,
        score=float(score),
        signal=sig,
        confidence=float(conf),
        features={k: float(v) for k, v in feats.items()},
        notes=notes,
        score_mode=mode,
        base_score=float(base_score),
        stacked_score=float(stack_score),
        stacked_meta=stack_meta,
        calibration_loaded=bool(cal_loaded),
        calibration_meta=meta,
    )


def compute_signal_for_symbol_from_candles(
    symbol: str,
    as_of: str,
    lookback: int,
    candles_all: List[Candle],
) -> SignalResponse:
    candles_upto = _filter_upto(candles_all, as_of)

    if len(candles_upto) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data to compute signal. Need at least 30 bars <= {as_of}, got {len(candles_upto)}.",
        )

    effective = min(int(lookback), len(candles_upto))
    window = candles_upto[-effective:]

    feats = compute_features(window)
    regime = detect_regime(feats)

    base_score, notes = score_from_features(feats)
    stack_score, stack_meta = stacked_score_from_features(feats, regime)

    mode = _score_mode()
    if mode == "stacked":
        score = float(stack_score)
        notes.append("Score mode: STACKED (multi-horizon).")
    else:
        score = float(base_score)
        notes.append("Score mode: BASE (legacy).")

    cal_loaded, cal = _load_calibration_for_symbol(symbol)
    thr = _thresholds_for_regime(regime, cal)

    sig = "HOLD"
    if score >= thr["buy"]:
        sig = "BUY"
    elif score <= thr["sell"]:
        sig = "SELL"

    conf = abs(score)
    if regime == "HIGH_VOL":
        conf *= 0.65
        notes.append("Confidence reduced due to HIGH_VOL regime.")

    if effective < lookback:
        notes.insert(
            0,
            f"Requested lookback={lookback}, but only {effective} bars available by {as_of}. Using effective_lookback={effective}.",
        )

    if regime == "TRENDING":
        notes.append("Regime: TRENDING.")
    elif regime == "CHOPPY":
        notes.append("Regime: CHOPPY.")
    else:
        notes.append("Regime: HIGH_VOL.")

    if cal_loaded:
        notes.append(
            "Using calibrated thresholds from FTIP_CALIBRATION_JSON_MAP/FTIP_CALIBRATION_JSON."
        )

    meta = None
    if cal_loaded and cal:
        meta = {
            "optimize_horizon": cal.get("optimize_horizon"),
            "created_at_utc": cal.get("created_at_utc"),
            "symbol": cal.get("symbol"),
            "train_range": cal.get("train_range"),
        }

    return SignalResponse(
        symbol=(symbol or "").strip().upper(),
        as_of=as_of,
        lookback=int(lookback),
        effective_lookback=int(effective),
        regime=regime,
        thresholds=thr,
        score=float(score),
        signal=sig,
        confidence=float(conf),
        features={k: float(v) for k, v in feats.items()},
        notes=notes,
        score_mode=mode,
        base_score=float(base_score),
        stacked_score=float(stack_score),
        stacked_meta=stack_meta,
        calibration_loaded=bool(cal_loaded),
        calibration_meta=meta,
    )


# =============================================================================
# Walk-forward + Calibration
# =============================================================================


def _forward_return(closes: List[float], idx: int, horizon: int) -> Optional[float]:
    j = idx + horizon
    if j >= len(closes):
        return None
    if closes[idx] == 0:
        return None
    return float(closes[j] / closes[idx] - 1.0)


def walk_forward_table(
    symbol: str, from_date: str, to_date: str, lookback: int, horizons: List[int]
) -> List[Dict[str, Any]]:
    candles_all = massive_fetch_daily_bars_cached(symbol, from_date, to_date)
    closes = [c.close for c in candles_all]
    dates = [c.timestamp for c in candles_all]

    mode = _score_mode()
    cal_loaded, cal = _load_calibration_for_symbol(symbol)

    rows: List[Dict[str, Any]] = []
    for i in range(len(candles_all)):
        start = max(0, i - int(lookback) + 1)
        window = candles_all[start : i + 1]
        if len(window) < 30:
            continue

        feats = compute_features(window)
        regime = detect_regime(feats)

        base_score, _ = score_from_features(feats)
        stack_score, _stack_meta = stacked_score_from_features(feats, regime)
        score = float(stack_score) if mode == "stacked" else float(base_score)

        thr = _thresholds_for_regime(regime, cal)

        sig = "HOLD"
        if score >= thr["buy"]:
            sig = "BUY"
        elif score <= thr["sell"]:
            sig = "SELL"

        conf = abs(score)
        if regime == "HIGH_VOL":
            conf *= 0.65

        row: Dict[str, Any] = {
            "date": dates[i],
            "signal": sig,
            "score": float(score),
            "confidence": float(conf),
            "regime": regime,
            "calibration_loaded": bool(cal_loaded),
        }

        for h in horizons:
            row[f"fwd_ret_{h}"] = _forward_return(closes, i, int(h))

        rows.append(row)

    return rows


def calibrate_thresholds(
    rows: List[Dict[str, Any]], optimize_horizon: int, min_trades_per_side: int
) -> Dict[str, Any]:
    created = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    regimes = sorted(set(r["regime"] for r in rows))
    thresholds_by_regime: Dict[str, Dict[str, float]] = {}
    diagnostics: Dict[str, Any] = {}

    buys = [0.15, 0.20, 0.25, 0.30, 0.35, 0.45]
    sells = [-0.15, -0.20, -0.25, -0.30, -0.35, -0.45]

    key = f"fwd_ret_{int(optimize_horizon)}"

    for reg in regimes:
        rrows = [r for r in rows if r["regime"] == reg and r.get(key) is not None]
        if not rrows:
            continue

        best: Optional[Dict[str, Any]] = None
        best_metric: Optional[float] = None

        for b in buys:
            for s in sells:
                buysig = [r for r in rrows if float(r["score"]) >= float(b)]
                sellsig = [r for r in rrows if float(r["score"]) <= float(s)]

                if len(buysig) < int(min_trades_per_side) or len(sellsig) < int(
                    min_trades_per_side
                ):
                    continue

                buy_ret = _mean(
                    [float(r[key]) for r in buysig if r.get(key) is not None]
                )
                sell_ret = _mean(
                    [float(r[key]) for r in sellsig if r.get(key) is not None]
                )

                metric = float(buy_ret - sell_ret)
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best = {
                        "buy": float(b),
                        "sell": float(s),
                        "metric": float(metric),
                        "buy_n": len(buysig),
                        "sell_n": len(sellsig),
                    }

        if best is None:
            if reg == "TRENDING":
                thresholds_by_regime[reg] = {"buy": 0.2, "sell": -0.2}
            elif reg == "CHOPPY":
                thresholds_by_regime[reg] = {"buy": 0.3, "sell": -0.3}
            else:
                thresholds_by_regime[reg] = {"buy": 0.45, "sell": -0.45}
            diagnostics[reg] = {
                "note": "No valid threshold pair met min_trades constraints; using defaults.",
                "row_count": len(rrows),
            }
        else:
            thresholds_by_regime[reg] = {
                "buy": float(best["buy"]),
                "sell": float(best["sell"]),
            }
            diagnostics[reg] = {"best": best, "row_count": len(rrows)}

    return {
        "created_at_utc": created,
        "optimize_horizon": int(optimize_horizon),
        "thresholds_by_regime": thresholds_by_regime,
        "diagnostics": diagnostics,
    }


# =============================================================================
# Portfolio layer helpers
# =============================================================================


def _portfolio_knobs(req: PortfolioRequest) -> Dict[str, float]:
    max_w = (
        req.max_weight
        if req.max_weight is not None
        else _env_float("FTIP_PORTFOLIO_MAX_WEIGHT", 0.30)
    )
    hold_mult = (
        req.hold_multiplier
        if req.hold_multiplier is not None
        else _env_float("FTIP_PORTFOLIO_HOLD_MULT", 0.35)
    )
    min_conf = (
        req.min_confidence
        if req.min_confidence is not None
        else _env_float("FTIP_PORTFOLIO_MIN_CONF", 0.10)
    )

    max_w = _clamp(float(max_w), 0.01, 1.0)
    hold_mult = _clamp(float(hold_mult), 0.0, 1.0)
    min_conf = _clamp(float(min_conf), 0.0, 1.0)
    return {
        "max_weight": max_w,
        "hold_multiplier": hold_mult,
        "min_confidence": min_conf,
    }


def _raw_weight_from_signal(
    sig: Dict[str, Any], hold_mult: float, min_conf: float, allow_shorts: bool
) -> float:
    signal = (sig.get("signal") or "HOLD").upper()
    conf = float(sig.get("confidence") or 0.0)
    vola = float(
        (sig.get("features") or {}).get("volatility_ann")
        or sig.get("volatility_ann")
        or 0.0
    )

    if conf < float(min_conf):
        return 0.0

    vol_adj = 1.0 / max(1e-6, vola)

    if signal == "BUY":
        return conf * vol_adj
    if signal == "HOLD":
        return conf * vol_adj * float(hold_mult)
    if signal == "SELL":
        return (-conf * vol_adj) if allow_shorts else 0.0
    return 0.0


def _apply_deadband(
    old_w: Dict[str, float], new_w: Dict[str, float], min_delta: float
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    syms = set(old_w.keys()) | set(new_w.keys())
    for s in syms:
        ow = float(old_w.get(s, 0.0))
        nw = float(new_w.get(s, 0.0))
        if abs(nw - ow) < float(min_delta):
            if abs(ow) > 1e-12:
                out[s] = ow
        else:
            if abs(nw) > 1e-12:
                out[s] = nw
    return out


def _cap_turnover(
    old_w: Dict[str, float], target_w: Dict[str, float], max_turnover: float
) -> Dict[str, float]:
    if max_turnover <= 0:
        return dict(old_w)

    syms = set(old_w.keys()) | set(target_w.keys())
    deltas: Dict[str, float] = {}
    turnover = 0.0
    for s in syms:
        d = float(target_w.get(s, 0.0)) - float(old_w.get(s, 0.0))
        deltas[s] = d
        turnover += abs(d)

    if turnover <= float(max_turnover) or turnover <= 1e-15:
        return dict(target_w)

    scale = float(max_turnover) / turnover
    out: Dict[str, float] = {}
    for s in syms:
        ow = float(old_w.get(s, 0.0))
        nw = ow + deltas[s] * scale
        if abs(nw) > 1e-12:
            out[s] = float(nw)
    return out


def _normalize_long_only_with_caps(
    raw_pos: Dict[str, float], max_w: float
) -> Tuple[Dict[str, float], float]:
    items = {k: float(v) for k, v in raw_pos.items() if float(v) > 0.0}
    if not items:
        return {}, 1.0

    total = sum(items.values())
    if total <= 0:
        return {}, 1.0

    pref = {k: v / total for k, v in items.items()}  # sums to 1

    final: Dict[str, float] = {k: 0.0 for k in pref.keys()}
    active = set(pref.keys())
    remaining = 1.0

    for _ in range(10000):
        if remaining <= 1e-12 or not active:
            break

        denom = sum(pref[k] for k in active)
        if denom <= 0:
            break

        progressed = False
        for k in list(active):
            target = remaining * (pref[k] / denom)
            room = float(max_w) - final[k]
            add = min(target, room)
            if add > 0:
                final[k] += add
                remaining -= add
                progressed = True
            if final[k] >= float(max_w) - 1e-12:
                active.remove(k)

        if not progressed:
            break

    final = {k: float(v) for k, v in final.items() if v > 1e-12}
    cash = float(_clamp(1.0 - sum(final.values()), 0.0, 1.0))
    return final, cash


def _normalize_allow_shorts_with_caps(
    raw: Dict[str, float], max_w: float
) -> Tuple[Dict[str, float], float]:
    if not raw:
        return {}, 1.0

    gross = sum(abs(v) for v in raw.values())
    if gross <= 0:
        return {}, 1.0

    wts: Dict[str, float] = {}
    for k, v in raw.items():
        w = v / gross
        w = _clamp(float(w), -float(max_w), float(max_w))
        if abs(w) > 1e-12:
            wts[k] = float(w)

    net = sum(wts.values())
    cash = float(_clamp(1.0 - max(net, 0.0), 0.0, 1.0))
    return wts, cash


def _daily_returns_from_closes(
    dates: List[str], closes: List[float]
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        out[dates[i]] = 0.0 if prev == 0 else float(cur / prev - 1.0)
    return out


# =============================================================================
# Portfolio backtest (OFFLINE signals from cached candles)
# =============================================================================


def backtest_portfolio(req: PortfolioBacktestRequest) -> PortfolioBacktestResponse:
    # ---- absolute safety: never return None ----
    try:
        min_symbols = max(1, _env_int("FTIP_PORTFOLIO_MIN_SYMBOLS", 2))
        skipped_symbols: List[Dict[str, Any]] = []

        start_dt = _parse_date(req.from_date)
        warmup_days = max(900, int(req.lookback) * 4)
        buffer_start = (start_dt - dt.timedelta(days=warmup_days)).isoformat()

        # Fetch candles once per symbol (cached)
        candles_by_sym: Dict[str, List[Candle]] = {}
        for sym in req.symbols:
            s = (sym or "").strip().upper()
            if not s:
                continue
            try:
                bars = massive_fetch_daily_bars_cached(s, buffer_start, req.to_date)
                if not bars:
                    raise ValueError("No market data returned")
                candles_by_sym[s] = bars
            except HTTPException as e:
                skipped_symbols.append(
                    {
                        "symbol": s,
                        "reason": f"HTTP {e.status_code}: {e.detail}",
                    }
                )
            except Exception as e:
                skipped_symbols.append(
                    {"symbol": s, "reason": f"{type(e).__name__}: {e}"}
                )

        audit_info: Dict[str, Any] = {}
        if skipped_symbols:
            audit_info["skipped_symbols"] = skipped_symbols

        if not candles_by_sym:
            raise HTTPException(
                status_code=502, detail="All symbols failed to fetch market data."
            )

        if len(candles_by_sym) < min_symbols:
            note = (
                f"Only {len(candles_by_sym)} symbols fetched successfully; "
                f"minimum required is {min_symbols}."
            )
            audit_info.setdefault("notes", []).append(note)
            return PortfolioBacktestResponse(
                total_return=0.0,
                annual_return=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                turnover=0.0,
                equity_curve={} if req.include_equity_curve else None,
                audit=audit_info or None,
            )

        # Common calendar intersection within requested range
        date_sets: List[set] = []
        for _s, cs in candles_by_sym.items():
            ds = {
                c.timestamp for c in cs if req.from_date <= c.timestamp <= req.to_date
            }
            if ds:
                date_sets.append(ds)

        if not date_sets:
            return PortfolioBacktestResponse(
                total_return=0.0,
                annual_return=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                turnover=0.0,
                equity_curve={} if req.include_equity_curve else None,
                audit=audit_info or None,
            )

        common_dates = sorted(set.intersection(*date_sets))
        if len(common_dates) < 2:
            return PortfolioBacktestResponse(
                total_return=0.0,
                annual_return=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                turnover=0.0,
                equity_curve={} if req.include_equity_curve else None,
                audit=audit_info or None,
            )

        common_set = set(common_dates)

        # Daily returns per symbol
        rets_by_sym: Dict[str, Dict[str, float]] = {}
        for s, cs in candles_by_sym.items():
            pairs = [(c.timestamp, c.close) for c in cs if c.timestamp in common_set]
            pairs.sort(key=lambda x: x[0])
            if len(pairs) < 2:
                continue
            ds2 = [p[0] for p in pairs]
            closes2 = [float(p[1]) for p in pairs]
            rets_by_sym[s] = _daily_returns_from_closes(ds2, closes2)

        mode = _score_mode()

        def _compute_signal_from_cached(
            symbol: str, as_of: str, lookback: int
        ) -> Optional[Dict[str, Any]]:
            cs_all = candles_by_sym.get(symbol, [])
            if not cs_all:
                return None

            cutoff = _parse_date(as_of)
            cs_upto: List[Candle] = []
            for c in cs_all:
                try:
                    d0 = _parse_date(c.timestamp)
                except Exception:
                    continue
                if d0 <= cutoff:
                    cs_upto.append(c)

            if len(cs_upto) < 30:
                return None

            eff = min(int(lookback), len(cs_upto))
            window = cs_upto[-eff:]

            feats = compute_features(window)
            regime = detect_regime(feats)

            base_score, notes = score_from_features(feats)
            stack_score, stack_meta = stacked_score_from_features(feats, regime)

            if mode == "stacked":
                score = float(stack_score)
                notes.append("Score mode: STACKED (multi-horizon).")
            else:
                score = float(base_score)
                notes.append("Score mode: BASE (legacy).")

            cal_loaded, cal = _load_calibration_for_symbol(symbol)
            thr = _thresholds_for_regime(regime, cal)

            sig = "HOLD"
            if score >= thr["buy"]:
                sig = "BUY"
            elif score <= thr["sell"]:
                sig = "SELL"

            conf = abs(score)
            if regime == "HIGH_VOL":
                conf *= 0.65
                notes.append("Confidence reduced due to HIGH_VOL regime.")

            if cal_loaded:
                notes.append(
                    "Using calibrated thresholds from FTIP_CALIBRATION_JSON_MAP/FTIP_CALIBRATION_JSON."
                )

            meta = None
            if cal_loaded and cal:
                meta = {
                    "optimize_horizon": cal.get("optimize_horizon"),
                    "created_at_utc": cal.get("created_at_utc"),
                    "symbol": cal.get("symbol"),
                    "train_range": cal.get("train_range"),
                }

            return {
                "symbol": symbol,
                "as_of": as_of,
                "lookback": int(lookback),
                "effective_lookback": int(eff),
                "regime": regime,
                "thresholds": thr,
                "score": float(score),
                "signal": sig,
                "confidence": float(conf),
                "features": {k: float(v) for k, v in feats.items()},
                "notes": notes,
                "score_mode": mode,
                "base_score": float(base_score),
                "stacked_score": float(stack_score),
                "stacked_meta": stack_meta,
                "calibration_loaded": bool(cal_loaded),
                "calibration_meta": meta,
            }

        def _weights_for_date(as_of: str) -> Tuple[Dict[str, float], float]:
            max_w = (
                float(req.max_weight)
                if req.max_weight is not None
                else _env_float("FTIP_PORTFOLIO_MAX_WEIGHT", 0.30)
            )
            hold_mult = (
                float(req.hold_multiplier)
                if req.hold_multiplier is not None
                else _env_float("FTIP_PORTFOLIO_HOLD_MULT", 0.35)
            )
            min_conf = (
                float(req.min_confidence)
                if req.min_confidence is not None
                else _env_float("FTIP_PORTFOLIO_MIN_CONF", 0.10)
            )
            allow_shorts = bool(req.allow_shorts)

            max_w = _clamp(max_w, 0.01, 1.0)
            hold_mult = _clamp(hold_mult, 0.0, 1.0)
            min_conf = _clamp(min_conf, 0.0, 1.0)

            per: Dict[str, Any] = {}
            for s in candles_by_sym.keys():
                sig = _compute_signal_from_cached(s, as_of, int(req.lookback))
                if sig is not None:
                    per[s] = sig

            raw: Dict[str, float] = {}
            for s, sig in per.items():
                rw = _raw_weight_from_signal(
                    sig,
                    hold_mult=hold_mult,
                    min_conf=min_conf,
                    allow_shorts=allow_shorts,
                )
                if rw != 0.0:
                    raw[s] = float(rw)

            if not raw:
                return {}, 1.0

            if allow_shorts:
                return _normalize_allow_shorts_with_caps(raw, max_w)

            long_raw = {k: v for k, v in raw.items() if v > 0.0}
            return _normalize_long_only_with_caps(long_raw, max_w)

        # Audit tracking
        audit_rows: List[Dict[str, Any]] = []
        audit_violations = 0
        ts_by_sym: Dict[str, List[str]] = {
            s: [c.timestamp for c in cs] for s, cs in candles_by_sym.items()
        }

        # Rebalance loop
        cost_rate = (float(req.trading_cost_bps) + float(req.slippage_bps)) / 10000.0
        equity = 1.0
        equity_curve: Dict[str, float] = {}
        daily_port_rets: List[float] = []

        weights: Dict[str, float] = {}
        total_turnover = 0.0
        reb_every = max(1, int(req.rebalance_every))

        max_w_caps = (
            float(req.max_weight)
            if req.max_weight is not None
            else _env_float("FTIP_PORTFOLIO_MAX_WEIGHT", 0.30)
        )
        max_w_caps = _clamp(max_w_caps, 0.01, 1.0)

        for i, d in enumerate(common_dates):
            if i == 0:
                equity_curve[d] = equity
                continue

            if (i - 1) % reb_every == 0:
                target_w, _cash = _weights_for_date(as_of=d)

                if req.audit_no_lookahead:
                    row = {"rebalance_date": d, "per_symbol_last_candle": {}}
                    for sym, ts in ts_by_sym.items():
                        last_ok = None
                        for t in ts:
                            if t <= d:
                                last_ok = t
                            else:
                                break
                        row["per_symbol_last_candle"][sym] = last_ok
                        if last_ok is None or last_ok > d:
                            audit_violations += 1
                    if len(audit_rows) < int(req.audit_max_rows):
                        audit_rows.append(row)

                target_w = _apply_deadband(
                    weights, target_w, min_delta=float(req.min_trade_delta)
                )
                capped_w = _cap_turnover(
                    weights,
                    target_w,
                    max_turnover=float(req.max_turnover_per_rebalance),
                )

                if req.allow_shorts:
                    capped_w, _cash2 = _normalize_allow_shorts_with_caps(
                        capped_w, max_w_caps
                    )
                else:
                    capped_w, _cash2 = _normalize_long_only_with_caps(
                        {k: v for k, v in capped_w.items() if v > 0.0}, max_w_caps
                    )

                turnover = sum(
                    abs(float(capped_w.get(s, 0.0)) - float(weights.get(s, 0.0)))
                    for s in set(weights) | set(capped_w)
                )

                equity *= 1.0 - turnover * cost_rate
                total_turnover += turnover
                weights = dict(capped_w)

            pr = 0.0
            for s, w in weights.items():
                pr += float(w) * float(rets_by_sym.get(s, {}).get(d, 0.0))

            equity *= 1.0 + pr
            equity_curve[d] = equity
            daily_port_rets.append(float(pr))

        total_return = equity - 1.0
        n = max(1, len(daily_port_rets))
        annual_return = (1.0 + total_return) ** (252.0 / n) - 1.0

        volatility = (
            _std(daily_port_rets) * math.sqrt(252.0)
            if len(daily_port_rets) > 2
            else 0.0
        )
        sharpe = (annual_return / volatility) if volatility > 0 else 0.0
        max_dd = _max_drawdown(list(equity_curve.values()))

        audit_payload: Dict[str, Any] = dict(audit_info)
        if req.audit_no_lookahead:
            audit_payload.update(
                {
                    "violations_count": int(audit_violations),
                    "rows_sample": audit_rows,
                    "note": "Each row shows the last candle timestamp <= rebalance_date per symbol.",
                }
            )

        return PortfolioBacktestResponse(
            total_return=float(total_return),
            annual_return=float(annual_return),
            sharpe=float(sharpe),
            max_drawdown=float(max_dd),
            volatility=float(volatility),
            turnover=float(total_turnover),
            equity_curve=equity_curve if req.include_equity_curve else None,
            audit=audit_payload or None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"portfolio_backtest crash: {type(e).__name__}: {e}"
        )


# =============================================================================
# FastAPI
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    _startup()
    yield


def _startup() -> List[str]:
    return lifecycle.startup()


app = FastAPI(title=APP_NAME, version="1.0.0", lifespan=lifespan)
security.add_cors_middleware(app)
security.log_auth_config(logger)


@app.middleware("http")
async def security_and_tracing_middleware(request: Request, call_next):
    trace_id = security.trace_id_from_request(request)
    start = time.perf_counter()
    logger.info(
        "request.start",
        extra={
            "path": request.url.path,
            "method": request.method,
            "trace_id": trace_id,
        },
    )

    auth_error = security.require_api_key_if_needed(request, trace_id)
    if auth_error:
        metrics_tracker.record_request(request.url.path, auth_error.status_code)
        return auth_error

    rate_limit_error = security.enforce_rate_limit(request, rate_limiter, trace_id)
    if rate_limit_error:
        metrics_tracker.record_request(request.url.path, rate_limit_error.status_code)
        return rate_limit_error

    try:
        response = await call_next(request)
    except DBError as exc:
        logger.warning("db.error", extra={"trace_id": trace_id, "message": str(exc)})
        response = security.json_error_response(
            "database_error", str(exc), trace_id, exc.status_code
        )
    except HTTPException as exc:
        if exc.status_code == 401:
            response = security.unauthorized_response(trace_id)
        else:
            detail_msg = (
                exc.detail.get("message")
                if isinstance(exc.detail, dict)
                else str(exc.detail)
            )
            response = security.json_error_response(
                "http_error", detail_msg, trace_id, exc.status_code
            )
    except RequestValidationError as exc:
        response = security.json_error_response(
            "validation_error", str(exc), trace_id, 422
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("unhandled.error", extra={"trace_id": trace_id})
        response = security.json_error_response(
            exc.__class__.__name__, str(exc), trace_id, 500
        )

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Trace-Id"] = trace_id
    logger.info(
        "request.end",
        extra={
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "trace_id": trace_id,
        },
    )
    metrics_tracker.record_request(request.url.path, response.status_code)
    return response


@app.exception_handler(DBError)
async def db_exception_handler(request: Request, exc: DBError):
    trace_id = security.trace_id_from_request(request)
    logger.warning("db.error", extra={"trace_id": trace_id, "message": str(exc)})
    return security.json_error_response(
        "database_error", str(exc), trace_id, exc.status_code
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    trace_id = security.trace_id_from_request(request)
    detail_msg = (
        exc.detail.get("message") if isinstance(exc.detail, dict) else str(exc.detail)
    )
    return security.json_error_response(
        "http_error", detail_msg, trace_id, exc.status_code
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    trace_id = security.trace_id_from_request(request)
    return security.json_error_response("validation_error", str(exc), trace_id, 422)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    trace_id = security.trace_id_from_request(request)
    logger.exception("unhandled.error", extra={"trace_id": trace_id})
    return security.json_error_response(exc.__class__.__name__, str(exc), trace_id, 500)


app.include_router(assistant_router, prefix="/assistant")
app.include_router(llm_router)
app.include_router(narrator_router, prefix="/narrator")
app.include_router(prosperity_router, prefix="/prosperity")
app.include_router(prosperity_jobs_router)
app.include_router(market_data_jobs_router)
app.include_router(features_jobs_router)
app.include_router(signals_jobs_router)
app.include_router(signals_router)
app.include_router(backtest_router)
app.include_router(ops_router)

WEBAPP_DIR = Path(__file__).resolve().parent / "webapp"
if WEBAPP_DIR.exists():
    app.mount(
        "/app/static", StaticFiles(directory=str(WEBAPP_DIR)), name="webapp-static"
    )


@app.get("/app")
def webapp_index() -> FileResponse:
    return FileResponse(WEBAPP_DIR / "index.html")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": APP_NAME,
        "status": "ok",
        "db_enabled": bool(DB_ENABLED),
        "db_pool_ready": _db_pool_ready(),
        "endpoints": [
            "/health",
            "/ready",
            "/db/health",
            "/db/save_signal",
            "/db/save_signals",
            "/db/run_snapshot",
            "/db/save_portfolio_backtest",
            "/db/universe/load_default",
            "/version",
            "/signal",
            "/signals",
            "/portfolio_signals",
            "/portfolio_backtest",
            "/walk_forward",
            "/calibrate",
            "/run_backtest",
            "/market/massive/bars",
            "/universe/upsert",
            "/universe/top",
            "/docs",
            "/assistant/health",
            "/assistant/chat",
            "/assistant/explain/signal",
            "/assistant/explain/backtest",
            "/assistant/title_session",
            "/narrator/signal",
            "/narrator/portfolio",
            "/narrator/ask",
            "/narrator/health",
            "/narrator/explain-signal",
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> Dict[str, Any]:
    if not db.db_enabled():
        raise HTTPException(status_code=503, detail="db disabled")

    try:
        row = db.fetch1("SELECT 1")
        if not row or row[0] != 1:
            raise HTTPException(status_code=503, detail="db not ready")
        return {"status": "ok", "db_enabled": True, "select_1": row[0]}
    except DBError as exc:
        raise HTTPException(status_code=503, detail=f"db not ready: {exc}") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"db not ready: {type(exc).__name__}: {exc}"
        ) from exc


@app.get("/auth/status")
async def auth_status(request: Request) -> Dict[str, Any]:
    trace_id = security.trace_id_from_request(request)
    if security.auth_enabled() and not security.auth_status_public():
        try:
            security.validate_api_key(request)
        except HTTPException:
            return security.unauthorized_response(trace_id)

    return security.auth_status_payload()


@app.get("/db/health")
def db_health() -> Dict[str, Any]:
    if not db.db_enabled():
        return {"status": "disabled", "db_enabled": False}

    try:
        row = db.fetch1("SELECT 1")
        schema_info: Dict[str, Any] = {
            "schema_migrations": None,
            "latest_migration": None,
        }
        try:
            table_row = db.fetch1("SELECT to_regclass('public.schema_migrations')")
            if table_row and table_row[0]:
                versions = [
                    r[0]
                    for r in db.fetchall(
                        "SELECT version FROM schema_migrations ORDER BY applied_at ASC"
                    )
                ]
                latest = db.fetch1(
                    "SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1"
                )
                schema_info = {
                    "schema_migrations": versions,
                    "latest_migration": latest[0] if latest else None,
                }
        except Exception as exc:
            schema_info = {"schema_migrations_error": f"{type(exc).__name__}: {exc}"}

        return {
            "status": "ok",
            "db_enabled": True,
            "select_1": row[0] if row else None,
            **schema_info,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"db_health failed: {type(e).__name__}: {e}"
        )


@app.post("/db/save_signal")
def db_save_signal(req: SaveSignalRequest) -> Dict[str, Any]:
    if not DB_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="DB is disabled. Set FTIP_DB_ENABLED=1 and DATABASE_URL.",
        )

    try:
        sig = compute_signal_for_symbol(req.symbol, req.as_of, req.lookback)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"compute_signal failed: {type(e).__name__}: {e}"
        )

    try:
        inserted, row_id = persist_signal_record(sig)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"save_signal failed: {type(e).__name__}: {e}"
        )

    return {
        "status": "ok",
        "inserted": bool(inserted),
        "id": row_id,
        "key": {
            "symbol": sig.symbol,
            "as_of": sig.as_of,
            "lookback": sig.lookback,
            "score_mode": sig.score_mode,
        },
    }


@app.post("/db/save_signals")
def db_save_signals(req: SaveSignalsRequest) -> Dict[str, Any]:
    if not DB_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="DB is disabled. Set FTIP_DB_ENABLED=1 and DATABASE_URL.",
        )

    results: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}
    inserted_count = 0
    updated_count = 0

    for sym in req.symbols:
        s = (sym or "").strip().upper()
        if not s:
            continue
        try:
            sig = compute_signal_for_symbol(s, req.as_of, req.lookback)
        except HTTPException as e:
            errors[s] = {"status_code": e.status_code, "detail": e.detail}
            continue
        except Exception as e:
            errors[s] = {"status_code": 500, "detail": str(e)}
            continue

        try:
            inserted, row_id = persist_signal_record(sig)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"save_signals failed on {s}: {type(e).__name__}: {e}",
            )

        results[s] = {
            "id": row_id,
            "inserted": bool(inserted),
            "score_mode": sig.score_mode,
            "as_of": sig.as_of,
            "lookback": sig.lookback,
        }
        if inserted:
            inserted_count += 1
        else:
            updated_count += 1

    return {
        "status": "ok",
        "as_of": req.as_of,
        "lookback": req.lookback,
        "count_ok": len(results),
        "count_error": len(errors),
        "inserted": inserted_count,
        "updated": updated_count,
        "results": results,
        "errors": errors,
    }


@app.post("/db/run_snapshot")
def db_run_snapshot(req: RunSnapshotRequest) -> Dict[str, Any]:
    if not DB_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="DB is disabled. Set FTIP_DB_ENABLED=1 and DATABASE_URL.",
        )

    try:
        db.get_pool()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB pool not available: {e}")

    try:
        limit_val = max(1, min(int(req.limit), 5000))
    except Exception:
        limit_val = 1000

    try:
        symbols = db.get_universe(active_only=req.active_only, limit=limit_val)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load universe: {type(e).__name__}: {e}"
        )

    saved_count = 0
    error_count = 0
    skipped_count = 0
    errors: Dict[str, str] = {}
    sleep_ms = max(0, _env_int("FTIP_SNAPSHOT_SLEEP_MS", 0))

    for sym in symbols:
        s = (sym or "").strip().upper()
        if not s:
            skipped_count += 1
            continue

        try:
            sig = compute_signal_for_symbol(s, req.as_of, req.lookback)
            inserted, row_id = persist_signal_record(sig)
            if row_id is None:
                error_count += 1
                if len(errors) < 25:
                    errors[s] = "persist_signal_record returned no id"
            else:
                saved_count += 1
        except HTTPException as e:
            error_count += 1
            if len(errors) < 25:
                errors[s] = str(e.detail)
        except Exception as e:
            error_count += 1
            if len(errors) < 25:
                errors[s] = f"{type(e).__name__}: {e}"

        if sleep_ms > 0:
            time.sleep(float(sleep_ms) / 1000.0)

    return {
        "status": "ok",
        "as_of": req.as_of,
        "lookback": req.lookback,
        "universe_count": len(symbols),
        "saved_count": saved_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "errors": errors,
    }


@app.post("/db/save_portfolio_backtest")
def db_save_portfolio_backtest(req: PortfolioBacktestRequest) -> Dict[str, Any]:
    if not DB_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="DB is disabled. Set FTIP_DB_ENABLED=1 and DATABASE_URL.",
        )

    try:
        out = backtest_portfolio(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"backtest_portfolio failed: {type(e).__name__}: {e}",
        )

    try:
        row_id = persist_portfolio_backtest_record(req, out)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"save_portfolio_backtest failed: {type(e).__name__}: {e}",
        )

    return {"status": "ok", "id": row_id, "perf": out.model_dump(exclude_none=True)}


DEFAULT_PROSPERITY_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "TSLA",
    "GOOGL",
    "META",
    "BRK.B",
    "JPM",
    "XOM",
]


@app.post("/db/universe/upsert")
def db_universe_upsert(req: UniverseDbUpsertRequest) -> Dict[str, Any]:
    if not DB_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="DB is disabled. Set FTIP_DB_ENABLED=1 and DATABASE_URL.",
        )

    try:
        received, upserted = db.upsert_universe(req.symbols, req.source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"universe_upsert failed: {type(e).__name__}: {e}"
        )

    return {"status": "ok", "received": received, "upserted": upserted}


@app.get("/db/universe")
def db_universe(active_only: bool = Query(True)) -> Dict[str, Any]:
    if not DB_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="DB is disabled. Set FTIP_DB_ENABLED=1 and DATABASE_URL.",
        )

    try:
        symbols = db.get_universe(active_only=active_only)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"universe_get failed: {type(e).__name__}: {e}"
        )

    return {"count": len(symbols), "symbols": symbols}


@app.post("/db/universe/load_default")
def db_universe_load_default() -> Dict[str, Any]:
    if not DB_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="DB is disabled. Set FTIP_DB_ENABLED=1 and DATABASE_URL.",
        )

    try:
        pool = db.get_pool()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB pool not available: {e}")

    try:
        with pool.connection(timeout=10) as conn:
            with conn.cursor() as cur:
                db.set_statement_timeout(cur, 5000)
                params = [
                    (sym, True, "default_top1000_seed")
                    for sym in DEFAULT_PROSPERITY_UNIVERSE
                ]
                cur.executemany(
                    """
                    INSERT INTO prosperity_universe(symbol, active, source)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (symbol)
                    DO UPDATE SET active=EXCLUDED.active, source=EXCLUDED.source
                    """,
                    params,
                )
                conn.commit()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"universe_load_default failed: {type(e).__name__}: {e}",
        )

    return {"status": "ok", "upserted": len(DEFAULT_PROSPERITY_UNIVERSE)}


@app.get("/version")
def version() -> Dict[str, Any]:
    return {"railway_git_commit_sha": _git_sha(), "railway_environment": _railway_env()}


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


@app.post("/run_backtest")
def run_backtest(req: BacktestRequest) -> Dict[str, Any]:
    if req.data and len(req.data) > 0:
        candles = req.data
    else:
        if not (req.symbol and req.from_date and req.to_date):
            raise HTTPException(
                status_code=400,
                detail="Provide either `data` OR (`symbol`, `from_date`, `to_date`).",
            )
        candles = massive_fetch_daily_bars(req.symbol, req.from_date, req.to_date)

    summary = run_backtest_core(candles, req.include_equity_curve, req.include_returns)
    return {"backtest_summary": summary.model_dump(exclude_none=True)}


@app.get("/signal")
def signal(
    symbol: str = Query(...),
    as_of: str = Query(..., description="YYYY-MM-DD"),
    lookback: int = Query(252, ge=30, le=2000),
) -> Dict[str, Any]:
    out = compute_signal_for_symbol(symbol, as_of, lookback)
    return out.model_dump(exclude_none=True)


@app.post("/signals")
def signals(req: SignalsRequest) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}

    for sym in req.symbols:
        s = (sym or "").strip().upper()
        if not s:
            continue
        try:
            out = compute_signal_for_symbol(s, req.as_of, req.lookback)
            results[s] = out.model_dump(exclude_none=True)
        except HTTPException as e:
            errors[s] = {"status_code": e.status_code, "detail": e.detail}
        except Exception as e:
            errors[s] = {"status_code": 500, "detail": str(e)}

    # ---- Phase 4: persist signal run + observations (best-effort) ----
    run_id = db_insert_signal_run(req.as_of, req.lookback, _score_mode())
    if run_id:
        for _sym, payload in results.items():
            db_insert_signal_observation(run_id, payload)

    return {
        "as_of": req.as_of,
        "lookback": req.lookback,
        "count_ok": len(results),
        "count_error": len(errors),
        "results": results,
        "errors": errors,
        "persisted": bool(run_id is not None),
        "signal_run_id": run_id,
    }


@app.post("/portfolio_signals")
def portfolio_signals(req: PortfolioRequest) -> Dict[str, Any]:
    knobs = _portfolio_knobs(req)
    max_w = knobs["max_weight"]
    hold_mult = knobs["hold_multiplier"]
    min_conf = knobs["min_confidence"]

    per: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}
    for sym in req.symbols:
        s = (sym or "").strip().upper()
        if not s:
            continue
        try:
            out = compute_signal_for_symbol(s, req.as_of, req.lookback).model_dump(
                exclude_none=True
            )
            per[s] = out
        except HTTPException as e:
            errors[s] = {"status_code": e.status_code, "detail": e.detail}
        except Exception as e:
            errors[s] = {"status_code": 500, "detail": str(e)}

    raw: Dict[str, float] = {}
    for s, sig in per.items():
        rw = _raw_weight_from_signal(
            sig, hold_mult=hold_mult, min_conf=min_conf, allow_shorts=req.allow_shorts
        )
        if rw != 0.0:
            raw[s] = float(rw)

    if not raw:
        return {
            "as_of": req.as_of,
            "lookback": req.lookback,
            "knobs": {
                "max_weight": max_w,
                "hold_multiplier": hold_mult,
                "min_confidence": min_conf,
                "allow_shorts": req.allow_shorts,
            },
            "portfolio": {},
            "cash": 1.0,
            "count_ok": len(per),
            "count_error": len(errors),
            "errors": errors,
            "signals": per,
            "notes": ["No positions met min_confidence / signal criteria. 100% cash."],
        }

    if req.allow_shorts:
        portfolio, cash = _normalize_allow_shorts_with_caps(raw, max_w)
    else:
        portfolio, cash = _normalize_long_only_with_caps(
            {k: v for k, v in raw.items() if v > 0.0}, max_w
        )

    return {
        "as_of": req.as_of,
        "lookback": req.lookback,
        "knobs": {
            "max_weight": max_w,
            "hold_multiplier": hold_mult,
            "min_confidence": min_conf,
            "allow_shorts": req.allow_shorts,
        },
        "portfolio": portfolio,
        "cash": cash,
        "count_ok": len(per),
        "count_error": len(errors),
        "errors": errors,
        "signals": per,
    }


@app.post("/portfolio_backtest")
def portfolio_backtest(req: PortfolioBacktestRequest) -> Dict[str, Any]:
    out = backtest_portfolio(req)  # never returns None; raises HTTPException on failure

    # ---- Phase 4: persist portfolio backtest (best-effort) ----
    db_insert_portfolio_backtest(req, out)

    return out.model_dump(exclude_none=True)


@app.post("/walk_forward")
def walk_forward(req: WalkForwardRequest) -> Dict[str, Any]:
    rows = walk_forward_table(
        req.symbol, req.from_date, req.to_date, req.lookback, req.horizons
    )
    return {
        "symbol": req.symbol.upper(),
        "from_date": req.from_date,
        "to_date": req.to_date,
        "lookback": req.lookback,
        "horizons": req.horizons,
        "rows_sample": rows[:50],
        "row_count": len(rows),
    }


@app.post("/calibrate")
def calibrate(req: CalibrateRequest) -> Dict[str, Any]:
    rows = walk_forward_table(
        req.symbol, req.from_date, req.to_date, req.lookback, req.horizons
    )
    cal_core = calibrate_thresholds(rows, req.optimize_horizon, req.min_trades_per_side)

    env_payload = {
        "created_at_utc": cal_core["created_at_utc"],
        "symbol": req.symbol.upper(),
        "train_range": {
            "from_date": req.from_date,
            "to_date": req.to_date,
            "eval_start": req.from_date,
            "eval_end": req.to_date,
        },
        "optimize_horizon": cal_core["optimize_horizon"],
        "thresholds_by_regime": cal_core["thresholds_by_regime"],
        "diagnostics": cal_core["diagnostics"],
    }
    env_value = json.dumps(env_payload, separators=(",", ":"))

    sym = (req.symbol or "").strip().upper()
    existing_map: Dict[str, Any] = {}
    raw_map = _env("FTIP_CALIBRATION_JSON_MAP")
    if raw_map:
        try:
            tmp = json.loads(raw_map)
            if isinstance(tmp, dict):
                existing_map = tmp
        except Exception:
            existing_map = {}

    existing_map[sym] = env_payload
    env_var_value_map = json.dumps(existing_map, separators=(",", ":"))

    # ---- Phase 4: persist calibration snapshot (best-effort) ----
    db_insert_calibration_snapshot(sym, env_payload)

    return {
        "calibration": env_payload,
        "env_var_name": "FTIP_CALIBRATION_JSON",
        "env_var_value": env_value,
        "next_step": "Paste env_var_value into Railway Variable FTIP_CALIBRATION_JSON then redeploy/restart.",
        "env_var_name_map": "FTIP_CALIBRATION_JSON_MAP",
        "env_var_value_map": env_var_value_map,
        "next_step_map": "Paste env_var_value_map into Railway Variable FTIP_CALIBRATION_JSON_MAP then redeploy/restart.",
    }


# =============================================================================
# Phase 4: Universe endpoints (Top 1,000)
# =============================================================================


@app.post("/universe/upsert")
def universe_upsert(req: UniverseUpsertRequest) -> Dict[str, Any]:
    if not DB_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="DB is disabled. Set FTIP_DB_ENABLED=1 and DATABASE_URL.",
        )

    syms = [(s or "").strip().upper() for s in (req.symbols or [])]
    syms = [s for s in syms if s]
    if not syms:
        raise HTTPException(status_code=400, detail="symbols list is empty.")

    try:
        pool = db.get_pool()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB pool not available: {e}")

    try:
        with pool.connection(timeout=10) as conn:
            with conn.cursor() as cur:
                db.set_statement_timeout(cur, 5000)
                cur.execute(
                    """
                    INSERT INTO universe_snapshot (as_of_date, name, source)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (as_of_date, name)
                    DO UPDATE SET source=EXCLUDED.source
                    RETURNING id
                    """,
                    (req.as_of_date, req.name, req.source),
                )
                snapshot_id = cur.fetchone()[0]

                cur.execute(
                    "DELETE FROM universe_member WHERE snapshot_id=%s", (snapshot_id,)
                )

                rows = [(snapshot_id, s, i) for i, s in enumerate(syms, start=1)]
                cur.executemany(
                    "INSERT INTO universe_member (snapshot_id, symbol, rank) VALUES (%s,%s,%s)",
                    rows,
                )
                conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"universe_upsert failed: {type(e).__name__}: {e}"
        )

    return {
        "status": "ok",
        "as_of_date": req.as_of_date,
        "name": req.name,
        "count": len(syms),
    }


@app.get("/universe/top")
def universe_top(
    as_of_date: str = Query(..., description="YYYY-MM-DD"),
    name: str = Query("TOP1000_US"),
    limit: int = Query(1000, ge=1, le=5000),
) -> Dict[str, Any]:
    if not DB_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="DB is disabled. Set FTIP_DB_ENABLED=1 and DATABASE_URL.",
        )

    try:
        pool = db.get_pool()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB pool not available: {e}")

    try:
        with pool.connection(timeout=10) as conn:
            with conn.cursor() as cur:
                db.set_statement_timeout(cur, 5000)
                cur.execute(
                    "SELECT id FROM universe_snapshot WHERE as_of_date=%s AND name=%s",
                    (as_of_date, name),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404, detail="Universe snapshot not found."
                    )

                sid = row[0]
                cur.execute(
                    """
                    SELECT symbol, rank
                    FROM universe_member
                    WHERE snapshot_id=%s
                    ORDER BY rank ASC
                    LIMIT %s
                    """,
                    (sid, int(limit)),
                )
                items = [{"symbol": r[0], "rank": int(r[1])} for r in cur.fetchall()]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"universe_top failed: {type(e).__name__}: {e}"
        )

    return {"as_of_date": as_of_date, "name": name, "count": len(items), "items": items}
