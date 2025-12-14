from __future__ import annotations

import os
import math
import json
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


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


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------

class Candle(BaseModel):
    timestamp: str  # ISO date "YYYY-MM-DD" (daily)
    close: float
    volume: Optional[float] = None

    # optional extra features (supported for future use)
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

    # calibration diagnostics
    calibration_loaded: bool = False
    calibration_meta: Optional[Dict[str, Any]] = None


class SignalsRequest(BaseModel):
    symbols: List[str]
    as_of: str
    lookback: int = 252


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
    min_trades_per_side: int = 15  # require enough BUY/SELL signals to be meaningful


# NEW: Portfolio request model
class PortfolioRequest(BaseModel):
    symbols: List[str]
    as_of: str
    lookback: int = 252

    # Optional knobs (can also be overridden via env vars)
    max_weight: Optional[float] = None          # cap per position
    hold_multiplier: Optional[float] = None     # HOLD gets smaller weight than BUY
    min_confidence: Optional[float] = None      # ignore weak signals
    allow_shorts: bool = False                  # default False (signals only; no short weights)


# ------------------------------------------------------------------------------
# Massive (Polygon legacy) REST API client
# ------------------------------------------------------------------------------

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
# ------------------------------------------------------------------------------

MASSIVE_S3_ENDPOINT = _env("MASSIVE_S3_ENDPOINT", "https://files.massive.com") or "https://files.massive.com"
MASSIVE_S3_ACCESS_KEY_ID = _env("MASSIVE_S3_ACCESS_KEY_ID")
MASSIVE_S3_SECRET_ACCESS_KEY = _env("MASSIVE_S3_SECRET_ACCESS_KEY")


# ------------------------------------------------------------------------------
# Math helpers
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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ------------------------------------------------------------------------------
# Backtest engine (simple buy-and-hold placeholder)
# ------------------------------------------------------------------------------

def run_backtest_core(candles: List[Candle], include_equity: bool, include_returns: bool) -> BacktestSummary:
    closes = [c.close for c in candles]
    dates = [c.timestamp for c in candles]

    rets = _pct_change(closes)  # daily returns
    equity = [1.0]
    for r in rets[1:]:
        equity.append(equity[-1] * (1.0 + r))

    total_return = equity[-1] - 1.0

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
# Feature engine + regime detection + thresholds
# ------------------------------------------------------------------------------

def _sma(values: List[float], window: int) -> float:
    if window <= 0 or len(values) < window:
        return float("nan")
    return _mean(values[-window:])


def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
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

    feat = {
        "mom_5": mom(5),
        "mom_21": mom(21),
        "mom_63": mom(63),
        "trend_sma20_50": trend,
        "volatility_ann": float(vol_ann),
        "rsi14": float(_rsi(closes, 14)),
        "volume_z20": float(_zscore_last(vols, 20)),
        "last_close": float(closes[-1]),
    }
    return feat


def detect_regime(features: Dict[str, float]) -> str:
    vol = features.get("volatility_ann", 0.0)
    trend = abs(features.get("trend_sma20_50", 0.0))

    if vol >= 0.45:
        return "HIGH_VOL"
    if trend >= 0.05:
        return "TRENDING"
    return "CHOPPY"


def score_from_features(features: Dict[str, float]) -> Tuple[float, List[str]]:
    notes: List[str] = []

    rsi = features["rsi14"]
    rsi_sig = _clamp((rsi - 50.0) / 25.0, -1.0, 1.0)

    mom = 0.4 * features["mom_21"] + 0.6 * features["mom_63"]
    mom_sig = _clamp(mom / 0.25, -1.0, 1.0)

    trend_sig = _clamp(features["trend_sma20_50"] / 0.10, -1.0, 1.0)
    if features["trend_sma20_50"] > 0:
        notes.append("Short-term trend (SMA20 vs SMA50) is positive.")
    elif features["trend_sma20_50"] < 0:
        notes.append("Short-term trend (SMA20 vs SMA50) is negative.")

    volz = features["volume_z20"]
    vol_sig = _clamp(volz / 3.0, -1.0, 1.0)

    vola = features["volatility_ann"]
    vola_pen = _clamp((vola - 0.25) / 0.50, 0.0, 0.5)  # 0..0.5

    raw = 0.45 * mom_sig + 0.30 * trend_sig + 0.20 * rsi_sig + 0.05 * vol_sig
    score = _clamp(raw * (1.0 - vola_pen), -1.0, 1.0)

    if vola >= 0.45:
        notes.append("Volatility is elevated; treat signal with caution.")
    return float(score), notes


# ------------------------------------------------------------------------------
# Calibration (optional) via env var
# ------------------------------------------------------------------------------

def _load_calibration() -> Tuple[bool, Optional[Dict[str, Any]]]:
    s = _env("FTIP_CALIBRATION_JSON")
    if not s:
        return False, None
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return False, None
        return True, obj
    except Exception:
        return False, None


def _thresholds_for_regime(regime: str, calibration: Optional[Dict[str, Any]]) -> Dict[str, float]:
    defaults = {
        "TRENDING": {"buy": 0.2, "sell": -0.2},
        "CHOPPY": {"buy": 0.3, "sell": -0.3},
        "HIGH_VOL": {"buy": 0.45, "sell": -0.45},
    }
    if not calibration:
        return defaults.get(regime, {"buy": 0.3, "sell": -0.3})

    thr_by_reg = calibration.get("thresholds_by_regime") or {}
    if regime in thr_by_reg and isinstance(thr_by_reg[regime], dict):
        b = float(thr_by_reg[regime].get("buy", defaults.get(regime, {"buy": 0.3})["buy"]))
        s = float(thr_by_reg[regime].get("sell", defaults.get(regime, {"sell": -0.3})["sell"]))
        return {"buy": b, "sell": s}

    return defaults.get(regime, {"buy": 0.3, "sell": -0.3})


# ------------------------------------------------------------------------------
# Signal computation (single symbol)
# ------------------------------------------------------------------------------

def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _filter_upto(candles: List[Candle], as_of: str) -> List[Candle]:
    cutoff = _parse_date(as_of)
    out = []
    for c in candles:
        d = _parse_date(c.timestamp)
        if d <= cutoff:
            out.append(c)
    return out


def compute_signal_for_symbol(symbol: str, as_of: str, lookback: int) -> SignalResponse:
    as_of_d = _parse_date(as_of)
    from_guess = (as_of_d - dt.timedelta(days=900)).isoformat()
    candles_all = massive_fetch_daily_bars(symbol, from_guess, as_of)

    candles_upto = _filter_upto(candles_all, as_of)
    if len(candles_upto) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data to compute signal. Need at least 30 bars <= {as_of}, got {len(candles_upto)}.",
        )

    effective = min(lookback, len(candles_upto))
    candles = candles_upto[-effective:]

    features = compute_features(candles)
    regime = detect_regime(features)

    score, notes = score_from_features(features)

    calibration_loaded, calibration = _load_calibration()
    thresholds = _thresholds_for_regime(regime, calibration)

    sig = "HOLD"
    if score >= thresholds["buy"]:
        sig = "BUY"
    elif score <= thresholds["sell"]:
        sig = "SELL"

    conf = abs(score)
    if regime == "HIGH_VOL":
        conf *= 0.65
        notes.append("Confidence reduced due to HIGH_VOL regime.")
    if effective < lookback:
        notes.insert(0, f"Requested lookback={lookback}, but only {effective} bars available by {as_of}. Using effective_lookback={effective}.")

    if regime == "TRENDING":
        notes.append("Regime: TRENDING.")
    elif regime == "CHOPPY":
        notes.append("Regime: CHOPPY.")
    else:
        notes.append("Regime: HIGH_VOL.")

    if calibration_loaded:
        notes.append("Using calibrated thresholds from FTIP_CALIBRATION_JSON.")

    meta = None
    if calibration_loaded and calibration:
        meta = {
            "optimize_horizon": calibration.get("optimize_horizon"),
            "created_at_utc": calibration.get("created_at_utc"),
            "symbol": calibration.get("symbol"),
            "train_range": calibration.get("train_range"),
        }

    return SignalResponse(
        symbol=symbol.upper(),
        as_of=as_of,
        lookback=lookback,
        effective_lookback=effective,
        regime=regime,
        thresholds=thresholds,
        score=float(score),
        signal=sig,
        confidence=float(conf),
        features={k: float(v) for k, v in features.items()},
        notes=notes,
        calibration_loaded=bool(calibration_loaded),
        calibration_meta=meta,
    )


# ------------------------------------------------------------------------------
# Walk-forward + Calibration endpoints (training helper)
# ------------------------------------------------------------------------------

def _forward_return(closes: List[float], idx: int, horizon: int) -> Optional[float]:
    j = idx + horizon
    if j >= len(closes):
        return None
    if closes[idx] == 0:
        return None
    return float(closes[j] / closes[idx] - 1.0)


def walk_forward_table(symbol: str, from_date: str, to_date: str, lookback: int, horizons: List[int]) -> List[Dict[str, Any]]:
    candles_all = massive_fetch_daily_bars(symbol, from_date, to_date)

    closes = [c.close for c in candles_all]
    dates = [c.timestamp for c in candles_all]

    rows: List[Dict[str, Any]] = []
    for i in range(len(candles_all)):
        start = max(0, i - lookback + 1)
        window = candles_all[start:i + 1]
        if len(window) < 30:
            continue

        feats = compute_features(window)
        regime = detect_regime(feats)
        score, _notes = score_from_features(feats)

        cal_loaded, cal = _load_calibration()
        thr = _thresholds_for_regime(regime, cal)

        sig = "HOLD"
        if score >= thr["buy"]:
            sig = "BUY"
        elif score <= thr["sell"]:
            sig = "SELL"

        row = {
            "date": dates[i],
            "signal": sig,
            "score": float(score),
            "confidence": float(abs(score) * (0.65 if regime == "HIGH_VOL" else 1.0)),
            "regime": regime,
        }
        for h in horizons:
            fr = _forward_return(closes, i, h)
            row[f"fwd_ret_{h}"] = fr
        rows.append(row)

    return rows


def calibrate_thresholds(rows: List[Dict[str, Any]], optimize_horizon: int, min_trades_per_side: int) -> Dict[str, Any]:
    created = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    regimes = sorted(set(r["regime"] for r in rows))
    thresholds_by_regime: Dict[str, Dict[str, float]] = {}
    diagnostics: Dict[str, Any] = {}

    buys = [0.15, 0.20, 0.25, 0.30, 0.35, 0.45]
    sells = [-0.15, -0.20, -0.25, -0.30, -0.35, -0.45]

    key = f"fwd_ret_{optimize_horizon}"

    for reg in regimes:
        rrows = [r for r in rows if r["regime"] == reg and r.get(key) is not None]
        if not rrows:
            continue

        best = None
        best_metric = None

        for b in buys:
            for s in sells:
                if s >= 0:
                    continue
                buysig = [r for r in rrows if r["score"] >= b]
                sellsig = [r for r in rrows if r["score"] <= s]

                if len(buysig) < min_trades_per_side or len(sellsig) < min_trades_per_side:
                    continue

                buy_ret = _mean([float(r[key]) for r in buysig])
                sell_ret = _mean([float(r[key]) for r in sellsig])

                metric = buy_ret - sell_ret
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best = {"buy": float(b), "sell": float(s), "metric": float(metric), "buy_n": len(buysig), "sell_n": len(sellsig)}

        if best is None:
            if reg == "TRENDING":
                thresholds_by_regime[reg] = {"buy": 0.2, "sell": -0.2}
            elif reg == "CHOPPY":
                thresholds_by_regime[reg] = {"buy": 0.3, "sell": -0.3}
            else:
                thresholds_by_regime[reg] = {"buy": 0.45, "sell": -0.45}

            diagnostics[reg] = {"note": "No valid threshold pair met min_trades constraints; using defaults.", "row_count": len(rrows)}
        else:
            thresholds_by_regime[reg] = {"buy": best["buy"], "sell": best["sell"]}
            diagnostics[reg] = {"best": best, "row_count": len(rrows)}

    return {
        "created_at_utc": created,
        "optimize_horizon": int(optimize_horizon),
        "thresholds_by_regime": thresholds_by_regime,
        "diagnostics": diagnostics,
    }


# ------------------------------------------------------------------------------
# NEW: Portfolio construction layer
# ------------------------------------------------------------------------------

def _portfolio_knobs(req: PortfolioRequest) -> Dict[str, float]:
    max_w = req.max_weight if req.max_weight is not None else _env_float("FTIP_PORTFOLIO_MAX_WEIGHT", 0.30)
    hold_mult = req.hold_multiplier if req.hold_multiplier is not None else _env_float("FTIP_PORTFOLIO_HOLD_MULT", 0.35)
    min_conf = req.min_confidence if req.min_confidence is not None else _env_float("FTIP_PORTFOLIO_MIN_CONF", 0.10)

    # sanity clamps
    max_w = _clamp(float(max_w), 0.01, 1.0)
    hold_mult = _clamp(float(hold_mult), 0.0, 1.0)
    min_conf = _clamp(float(min_conf), 0.0, 1.0)
    return {"max_weight": max_w, "hold_multiplier": hold_mult, "min_confidence": min_conf}


def _raw_weight_from_signal(sig: Dict[str, Any], hold_mult: float, min_conf: float, allow_shorts: bool) -> float:
    signal = (sig.get("signal") or "HOLD").upper()
    conf = float(sig.get("confidence") or 0.0)
    vola = float(sig.get("features", {}).get("volatility_ann") or sig.get("features", {}).get("volatility") or sig.get("volatility_ann") or 0.0)

    # Filter weak / noisy outputs
    if conf < min_conf:
        return 0.0

    # Volatility targeting: higher vol => lower size
    vol_adj = 1.0 / (max(1e-6, vola))

    if signal == "BUY":
        return conf * vol_adj
    if signal == "HOLD":
        return conf * vol_adj * hold_mult
    if signal == "SELL":
        return (-conf * vol_adj) if allow_shorts else 0.0
    return 0.0


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
            "/signal",
            "/signals",
            "/portfolio_signals",
            "/walk_forward",
            "/calibrate",
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
    return {
        "s3_endpoint": MASSIVE_S3_ENDPOINT,
        "access_key_set": bool(MASSIVE_S3_ACCESS_KEY_ID),
        "secret_key_set": bool(MASSIVE_S3_SECRET_ACCESS_KEY),
    }


@app.post("/run_backtest")
def run_backtest(req: BacktestRequest) -> Dict[str, Any]:
    if req.data and len(req.data) > 0:
        candles = req.data
    else:
        if not (req.symbol and req.from_date and req.to_date):
            raise HTTPException(
                status_code=400,
                detail="Provide either `data` OR (`symbol`, `from_date`, `to_date`) to fetch from Massive.",
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

    return {
        "as_of": req.as_of,
        "lookback": req.lookback,
        "count_ok": len(results),
        "count_error": len(errors),
        "results": results,
        "errors": errors,
    }


# NEW: Portfolio endpoint
@app.post("/portfolio_signals")
def portfolio_signals(req: PortfolioRequest) -> Dict[str, Any]:
    knobs = _portfolio_knobs(req)
    max_w = knobs["max_weight"]
    hold_mult = knobs["hold_multiplier"]
    min_conf = knobs["min_confidence"]

    # Step 1: compute per-symbol signals (re-using your engine)
    per: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}
    for sym in req.symbols:
        s = (sym or "").strip().upper()
        if not s:
            continue
        try:
            out = compute_signal_for_symbol(s, req.as_of, req.lookback).model_dump(exclude_none=True)
            per[s] = out
        except HTTPException as e:
            errors[s] = {"status_code": e.status_code, "detail": e.detail}
        except Exception as e:
            errors[s] = {"status_code": 500, "detail": str(e)}

    # Step 2: compute raw weights (confidence * 1/vol, HOLD dampened, SELL optional)
    raw: Dict[str, float] = {}
    for s, sig in per.items():
        rw = _raw_weight_from_signal(sig, hold_mult=hold_mult, min_conf=min_conf, allow_shorts=req.allow_shorts)
        if rw != 0.0:
            raw[s] = float(rw)

    # Step 3: normalize to portfolio weights with caps
    # If shorts disabled -> long-only normalization and remainder to cash.
    portfolio: Dict[str, float] = {}
    cash = 1.0

    if not raw:
        return {
            "as_of": req.as_of,
            "lookback": req.lookback,
            "method": "confidence_vol_targeted",
            "knobs": {"max_weight": max_w, "hold_multiplier": hold_mult, "min_confidence": min_conf, "allow_shorts": req.allow_shorts},
            "portfolio": {},
            "cash": 1.0,
            "count_ok": len(per),
            "count_error": len(errors),
            "errors": errors,
            "signals": per,
            "meta": {"count_buy": 0, "count_hold": 0, "count_sell": 0},
            "notes": ["No positions met min_confidence / signal criteria. 100% cash."],
        }

    if req.allow_shorts:
        # If shorts allowed: normalize by gross exposure, keep net exposure and compute cash as 1 - net_long (clamped)
        gross = sum(abs(v) for v in raw.values())
        if gross <= 0:
            gross = 1.0
        for s, v in raw.items():
            w = v / gross
            # apply cap in absolute terms
            w = _clamp(w, -max_w, max_w)
            portfolio[s] = float(w)
        net = sum(portfolio.values())
        cash = float(_clamp(1.0 - max(net, 0.0), 0.0, 1.0))
    else:
        # Long-only: ignore negative raw (shouldn't exist), cap, renormalize, cash = leftover
        long_raw = {s: v for s, v in raw.items() if v > 0}
        total = sum(long_raw.values())
        if total <= 0:
            total = 1.0
        tmp = {}
        for s, v in long_raw.items():
            tmp[s] = min(v / total, max_w)

        # If caps caused sum < 1, redistribute remaining proportionally among uncapped
        def _sum(d): return sum(d.values())
        capped_sum = _sum(tmp)
        if capped_sum > 0:
            # renormalize to <=1
            # if everything capped and sum < 1, keep cash
            # else scale weights to sum to min(1, capped_sum)
            scale = min(1.0, capped_sum) / capped_sum
            for s, v in tmp.items():
                portfolio[s] = float(v * scale)

        cash = float(_clamp(1.0 - sum(portfolio.values()), 0.0, 1.0))

    # Meta counts
    count_buy = sum(1 for s in per.values() if (s.get("signal") == "BUY"))
    count_hold = sum(1 for s in per.values() if (s.get("signal") == "HOLD"))
    count_sell = sum(1 for s in per.values() if (s.get("signal") == "SELL"))

    return {
        "as_of": req.as_of,
        "lookback": req.lookback,
        "method": "confidence_vol_targeted",
        "knobs": {"max_weight": max_w, "hold_multiplier": hold_mult, "min_confidence": min_conf, "allow_shorts": req.allow_shorts},
        "portfolio": portfolio,
        "cash": cash,
        "count_ok": len(per),
        "count_error": len(errors),
        "errors": errors,
        "signals": per,
        "meta": {"count_buy": count_buy, "count_hold": count_hold, "count_sell": count_sell},
    }


@app.post("/walk_forward")
def walk_forward(req: WalkForwardRequest) -> Dict[str, Any]:
    rows = walk_forward_table(req.symbol, req.from_date, req.to_date, req.lookback, req.horizons)
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
    rows = walk_forward_table(req.symbol, req.from_date, req.to_date, req.lookback, req.horizons)
    cal = calibrate_thresholds(rows, req.optimize_horizon, req.min_trades_per_side)

    env_payload = {
        "created_at_utc": cal["created_at_utc"],
        "symbol": req.symbol.upper(),
        "train_range": {
            "from_date": req.from_date,
            "to_date": req.to_date,
            "eval_start": req.from_date,
            "eval_end": req.to_date,
        },
        "optimize_horizon": cal["optimize_horizon"],
        "thresholds_by_regime": cal["thresholds_by_regime"],
        "diagnostics": cal["diagnostics"],
    }
    env_value = json.dumps(env_payload, separators=(",", ":"))

    return {
        "calibration": env_payload,
        "env_var_name": "FTIP_CALIBRATION_JSON",
        "env_var_value": env_value,
        "next_step": "Paste env_var_value into Railway Variable FTIP_CALIBRATION_JSON then redeploy/restart.",
    }


@app.post("/run_scores")
def run_scores(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_scores endpoint placeholder", "received_keys": list(payload.keys())}


@app.post("/run_all")
def run_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "detail": "run_all endpoint placeholder", "received_keys": list(payload.keys())}

