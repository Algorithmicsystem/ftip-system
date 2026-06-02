"""Session 18: Realized signal P&L tracker.

Computes forward-return attribution per signal/date by pairing stored signals
from prosperity_signals_daily with prices from prosperity_daily_bars.

POST /jobs/pnl/compute  — compute & store P&L rows for a given as_of_date
GET  /jobs/pnl/summary  — load aggregate attribution stats
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api import db, security

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS: List[int] = [5, 21, 63]


# ---------------------------------------------------------------------------
# 1.10 De Prado Triple Barrier Labeling
# ---------------------------------------------------------------------------

def compute_triple_barrier_label(
    price_series: list,
    entry_price: float,
    profit_factor: float = 0.02,
    stop_factor: float = 0.01,
    horizon_days: int = 5,
) -> int:
    """De Prado Triple Barrier Labeling.

    Grounded in Advances in Financial Machine Learning (López de Prado).

    Returns:
        1  — profit target hit first (upper barrier)
       -1  — stop loss hit first (lower barrier)
        0  — time stop (horizon elapsed without barrier touch)
    """
    if not price_series or not entry_price or float(entry_price) <= 0:
        return 0

    entry = float(entry_price)
    profit_barrier = entry * (1.0 + profit_factor)
    stop_barrier = entry * (1.0 - stop_factor)

    prices = [float(p) for p in price_series[:horizon_days] if p is not None]
    for price in prices:
        if price >= profit_barrier:
            return 1
        if price <= stop_barrier:
            return -1
    return 0  # time stop


# ---------------------------------------------------------------------------
# P&L computation
# ---------------------------------------------------------------------------

def compute_signal_pnl(
    as_of_date: dt.date,
    horizons: List[int] = DEFAULT_HORIZONS,
) -> List[Dict[str, Any]]:
    """
    For each horizon H, pairs signals from (as_of_date - H calendar days)
    with prices on both the signal date and as_of_date.

    Returns a list of dicts ready for upsert into signal_pnl_daily.
    Rows with missing price data still appear (return_pct=None) so the
    hit-rate denominator stays honest.
    """
    if not db.db_read_enabled():
        return []

    rows: List[Dict[str, Any]] = []

    for h in horizons:
        signal_date = as_of_date - dt.timedelta(days=h)

        signals = db.safe_fetchall(
            """
            SELECT
                psd.symbol,
                psd.signal,
                psd.score,
                COALESCE(psd.lookback, 252)       AS lookback,
                asd.deployable_alpha_utility,
                asd.regime_label,
                asd.overall_confidence
            FROM prosperity_signals_daily psd
            LEFT JOIN axiom_scores_daily asd
                ON  asd.symbol     = psd.symbol
                AND asd.as_of_date = psd.as_of
            WHERE psd.as_of = %s
            """,
            (signal_date,),
        )
        if not signals:
            continue

        symbols = list({r[0] for r in signals})

        bars = db.safe_fetchall(
            """
            SELECT symbol, date, close
            FROM prosperity_daily_bars
            WHERE symbol = ANY(%s) AND date IN (%s, %s)
            """,
            (symbols, signal_date, as_of_date),
        )

        price_map: Dict[str, Dict[Any, float]] = {}
        for sym, date, close in bars:
            price_map.setdefault(sym, {})[date] = float(close)

        for sym, signal, score, lookback, dau, regime, _conf in signals:
            p0 = price_map.get(sym, {}).get(signal_date)
            p1 = price_map.get(sym, {}).get(as_of_date)

            ret_pct: Optional[float] = None
            hit: Optional[bool] = None
            triple_barrier_outcome: Optional[int] = None
            if p0 and p1 and p0 > 0:
                ret_pct = round(((p1 / p0) - 1.0) * 100.0, 4)
                if signal == "BUY":
                    hit = ret_pct > 0
                elif signal == "SELL":
                    hit = ret_pct < 0
                # Triple barrier labeling using available prices as path proxy
                _price_path = [v for v in [p0, p1] if v is not None]
                triple_barrier_outcome = compute_triple_barrier_label(
                    price_series=_price_path,
                    entry_price=p0,
                    profit_factor=0.02,
                    stop_factor=0.01,
                    horizon_days=h,
                )

            rows.append(
                {
                    "symbol": sym,
                    "signal_date": signal_date,
                    "horizon_days": h,
                    "lookback": int(lookback or 252),
                    "signal_label": signal,
                    "signal_score": float(score) if score is not None else None,
                    "dau": float(dau) if dau is not None else None,
                    "regime_label": regime,
                    "price_at_signal": p0,
                    "horizon_date": as_of_date,
                    "price_at_horizon": p1,
                    "return_pct": ret_pct,
                    "hit": hit,
                    "triple_barrier_outcome": triple_barrier_outcome,
                    "computed_at": as_of_date,
                }
            )

    return rows


def store_signal_pnl(rows: List[Dict[str, Any]]) -> int:
    """Upsert P&L rows; returns number of rows written."""
    if not db.db_write_enabled() or not rows:
        return 0

    written = 0
    for r in rows:
        try:
            db.exec1(
                """
                INSERT INTO signal_pnl_daily (
                    symbol, signal_date, horizon_days, lookback, signal_label,
                    signal_score, dau, regime_label, price_at_signal,
                    horizon_date, price_at_horizon, return_pct, hit,
                    triple_barrier_outcome, computed_at, updated_at
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (symbol, signal_date, horizon_days)
                DO UPDATE SET
                    price_at_signal        = EXCLUDED.price_at_signal,
                    price_at_horizon       = EXCLUDED.price_at_horizon,
                    horizon_date           = EXCLUDED.horizon_date,
                    return_pct             = EXCLUDED.return_pct,
                    hit                    = EXCLUDED.hit,
                    triple_barrier_outcome = EXCLUDED.triple_barrier_outcome,
                    computed_at            = EXCLUDED.computed_at,
                    updated_at             = now()
                """,
                (
                    r["symbol"], r["signal_date"], r["horizon_days"], r["lookback"],
                    r["signal_label"], r["signal_score"], r["dau"], r["regime_label"],
                    r["price_at_signal"], r["horizon_date"], r["price_at_horizon"],
                    r["return_pct"], r["hit"], r.get("triple_barrier_outcome"),
                    r["computed_at"],
                ),
            )
            written += 1
        except Exception as exc:
            logger.warning("pnl.store_failed symbol=%s error=%s", r.get("symbol"), exc)

    return written


# ---------------------------------------------------------------------------
# Pure-Python Spearman rank correlation
# ---------------------------------------------------------------------------

def _rank(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    for rank_pos, (orig_idx, _) in enumerate(indexed, 1):
        ranks[orig_idx] = float(rank_pos)
    return ranks


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    return float(cov / denom) if denom > 0 else 0.0


def spearman_ic(scores: List[float], returns: List[float]) -> float:
    return _pearson(_rank(scores), _rank(returns))


# ---------------------------------------------------------------------------
# Summary / analytics
# ---------------------------------------------------------------------------

def load_pnl_summary(
    as_of_date: dt.date,
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """Return attribution stats over the last `lookback_days` calendar days."""
    if not db.db_read_enabled():
        return {"status": "db_disabled"}

    since = as_of_date - dt.timedelta(days=lookback_days)

    rows = db.safe_fetchall(
        """
        SELECT
            symbol, signal_date, horizon_days, signal_label,
            signal_score, dau, regime_label,
            return_pct, hit, computed_at
        FROM signal_pnl_daily
        WHERE signal_date >= %s
        ORDER BY signal_date DESC, symbol
        """,
        (since,),
    )

    total = len(rows)
    rated = [r for r in rows if r[7] is not None]

    empty = {
        "n": 0,
        "avg_return_pct": None,
        "hit_rate": None,
        "win_pct": None,
    }

    def _stats(subset) -> Dict[str, Any]:
        if not subset:
            return dict(empty)
        rets = [float(r[7]) for r in subset]
        hits = [r[8] for r in subset if r[8] is not None]
        hr = sum(1 for x in hits if x) / len(hits) if hits else None
        return {
            "n": len(subset),
            "avg_return_pct": round(sum(rets) / len(rets), 3),
            "hit_rate": round(hr, 3) if hr is not None else None,
            "win_pct": round(hr * 100, 1) if hr is not None else None,
        }

    by_horizon: Dict[str, Any] = {}
    for h in DEFAULT_HORIZONS:
        h_rows = [r for r in rated if r[2] == h]
        if h_rows:
            by_horizon[str(h)] = _stats(h_rows)

    by_signal: Dict[str, Any] = {}
    for label in ("BUY", "SELL", "HOLD"):
        s_rows = [r for r in rated if r[3] == label]
        if s_rows:
            by_signal[label] = _stats(s_rows)

    # Spearman IC: sign-adjusted score vs. return (BUY/SELL only)
    ic_candidates = [r for r in rated if r[3] in ("BUY", "SELL") and r[4] is not None]
    ic: Optional[float] = None
    if len(ic_candidates) >= 5:
        # Flip SELL scores so higher score always predicts positive return
        adj_scores = [
            float(r[4]) if r[3] == "BUY" else -float(r[4])
            for r in ic_candidates
        ]
        rets = [float(r[7]) for r in ic_candidates]
        ic = round(spearman_ic(adj_scores, rets), 4)

    recent = [
        {
            "symbol": r[0],
            "signal_date": r[1].isoformat() if hasattr(r[1], "isoformat") else str(r[1]),
            "horizon_days": r[2],
            "signal_label": r[3],
            "return_pct": round(float(r[7]), 2),
            "hit": r[8],
            "regime_label": r[6],
        }
        for r in rated[:20]
    ]

    return {
        "status": "ok",
        "as_of_date": as_of_date.isoformat(),
        "lookback_days": lookback_days,
        "total_rows": total,
        "rows_with_return": len(rated),
        "by_horizon": by_horizon,
        "by_signal": by_signal,
        "spearman_ic": ic,
        "recent": recent,
    }


# ---------------------------------------------------------------------------
# Attribution by regime / sector / signal_version
# ---------------------------------------------------------------------------

def load_pnl_attribution(
    as_of_date: dt.date,
    *,
    lookback_days: int = 90,
    horizon_days: int = 21,
    group_by: str = "regime",   # "regime" | "sector" | "signal_version"
) -> Dict[str, Any]:
    """Group realised P&L rows into attribution buckets.

    Returns win_rate, avg_return, max_drawdown (worst single return), and n
    per bucket. Only rows with a resolved return_pct are included in stats;
    n counts all rows in the bucket.
    """
    if not db.db_read_enabled():
        return {"status": "db_disabled", "group_by": group_by, "buckets": []}

    since = as_of_date - dt.timedelta(days=lookback_days)

    allowed_groups = {"regime", "sector", "signal_version"}
    if group_by not in allowed_groups:
        group_by = "regime"

    rows = db.safe_fetchall(
        """
        SELECT
            p.regime_label,
            COALESCE(m.sector, 'Unknown')                       AS sector,
            COALESCE(psd.signal_version, 'unknown')             AS signal_version,
            p.return_pct,
            p.hit
        FROM signal_pnl_daily p
        LEFT JOIN market_symbols m
            ON m.symbol = p.symbol
        LEFT JOIN prosperity_signals_daily psd
            ON psd.symbol = p.symbol AND psd.as_of = p.signal_date
        WHERE p.signal_date >= %s
          AND p.horizon_days = %s
        """,
        (since, horizon_days),
    )

    buckets: Dict[str, Dict[str, Any]] = {}

    for regime, sector, sig_ver, ret_pct, hit in rows:
        if group_by == "regime":
            key = str(regime or "unknown")
        elif group_by == "sector":
            key = str(sector or "Unknown")
        else:
            key = str(sig_ver or "unknown")

        if key not in buckets:
            buckets[key] = {"n": 0, "rets": [], "hits": []}
        b = buckets[key]
        b["n"] += 1
        if ret_pct is not None:
            b["rets"].append(float(ret_pct))
        if hit is not None:
            b["hits"].append(bool(hit))

    result_buckets = []
    for label, b in sorted(buckets.items()):
        rets = b["rets"]
        hits = b["hits"]
        avg_return  = round(sum(rets) / len(rets), 3) if rets else None
        max_drawdown = round(min(rets), 3) if rets else None
        win_rate    = round(sum(1 for h in hits if h) / len(hits), 3) if hits else None
        result_buckets.append({
            "bucket": label,
            "n": b["n"],
            "avg_return_pct": avg_return,
            "win_rate": win_rate,
            "max_drawdown_pct": max_drawdown,
        })

    # Sort by win_rate desc, then avg_return desc
    result_buckets.sort(
        key=lambda x: (x["win_rate"] or 0.0, x["avg_return_pct"] or 0.0),
        reverse=True,
    )

    return {
        "status": "ok",
        "as_of_date": as_of_date.isoformat(),
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "group_by": group_by,
        "bucket_count": len(result_buckets),
        "buckets": result_buckets,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class PnLComputeRequest(BaseModel):
    as_of_date: Optional[str] = None
    horizons: List[int] = Field(default_factory=lambda: list(DEFAULT_HORIZONS))
    store: bool = True


class PnLSummaryRequest(BaseModel):
    as_of_date: Optional[str] = None
    lookback_days: int = Field(default=90, ge=7, le=365)


def _resolve_date(raw: Optional[str]) -> dt.date:
    if raw:
        try:
            return dt.date.fromisoformat(raw)
        except ValueError:
            pass
    return dt.date.today() - dt.timedelta(days=1)


@router.post("/pnl/compute")
def pnl_compute(req: PnLComputeRequest) -> Dict[str, Any]:
    as_of = _resolve_date(req.as_of_date)
    horizons = [h for h in req.horizons if 1 <= h <= 252]
    if not horizons:
        horizons = list(DEFAULT_HORIZONS)

    rows = compute_signal_pnl(as_of, horizons)
    stored = store_signal_pnl(rows) if req.store and db.db_write_enabled() else 0

    with_return = sum(1 for r in rows if r["return_pct"] is not None)
    hits = sum(1 for r in rows if r.get("hit") is True)

    logger.info(
        "pnl.compute as_of=%s rows=%d with_return=%d hits=%d stored=%d",
        as_of, len(rows), with_return, hits, stored,
    )

    return {
        "status": "ok",
        "as_of_date": as_of.isoformat(),
        "horizons": horizons,
        "total_rows": len(rows),
        "rows_with_return": with_return,
        "hits": hits,
        "stored": stored,
        "rows": rows,
    }


@router.get("/pnl/summary")
def pnl_summary(
    as_of_date: Optional[str] = Query(default=None),
    lookback_days: int = Query(default=90, ge=7, le=365),
) -> Dict[str, Any]:
    as_of = _resolve_date(as_of_date)
    return load_pnl_summary(as_of, lookback_days)


@router.get("/pnl/attribution")
def pnl_attribution(
    as_of_date: Optional[str] = Query(default=None),
    lookback_days: int = Query(default=90, ge=7, le=365),
    horizon_days: int = Query(default=21, ge=1, le=252),
    group_by: str = Query(default="regime", pattern="^(regime|sector|signal_version)$"),
) -> Dict[str, Any]:
    as_of = _resolve_date(as_of_date)
    return load_pnl_attribution(
        as_of,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        group_by=group_by,
    )


# ---------------------------------------------------------------------------
# Calibration auto-update
# ---------------------------------------------------------------------------

_CAL_VERSION = "pnl_auto_v1"


def compute_calibration_snapshot(
    as_of_date: dt.date,
    *,
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """Derive calibration stats from signal_pnl_daily and return a snapshot payload.

    Computes overall hit_rate and per-bucket (signal_label × horizon_days) stats.
    Ready to upsert into axiom_calibration_snapshots.
    """
    if not db.db_read_enabled():
        return {"status": "db_disabled"}

    since = as_of_date - dt.timedelta(days=lookback_days)

    rows = db.safe_fetchall(
        """
        SELECT signal_label, horizon_days, regime_label, return_pct, hit
        FROM signal_pnl_daily
        WHERE signal_date >= %s
          AND return_pct IS NOT NULL
        """,
        (since,),
    )

    if not rows:
        return {"status": "no_data", "as_of_date": as_of_date.isoformat()}

    all_hits = [bool(r[4]) for r in rows if r[4] is not None]
    overall_hit_rate = round(sum(all_hits) / len(all_hits), 4) if all_hits else None
    overall_avg_return = round(sum(float(r[3]) for r in rows) / len(rows), 4)

    # Per (signal_label, horizon_days) bucket
    buckets: Dict[str, Dict] = {}
    for sig, h, regime, ret_pct, hit in rows:
        key = f"{sig or 'UNKNOWN'}_{h}d"
        if key not in buckets:
            buckets[key] = {"signal_label": sig, "horizon_days": h, "rets": [], "hits": []}
        b = buckets[key]
        b["rets"].append(float(ret_pct))
        if hit is not None:
            b["hits"].append(bool(hit))

    bucket_stats = []
    for key, b in sorted(buckets.items()):
        hits = b["hits"]
        rets = b["rets"]
        hr = round(sum(1 for h in hits if h) / len(hits), 4) if hits else None
        bucket_stats.append({
            "bucket_key": key,
            "signal_label": b["signal_label"],
            "horizon_days": b["horizon_days"],
            "n": len(rets),
            "hit_rate": hr,
            "avg_return_pct": round(sum(rets) / len(rets), 4),
        })

    payload = {
        "overall_hit_rate": overall_hit_rate,
        "overall_avg_return_pct": overall_avg_return,
        "sample_count": len(rows),
        "lookback_days": lookback_days,
        "as_of_date": as_of_date.isoformat(),
        "calibration_version": _CAL_VERSION,
        "buckets": bucket_stats,
    }

    return {"status": "ok", "payload": payload}


def store_calibration_snapshot(
    as_of_date: dt.date,
    payload: Dict[str, Any],
    *,
    horizon_label: str = "auto",
) -> bool:
    """Upsert a calibration snapshot row. Returns True on success."""
    if not db.db_write_enabled():
        return False
    snapshot_key = f"pnl_auto_v1:{as_of_date.isoformat()}:{horizon_label}"
    try:
        db.safe_execute(
            """
            INSERT INTO axiom_calibration_snapshots
                (snapshot_key, as_of_date, horizon_label, framework_version, payload)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (snapshot_key) DO UPDATE SET
                payload    = EXCLUDED.payload,
                updated_at = now()
            """,
            (snapshot_key, as_of_date, horizon_label, _CAL_VERSION, __import__("json").dumps(payload)),
        )
        return True
    except Exception as exc:
        logger.warning("calibration.store_failed error=%s", exc)
        return False


class CalibrationUpdateRequest(BaseModel):
    as_of_date: Optional[str] = None
    lookback_days: int = Field(default=90, ge=7, le=365)
    store: bool = True


@router.post("/calibration/daily")
def calibration_daily(req: CalibrationUpdateRequest) -> Dict[str, Any]:
    """Compute calibration stats from realized P&L and upsert into axiom_calibration_snapshots."""
    as_of = _resolve_date(req.as_of_date)
    result = compute_calibration_snapshot(as_of, lookback_days=req.lookback_days)

    if result.get("status") not in ("ok",):
        return result

    stored = False
    if req.store:
        stored = store_calibration_snapshot(as_of, result["payload"])

    logger.info(
        "calibration.daily as_of=%s n=%d hit_rate=%s stored=%s",
        as_of,
        result["payload"].get("sample_count", 0),
        result["payload"].get("overall_hit_rate"),
        stored,
    )

    return {
        "status": "ok",
        "as_of_date": as_of.isoformat(),
        "stored": stored,
        **result["payload"],
    }
