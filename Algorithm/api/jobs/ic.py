"""Session 11: Information Coefficient (IC) decay pipeline.

Computes Spearman rank-correlation between AXIOM composite/engine scores and
forward returns at 5d / 21d / 63d horizons, storing daily snapshots in
signal_ic_daily. Exposes POST /jobs/ic/daily-snapshot for scheduled ingestion.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api import db, security

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZONS: Dict[str, int] = {"5d": 5, "21d": 21, "63d": 63}

# Per-symbol IC threshold for effective breadth (Grinold-Kahn)
_EFFECTIVE_BREADTH_IC_THRESHOLD = 0.02
_EFFECTIVE_BREADTH_WINDOW_DAYS = 63

# Approximate calendar-day equivalent for each trading-day horizon
# Using 365/252 ≈ 1.449 scaling
_CALENDAR_DAYS: Dict[str, int] = {k: round(v * 365.0 / 252.0) + 1 for k, v in HORIZONS.items()}

SCORE_FIELDS = [
    "composite",
    "fundamental_reality",
    "state_pricing",
    "behavioral_distortion",
    "flow_transmission",
    "liquidity_convexity",
    "critical_fragility",
    "research_integrity",
]

_MIN_SAMPLE = 5   # minimum symbols to compute a valid IC

# ICIR thresholds for ic_state classification
_ICIR_STRONG = 0.5
_ICIR_MODERATE = 0.25
_ICIR_WEAK = 0.0


# ---------------------------------------------------------------------------
# Pure computation: Spearman IC
# ---------------------------------------------------------------------------

def _rank(values: List[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # handle ties via average rank
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        if j > i + 1:
            avg = float(np.mean(np.arange(i + 1, j + 1, dtype=float)))
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_ic(scores: List[float], returns: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return (ic_value, p_value). Returns (None, None) if sample too small."""
    n = len(scores)
    if n < _MIN_SAMPLE or len(returns) != n:
        return None, None
    rx = _rank(scores)
    ry = _rank(returns)
    rx_m = rx - rx.mean()
    ry_m = ry - ry.mean()
    denom = math.sqrt(float(np.dot(rx_m, rx_m)) * float(np.dot(ry_m, ry_m)))
    if denom == 0:
        return None, None
    ic = float(np.dot(rx_m, ry_m)) / denom
    # two-tailed p-value via t approximation
    if abs(ic) >= 1.0 or n <= 2:
        p_value = 0.0 if abs(ic) >= 1.0 else None
    else:
        t = ic * math.sqrt(n - 2) / math.sqrt(1.0 - ic ** 2)
        # approximate p-value: 2 * P(T > |t|) using normal approx for df > 30
        if n > 30:
            p_value = 2.0 * (1.0 - _normal_cdf(abs(t)))
        else:
            # conservative: just store t-stat, skip p approximation for small n
            p_value = None
    return round(ic, 6), (round(p_value, 6) if p_value is not None else None)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Score extraction from axiom_scores_daily payload
# ---------------------------------------------------------------------------

def _extract_score(payload: Dict[str, Any], field: str) -> Optional[float]:
    if field == "composite":
        val = payload.get("deployable_alpha_utility")
        if val is None:
            val = payload.get("gross_opportunity")
        return _safe_float(val)
    engine_scores = payload.get("engine_scores") or {}
    engine_block = engine_scores.get(field) or {}
    return _safe_float(engine_block.get("score"))


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# DB: fetch AXIOM scores for a given date
# ---------------------------------------------------------------------------

def _fetch_axiom_scores(as_of_date: dt.date) -> List[Dict[str, Any]]:
    rows = db.safe_fetchall(
        """
        SELECT symbol, payload
        FROM axiom_scores_daily
        WHERE as_of_date = %s
        """,
        (as_of_date,),
    )
    if not rows:
        return []
    result = []
    for row in rows:
        symbol, payload = row[0], row[1]
        if not payload:
            continue
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        result.append({"symbol": symbol, "payload": payload})
    return result


# ---------------------------------------------------------------------------
# DB: fetch forward returns from market_bars_daily
# ---------------------------------------------------------------------------

def _fetch_forward_returns(
    symbols: List[str],
    entry_date: dt.date,
    horizon_label: str,
) -> Dict[str, Optional[float]]:
    """Return {symbol: fwd_return} for symbols that have both entry and exit prices."""
    calendar_days = _CALENDAR_DAYS[horizon_label]
    target_exit = entry_date + dt.timedelta(days=calendar_days)

    rows = db.safe_fetchall(
        """
        SELECT b1.symbol,
               b1.close AS entry_close,
               b2.exit_close
        FROM market_bars_daily b1
        JOIN LATERAL (
            SELECT close AS exit_close
            FROM market_bars_daily
            WHERE symbol = b1.symbol
              AND as_of_date >= %s
              AND close IS NOT NULL
            ORDER BY as_of_date ASC
            LIMIT 1
        ) b2 ON true
        WHERE b1.as_of_date = %s
          AND b1.close IS NOT NULL
          AND b1.symbol = ANY(%s)
        """,
        (target_exit, entry_date, symbols),
    )
    result: Dict[str, Optional[float]] = {}
    for row in rows:
        symbol, entry_close, exit_close = row[0], row[1], row[2]
        if entry_close and exit_close and float(entry_close) > 0:
            fwd = float(exit_close) / float(entry_close) - 1.0
            result[symbol] = round(fwd, 8)
    return result


# ---------------------------------------------------------------------------
# Effective breadth (Grinold-Kahn): per-symbol time-series IC
# ---------------------------------------------------------------------------

def _fetch_per_symbol_scores_and_returns(
    symbols: List[str],
    as_of_date: dt.date,
    horizon_label: str,
    window_days: int = 63,
) -> Dict[str, List[Tuple[float, float]]]:
    """Return {symbol: [(score, fwd_return), ...]} over the rolling window."""
    if not db.db_read_enabled() or not symbols:
        return {}
    calendar_lookback = round(window_days * 365.0 / 252.0) + 10
    since = as_of_date - dt.timedelta(days=calendar_lookback)
    calendar_horizon = _CALENDAR_DAYS[horizon_label]

    try:
        rows = db.safe_fetchall(
            """
            SELECT a.symbol, a.as_of_date,
                   (a.payload->>'deployable_alpha_utility')::numeric AS score,
                   b.close AS entry_close,
                   b2.exit_close
            FROM axiom_scores_daily a
            JOIN market_bars_daily b
                ON b.symbol = a.symbol AND b.as_of_date = a.as_of_date AND b.close IS NOT NULL
            JOIN LATERAL (
                SELECT close AS exit_close
                FROM market_bars_daily
                WHERE symbol = a.symbol
                  AND as_of_date >= a.as_of_date + %s
                  AND close IS NOT NULL
                ORDER BY as_of_date ASC
                LIMIT 1
            ) b2 ON true
            WHERE a.symbol = ANY(%s)
              AND a.as_of_date >= %s AND a.as_of_date < %s
              AND (a.payload->>'deployable_alpha_utility') IS NOT NULL
            ORDER BY a.symbol, a.as_of_date
            """,
            (dt.timedelta(days=calendar_horizon - 1), symbols, since, as_of_date),
        )
    except Exception:
        return {}

    result: Dict[str, List[Tuple[float, float]]] = {}
    for row in (rows or []):
        if len(row) < 5:
            continue
        sym = str(row[0])
        try:
            score = float(row[2]) if row[2] is not None else None
            entry = float(row[3]) if row[3] is not None else None
            exit_ = float(row[4]) if row[4] is not None else None
        except (TypeError, ValueError):
            continue
        if score is None or entry is None or exit_ is None or entry <= 0:
            continue
        fwd = exit_ / entry - 1.0
        result.setdefault(sym, []).append((score, fwd))
    return result


def compute_effective_breadth(
    symbols: List[str],
    as_of_date: dt.date,
    horizon_label: str = "21d",
) -> int:
    """Grinold-Kahn effective breadth: count of symbols with per-symbol rolling IC > threshold.

    For each symbol, computes Spearman IC between its time-series AXIOM scores
    and its forward returns over the rolling 63-day window. Returns the count
    of symbols where this IC exceeds _EFFECTIVE_BREADTH_IC_THRESHOLD (0.02).
    """
    if not symbols or not db.db_read_enabled():
        return 0
    series_map = _fetch_per_symbol_scores_and_returns(
        symbols, as_of_date, horizon_label, window_days=_EFFECTIVE_BREADTH_WINDOW_DAYS
    )
    count = 0
    for sym, pairs in series_map.items():
        if len(pairs) < _MIN_SAMPLE:
            continue
        scores_list = [p[0] for p in pairs]
        returns_list = [p[1] for p in pairs]
        ic_val, _ = spearman_ic(scores_list, returns_list)
        if ic_val is not None and ic_val > _EFFECTIVE_BREADTH_IC_THRESHOLD:
            count += 1
    return count


# ---------------------------------------------------------------------------
# IC decay summary helpers
# ---------------------------------------------------------------------------

def _ic_state(icir: Optional[float]) -> str:
    if icir is None:
        return "INSUFFICIENT"
    if icir >= _ICIR_STRONG:
        return "STRONG"
    if icir >= _ICIR_MODERATE:
        return "MODERATE"
    if icir >= _ICIR_WEAK:
        return "WEAK"
    return "DEGRADED"


def _rolling_icir(ic_values: List[float], window: int) -> Optional[float]:
    tail = ic_values[-window:]
    if len(tail) < 3:
        return None
    mean_ic = float(np.mean(tail))
    std_ic = float(np.std(tail, ddof=1))
    if std_ic == 0:
        return None
    return round(mean_ic / std_ic, 4)


def _rolling_mean(ic_values: List[float], window: int) -> Optional[float]:
    tail = ic_values[-window:]
    if not tail:
        return None
    return round(float(np.mean(tail)), 6)


# ---------------------------------------------------------------------------
# Public: compute IC snapshot
# ---------------------------------------------------------------------------

def compute_ic_snapshot(
    as_of_date: dt.date,
) -> Dict[str, Any]:
    """
    Compute IC for all SCORE_FIELDS × HORIZONS for a given as_of_date.

    Returns dict keyed by (score_field, horizon_label) → {ic_value, sample_size,
    p_value, t_stat, ic_state, effective_breadth}.  Returns {} if no axiom scores
    exist for that date.
    """
    score_rows = _fetch_axiom_scores(as_of_date)
    if not score_rows:
        return {}

    symbols = [row["symbol"] for row in score_rows]

    # Extract per-field scores once
    field_scores: Dict[str, Dict[str, Optional[float]]] = {
        field: {row["symbol"]: _extract_score(row["payload"], field) for row in score_rows}
        for field in SCORE_FIELDS
    }

    # Compute effective breadth per horizon (shared across score fields)
    effective_breadths: Dict[str, int] = {}
    for horizon_label in HORIZONS:
        effective_breadths[horizon_label] = compute_effective_breadth(symbols, as_of_date, horizon_label)

    results: Dict[str, Any] = {}

    for horizon_label in HORIZONS:
        fwd_returns = _fetch_forward_returns(symbols, as_of_date, horizon_label)
        if not fwd_returns:
            continue

        for field in SCORE_FIELDS:
            scores_map = field_scores[field]
            pairs = [
                (scores_map[sym], fwd_returns[sym])
                for sym in symbols
                if scores_map.get(sym) is not None and fwd_returns.get(sym) is not None
            ]
            if len(pairs) < _MIN_SAMPLE:
                results[(field, horizon_label)] = {
                    "ic_value": None,
                    "sample_size": len(pairs),
                    "p_value": None,
                    "t_stat": None,
                    "ic_state": "INSUFFICIENT",
                    "effective_breadth": effective_breadths.get(horizon_label, 0),
                }
                continue

            scores_list = [p[0] for p in pairs]
            returns_list = [p[1] for p in pairs]
            ic_val, p_val = spearman_ic(scores_list, returns_list)

            t_stat = None
            if ic_val is not None and len(pairs) > 2:
                denom = math.sqrt(1.0 - ic_val ** 2) if abs(ic_val) < 1.0 else 0.0
                t_stat = round(ic_val * math.sqrt(len(pairs) - 2) / denom, 4) if denom != 0 else None

            results[(field, horizon_label)] = {
                "ic_value": ic_val,
                "sample_size": len(pairs),
                "p_value": p_val,
                "t_stat": t_stat,
                "ic_state": "INSUFFICIENT" if ic_val is None else None,  # final state set during store
                "effective_breadth": effective_breadths.get(horizon_label, 0),
            }

    return results


# ---------------------------------------------------------------------------
# Public: store IC snapshot
# ---------------------------------------------------------------------------

def store_ic_snapshot(as_of_date: dt.date, snapshot: Dict[Tuple[str, str], Dict[str, Any]]) -> int:
    """Upsert snapshot rows to signal_ic_daily. Returns number of rows written."""
    if not snapshot or not db.db_write_enabled():
        return 0
    written = 0
    for (field, horizon_label), row in snapshot.items():
        ic_val = row.get("ic_value")
        t_stat = row.get("t_stat")
        ic_state = row.get("ic_state") or (_ic_state(None) if ic_val is None else None)
        effective_breadth = row.get("effective_breadth") or 0
        db.safe_execute(
            """
            INSERT INTO signal_ic_daily (
                as_of_date, score_field, horizon_label,
                ic_value, sample_size, p_value, t_stat, ic_state,
                effective_breadth, meta
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (as_of_date, score_field, horizon_label) DO UPDATE SET
                ic_value         = EXCLUDED.ic_value,
                sample_size      = EXCLUDED.sample_size,
                p_value          = EXCLUDED.p_value,
                t_stat           = EXCLUDED.t_stat,
                ic_state         = EXCLUDED.ic_state,
                effective_breadth= EXCLUDED.effective_breadth,
                meta             = EXCLUDED.meta,
                updated_at       = now()
            """,
            (
                as_of_date,
                field,
                horizon_label,
                ic_val,
                row.get("sample_size"),
                row.get("p_value"),
                t_stat,
                ic_state,
                effective_breadth,
                "{}",
            ),
        )
        written += 1
    return written


# ---------------------------------------------------------------------------
# Public: load IC history + decay summary
# ---------------------------------------------------------------------------

def load_ic_history(
    score_field: str,
    horizon_label: str,
    *,
    window_days: int = 252,
) -> List[Dict[str, Any]]:
    """Return [{as_of_date, ic_value}] sorted by date ascending, up to window_days rows."""
    if not db.db_read_enabled():
        return []
    rows = db.safe_fetchall(
        """
        SELECT as_of_date, ic_value, sample_size, ic_state
        FROM signal_ic_daily
        WHERE score_field = %s
          AND horizon_label = %s
          AND ic_value IS NOT NULL
        ORDER BY as_of_date ASC
        LIMIT %s
        """,
        (score_field, horizon_label, window_days),
    )
    return [
        {
            "as_of_date": str(row[0]),
            "ic_value": float(row[1]) if row[1] is not None else None,
            "sample_size": row[2],
            "ic_state": row[3],
        }
        for row in rows
        if row and row[1] is not None
    ]


def compute_ic_decay_summary(
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    From a time-ordered list of {as_of_date, ic_value}, compute:
    mean_ic, std_ic, icir, t_stat, rolling means and ICIRs, ic_state.
    """
    ic_values = [float(row["ic_value"]) for row in history if row.get("ic_value") is not None]
    n = len(ic_values)
    if n < 3:
        return {
            "sample_count": n,
            "mean_ic": None,
            "std_ic": None,
            "icir": None,
            "t_stat": None,
            "ic_mean_21d": None,
            "ic_mean_63d": None,
            "icir_21d": None,
            "icir_63d": None,
            "ic_state": "INSUFFICIENT",
        }
    mean_ic = float(np.mean(ic_values))
    std_ic = float(np.std(ic_values, ddof=1))
    icir = round(mean_ic / std_ic, 4) if std_ic > 0 else None
    t_stat = round(icir * math.sqrt(n), 4) if icir is not None else None
    return {
        "sample_count": n,
        "mean_ic": round(mean_ic, 6),
        "std_ic": round(std_ic, 6),
        "icir": icir,
        "t_stat": t_stat,
        "ic_mean_21d": _rolling_mean(ic_values, 21),
        "ic_mean_63d": _rolling_mean(ic_values, 63),
        "icir_21d": _rolling_icir(ic_values, 21),
        "icir_63d": _rolling_icir(ic_values, 63),
        "ic_state": _ic_state(icir),
    }


# ---------------------------------------------------------------------------
# Public: translate IC history → axiom_calibration_snapshots
# ---------------------------------------------------------------------------

def snapshot_ic_as_calibration(
    as_of_date: dt.date,
    *,
    write_ok: bool = True,
    horizon_label: str = "21d",
    score_field: str = "composite",
) -> Dict[str, Any]:
    """Derive a Kelly-compatible hit_rate from the IC history and persist it.

    Reads the rolling IC history for (score_field, horizon_label) from
    signal_ic_daily, computes the decay summary, and converts mean_ic to an
    approximate hit_rate via the Gaussian CDF:

        hit_rate ≈ Φ(mean_ic) = 0.5 + 0.5 * erf(mean_ic / √2)

    For mean_ic = 0 → hit_rate = 0.50 (no edge, Kelly = 0).
    For mean_ic = 0.10 → hit_rate ≈ 0.54 (mild edge).
    For mean_ic = 0.20 → hit_rate ≈ 0.58 (solid edge).

    Writes to axiom_calibration_snapshots so _load_hit_rate() always finds a
    current value after the nightly IC stage runs.
    """
    from api.axiom.persistence import persist_axiom_calibration_snapshot

    history = load_ic_history(score_field, horizon_label, window_days=252)
    summary = compute_ic_decay_summary(history)

    mean_ic = float(summary.get("mean_ic") or 0.0)
    # Gaussian CDF approximation: hit_rate = Φ(mean_ic)
    hit_rate = 0.5 + 0.5 * math.erf(mean_ic / math.sqrt(2.0))
    hit_rate = max(0.01, min(0.99, round(hit_rate, 4)))

    payload = {
        "as_of_date": as_of_date.isoformat(),
        "horizon_label": horizon_label,
        "framework_version": "ic_daily_v1",
        "diagnostics": {
            "overall_outcome_metrics": {
                "hit_rate": hit_rate,
                "mean_ic": summary.get("mean_ic"),
                "icir": summary.get("icir"),
                "ic_state": summary.get("ic_state", "INSUFFICIENT"),
                "sample_count": summary.get("sample_count", 0),
            }
        },
    }

    snapshot_key = None
    if write_ok:
        snapshot_key = persist_axiom_calibration_snapshot(
            payload,
            snapshot_key=f"ic_daily_v1:{horizon_label}:{as_of_date.isoformat()}",
        )

    return {
        "hit_rate": hit_rate,
        "mean_ic": summary.get("mean_ic"),
        "ic_state": summary.get("ic_state", "INSUFFICIENT"),
        "sample_count": summary.get("sample_count", 0),
        "snapshot_stored": snapshot_key is not None,
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

class IcSnapshotRequest(BaseModel):
    as_of_date: Optional[str] = None


@router.post("/ic/daily-snapshot")
def ic_daily_snapshot(req: IcSnapshotRequest) -> Dict[str, Any]:
    as_of_date: dt.date
    if req.as_of_date:
        as_of_date = dt.date.fromisoformat(req.as_of_date)
    else:
        as_of_date = dt.date.today()

    snapshot = compute_ic_snapshot(as_of_date)
    if not snapshot:
        return {"status": "no_data", "as_of_date": as_of_date.isoformat(), "stored": 0}

    stored = store_ic_snapshot(as_of_date, snapshot)

    # Build a compact summary for the response
    summary = {
        f"{field}_{horizon}": {
            "ic": row.get("ic_value"),
            "n": row.get("sample_size"),
            "state": row.get("ic_state"),
        }
        for (field, horizon), row in snapshot.items()
    }

    return {
        "status": "ok",
        "as_of_date": as_of_date.isoformat(),
        "stored": stored,
        "snapshot": summary,
    }
