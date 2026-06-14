"""Phase 12.1: Signal Performance Memory and Genealogy.

Tracks every signal's performance across all horizons and builds genealogy
tracing which features drove which signals that produced which returns.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

_SPY_BASELINE = 0.52  # buy-and-hold SPY hit rate equivalent


@dataclass
class SignalPerformanceRecord:
    symbol: str
    signal_date: dt.date
    signal_label: str
    dau_at_signal: float
    regime_at_signal: str
    factor_composite_at_signal: float
    primary_factor_driver: str
    horizon_5d_return: Optional[float]
    horizon_21d_return: Optional[float]
    horizon_63d_return: Optional[float]
    horizon_126d_return: Optional[float]
    triple_barrier_5d: Optional[int]
    triple_barrier_21d: Optional[int]
    batting_average: Optional[float]
    slugging_average: Optional[float]
    signal_war: Optional[float]
    genealogy: dict = field(default_factory=dict)


def _compute_stats_from_pnl_rows(rows: List[Dict]) -> Dict:
    """Pure-computation batting stats from pre-loaded PnL rows.

    Each row must contain: hit (bool|None), return_pct (float|None),
    regime (str), horizon_days (int).
    """
    if len(rows) < 5:
        return {
            "batting_average_5d": None,
            "batting_average_21d": None,
            "slugging_5d": None,
            "slugging_21d": None,
            "signal_war": None,
            "sample_count": len(rows),
            "regime_breakdown": {},
        }

    rows_5d = [r for r in rows if r.get("horizon_days") == 5]
    rows_21d = [r for r in rows if r.get("horizon_days") == 21]

    def _batting(r_list: List[Dict]) -> Optional[float]:
        valid = [r for r in r_list if r.get("hit") is not None]
        if not valid:
            return None
        return sum(1 for r in valid if r["hit"] is True) / len(valid)

    def _slugging(r_list: List[Dict]) -> Optional[float]:
        winners = [
            r["return_pct"]
            for r in r_list
            if r.get("hit") is True and r.get("return_pct") is not None
        ]
        return sum(winners) / len(winners) if winners else None

    ba_5d = _batting(rows_5d) if rows_5d else _batting(rows)
    ba_21d = _batting(rows_21d) if rows_21d else _batting(rows)
    slug_5d = _slugging(rows_5d) if rows_5d else _slugging(rows)
    slug_21d = _slugging(rows_21d) if rows_21d else _slugging(rows)

    overall_ba = _batting(rows) or 0.0
    n = len(rows)

    # Per-symbol IC: Spearman correlation between DAU proxy (batting) and return
    # Use return_pct as both score and outcome proxy for now
    ret_list = [r["return_pct"] for r in rows if r.get("return_pct") is not None]
    hit_list = [1.0 if r.get("hit") else 0.0 for r in rows if r.get("return_pct") is not None]
    ic = 0.0
    if len(ret_list) >= 5:
        try:
            import math
            n_ic = len(ret_list)
            mean_r = sum(ret_list) / n_ic
            mean_h = sum(hit_list) / n_ic
            cov = sum((ret_list[i] - mean_r) * (hit_list[i] - mean_h) for i in range(n_ic))
            std_r = math.sqrt(sum((x - mean_r) ** 2 for x in ret_list) / n_ic) or 1e-9
            std_h = math.sqrt(sum((x - mean_h) ** 2 for x in hit_list) / n_ic) or 1e-9
            ic = cov / (n_ic * std_r * std_h)
            ic = max(-1.0, min(1.0, ic))
        except Exception:
            ic = 0.0

    # WAR formula: (BA - league_avg) * (1 + max(0, IC))
    # IC component boosts WAR when signal quality is high; sqrt(N) excluded
    # to keep the metric interpretable (excess hit rate, IC-adjusted).
    signal_war = round((overall_ba - _SPY_BASELINE) * (1.0 + max(0.0, ic)), 4)

    regime_map: Dict[str, Dict[str, int]] = {}
    for r in rows:
        regime = str(r.get("regime") or "unknown")
        regime_map.setdefault(regime, {"hits": 0, "total": 0})
        regime_map[regime]["total"] += 1
        if r.get("hit") is True:
            regime_map[regime]["hits"] += 1

    regime_breakdown = {
        reg: round(v["hits"] / v["total"], 4)
        for reg, v in regime_map.items()
        if v["total"] > 0
    }

    return {
        "batting_average_5d": round(ba_5d, 4) if ba_5d is not None else None,
        "batting_average_21d": round(ba_21d, 4) if ba_21d is not None else None,
        "slugging_5d": round(slug_5d, 4) if slug_5d is not None else None,
        "slugging_21d": round(slug_21d, 4) if slug_21d is not None else None,
        "signal_war": signal_war,
        "war_ic_component": round(ic, 4),
        "league_avg": _SPY_BASELINE,
        "sample_count": len(rows),
        "regime_breakdown": regime_breakdown,
    }


def compute_signal_batting_average(
    symbol: str,
    lookback_days: int = 252,
) -> Dict:
    """Compute batting average, slugging, and signal_war for a symbol."""
    empty: Dict = {
        "symbol": symbol,
        "batting_average_5d": None,
        "batting_average_21d": None,
        "slugging_5d": None,
        "slugging_21d": None,
        "signal_war": None,
        "sample_count": 0,
        "regime_breakdown": {},
    }
    if not db.db_read_enabled():
        return empty

    try:
        since = dt.date.today() - dt.timedelta(days=lookback_days)
        rows = db.safe_fetchall(
            """
            SELECT p.horizon_days, p.return_pct, p.hit,
                   COALESCE(a.payload->>'regime_label', 'unknown') AS regime
              FROM signal_pnl_daily p
              LEFT JOIN axiom_scores_daily a
                ON a.symbol = p.symbol AND a.as_of_date = p.signal_date
             WHERE p.symbol = %s
               AND p.signal_date >= %s
               AND p.hit IS NOT NULL
             ORDER BY p.signal_date DESC
            """,
            (symbol, since),
        ) or []
    except Exception:
        return empty

    raw = [
        {
            "horizon_days": int(r[0]),
            "return_pct": float(r[1]) if r[1] is not None else 0.0,
            "hit": bool(r[2]),
            "regime": str(r[3] or "unknown"),
        }
        for r in rows
    ]
    stats = _compute_stats_from_pnl_rows(raw)
    return {"symbol": symbol, **stats}


def load_signal_performance_history(
    symbol: Optional[str] = None,
    lookback_days: int = 252,
    min_dau: float = 50.0,
) -> List[SignalPerformanceRecord]:
    """Load signal performance records from the archive."""
    if not db.db_read_enabled():
        return []
    try:
        since = dt.date.today() - dt.timedelta(days=lookback_days)
        if symbol:
            rows = db.safe_fetchall(
                """
                SELECT symbol, signal_date, signal_label, dau_at_signal,
                       regime_at_signal, primary_factor_driver,
                       horizon_5d_return, horizon_21d_return,
                       batting_average, slugging_average, signal_war, genealogy
                  FROM signal_performance_archive
                 WHERE symbol = %s AND signal_date >= %s AND dau_at_signal >= %s
                 ORDER BY signal_date DESC LIMIT 500
                """,
                (symbol, since, min_dau),
            ) or []
        else:
            rows = db.safe_fetchall(
                """
                SELECT symbol, signal_date, signal_label, dau_at_signal,
                       regime_at_signal, primary_factor_driver,
                       horizon_5d_return, horizon_21d_return,
                       batting_average, slugging_average, signal_war, genealogy
                  FROM signal_performance_archive
                 WHERE signal_date >= %s AND dau_at_signal >= %s
                 ORDER BY signal_date DESC LIMIT 1000
                """,
                (since, min_dau),
            ) or []

        return [
            SignalPerformanceRecord(
                symbol=str(r[0]),
                signal_date=r[1],
                signal_label=str(r[2] or "BUY"),
                dau_at_signal=float(r[3] or 0),
                regime_at_signal=str(r[4] or "unknown"),
                factor_composite_at_signal=0.0,
                primary_factor_driver=str(r[5] or ""),
                horizon_5d_return=float(r[6]) if r[6] is not None else None,
                horizon_21d_return=float(r[7]) if r[7] is not None else None,
                horizon_63d_return=None,
                horizon_126d_return=None,
                triple_barrier_5d=None,
                triple_barrier_21d=None,
                batting_average=float(r[8]) if r[8] is not None else None,
                slugging_average=float(r[9]) if r[9] is not None else None,
                signal_war=float(r[10]) if r[10] is not None else None,
                genealogy=r[11] if isinstance(r[11], dict) else {},
            )
            for r in rows
        ]
    except Exception:
        return []


def update_signal_performance_archive(as_of_date: dt.date) -> Dict:
    """Archive matured 21-day signals into signal_performance_archive."""
    if not db.db_read_enabled():
        return {"updated": 0, "as_of_date": as_of_date.isoformat()}

    try:
        rows = db.safe_fetchall(
            """
            SELECT p.symbol, p.signal_date, p.return_pct, p.hit,
                   COALESCE(a.payload->>'regime_label', 'unknown') AS regime,
                   COALESCE((a.payload->>'deployable_alpha_utility')::numeric, 0) AS dau,
                   a.payload->>'alpha_decomposition' AS alpha_json
              FROM signal_pnl_daily p
              LEFT JOIN axiom_scores_daily a
                ON a.symbol = p.symbol AND a.as_of_date = p.signal_date
             WHERE p.signal_date <= %s AND p.horizon_days = 21 AND p.hit IS NOT NULL
             ORDER BY p.signal_date DESC LIMIT 500
            """,
            (as_of_date,),
        ) or []

        updated = 0
        for r in rows:
            symbol, signal_date = str(r[0]), r[1]
            return_pct = float(r[2]) if r[2] is not None else 0.0
            hit = bool(r[3])
            regime, dau = str(r[4] or "unknown"), float(r[5] or 0)

            primary_driver, genealogy = "unknown", {}
            if r[6]:
                try:
                    ad = json.loads(r[6]) if isinstance(r[6], str) else r[6]
                    primary_driver = str((ad or {}).get("primary_driver") or "unknown")
                    genealogy = {k: v for k, v in (ad or {}).items() if k != "primary_driver"}
                except Exception:
                    pass

            batting = 1.0 if hit else 0.0
            slugging = return_pct if hit else 0.0
            war = round(batting - _SPY_BASELINE, 4)

            db.safe_execute(
                """
                INSERT INTO signal_performance_archive
                    (symbol, signal_date, signal_label, dau_at_signal, regime_at_signal,
                     primary_factor_driver, horizon_21d_return, batting_average,
                     slugging_average, signal_war, genealogy, computed_at)
                VALUES (%s, %s, 'BUY', %s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
                ON CONFLICT (symbol, signal_date) DO UPDATE
                   SET batting_average = EXCLUDED.batting_average,
                       signal_war      = EXCLUDED.signal_war,
                       computed_at     = now()
                """,
                (symbol, signal_date, dau, regime, primary_driver,
                 return_pct, batting, slugging, war, json.dumps(genealogy)),
            )
            updated += 1

        return {"updated": updated, "as_of_date": as_of_date.isoformat()}
    except Exception as exc:
        logger.warning("signal_archive_update_failed err=%s", exc)
        return {"updated": 0, "as_of_date": as_of_date.isoformat()}


def get_signal_leaderboard(limit: int = 10) -> List[Dict]:
    """Return top symbols by signal_war from the archive."""
    if not db.db_read_enabled():
        return []
    try:
        rows = db.safe_fetchall(
            """
            SELECT symbol,
                   AVG(signal_war)       AS avg_war,
                   AVG(batting_average)  AS avg_batting,
                   COUNT(*)              AS sample_count
              FROM signal_performance_archive
             GROUP BY symbol
            HAVING COUNT(*) >= 5
             ORDER BY avg_war DESC NULLS LAST
             LIMIT %s
            """,
            (limit,),
        ) or []
        return [
            {
                "symbol": str(r[0]),
                "avg_signal_war": round(float(r[1] or 0), 4),
                "avg_batting_average": round(float(r[2] or 0), 4),
                "sample_count": int(r[3]),
            }
            for r in rows
        ]
    except Exception:
        return []
