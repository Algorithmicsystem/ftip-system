"""Regime Transition Intelligence — probability scoring and warning signals."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

_MIN_TRANSITIONS = 3

# Warning signal thresholds
_IC_DEGRADATION_THRESHOLD = "WEAK"
_FRAGILITY_SPIKE_THRESHOLD = 70.0
_SRI_ELEVATED = 55.0


def compute_regime_transition_probabilities(
    current_regime: str,
    lookback_days: int = 504,
) -> Dict[str, Any]:
    """Compute probability distribution over next regimes from historical transitions."""
    if not db.db_read_enabled():
        return {
            "current_regime": current_regime,
            "transition_probabilities": {},
            "sample_count": 0,
            "avg_duration_days": 0.0,
            "confidence": "low",
        }

    try:
        since = dt.date.today() - dt.timedelta(days=lookback_days)
        rows = db.safe_fetchall(
            """
            SELECT to_regime, as_of_date
              FROM regime_transitions
             WHERE from_regime = %s AND as_of_date >= %s
             ORDER BY as_of_date DESC
            """,
            (current_regime, since),
        ) or []

        if not rows:
            return {
                "current_regime": current_regime,
                "transition_probabilities": {},
                "sample_count": 0,
                "avg_duration_days": 0.0,
                "confidence": "low",
            }

        counts: Dict[str, int] = {}
        for r in rows:
            to_regime = str(r[0] or "unknown")
            counts[to_regime] = counts.get(to_regime, 0) + 1

        total = sum(counts.values())
        probs = {k: round(v / total, 4) for k, v in sorted(counts.items(), key=lambda x: -x[1])}

        # Average duration in current regime
        dur_rows = db.safe_fetchall(
            """
            SELECT as_of_date
              FROM regime_transitions
             WHERE to_regime = %s AND as_of_date >= %s
             ORDER BY as_of_date ASC
            """,
            (current_regime, since),
        ) or []

        durations: List[int] = []
        for i in range(len(dur_rows) - 1):
            d1 = dur_rows[i][0]
            d2 = dur_rows[i + 1][0]
            if hasattr(d1, "days") or isinstance(d1, dt.date):
                diff = (d2 - d1).days if isinstance(d2, dt.date) else 0
                if diff > 0:
                    durations.append(diff)

        avg_duration = round(sum(durations) / len(durations), 1) if durations else 0.0
        confidence = "high" if total >= 10 else "moderate" if total >= _MIN_TRANSITIONS else "low"

        return {
            "current_regime": current_regime,
            "transition_probabilities": probs,
            "most_likely_next": max(probs, key=probs.__getitem__) if probs else "unknown",
            "most_likely_probability": max(probs.values()) if probs else 0.0,
            "sample_count": total,
            "avg_duration_days": avg_duration,
            "confidence": confidence,
        }
    except Exception as exc:
        logger.warning("compute_regime_transition_probabilities_failed err=%s", exc)
        return {
            "current_regime": current_regime,
            "transition_probabilities": {},
            "sample_count": 0,
            "avg_duration_days": 0.0,
            "confidence": "low",
        }


def identify_warning_signals(as_of_date: dt.date) -> List[Dict[str, Any]]:
    """Return list of active warning signals that precede regime transitions."""
    warnings: List[Dict[str, Any]] = []
    if not db.db_read_enabled():
        return warnings

    try:
        # Warning 1: IC degradation
        ic_row = db.safe_fetchone(
            """
            SELECT ic_state, ic_value
              FROM signal_ic_daily
             WHERE as_of_date <= %s
             ORDER BY as_of_date DESC LIMIT 1
            """,
            (as_of_date,),
        )
        if ic_row and ic_row[0] in ("WEAK", "DEGRADED", "INSUFFICIENT"):
            warnings.append({
                "type": "ic_degradation",
                "severity": "high" if ic_row[0] == "DEGRADED" else "moderate",
                "detail": f"IC state: {ic_row[0]}, value: {ic_row[1]}",
            })

        # Warning 2: Fragility spike
        frag_row = db.safe_fetchone(
            """
            SELECT AVG((payload->'engine_scores'->'critical_fragility'->>'score')::numeric)
              FROM axiom_scores_daily
             WHERE as_of_date = %s
            """,
            (as_of_date,),
        )
        if frag_row and frag_row[0] is not None:
            avg_frag = float(frag_row[0])
            if avg_frag >= _FRAGILITY_SPIKE_THRESHOLD:
                warnings.append({
                    "type": "fragility_spike",
                    "severity": "high" if avg_frag >= 80 else "moderate",
                    "detail": f"Universe avg fragility: {avg_frag:.1f}",
                })

        # Warning 3: SRI elevated
        sri_row = db.safe_fetchone(
            "SELECT sri FROM market_breadth_daily ORDER BY as_of_date DESC LIMIT 1",
        )
        if sri_row and sri_row[0] is not None:
            sri = float(sri_row[0])
            if sri >= _SRI_ELEVATED:
                warnings.append({
                    "type": "sri_elevated",
                    "severity": "critical" if sri >= 75 else "high" if sri >= 60 else "moderate",
                    "detail": f"Systemic risk index: {sri:.1f}",
                })

    except Exception as exc:
        logger.warning("identify_warning_signals_failed err=%s", exc)

    return warnings
