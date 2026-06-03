"""Phase 12.2: Regime Intelligence Playbook.

Learns from history which signals to trust in each regime.
Every regime transition updates the playbook automatically.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 5


@dataclass
class RegimePlaybookEntry:
    regime_label: str
    recommended_signal_types: List[str]
    avoided_signal_types: List[str]
    factor_weights: Dict[str, float]
    historical_accuracy: float
    sample_count: int
    regime_duration_avg_days: float
    transition_probability: Dict[str, float]


def _compute_transition_probabilities(
    from_regime: str,
    transitions: List[Dict],
) -> Dict[str, float]:
    """Compute next-regime probabilities from a list of transition dicts."""
    relevant = [t for t in transitions if str(t.get("from_regime")) == from_regime]
    if not relevant:
        return {}
    counts: Dict[str, int] = {}
    for t in relevant:
        to = str(t.get("to_regime") or "unknown")
        counts[to] = counts.get(to, 0) + 1
    total = sum(counts.values())
    return {k: round(v / total, 4) for k, v in counts.items()}


def _build_entry_from_signal_rows(
    regime_label: str,
    signal_rows: List,
    transition_rows: List[Dict],
    duration_avg: float = 0.0,
) -> Optional[RegimePlaybookEntry]:
    """Build a RegimePlaybookEntry from pre-loaded rows."""
    if len(signal_rows) < _MIN_SAMPLES:
        return None

    hits = sum(1 for r in signal_rows if r[8] is not None and float(r[8]) >= 0.5)
    accuracy = hits / len(signal_rows)

    # Factor weights: count which primary_factor_driver appears in high-accuracy signals
    factor_counts: Dict[str, int] = {}
    for r in signal_rows:
        driver = str(r[5] or "unknown")
        if r[8] is not None and float(r[8]) >= 0.5:
            factor_counts[driver] = factor_counts.get(driver, 0) + 1

    total_drivers = sum(factor_counts.values()) or 1
    factor_weights = {k: round(v / total_drivers, 4) for k, v in sorted(factor_counts.items(), key=lambda x: -x[1])[:5]}

    recommended = ["BUY"] if accuracy >= 0.52 else []
    avoided = ["SELL"] if accuracy < 0.48 else []

    trans_probs = _compute_transition_probabilities(regime_label, transition_rows)

    return RegimePlaybookEntry(
        regime_label=regime_label,
        recommended_signal_types=recommended,
        avoided_signal_types=avoided,
        factor_weights=factor_weights,
        historical_accuracy=round(accuracy, 4),
        sample_count=len(signal_rows),
        regime_duration_avg_days=round(duration_avg, 1),
        transition_probability=trans_probs,
    )


def build_regime_playbook(lookback_days: int = 504) -> Dict[str, RegimePlaybookEntry]:
    """Build playbook entries for each regime with sufficient data."""
    if not db.db_read_enabled():
        return {}

    try:
        since = dt.date.today() - dt.timedelta(days=lookback_days)
        # Load signal_performance_archive grouped by regime
        all_signal_rows = db.safe_fetchall(
            """
            SELECT symbol, signal_date, signal_label, dau_at_signal,
                   regime_at_signal, primary_factor_driver,
                   horizon_5d_return, horizon_21d_return,
                   batting_average, slugging_average, signal_war
              FROM signal_performance_archive
             WHERE signal_date >= %s
             ORDER BY regime_at_signal, signal_date
            """,
            (since,),
        ) or []

        # Load regime_transitions for probabilities and durations
        transition_rows = db.safe_fetchall(
            """
            SELECT from_regime, to_regime, as_of_date
              FROM regime_transitions
             WHERE as_of_date >= %s
             ORDER BY as_of_date
            """,
            (since,),
        ) or []

        transition_dicts = [
            {"from_regime": str(r[0]), "to_regime": str(r[1]), "as_of_date": r[2]}
            for r in transition_rows
        ]

    except Exception as exc:
        logger.warning("build_regime_playbook_failed err=%s", exc)
        return {}

    # Group signal rows by regime
    by_regime: Dict[str, List] = {}
    for r in all_signal_rows:
        regime = str(r[4] or "unknown")
        by_regime.setdefault(regime, []).append(r)

    # Compute average regime durations from transitions
    regime_durations: Dict[str, List[int]] = {}
    last_start: Dict[str, dt.date] = {}
    for td in sorted(transition_dicts, key=lambda x: x["as_of_date"]):
        from_r = td["from_regime"]
        to_r = td["to_regime"]
        aod = td["as_of_date"]
        if isinstance(aod, str):
            aod = dt.date.fromisoformat(aod)
        if from_r in last_start:
            days = (aod - last_start[from_r]).days
            regime_durations.setdefault(from_r, []).append(days)
        last_start[to_r] = aod

    playbook: Dict[str, RegimePlaybookEntry] = {}
    for regime, rows in by_regime.items():
        durations = regime_durations.get(regime, [])
        avg_dur = sum(durations) / len(durations) if durations else 0.0
        entry = _build_entry_from_signal_rows(regime, rows, transition_dicts, avg_dur)
        if entry is not None:
            playbook[regime] = entry

    return playbook


def get_regime_recommendation(
    current_regime: str,
    current_playbook: Dict[str, RegimePlaybookEntry],
) -> Dict:
    """Return actionable recommendations for the current regime."""
    entry = current_playbook.get(current_regime)
    if entry is None:
        return {
            "regime": current_regime,
            "recommended_signals": [],
            "top_factors_for_regime": [],
            "expected_batting_average": 0.5,
            "expected_duration_days": 0.0,
            "most_likely_next_regime": "unknown",
            "next_regime_probability": 0.0,
        }

    top_factors = sorted(entry.factor_weights, key=entry.factor_weights.__getitem__, reverse=True)[:3]

    next_regime, next_prob = "unknown", 0.0
    if entry.transition_probability:
        next_regime = max(entry.transition_probability, key=entry.transition_probability.__getitem__)
        next_prob = entry.transition_probability[next_regime]

    return {
        "regime": current_regime,
        "recommended_signals": entry.recommended_signal_types,
        "top_factors_for_regime": top_factors,
        "expected_batting_average": entry.historical_accuracy,
        "expected_duration_days": entry.regime_duration_avg_days,
        "most_likely_next_regime": next_regime,
        "next_regime_probability": next_prob,
    }


def update_regime_playbook(as_of_date: dt.date) -> Dict:
    """Rebuild and store the full playbook, returning count of regimes updated."""
    playbook = build_regime_playbook()
    if not db.db_read_enabled():
        return {"regimes_updated": 0, "as_of_date": as_of_date.isoformat()}

    updated = 0
    for regime, entry in playbook.items():
        try:
            db.safe_execute(
                """
                INSERT INTO regime_playbook
                    (regime_label, recommended_signal_types, avoided_signal_types,
                     factor_weights, historical_accuracy, sample_count,
                     regime_duration_avg_days, transition_probability, last_updated)
                VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s::jsonb, now())
                ON CONFLICT (regime_label) DO UPDATE
                   SET recommended_signal_types = EXCLUDED.recommended_signal_types,
                       factor_weights            = EXCLUDED.factor_weights,
                       historical_accuracy       = EXCLUDED.historical_accuracy,
                       sample_count              = EXCLUDED.sample_count,
                       transition_probability    = EXCLUDED.transition_probability,
                       last_updated              = now()
                """,
                (
                    regime,
                    json.dumps(entry.recommended_signal_types),
                    json.dumps(entry.avoided_signal_types),
                    json.dumps(entry.factor_weights),
                    entry.historical_accuracy,
                    entry.sample_count,
                    entry.regime_duration_avg_days,
                    json.dumps(entry.transition_probability),
                ),
            )
            updated += 1
        except Exception as exc:
            logger.warning("regime_playbook_upsert_failed regime=%s err=%s", regime, exc)

    return {"regimes_updated": updated, "as_of_date": as_of_date.isoformat()}
