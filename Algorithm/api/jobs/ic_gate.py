"""Phase 22: IC Auto-Gate.

Loads the latest IC state from signal_ic_daily and applies a confidence
reduction when IC is degraded or insufficient.  Called once per signals-daily
job run (not per symbol) and returns a gate result that callers merge into
each signal's signal_meta before persistence.

Gate rules:
  STRONG     → no reduction (gate open)
  MODERATE   → no reduction (gate open)
  WEAK       → confidence × 0.70, gate flag set
  DEGRADED   → confidence × 0.40, gate flag set
  INSUFFICIENT → confidence × 0.85 (slight caution), gate flag set when
                 sample_count < MIN_IC_SAMPLES

Design: pure functions with no DB side-effects so tests can call
load_ic_gate_state() with a monkeypatched DB and verify the math.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Optional

from api import db
from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)

# Minimum sample count before INSUFFICIENT triggers the caution flag
MIN_IC_SAMPLES = 10

# Confidence multipliers by IC state
_IC_CONFIDENCE_MULT: Dict[str, float] = {
    "STRONG":       1.00,
    "MODERATE":     1.00,
    "WEAK":         0.70,
    "DEGRADED":     0.40,
    "INSUFFICIENT": 1.00,   # multiplier; flag only set when sample_count < threshold
}

_IC_GATE_NOTE: Dict[str, str] = {
    "STRONG":       "",
    "MODERATE":     "",
    "WEAK":         "ic_quality_gate_weak",
    "DEGRADED":     "ic_quality_gate_degraded",
    "INSUFFICIENT": "ic_quality_gate_insufficient_samples",
}


# ---------------------------------------------------------------------------
# 1.9 Grinold-Kahn Active Management Quality Score
# ---------------------------------------------------------------------------

def compute_amqs(ic_value: float, breadth: int, tracking_error: float) -> float:
    """Grinold-Kahn Active Management Quality Score (0–100).

    Grounded in Active Portfolio Management (Grinold & Kahn).
    Fundamental Law of Active Management: IR = IC × sqrt(breadth) / TE

    More breadth (more symbols) → higher AMQS for same IC and TE.
    Score normalised from IR range [-2.0, 3.0] → [0, 100].
    """
    ic = clamp(float(ic_value or 0.0), -1.0, 1.0)
    br = max(int(breadth or 1), 1)
    te = max(float(tracking_error or 0.01), 0.001)

    ir = ic * (br ** 0.5) / te
    score = clamp((ir + 2.0) / 5.0 * 100.0, 0.0, 100.0)
    return round(score, 2)


def load_ic_gate_state(
    as_of_date: dt.date,
    *,
    horizon_label: str = "21d",
    score_field: str = "composite",
) -> Dict[str, Any]:
    """Return the current IC gate parameters from signal_ic_daily.

    Returns a dict with:
      ic_state        — "STRONG" | "MODERATE" | "WEAK" | "DEGRADED" | "INSUFFICIENT"
      sample_count    — number of IC observations
      mean_ic         — float or None
      confidence_mult — multiplier to apply to signal confidence (0.0–1.0)
      gate_active     — bool: True when any reduction is applied
      gate_note       — str note to add to signal_meta (empty if gate open)
    """
    if not db.db_read_enabled():
        return _open_gate("INSUFFICIENT", 0, None)

    try:
        row = db.safe_fetchone(
            """
            SELECT ic_state, sample_size, ic_mean_21d, effective_breadth
            FROM signal_ic_daily
            WHERE score_field = %s AND horizon_label = %s
              AND as_of_date <= %s
            ORDER BY as_of_date DESC
            LIMIT 1
            """,
            (score_field, horizon_label, as_of_date),
        )
    except Exception as exc:
        logger.warning("ic_gate.load_failed error=%s", exc)
        return _open_gate("INSUFFICIENT", 0, None)

    if not row:
        return _open_gate("INSUFFICIENT", 0, None)

    ic_state = str(row[0] or "INSUFFICIENT").upper()
    try:
        sample_count = int(row[1] or 0)
    except (TypeError, ValueError):
        sample_count = 0
    try:
        mean_ic = float(row[2]) if row[2] is not None else None
    except (TypeError, ValueError):
        mean_ic = None
    try:
        effective_breadth = int(row[3]) if len(row) > 3 and row[3] is not None else sample_count
    except (TypeError, ValueError, IndexError):
        effective_breadth = sample_count

    mult = _IC_CONFIDENCE_MULT.get(ic_state, 1.00)
    note = _IC_GATE_NOTE.get(ic_state, "")

    # INSUFFICIENT: only flag when sample_count is low
    if ic_state == "INSUFFICIENT":
        if sample_count < MIN_IC_SAMPLES:
            mult = 0.85
            note = _IC_GATE_NOTE["INSUFFICIENT"]
        else:
            mult = 1.00
            note = ""

    gate_active = mult < 1.0

    # AMQS uses effective_breadth (symbols with IC > 0.02) when available,
    # falling back to sample_count as proxy
    tracking_error = 0.05
    amqs_score = compute_amqs(
        ic_value=mean_ic if mean_ic is not None else 0.0,
        breadth=effective_breadth,
        tracking_error=tracking_error,
    )

    return {
        "ic_state": ic_state,
        "sample_count": sample_count,
        "effective_breadth": effective_breadth,
        "mean_ic": mean_ic,
        "confidence_mult": mult,
        "gate_active": gate_active,
        "gate_note": note,
        "amqs_score": amqs_score,
    }


def apply_ic_gate(
    signal: Dict[str, Any],
    gate: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a shallow copy of *signal* with IC gate applied to confidence.

    Merges quality_gate metadata into signal_meta without mutating original.
    """
    if not gate.get("gate_active"):
        return signal

    out = dict(signal)
    original_conf = float(out.get("confidence") or 0.0)
    mult = gate["confidence_mult"]
    out["confidence"] = round(original_conf * mult, 4)

    meta = dict(out.get("signal_meta") or {})
    meta["quality_gate_applied"] = True
    meta["ic_gate_state"] = gate["ic_state"]
    meta["ic_gate_confidence_mult"] = mult
    meta["ic_gate_note"] = gate["gate_note"]
    meta["ic_gate_original_confidence"] = original_conf
    out["signal_meta"] = meta

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_gate(ic_state: str, sample_count: int, mean_ic: Optional[float]) -> Dict[str, Any]:
    amqs_score = compute_amqs(
        ic_value=mean_ic if mean_ic is not None else 0.0,
        breadth=sample_count,
        tracking_error=0.05,
    )
    return {
        "ic_state": ic_state,
        "sample_count": sample_count,
        "effective_breadth": sample_count,
        "mean_ic": mean_ic,
        "confidence_mult": 1.0,
        "gate_active": False,
        "gate_note": "",
        "amqs_score": amqs_score,
    }
