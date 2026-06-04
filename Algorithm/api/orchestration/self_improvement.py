"""Phase 17.3: Self-Improvement Engine Wiring."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelfImprovementStatus:
    last_model_training: Optional[dt.datetime]
    last_model_version: str
    training_sample_count: int
    current_psi_score: float
    drift_warning: bool
    last_weight_optimization: Optional[dt.datetime]
    weight_optimization_pending: bool
    last_regime_model_update: Optional[dt.datetime]
    effective_breadth: int
    amqs_score: float
    next_recommended_action: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_new_labeled_samples(since: Optional[dt.date] = None) -> int:
    if not db.db_read_enabled():
        return 0
    try:
        since = since or (dt.date.today() - dt.timedelta(days=30))
        row = db.safe_fetchone(
            """
            SELECT COUNT(*)
              FROM signal_pnl_daily
             WHERE hit IS NOT NULL AND signal_date >= %s
            """,
            (since,),
        )
        return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return 0


def _read_latest_model_info() -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {}
    try:
        row = db.safe_fetchone(
            """
            SELECT model_version, trained_at,
                   (metadata->>'sample_count')::int AS sample_count,
                   (metadata->>'psi_score')::numeric AS psi_score,
                   (metadata->>'drift_warning')::boolean AS drift_warning
              FROM ml_model_registry
             WHERE regime_label IS NULL
             ORDER BY trained_at DESC LIMIT 1
            """
        )
        if not row:
            return {}
        return {
            "model_version": str(row[0] or ""),
            "trained_at": row[1],
            "sample_count": int(row[2] or 0),
            "psi_score": float(row[3] or 0.0),
            "drift_warning": bool(row[4] or False),
        }
    except Exception:
        return {}


def _read_latest_ic() -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {}
    try:
        row = db.safe_fetchone(
            """
            SELECT amqs_score, effective_breadth
              FROM signal_ic_daily
             ORDER BY as_of_date DESC LIMIT 1
            """
        )
        if not row:
            return {}
        return {
            "amqs_score": float(row[0] or 0.0),
            "effective_breadth": int(row[1] or 1),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def check_self_improvement_status() -> SelfImprovementStatus:
    if not db.db_read_enabled():
        return SelfImprovementStatus(
            last_model_training=None,
            last_model_version="no_model_trained",
            training_sample_count=0,
            current_psi_score=0.0,
            drift_warning=False,
            last_weight_optimization=None,
            weight_optimization_pending=False,
            last_regime_model_update=None,
            effective_breadth=1,
            amqs_score=0.0,
            next_recommended_action="enable_database_connection",
        )

    model_info = _read_latest_model_info()
    ic_info = _read_latest_ic()

    model_version = model_info.get("model_version") or "no_model_trained"
    trained_at = model_info.get("trained_at")
    sample_count = model_info.get("sample_count", 0)
    psi_score = model_info.get("psi_score", 0.0)
    drift_warning = model_info.get("drift_warning", False) or (psi_score > 0.25)

    amqs = ic_info.get("amqs_score", 0.0)
    eff_breadth = ic_info.get("effective_breadth", 1)

    # Determine recommended action
    if model_version == "no_model_trained":
        action = "train_initial_model"
    elif drift_warning:
        action = "retrain_model_drift_detected"
    elif sample_count < 50:
        action = "collect_more_labeled_data"
    else:
        action = "monitor_performance"

    return SelfImprovementStatus(
        last_model_training=trained_at,
        last_model_version=model_version,
        training_sample_count=sample_count,
        current_psi_score=psi_score,
        drift_warning=drift_warning,
        last_weight_optimization=None,
        weight_optimization_pending=drift_warning,
        last_regime_model_update=trained_at,
        effective_breadth=eff_breadth,
        amqs_score=amqs,
        next_recommended_action=action,
    )


def trigger_self_improvement(
    force_retrain: bool = False,
    min_new_samples: int = 20,
) -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {"status": "skipped", "reason": "db_disabled"}

    new_samples = _count_new_labeled_samples()
    if not force_retrain and new_samples < min_new_samples:
        return {
            "status": "skipped",
            "reason": "insufficient_samples",
            "new_samples": new_samples,
            "min_required": min_new_samples,
        }

    try:
        from api.axiom.ml.training_job import run_training_job
        training_result = run_training_job(min_samples=min_new_samples)
        psi = training_result.get("psi_score", 0.0)
        drift = training_result.get("drift_warning", False)

        # Update AMQS
        ic_info = _read_latest_ic()

        return {
            "status": training_result.get("status", "trained"),
            "model_version": training_result.get("model_version"),
            "sample_count": training_result.get("sample_count", 0),
            "psi_score": psi,
            "drift_warning": drift,
            "amqs_score": ic_info.get("amqs_score", 0.0),
            "effective_breadth": ic_info.get("effective_breadth", 1),
        }
    except Exception as exc:
        logger.warning("trigger_self_improvement_failed err=%s", exc)
        return {"status": "error", "error": str(exc)}


def get_improvement_history(lookback_days: int = 90) -> List[Dict[str, Any]]:
    if not db.db_read_enabled():
        return []
    try:
        since = dt.date.today() - dt.timedelta(days=lookback_days)
        rows = db.safe_fetchall(
            """
            SELECT model_version, trained_at, regime_label,
                   (metadata->>'sample_count')::int AS sample_count,
                   (metadata->>'psi_score')::numeric AS psi_score,
                   (metadata->>'accuracy')::numeric AS accuracy
              FROM ml_model_registry
             WHERE trained_at >= %s
             ORDER BY trained_at DESC LIMIT 100
            """,
            (since,),
        ) or []
        return [
            {
                "model_version": str(r[0] or ""),
                "trained_at": str(r[1]) if r[1] else None,
                "regime_label": str(r[2]) if r[2] else "global",
                "sample_count": int(r[3] or 0),
                "psi_score": float(r[4] or 0.0),
                "accuracy": float(r[5] or 0.0),
            }
            for r in rows
        ]
    except Exception:
        return []
