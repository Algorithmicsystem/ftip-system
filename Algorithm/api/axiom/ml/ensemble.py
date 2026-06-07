"""AXIOM Ensemble: IC-weighted blend of rule-based DAU and ML probability."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

_RULE_WEIGHT_DEFAULT = 0.70  # DAU weight when no model available
_ML_WEIGHT_DEFAULT = 0.30


@dataclass
class EnsembleResult:
    symbol: str
    as_of_date: dt.date
    rule_dau: float
    ml_probability: Optional[float]
    ensemble_dau: float
    rule_weight: float
    ml_weight: float
    ic_composite: Optional[float]
    model_version: str
    blend_method: str


def _load_latest_ic(as_of_date: dt.date) -> Optional[float]:
    """Return most recent composite IC value from signal_ic_daily."""
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            """
            SELECT ic_value
              FROM signal_ic_daily
             WHERE score_field = 'composite'
               AND horizon_label = '21d'
               AND ic_value IS NOT NULL
               AND as_of_date <= %s
             ORDER BY as_of_date DESC LIMIT 1
            """,
            (as_of_date,),
        )
        return float(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def _load_ml_probability(symbol: str, as_of_date: dt.date) -> tuple:
    """Return (ml_probability, model_version) from ml_signal_predictions if available."""
    if not db.db_read_enabled():
        return None, "no_model"
    try:
        row = db.safe_fetchone(
            """
            SELECT prediction_score, model_version
              FROM ml_signal_predictions
             WHERE symbol = %s AND as_of_date <= %s
             ORDER BY as_of_date DESC LIMIT 1
            """,
            (symbol, as_of_date),
        )
        if row and row[0] is not None:
            return float(row[0]), str(row[1] or "unknown")
    except Exception:
        pass

    # Fall back to model registry active model (no predictions stored)
    try:
        from api.axiom.ml.model_registry import get_model_version
        version = get_model_version()
        return None, version
    except Exception:
        return None, "no_model"


def compute_ensemble_dau(
    symbol: str,
    rule_dau: float,
    as_of_date: Optional[dt.date] = None,
) -> EnsembleResult:
    """Blend rule-based DAU with ML probability using IC as confidence weight."""
    aod = as_of_date or dt.date.today()

    ic = _load_latest_ic(aod)
    ml_prob, model_version = _load_ml_probability(symbol, aod)

    # IC-derived weights: higher IC → trust rule-based more
    if ic is not None and ml_prob is not None:
        # IC ranges roughly -0.1 to 0.3; map to rule_weight 0.60–0.85
        rule_weight = max(0.60, min(0.85, 0.70 + ic * 0.5))
        ml_weight = 1.0 - rule_weight
        blend_method = "ic_weighted"
    elif ml_prob is not None:
        rule_weight = _RULE_WEIGHT_DEFAULT
        ml_weight = _ML_WEIGHT_DEFAULT
        blend_method = "fixed_blend"
    else:
        rule_weight = 1.0
        ml_weight = 0.0
        blend_method = "rule_only"

    if ml_prob is not None:
        # ml_prob is [0,1]; scale to DAU range [0,100]
        ml_dau = ml_prob * 100.0
        ensemble_dau = round(rule_weight * rule_dau + ml_weight * ml_dau, 2)
    else:
        ensemble_dau = round(rule_dau, 2)

    return EnsembleResult(
        symbol=symbol,
        as_of_date=aod,
        rule_dau=round(rule_dau, 2),
        ml_probability=round(ml_prob, 4) if ml_prob is not None else None,
        ensemble_dau=ensemble_dau,
        rule_weight=round(rule_weight, 4),
        ml_weight=round(ml_weight, 4),
        ic_composite=round(ic, 4) if ic is not None else None,
        model_version=model_version,
        blend_method=blend_method,
    )


def invalidate_ensemble_cache() -> None:
    """No-op stub: called after ML training to signal cache should refresh."""
    pass


def get_ensemble_status(as_of_date: Optional[dt.date] = None) -> Dict[str, Any]:
    """Return current ensemble configuration and IC state."""
    aod = as_of_date or dt.date.today()
    ic = _load_latest_ic(aod)

    try:
        from api.axiom.ml.model_registry import get_model_version
        model_version = get_model_version()
    except Exception:
        model_version = "no_model"

    return {
        "as_of_date": aod.isoformat(),
        "model_version": model_version,
        "ic_composite": round(ic, 4) if ic is not None else None,
        "blend_method": "ic_weighted" if ic is not None else "rule_only",
        "rule_weight": max(0.60, min(0.85, 0.70 + (ic or 0.0) * 0.5)),
        "ml_weight": 1.0 - max(0.60, min(0.85, 0.70 + (ic or 0.0) * 0.5)),
    }
