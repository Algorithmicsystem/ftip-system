"""Phase 9.3: Model Registry.

Tracks which ML model is currently active and manages model versioning.
All DB operations are guarded by db.db_read_enabled().
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from api import db
from api.axiom.ml.signal_model import Pipeline, load_model

logger = logging.getLogger(__name__)

# Absolute path so it works regardless of CWD
_HERE = Path(__file__).resolve().parent.parent.parent.parent  # Algorithm/
ML_MODEL_DIR = _HERE / "models"
ML_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def register_model(
    model_path: str,
    model_version: str,
    regime_label: Optional[str],
    metrics: Dict[str, Any],
) -> bool:
    """Register a newly trained model in ml_model_registry.

    Deactivates previous models for the same regime_label first.
    """
    if not db.db_read_enabled():
        return False

    try:
        # Deactivate existing active models for this regime
        db.safe_execute(
            """
            UPDATE ml_model_registry
               SET is_active = FALSE
             WHERE regime_label IS NOT DISTINCT FROM %s
               AND is_active = TRUE
            """,
            (regime_label,),
        )

        db.safe_execute(
            """
            INSERT INTO ml_model_registry
                (model_id, model_version, regime_label, model_path, is_active,
                 trained_at, sample_count, test_accuracy, test_roc_auc,
                 psi_score, metadata)
            VALUES (%s, %s, %s, %s, TRUE, now(), %s, %s, %s, %s, %s)
            ON CONFLICT (model_id) DO UPDATE
               SET is_active = TRUE,
                   trained_at = EXCLUDED.trained_at,
                   sample_count = EXCLUDED.sample_count,
                   test_accuracy = EXCLUDED.test_accuracy,
                   test_roc_auc = EXCLUDED.test_roc_auc,
                   psi_score = EXCLUDED.psi_score,
                   metadata = EXCLUDED.metadata
            """,
            (
                model_version,
                model_version,
                regime_label,
                model_path,
                metrics.get("sample_count", 0),
                metrics.get("accuracy"),
                metrics.get("roc_auc"),
                metrics.get("psi_score"),
                json.dumps(metrics),
            ),
        )
        return True
    except Exception as exc:
        logger.warning("model_registry.register_failed version=%s error=%s", model_version, exc)
        return False


def get_active_model(
    regime_label: Optional[str] = None,
) -> Tuple[Optional[Pipeline], Dict[str, Any]]:
    """Return (model, metadata) for the most recent active model.

    Falls back to global model (regime_label IS NULL) if no regime-specific one.
    Returns (None, {}) if no model available.
    """
    if not db.db_read_enabled():
        return None, {}

    try:
        row = db.safe_fetchone(
            """
            SELECT model_path, metadata
              FROM ml_model_registry
             WHERE is_active = TRUE
               AND (regime_label = %s OR (regime_label IS NULL AND %s IS NULL))
             ORDER BY trained_at DESC
             LIMIT 1
            """,
            (regime_label, regime_label),
        )
        if not row:
            # Fall back to global model
            row = db.safe_fetchone(
                """
                SELECT model_path, metadata
                  FROM ml_model_registry
                 WHERE is_active = TRUE
                   AND regime_label IS NULL
                 ORDER BY trained_at DESC
                 LIMIT 1
                """,
                (),
            )
        if not row:
            return None, {}

        model_path = str(row[0])
        meta_raw = row[1]
        metadata: Dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}

        model, file_meta = load_model(model_path)
        if file_meta:
            metadata.update(file_meta)
        return model, metadata

    except Exception as exc:
        logger.warning("model_registry.get_failed regime=%s error=%s", regime_label, exc)
        return None, {}


def get_model_version() -> str:
    """Return current active model version string."""
    if not db.db_read_enabled():
        return "no_model_trained"

    try:
        row = db.safe_fetchone(
            """
            SELECT model_version
              FROM ml_model_registry
             WHERE is_active = TRUE
               AND regime_label IS NULL
             ORDER BY trained_at DESC
             LIMIT 1
            """,
            (),
        )
        return str(row[0]) if row else "no_model_trained"
    except Exception:
        return "no_model_trained"
