"""Phase 9.5: Model Drift Monitor.

Detects feature distribution shift between training time and live data.
Uses Population Stability Index (PSI) over the 46-feature vector.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, List

import numpy as np

from api import db
from api.axiom.ml.feature_builder import (
    build_feature_vector,
    feature_vector_to_array,
    get_feature_names,
)
from api.axiom.ml.model_registry import get_active_model
from api.axiom.ml.training_job import compute_psi

logger = logging.getLogger(__name__)

_DRIFT_LOOKBACK_DAYS = 21
_PSI_RETRAIN = 0.25
_PSI_MONITOR = 0.10


def check_model_drift(as_of_date: dt.date) -> Dict[str, Any]:
    """Compare live feature distributions against training-time reference.

    Computes per-feature PSI and identifies the top 5 most-drifted features.
    Returns drift assessment and recommendation.
    """
    _stable = {
        "drift_detected": False,
        "overall_psi": 0.0,
        "top_drifted_features": [],
        "recommendation": "stable",
    }

    if not db.db_read_enabled():
        return _stable

    try:
        _, metadata = get_active_model(regime_label=None)
        if not metadata:
            return _stable

        feature_ref_hist = metadata.get("feature_ref_hist") or {}
        feature_ref_bins = metadata.get("feature_ref_bins") or {}
        if not feature_ref_hist:
            return _stable

        # Load recent axiom_scores_daily payloads
        since = as_of_date - dt.timedelta(days=_DRIFT_LOOKBACK_DAYS)
        rows = db.safe_fetchall(
            """
            SELECT payload
              FROM axiom_scores_daily
             WHERE as_of_date >= %s
               AND as_of_date <= %s
               AND payload IS NOT NULL
             LIMIT 500
            """,
            (since, as_of_date),
        )

        if not rows or len(rows) < 10:
            return _stable

        # Build feature vectors for live data
        feature_names = get_feature_names()
        live_rows: List[np.ndarray] = []
        for row in rows:
            payload_raw = row[0]
            if isinstance(payload_raw, str):
                try:
                    payload = json.loads(payload_raw)
                except Exception:
                    continue
            else:
                payload = payload_raw or {}
            if not isinstance(payload, dict):
                continue
            factor_loadings = payload.get("factor_loadings_summary") or []
            try:
                fv = build_feature_vector(payload, factor_loadings, ic_state=None)
                arr = feature_vector_to_array(fv, fill_value=0.5)
                live_rows.append(arr)
            except Exception:
                continue

        if len(live_rows) < 5:
            return _stable

        live_X = np.array(live_rows, dtype=np.float64)

        # Compute PSI for each feature
        feature_psis: Dict[str, float] = {}
        for i, fname in enumerate(feature_names):
            ref_hist = feature_ref_hist.get(fname)
            ref_bins = feature_ref_bins.get(fname)
            if not ref_hist or not ref_bins:
                feature_psis[fname] = 0.0
                continue
            live_col = live_X[:, i]
            bins = np.array(ref_bins)
            cur_counts, _ = np.histogram(live_col, bins=bins)
            cur_pct = (cur_counts / max(len(live_col), 1)).tolist()
            feature_psis[fname] = compute_psi(
                np.array(ref_hist) * max(1, len(live_col)),
                cur_counts.astype(np.float64),
            )

        overall_psi = float(np.mean(list(feature_psis.values()))) if feature_psis else 0.0
        top_drifted = sorted(feature_psis.items(), key=lambda kv: -kv[1])[:5]

        if overall_psi > _PSI_RETRAIN:
            recommendation = "retrain"
        elif overall_psi > _PSI_MONITOR:
            recommendation = "monitor"
        else:
            recommendation = "stable"

        return {
            "drift_detected": overall_psi > _PSI_RETRAIN,
            "overall_psi": round(overall_psi, 6),
            "top_drifted_features": [
                {"feature": k, "psi": round(v, 6)} for k, v in top_drifted
            ],
            "recommendation": recommendation,
        }

    except Exception as exc:
        logger.warning("check_model_drift failed error=%s", exc)
        return _stable
