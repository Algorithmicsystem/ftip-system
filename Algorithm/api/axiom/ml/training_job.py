"""Phase 9.5: Training Job and PSI Drift Detection.

Orchestrates the full model training lifecycle:
  load data → split → train global + regime models → evaluate → save → register → PSI check
"""
from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from api.axiom.ml.feature_builder import get_feature_names
from api.axiom.ml.model_registry import ML_MODEL_DIR, get_active_model, register_model
from api.axiom.ml.signal_model import evaluate_model, save_model, train_signal_model
from api.axiom.ml.training_data import load_training_dataset, split_train_test_purged

logger = logging.getLogger(__name__)

_REGIME_LABELS = ["TRENDING", "CHOPPY", "HIGH_VOL", "RECOVERY"]
_MIN_REGIME_SAMPLES = 20


def compute_psi(
    reference_distribution: np.ndarray,
    current_distribution: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index.

    PSI = sum((actual_pct - expected_pct) × ln(actual_pct / expected_pct))
    PSI < 0.10 = stable, 0.10–0.25 = slight shift, > 0.25 = major drift.
    """
    if len(reference_distribution) == 0 or len(current_distribution) == 0:
        return 0.0

    ref = np.asarray(reference_distribution, dtype=np.float64)
    cur = np.asarray(current_distribution, dtype=np.float64)

    all_vals = np.concatenate([ref, cur])
    min_v = float(all_vals.min())
    max_v = float(all_vals.max())

    if max_v == min_v:
        return 0.0

    bins = np.linspace(min_v, max_v, n_bins + 1)
    bins[-1] += 1e-9  # ensure max is included

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    ref_n = max(len(ref), 1)
    cur_n = max(len(cur), 1)

    psi = 0.0
    for e_cnt, a_cnt in zip(ref_counts, cur_counts):
        e = float(e_cnt) / ref_n
        a = float(a_cnt) / cur_n
        # Identical distributions → terms are 0 exactly (since a - e = 0)
        if e == a:
            continue
        e_safe = e if e > 0 else 1e-10
        a_safe = a if a > 0 else 1e-10
        psi += (a - e) * math.log(a_safe / e_safe)

    return float(psi)


def _build_model_version(as_of_date: dt.date, regime_label: Optional[str] = None) -> str:
    ts = as_of_date.isoformat().replace("-", "")
    suffix = f"_{regime_label.lower()}" if regime_label else "_global"
    return f"axiom_ml_{ts}{suffix}"


def run_training_job(
    as_of_date: Optional[dt.date] = None,
    min_samples: int = 50,
    retrain_regime_models: bool = True,
) -> Dict[str, Any]:
    """Full model training lifecycle.

    Returns a training report with status, metrics, PSI, and regime model results.
    """
    as_of = as_of_date or dt.date.today()

    # Step 1: Load training data
    X, y, symbols = load_training_dataset(as_of, min_samples=min_samples)

    if X is None:
        n = 0
        return {
            "status": "insufficient_data",
            "sample_count": n,
            "model_version": None,
            "test_metrics": {},
            "regime_models_trained": [],
            "psi_score": 0.0,
            "drift_warning": False,
        }

    sample_count = len(X)

    # Step 2: Split
    X_train, X_test, y_train, y_test = split_train_test_purged(X, y, symbols)

    # Step 3: Train global model
    try:
        model = train_signal_model(X_train, y_train, regime_label=None)
    except Exception as exc:
        logger.error("training_job.train_failed error=%s", exc)
        return {"status": "error", "error": str(exc), "sample_count": sample_count}

    # Step 4: Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    metrics["sample_count"] = sample_count

    # Step 5: Compute reference feature distribution for drift detection
    feature_names = get_feature_names()
    feature_ref_hist: Dict[str, list] = {}
    feature_ref_bins: Dict[str, list] = {}
    feature_importances: Dict[str, float] = {}

    for i, fname in enumerate(feature_names):
        col = X_train[:, i]
        counts, bins = np.histogram(col, bins=10)
        n_train = max(len(col), 1)
        feature_ref_hist[fname] = (counts / n_train).tolist()
        feature_ref_bins[fname] = bins.tolist()

    # Step 6: Compute permutation feature importances
    try:
        from api.axiom.ml.explainability import compute_feature_importance
        if len(X_test) >= 5:
            feature_importances = compute_feature_importance(
                model, X_test, y_test, feature_names
            )
    except Exception:
        feature_importances = {}

    # Step 7: Compute PSI comparing train vs test distributions
    psi_vals = []
    for i, fname in enumerate(feature_names):
        col_train = X_train[:, i]
        col_test = X_test[:, i] if len(X_test) > 0 else col_train
        psi_vals.append(compute_psi(col_train, col_test))
    psi_score = float(np.mean(psi_vals)) if psi_vals else 0.0
    drift_warning = psi_score > 0.25

    # Step 8: Save and register global model
    model_version = _build_model_version(as_of, regime_label=None)
    model_path = str(ML_MODEL_DIR / f"{model_version}.pkl")
    model_metadata = {
        "model_version": model_version,
        "regime_label": None,
        "trained_at": as_of.isoformat(),
        "sample_count": sample_count,
        "feature_ref_hist": feature_ref_hist,
        "feature_ref_bins": feature_ref_bins,
        "feature_importances": feature_importances,
        **metrics,
    }
    save_model(model, model_path, model_metadata)
    register_model(
        model_path=model_path,
        model_version=model_version,
        regime_label=None,
        metrics={**metrics, "psi_score": psi_score},
    )

    # Step 9: Regime-specific models
    regime_models_trained: List[str] = []
    if retrain_regime_models:
        # Regime label is in the feature vector: regime_is_trending / regime_is_high_vol
        trend_idx = feature_names.index("regime_is_trending")
        highvol_idx = feature_names.index("regime_is_high_vol")

        regime_masks = {
            "TRENDING": X[:, trend_idx] > 0.5,
            "HIGH_VOL": X[:, highvol_idx] > 0.5,
            "RECOVERY": (X[:, trend_idx] <= 0.5) & (X[:, highvol_idx] <= 0.5) & (X[:, highvol_idx] >= 0.0),
            "CHOPPY": (X[:, trend_idx] <= 0.5) & (X[:, highvol_idx] <= 0.5),
        }

        for regime, mask in regime_masks.items():
            X_r = X[mask]
            y_r = y[mask]
            if len(X_r) < _MIN_REGIME_SAMPLES:
                continue
            try:
                X_r_train, X_r_test, y_r_train, y_r_test = split_train_test_purged(X_r, y_r, [])
                m_r = train_signal_model(X_r_train, y_r_train, regime_label=regime)
                m_r_metrics = evaluate_model(m_r, X_r_test, y_r_test) if len(X_r_test) > 0 else {}
                m_r_metrics["sample_count"] = len(X_r)
                rv = _build_model_version(as_of, regime_label=regime)
                rpath = str(ML_MODEL_DIR / f"{rv}.pkl")
                save_model(m_r, rpath, {"model_version": rv, "regime_label": regime, **m_r_metrics})
                register_model(rpath, rv, regime, m_r_metrics)
                regime_models_trained.append(regime)
            except Exception as exc:
                logger.warning("training_job.regime_train_failed regime=%s error=%s", regime, exc)

    return {
        "status": "trained",
        "model_version": model_version,
        "sample_count": sample_count,
        "test_metrics": metrics,
        "regime_models_trained": regime_models_trained,
        "psi_score": round(psi_score, 6),
        "drift_warning": drift_warning,
    }
