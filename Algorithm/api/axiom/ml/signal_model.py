"""Phase 9.3: Model Training and Persistence.

GradientBoostingClassifier pipeline with StandardScaler.
scikit-learn only — no additional heavy dependencies.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def train_signal_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    regime_label: Optional[str] = None,
) -> Pipeline:
    """Train a StandardScaler → GradientBoostingClassifier pipeline.

    regime_label is metadata-only; the same architecture is used for all regimes.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            min_samples_leaf=10,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Return standard classification metrics for the test set."""
    y_pred = model.predict(X_test)
    sample_count = len(y_test)
    positive_rate = float(y_test.sum()) / max(sample_count, 1)

    # roc_auc requires both classes present
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = float(roc_auc_score(y_test, y_proba))
        if not math.isfinite(roc):
            roc = 0.5
    except (ValueError, Exception):
        roc = 0.5  # degenerate test set (single class)

    # precision/recall/f1 with zero_division guard
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc,
        "sample_count": sample_count,
        "positive_rate": positive_rate,
    }


def save_model(
    model: Pipeline,
    model_path: str,
    metadata: Dict[str, Any],
) -> bool:
    """Save model to pickle and metadata to companion JSON file.

    Returns True on success.
    """
    try:
        model_p = Path(model_path)
        model_p.parent.mkdir(parents=True, exist_ok=True)

        with open(model_p, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta_path = model_p.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return True
    except Exception as exc:
        logger.error("save_model failed path=%s error=%s", model_path, exc)
        return False


def load_model(model_path: str) -> Tuple[Optional[Pipeline], Dict[str, Any]]:
    """Load model from pickle + companion JSON metadata.

    Returns (model, metadata) or (None, {}) on failure.
    """
    try:
        model_p = Path(model_path)
        if not model_p.exists():
            return None, {}

        with open(model_p, "rb") as f:
            model = pickle.load(f)

        meta_path = model_p.with_suffix(".json")
        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        return model, metadata
    except Exception as exc:
        logger.error("load_model failed path=%s error=%s", model_path, exc)
        return None, {}
