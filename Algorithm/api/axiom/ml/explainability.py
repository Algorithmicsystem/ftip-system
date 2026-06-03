"""Phase 9.6: Lightweight Permutation Importance Explainability.

Avoids SHAP dependency by using permutation importance (accuracy drop per feature)
and local importance (global importance × feature deviation from neutral 0.5).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def compute_feature_importance(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Compute permutation importance for each feature.

    importance_score = baseline_accuracy - accuracy_with_feature_i_shuffled.
    Sorted descending by importance score.
    """
    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    importances: List[float] = []
    rng = random.Random(42)

    for i in range(X_test.shape[1]):
        X_perm = X_test.copy()
        col = X_perm[:, i].tolist()
        rng.shuffle(col)
        X_perm[:, i] = col
        perm_acc = accuracy_score(y_test, model.predict(X_perm))
        importances.append(baseline_acc - perm_acc)

    return dict(
        sorted(
            zip(feature_names, importances),
            key=lambda kv: -kv[1],
        )
    )


def explain_prediction(
    feature_array: np.ndarray,
    feature_names: List[str],
    feature_importances: List[float],
    top_n: int = 5,
) -> Dict[str, Any]:
    """Compute local explanation for a single prediction.

    local_importance[i] = feature_importances[i] × |feature_array[i] - 0.5|

    Returns top_n features driving this specific prediction.
    """
    local_scores: List[tuple] = []
    for i, (name, imp) in enumerate(zip(feature_names, feature_importances)):
        val = float(feature_array[i]) if i < len(feature_array) else 0.5
        local_imp = float(imp) * abs(val - 0.5)
        direction = "bullish" if val > 0.5 else "bearish"
        local_scores.append((name, local_imp, direction, imp))

    local_scores.sort(key=lambda x: -x[1])
    top = local_scores[:top_n]

    drivers = [
        {
            "feature": name,
            "direction": direction,
            "importance": round(local_imp, 6),
        }
        for name, local_imp, direction, _ in top
    ]

    if drivers:
        top_names = [d["feature"] for d in drivers[:2]]
        explanation_text = (
            f"ML signal driven primarily by {top_names[0]}"
            + (f" and {top_names[1]}" if len(top_names) > 1 else "")
        )
    else:
        explanation_text = "ML signal — insufficient feature data"

    return {
        "top_drivers": drivers,
        "explanation_text": explanation_text,
    }
