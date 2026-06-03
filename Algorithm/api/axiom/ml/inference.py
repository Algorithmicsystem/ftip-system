"""Phase 9.4: Inference Engine.

Loads the active ML model and generates predictions for assembled AXIOM features.
Fails gracefully — when no model is trained, returns ml_available=False with no errors.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from api.axiom.ml.feature_builder import AxiomMLFeatureVector, feature_vector_to_array
from api.axiom.ml.model_registry import get_active_model, get_model_version

logger = logging.getLogger(__name__)

_ML_PREDICT_THRESHOLD = 0.55  # conservative threshold for positive class


def predict_signal(
    feature_vector: AxiomMLFeatureVector,
    regime_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ML inference on the given feature vector.

    Returns a dict with ml_prediction, ml_confidence, ml_model_version, ml_available.
    When no model is trained: ml_available=False, no errors raised.
    """
    _no_model = {
        "ml_prediction": None,
        "ml_confidence": None,
        "ml_model_version": "no_model_trained",
        "ml_available": False,
        "ml_agrees_with_axiom": None,
        "ml_explanation": None,
    }

    try:
        model, metadata = get_active_model(regime_label)
        if model is None:
            return _no_model

        model_version = metadata.get("model_version") or get_model_version()
        X = feature_vector_to_array(feature_vector, fill_value=0.5).reshape(1, -1)
        proba = float(model.predict_proba(X)[0, 1])
        prediction = 1 if proba > _ML_PREDICT_THRESHOLD else 0

        # DAU is at index 10 in the feature vector (deployable_alpha_utility, normalized)
        dau_normalized = feature_vector.deployable_alpha_utility
        axiom_bullish = (dau_normalized is not None and dau_normalized > 0.5) or False
        ml_agrees = (prediction == 1 and axiom_bullish) or (prediction == 0 and not axiom_bullish)

        # Local explanation
        try:
            from api.axiom.ml.explainability import explain_prediction
            from api.axiom.ml.feature_builder import get_feature_names
            feature_names = get_feature_names()
            # Use feature importances from metadata if available
            feat_imp = metadata.get("feature_importances") or {}
            if feat_imp:
                imp_list = [feat_imp.get(n, 0.0) for n in feature_names]
                explanation = explain_prediction(
                    feature_array=X[0],
                    feature_names=feature_names,
                    feature_importances=imp_list,
                    top_n=5,
                )
            else:
                explanation = None
        except Exception:
            explanation = None

        return {
            "ml_prediction": prediction,
            "ml_confidence": round(proba, 4),
            "ml_model_version": str(model_version),
            "ml_available": True,
            "ml_agrees_with_axiom": ml_agrees,
            "ml_explanation": explanation,
        }

    except Exception as exc:
        logger.warning("predict_signal failed regime=%s error=%s", regime_label, exc)
        return _no_model


def compute_ml_signal_boost(
    axiom_dau: float,
    ml_prediction: Dict[str, Any],
) -> float:
    """Compute a DAU adjustment based on ML confirmation/disagreement.

    Only adjusts when ML has high-confidence signal (>= 0.60).
    +5.0: ML agrees with AXIOM at high confidence (confirmation)
    -8.0: ML disagrees with AXIOM at high confidence (override signal)
    0.0: ML unavailable or low confidence
    """
    if not ml_prediction.get("ml_available"):
        return 0.0

    confidence = ml_prediction.get("ml_confidence") or 0.0
    if confidence < 0.60:
        return 0.0

    agrees = ml_prediction.get("ml_agrees_with_axiom")
    if agrees is True:
        return 5.0
    if agrees is False:
        return -8.0
    return 0.0
