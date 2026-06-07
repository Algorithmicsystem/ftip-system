"""Phase 9.5: ML Signal Layer API Routes.

Endpoints:
  POST /axiom/ml/train            — trigger training job
  GET  /axiom/ml/status           — model version + drift status
  GET  /axiom/ml/drift            — full drift analysis
  POST /axiom/ml/predict/{symbol} — inference for a symbol's latest scores
  GET  /axiom/ml/explain/{symbol} — ML explanation for most recent prediction
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query

from api import db, security
from api.axiom.ml.drift_monitor import check_model_drift
from api.axiom.ml.feature_builder import build_feature_vector, get_feature_names
from api.axiom.ml.inference import compute_ml_signal_boost, predict_signal
from api.axiom.ml.model_registry import get_active_model, get_model_version
from api.axiom.ml.training_data import MINIMUM_SAMPLES_INITIAL, MINIMUM_SAMPLES_PRODUCTION
from api.axiom.ml.training_job import run_training_job, _KELLY_MAX_BY_QUALITY

router = APIRouter(
    prefix="/axiom/ml",
    tags=["axiom_ml"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)
logger = logging.getLogger(__name__)


@router.post("/train")
def ml_train(
    min_samples: int = Query(default=MINIMUM_SAMPLES_INITIAL, ge=10),
    regime_models: bool = Query(default=True),
) -> Dict[str, Any]:
    """Trigger the ML training job for the current date."""
    report = run_training_job(
        as_of_date=dt.date.today(),
        min_samples=min_samples,
        retrain_regime_models=regime_models,
    )
    return report


@router.get("/status")
def ml_status() -> Dict[str, Any]:
    """Return current model version, active state, and basic drift indicator."""
    version = get_model_version()
    _, metadata = get_active_model(regime_label=None)
    has_model = version != "no_model_trained"

    result: Dict[str, Any] = {
        "model_version": version,
        "model_active": has_model,
    }
    if has_model and metadata:
        result["trained_at"] = metadata.get("trained_at")
        result["sample_count"] = metadata.get("sample_count")
        result["test_accuracy"] = metadata.get("accuracy")
        result["test_roc_auc"] = metadata.get("roc_auc")

    return result


@router.get("/training-status")
def ml_training_status() -> Dict[str, Any]:
    """Return rich training status including quality tier, Kelly cap, next training time."""
    version = get_model_version()
    _, metadata = get_active_model(regime_label=None)
    has_model = version != "no_model_trained"

    model_quality = metadata.get("model_quality", "insufficient") if has_model else "insufficient"
    sample_count = int(metadata.get("sample_count") or 0)
    cv_auc = float(metadata.get("roc_auc") or 0.0)
    feature_count = len(get_feature_names())
    kelly_max = _KELLY_MAX_BY_QUALITY.get(model_quality, 0.25)

    # Next Monday at 18:30 ET
    import datetime as _dt
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        tz = _dt.timezone.utc
    now = _dt.datetime.now(tz)
    days_until_monday = (7 - now.weekday()) % 7 or 7
    next_monday = (now + _dt.timedelta(days=days_until_monday)).replace(
        hour=18, minute=30, second=0, microsecond=0
    )
    next_training_at = next_monday.isoformat()

    # Ensemble mode
    ensemble_mode = "axiom_only"
    try:
        from api.axiom.ml.ensemble import get_ensemble_status
        ens = get_ensemble_status()
        ensemble_mode = ens.get("blend_method", "axiom_only")
    except Exception:
        pass

    return {
        "model_trained": has_model,
        "model_version": version,
        "model_quality": model_quality,
        "training_samples": sample_count,
        "samples_for_production": MINIMUM_SAMPLES_PRODUCTION,
        "samples_for_bootstrap": MINIMUM_SAMPLES_INITIAL,
        "cross_val_auc": round(cv_auc, 4),
        "feature_count": feature_count,
        "last_trained_at": metadata.get("trained_at") if has_model else None,
        "kelly_max_from_quality": kelly_max,
        "ensemble_mode": ensemble_mode,
        "next_training_at": next_training_at,
    }


@router.get("/drift")
def ml_drift(
    as_of_date: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Return full drift analysis comparing live vs training feature distributions."""
    if as_of_date:
        try:
            aod = dt.date.fromisoformat(as_of_date)
        except ValueError:
            aod = dt.date.today()
    else:
        aod = dt.date.today()

    return check_model_drift(aod)


@router.post("/predict/{symbol}")
def ml_predict(symbol: str) -> Dict[str, Any]:
    """Run ML inference for a symbol using its latest axiom_scores_daily row."""
    if not db.db_read_enabled():
        return {"ml_available": False, "symbol": symbol, "error": "db_disabled"}

    row = db.safe_fetchone(
        """
        SELECT payload, as_of_date
          FROM axiom_scores_daily
         WHERE symbol = %s
         ORDER BY as_of_date DESC
         LIMIT 1
        """,
        (symbol.upper(),),
    )
    if not row:
        return {"ml_available": False, "symbol": symbol, "error": "no_data"}

    payload_raw = row[0]
    as_of = row[1]
    if isinstance(payload_raw, str):
        try:
            payload = json.loads(payload_raw)
        except Exception:
            return {"ml_available": False, "symbol": symbol, "error": "payload_parse_error"}
    else:
        payload = payload_raw or {}

    factor_loadings = payload.get("factor_loadings_summary") or []
    try:
        fv = build_feature_vector(payload, factor_loadings, ic_state=None)
        regime = str(payload.get("regime_label") or "CHOPPY").upper()
        prediction = predict_signal(fv, regime_label=regime)
        dau = float(payload.get("deployable_alpha_utility") or 0.0)
        boost = compute_ml_signal_boost(dau, prediction)
    except Exception as exc:
        logger.warning("ml_predict failed symbol=%s error=%s", symbol, exc)
        return {"ml_available": False, "symbol": symbol, "error": str(exc)}

    return {
        "symbol": symbol,
        "as_of_date": str(as_of),
        "axiom_dau": dau,
        "ml_signal_boost": boost,
        **prediction,
    }


@router.get("/explain/{symbol}")
def ml_explain(symbol: str) -> Dict[str, Any]:
    """Return ML explanation for the most recent prediction for a symbol."""
    if not db.db_read_enabled():
        return {"ml_available": False, "symbol": symbol, "error": "db_disabled"}

    row = db.safe_fetchone(
        """
        SELECT payload
          FROM axiom_scores_daily
         WHERE symbol = %s
         ORDER BY as_of_date DESC
         LIMIT 1
        """,
        (symbol.upper(),),
    )
    if not row:
        return {"ml_available": False, "symbol": symbol, "error": "no_data"}

    payload_raw = row[0]
    if isinstance(payload_raw, str):
        try:
            payload = json.loads(payload_raw)
        except Exception:
            return {"ml_available": False, "symbol": symbol, "error": "payload_parse_error"}
    else:
        payload = payload_raw or {}

    factor_loadings = payload.get("factor_loadings_summary") or []
    try:
        fv = build_feature_vector(payload, factor_loadings, ic_state=None)
        regime = str(payload.get("regime_label") or "CHOPPY").upper()
        prediction = predict_signal(fv, regime_label=regime)
    except Exception as exc:
        return {"ml_available": False, "symbol": symbol, "error": str(exc)}

    explanation = prediction.get("ml_explanation")
    if not explanation and prediction.get("ml_available"):
        # Try to compute explanation directly
        model, metadata = get_active_model(regime_label=None)
        if model is not None:
            from api.axiom.ml.explainability import explain_prediction
            from api.axiom.ml.feature_builder import feature_vector_to_array
            feat_imp = metadata.get("feature_importances") or {}
            feature_names = get_feature_names()
            imp_list = [feat_imp.get(n, 0.0) for n in feature_names]
            arr = feature_vector_to_array(fv, fill_value=0.5)
            explanation = explain_prediction(arr, feature_names, imp_list, top_n=5)

    return {
        "symbol": symbol,
        "ml_available": prediction.get("ml_available", False),
        "ml_model_version": prediction.get("ml_model_version"),
        "ml_explanation": explanation,
    }
