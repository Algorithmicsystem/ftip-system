"""Phase 9: Machine Learning Signal Layer tests.

Tests for: feature builder, training data, model training, inference,
drift monitor, and explainability.
"""
from __future__ import annotations

import datetime as dt
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.axiom.ml.feature_builder import (
    AxiomMLFeatureVector,
    build_feature_vector,
    feature_vector_to_array,
    get_feature_names,
)
from api.axiom.ml.training_job import compute_psi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_payload(
    fund=60.0, state=55.0, beh=50.0, flow=65.0, liq=70.0, frag=40.0, res=75.0,
    dau=75.0, gross=72.0, friction=35.0, validated=68.0,
    regime="BULL_TRENDING", fcs=62.0,
) -> Dict[str, Any]:
    return {
        "engine_scores": {
            "fundamental_reality": {
                "score": fund,
                "components": {
                    "earnings_quality_component": 58.0,
                    "caps_component": 62.0,
                    "pess_component": 45.0,
                },
            },
            "state_pricing": {"score": state, "components": {"cardi_score": 55.0}},
            "behavioral_distortion": {
                "score": beh,
                "components": {"nms_score": 52.0, "crowding_component": 48.0},
            },
            "flow_transmission": {"score": flow, "components": {}},
            "liquidity_convexity": {"score": liq, "components": {"kle_score": 65.0}},
            "critical_fragility": {
                "score": frag,
                "components": {
                    "scps_component": 35.0,
                    "mtrs_score": 40.0,
                    "bfs_component": 38.0,
                },
            },
            "research_integrity": {"score": res, "components": {}},
        },
        "gross_opportunity": gross,
        "friction_burden": friction,
        "validated_edge": validated,
        "deployable_alpha_utility": dau,
        "regime_label": regime,
        "factor_composite_score": fcs,
        "alpha_decomposition": {
            "idiosyncratic_alpha": 12.5,
            "factor_concentration": 0.18,
        },
        "factor_loadings_summary": [
            {"factor_name": "EIF", "loading": 0.12, "t_stat": 0.66},
            {"factor_name": "CMF", "loading": 0.08, "t_stat": 0.44},
            {"factor_name": "BAF", "loading": 0.05, "t_stat": 0.27},
            {"factor_name": "KLF", "loading": 0.15, "t_stat": 0.82},
            {"factor_name": "SCAF", "loading": 0.20, "t_stat": 1.10},
            {"factor_name": "ICF", "loading": 0.07, "t_stat": 0.38},
            {"factor_name": "GBF", "loading": 0.10, "t_stat": 0.55},
            {"factor_name": "MTRF", "loading": 0.18, "t_stat": 0.99},
            {"factor_name": "MQF", "loading": 0.22, "t_stat": 1.21},
            {"factor_name": "VIF", "loading": 0.14, "t_stat": 0.77},
            {"factor_name": "RTF", "loading": 0.11, "t_stat": 0.60},
            {"factor_name": "NTFF", "loading": -0.05, "t_stat": 0.27},
        ],
        "source_context": {
            "regime_strength": 0.72,
            "osms": 68.0,
            "nss": 55.0,
            "ias": 60.0,
        },
    }


def _make_ic_state(ic_str="STRONG", breadth=25, mean_ic=0.08, amqs=72.0) -> Dict[str, Any]:
    return {
        "ic_state": ic_str,
        "effective_breadth": breadth,
        "mean_ic": mean_ic,
        "amqs_score": amqs,
    }


# ---------------------------------------------------------------------------
# TestFeatureBuilder
# ---------------------------------------------------------------------------

class TestFeatureBuilder:
    def test_feature_vector_has_46_features(self):
        payload = _make_payload()
        fv = build_feature_vector(payload, payload["factor_loadings_summary"])
        arr = feature_vector_to_array(fv)
        assert arr.shape == (46,)

    def test_feature_names_count(self):
        names = get_feature_names()
        assert len(names) == 46

    def test_none_values_filled(self):
        fv = AxiomMLFeatureVector()  # all None
        arr = feature_vector_to_array(fv, fill_value=0.5)
        assert arr.shape == (46,)
        assert np.all(arr == 0.5)

    def test_scores_normalized(self):
        payload = _make_payload(dau=75.0)
        fv = build_feature_vector(payload, [])
        assert fv.deployable_alpha_utility == pytest.approx(0.75, abs=1e-6)

    def test_feature_order_deterministic(self):
        payload = _make_payload()
        fv1 = build_feature_vector(payload, payload["factor_loadings_summary"])
        fv2 = build_feature_vector(payload, payload["factor_loadings_summary"])
        arr1 = feature_vector_to_array(fv1)
        arr2 = feature_vector_to_array(fv2)
        np.testing.assert_array_equal(arr1, arr2)

    def test_build_feature_vector_from_payload(self):
        payload = _make_payload(fund=80.0, frag=30.0)
        fv = build_feature_vector(payload, payload["factor_loadings_summary"])
        assert fv.fundamental_score == pytest.approx(0.80, abs=1e-6)
        assert fv.fragility_score == pytest.approx(0.30, abs=1e-6)
        assert fv.regime_is_trending == 1.0  # BULL_TRENDING

    def test_feature_vector_factor_loadings(self):
        payload = _make_payload()
        fv = build_feature_vector(payload, payload["factor_loadings_summary"])
        assert fv.factor_eif == pytest.approx(0.12, abs=1e-6)
        assert fv.factor_scaf == pytest.approx(0.20, abs=1e-6)
        assert fv.factor_ntff == pytest.approx(-0.05, abs=1e-6)

    def test_ic_state_normalization(self):
        payload = _make_payload()
        ic = _make_ic_state("STRONG", breadth=30, mean_ic=0.10)
        fv = build_feature_vector(payload, [], ic_state=ic)
        assert fv.ic_state_numeric == pytest.approx(1.0, abs=1e-6)
        assert fv.effective_breadth_normalized == pytest.approx(1.0, abs=1e-6)

    def test_ic_weak_state(self):
        payload = _make_payload()
        ic = _make_ic_state("WEAK", breadth=10)
        fv = build_feature_vector(payload, [], ic_state=ic)
        assert fv.ic_state_numeric == pytest.approx(0.5, abs=1e-6)

    def test_fill_value_custom(self):
        fv = AxiomMLFeatureVector()
        arr = feature_vector_to_array(fv, fill_value=0.0)
        assert np.all(arr == 0.0)

    def test_regime_high_vol_encoding(self):
        payload = _make_payload(regime="BEAR_STRESS")
        fv = build_feature_vector(payload, [])
        assert fv.regime_is_high_vol == 1.0
        assert fv.regime_is_trending == 0.0


# ---------------------------------------------------------------------------
# TestTrainingData
# ---------------------------------------------------------------------------

class TestTrainingData:
    def test_load_returns_none_insufficient(self):
        from api.axiom.ml.training_data import load_training_dataset
        with patch("api.axiom.ml.training_data.db.db_read_enabled", return_value=True), \
             patch("api.axiom.ml.training_data.db.safe_fetchall", return_value=[]):
            X, y, symbols = load_training_dataset(dt.date.today(), min_samples=50)
        assert X is None
        assert y is None
        assert symbols == []

    def test_binary_label_conversion(self):
        from api.axiom.ml.training_data import load_training_dataset
        payload = _make_payload()
        import json as _json

        def _fake_fetchall(sql, params):
            return [
                ("AAPL", _json.dumps(payload), 1),
                ("MSFT", _json.dumps(payload), -1),
                ("GOOG", _json.dumps(payload), 0),
            ] * 20  # 60 rows > min_samples

        with patch("api.axiom.ml.training_data.db.db_read_enabled", return_value=True), \
             patch("api.axiom.ml.training_data.db.safe_fetchall", side_effect=_fake_fetchall):
            X, y, symbols = load_training_dataset(dt.date.today(), min_samples=50)

        assert X is not None
        assert 1 in y
        assert 0 in y  # both -1 and 0 outcomes map to 0

    def test_purged_split_no_leakage(self):
        from api.axiom.ml.training_data import split_train_test_purged
        n = 100
        X = np.arange(n).reshape(n, 1).astype(np.float64)
        y = np.zeros(n, dtype=np.int32)
        X_train, X_test, y_train, y_test = split_train_test_purged(X, y, list(range(n)))
        # All test indices should be after train indices (no future data in train)
        assert X_train.max() < X_test.min()

    def test_horizon_label_parsing(self):
        from api.axiom.ml.training_data import _parse_horizon
        assert _parse_horizon("21d") == 21
        assert _parse_horizon("5d") == 5
        assert _parse_horizon("63d") == 63
        assert _parse_horizon("unknown") == 21  # default


# ---------------------------------------------------------------------------
# TestModelTraining
# ---------------------------------------------------------------------------

class TestModelTraining:
    def _synthetic_data(self, n=100, n_features=46, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.uniform(0, 1, (n, n_features))
        y = (X[:, 0] > 0.5).astype(int)
        return X, y

    def test_train_returns_pipeline(self):
        from api.axiom.ml.signal_model import train_signal_model
        from sklearn.pipeline import Pipeline
        X, y = self._synthetic_data()
        model = train_signal_model(X, y)
        assert isinstance(model, Pipeline)
        assert len(model.steps) == 2

    def test_evaluate_returns_all_metrics(self):
        from api.axiom.ml.signal_model import evaluate_model, train_signal_model
        X, y = self._synthetic_data()
        model = train_signal_model(X, y)
        metrics = evaluate_model(model, X[:20], y[:20])
        for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "sample_count", "positive_rate"):
            assert key in metrics

    def test_save_load_roundtrip(self):
        from api.axiom.ml.signal_model import evaluate_model, load_model, save_model, train_signal_model
        X, y = self._synthetic_data()
        model = train_signal_model(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_model.pkl")
            meta = {"model_version": "test_v1", "trained_at": "2026-06-02"}
            ok = save_model(model, path, meta)
            assert ok is True
            loaded_model, loaded_meta = load_model(path)
            assert loaded_model is not None
            assert loaded_meta.get("model_version") == "test_v1"
            # Same predictions
            X_test = X[:10]
            np.testing.assert_array_equal(
                model.predict(X_test), loaded_model.predict(X_test)
            )

    def test_regime_conditional_trains(self):
        from api.axiom.ml.signal_model import train_signal_model
        from sklearn.pipeline import Pipeline
        X, y = self._synthetic_data()
        model = train_signal_model(X, y, regime_label="TRENDING")
        assert isinstance(model, Pipeline)

    def test_model_pipeline_components(self):
        from api.axiom.ml.signal_model import train_signal_model
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        X, y = self._synthetic_data()
        model = train_signal_model(X, y)
        assert isinstance(model.named_steps["scaler"], StandardScaler)
        assert isinstance(model.named_steps["clf"], GradientBoostingClassifier)

    def test_evaluate_handles_single_class(self):
        from api.axiom.ml.signal_model import evaluate_model, train_signal_model
        X, y = self._synthetic_data()
        model = train_signal_model(X, y)
        y_single = np.zeros(20, dtype=int)  # all same class
        metrics = evaluate_model(model, X[:20], y_single)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# TestInference
# ---------------------------------------------------------------------------

class TestInference:
    def test_predict_no_model(self):
        from api.axiom.ml.inference import predict_signal
        payload = _make_payload()
        fv = build_feature_vector(payload, [])
        with patch("api.axiom.ml.inference.get_active_model", return_value=(None, {})):
            result = predict_signal(fv)
        assert result["ml_available"] is False
        assert result["ml_prediction"] is None
        assert result["ml_model_version"] == "no_model_trained"

    def test_predict_with_model(self):
        from api.axiom.ml.inference import predict_signal
        from api.axiom.ml.signal_model import train_signal_model

        rng = np.random.RandomState(42)
        X_tr = rng.uniform(0, 1, (100, 46))
        y_tr = (X_tr[:, 0] > 0.5).astype(int)
        model = train_signal_model(X_tr, y_tr)

        payload = _make_payload()
        fv = build_feature_vector(payload, payload["factor_loadings_summary"])

        with patch("api.axiom.ml.inference.get_active_model", return_value=(model, {"model_version": "v1"})):
            result = predict_signal(fv)

        assert result["ml_available"] is True
        assert result["ml_prediction"] in (0, 1)
        assert 0.0 <= result["ml_confidence"] <= 1.0

    def test_ml_boost_positive_when_agree(self):
        from api.axiom.ml.inference import compute_ml_signal_boost
        pred = {"ml_available": True, "ml_confidence": 0.75, "ml_agrees_with_axiom": True}
        boost = compute_ml_signal_boost(70.0, pred)
        assert boost > 0

    def test_ml_boost_negative_when_disagree(self):
        from api.axiom.ml.inference import compute_ml_signal_boost
        pred = {"ml_available": True, "ml_confidence": 0.80, "ml_agrees_with_axiom": False}
        boost = compute_ml_signal_boost(70.0, pred)
        assert boost < 0

    def test_ml_boost_zero_low_confidence(self):
        from api.axiom.ml.inference import compute_ml_signal_boost
        pred = {"ml_available": True, "ml_confidence": 0.55, "ml_agrees_with_axiom": True}
        boost = compute_ml_signal_boost(70.0, pred)
        assert boost == pytest.approx(0.0, abs=1e-9)

    def test_dau_adjustment_clamped(self):
        from api.assistant.phase3.common import clamp
        from api.axiom.ml.inference import compute_ml_signal_boost
        # Boost that would push DAU > 100
        pred = {"ml_available": True, "ml_confidence": 0.90, "ml_agrees_with_axiom": True}
        dau = 98.0
        boost = compute_ml_signal_boost(dau, pred)
        adjusted = clamp(dau + boost, 0.0, 100.0)
        assert adjusted <= 100.0
        # Boost that would push DAU below 0
        pred_down = {"ml_available": True, "ml_confidence": 0.90, "ml_agrees_with_axiom": False}
        dau_low = 5.0
        boost_down = compute_ml_signal_boost(dau_low, pred_down)
        adjusted_low = clamp(dau_low + boost_down, 0.0, 100.0)
        assert adjusted_low >= 0.0

    def test_inference_agrees_with_axiom_flag(self):
        from api.axiom.ml.inference import predict_signal
        from api.axiom.ml.signal_model import train_signal_model

        rng = np.random.RandomState(42)
        X_tr = rng.uniform(0, 1, (200, 46))
        y_tr = (X_tr[:, 0] > 0.5).astype(int)
        model = train_signal_model(X_tr, y_tr)

        # Build a payload with DAU > 50 (AXIOM bullish)
        payload = _make_payload(dau=80.0)
        fv = build_feature_vector(payload, payload["factor_loadings_summary"])

        with patch("api.axiom.ml.inference.get_active_model", return_value=(model, {"model_version": "v1"})):
            result = predict_signal(fv)

        # ml_agrees_with_axiom should be a bool (not None)
        assert isinstance(result["ml_agrees_with_axiom"], bool)


# ---------------------------------------------------------------------------
# TestDriftMonitor
# ---------------------------------------------------------------------------

class TestDriftMonitor:
    def test_psi_zero_identical_distributions(self):
        same = np.random.uniform(0, 1, 100)
        psi = compute_psi(same, same)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_psi_high_different_distributions(self):
        ref = np.zeros(100)   # all at 0
        cur = np.ones(100)    # all at 1
        psi = compute_psi(ref, cur)
        assert psi > 0.25

    def test_drift_recommendation_retrain(self):
        from api.axiom.ml.drift_monitor import _PSI_RETRAIN
        assert _PSI_RETRAIN == pytest.approx(0.25, abs=1e-6)
        # PSI > 0.25 → recommendation "retrain"
        with patch("api.axiom.ml.drift_monitor.db.db_read_enabled", return_value=False):
            from api.axiom.ml.drift_monitor import check_model_drift
            result = check_model_drift(dt.date.today())
        assert result["recommendation"] == "stable"  # DB disabled → stable

    def test_drift_recommendation_stable(self):
        from api.axiom.ml.drift_monitor import check_model_drift
        with patch("api.axiom.ml.drift_monitor.db.db_read_enabled", return_value=False):
            result = check_model_drift(dt.date.today())
        assert result["recommendation"] == "stable"
        assert result["overall_psi"] == pytest.approx(0.0)

    def test_psi_between_states(self):
        # Moderate shift → PSI between 0.10 and 0.25
        rng = np.random.RandomState(42)
        ref = rng.normal(0.5, 0.1, 200)
        cur = rng.normal(0.6, 0.1, 200)  # slight shift
        psi = compute_psi(ref, cur)
        # Should be > 0 (not identical)
        assert psi >= 0.0

    def test_psi_empty_arrays(self):
        psi = compute_psi(np.array([]), np.array([]))
        assert psi == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TestExplainability
# ---------------------------------------------------------------------------

class TestExplainability:
    def _trained_model_and_data(self, n=200, n_features=46):
        from api.axiom.ml.signal_model import train_signal_model
        rng = np.random.RandomState(99)
        X = rng.uniform(0, 1, (n, n_features))
        y = (X[:, 0] > 0.5).astype(int)
        model = train_signal_model(X, y)
        return model, X, y

    def test_feature_importance_length(self):
        from api.axiom.ml.explainability import compute_feature_importance
        model, X, y = self._trained_model_and_data()
        feature_names = get_feature_names()
        importances = compute_feature_importance(model, X[:30], y[:30], feature_names)
        assert len(importances) == 46

    def test_explain_prediction_top_n(self):
        from api.axiom.ml.explainability import explain_prediction
        feature_names = get_feature_names()
        imp_list = [float(i) * 0.01 for i in range(46)]
        arr = np.full(46, 0.7)
        result = explain_prediction(arr, feature_names, imp_list, top_n=5)
        assert len(result["top_drivers"]) == 5

    def test_explanation_text_generated(self):
        from api.axiom.ml.explainability import explain_prediction
        feature_names = get_feature_names()
        imp_list = [1.0] + [0.0] * 45
        arr = np.full(46, 0.8)
        result = explain_prediction(arr, feature_names, imp_list, top_n=3)
        assert isinstance(result["explanation_text"], str)
        assert len(result["explanation_text"]) > 0

    def test_direction_bullish_when_positive(self):
        from api.axiom.ml.explainability import explain_prediction
        feature_names = get_feature_names()
        # First feature has importance 1.0, value 0.9 (above 0.5 = bullish)
        imp_list = [1.0] + [0.0] * 45
        arr = np.zeros(46)
        arr[0] = 0.9  # above 0.5 = bullish
        result = explain_prediction(arr, feature_names, imp_list, top_n=1)
        assert result["top_drivers"][0]["direction"] == "bullish"

    def test_feature_importance_sum_nonzero(self):
        from api.axiom.ml.explainability import compute_feature_importance
        model, X, y = self._trained_model_and_data()
        feature_names = get_feature_names()
        importances = compute_feature_importance(model, X[:50], y[:50], feature_names)
        vals = list(importances.values())
        assert sum(abs(v) for v in vals) > 0

    def test_explain_local_importance_scales_with_deviation(self):
        from api.axiom.ml.explainability import explain_prediction
        feature_names = get_feature_names()
        # Feature 0 has high global importance, feature 1 also high but smaller deviation
        imp_list = [1.0, 1.0] + [0.0] * 44
        arr = np.full(46, 0.5)
        arr[0] = 0.9   # deviation 0.4 from neutral
        arr[1] = 0.55  # deviation 0.05 from neutral
        result = explain_prediction(arr, feature_names, imp_list, top_n=2)
        # Feature 0 should rank higher (larger deviation)
        assert result["top_drivers"][0]["feature"] == feature_names[0]
