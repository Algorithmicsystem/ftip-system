"""Prompt 2 integrity tests: look-ahead bias, IP protection, friction, WAR engine,
regime transitions, dossier IQ, ensemble, moat score."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

MIGRATIONS = Path(__file__).resolve().parents[1] / "api" / "migrations"
WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"


# ---------------------------------------------------------------------------
# Section 1 — Look-ahead bias fix
# ---------------------------------------------------------------------------

class TestLookAheadBiasFix:

    def test_snapshot_uses_45day_lag_for_null_report_date(self):
        """_load_fundamentals must not allow report_date IS NULL to bypass lag."""
        src = (Path(__file__).parents[1] / "api" / "research" / "snapshot.py").read_text()
        assert "report_date IS NULL AND fiscal_period_end" in src, \
            "snapshot.py must use 45-day lag when report_date IS NULL"
        assert "INTERVAL '45 days'" in src, \
            "snapshot.py must apply 45-day lag for NULL report_date"

    def test_snapshot_does_not_use_old_null_or_pattern(self):
        src = (Path(__file__).parents[1] / "api" / "research" / "snapshot.py").read_text()
        assert "report_date IS NULL OR report_date" not in src, \
            "Old look-ahead pattern must be removed from snapshot.py"

    def test_migration_096_exists(self):
        assert (MIGRATIONS / "096_lookahead_bias_fix.sql").exists()


# ---------------------------------------------------------------------------
# Section 2 — IP Protection
# ---------------------------------------------------------------------------

class TestIPProtection:

    def test_signal_serializer_exists(self):
        path = Path(__file__).parents[1] / "api" / "axiom" / "signal_serializer.py"
        assert path.exists(), "api/axiom/signal_serializer.py must exist"

    def test_safe_signal_response_strips_thresholds(self):
        from api.axiom.signal_serializer import safe_signal_response
        payload = {
            "symbol": "AAPL",
            "signal": "BUY",
            "score": 0.3,
            "thresholds": {"buy": 0.20, "sell": -0.20},
            "stacked_meta": {"stack_weights": {}, "components": {}},
            "features": {"rsi14": 55.0},
        }
        result = safe_signal_response(payload)
        assert "thresholds" not in result
        assert "stacked_meta" not in result
        assert "features" not in result
        assert result["symbol"] == "AAPL"
        assert result["signal"] == "BUY"

    def test_validate_response_safety_detects_leak(self):
        from api.axiom.signal_serializer import validate_response_safety
        safe = {"symbol": "AAPL", "signal": "BUY", "dau": 70.0}
        unsafe = {"symbol": "AAPL", "thresholds": {"buy": 0.2}}
        assert validate_response_safety(safe) is True
        assert validate_response_safety(unsafe) is False

    def test_internal_fields_defined(self):
        from api.axiom.signal_serializer import _INTERNAL_FIELDS
        for field in ("thresholds", "stacked_meta", "calibration_meta", "component_support"):
            assert field in _INTERNAL_FIELDS


# ---------------------------------------------------------------------------
# Section 3 — Friction Engine
# ---------------------------------------------------------------------------

class TestFrictionEngine:

    def test_fallback_friction_class_exists(self):
        from api.jobs.axiom_backtest import FallbackFriction
        assert FallbackFriction is not None

    def test_fallback_friction_returns_float(self):
        from api.jobs.axiom_backtest import FallbackFriction
        bps = FallbackFriction.estimate_bps(70.0)
        assert isinstance(bps, float)
        assert bps > 0

    def test_compute_net_return_deducts_friction(self):
        from api.jobs.axiom_backtest import compute_net_return
        net, bps = compute_net_return(0.01, 70.0)
        assert net < 0.01, "Net return must be less than gross return after friction"
        assert bps > 0

    def test_backtest_stats_has_sharpe_ratio_net(self):
        from api.jobs.axiom_backtest import compute_backtest_stats
        signals = [
            {"symbol": "AAPL", "as_of_date": None, "regime_label": "TRENDING",
             "dau": 70.0, "signal_label": "BUY"},
        ]
        import datetime as dt
        d0 = dt.date(2024, 1, 2)
        d1 = dt.date(2024, 1, 23)
        price_map = {"AAPL": {d0: 180.0, d1: 190.0}}
        signals[0]["as_of_date"] = d0
        stats = compute_backtest_stats(signals, price_map, 21, ["BUY", "SELL"])
        assert "sharpe_ratio_net" in stats
        assert "friction_applied_bps" in stats

    def test_backtest_stats_has_avg_net_return_pct(self):
        from api.jobs.axiom_backtest import compute_backtest_stats
        import datetime as dt
        d0 = dt.date(2024, 1, 2)
        d1 = dt.date(2024, 1, 23)
        signals = [{"symbol": "AAPL", "as_of_date": d0, "regime_label": "TRENDING",
                    "dau": 70.0, "signal_label": "BUY"}]
        price_map = {"AAPL": {d0: 180.0, d1: 190.0}}
        stats = compute_backtest_stats(signals, price_map, 21, ["BUY", "SELL"])
        assert "avg_net_return_pct" in stats


# ---------------------------------------------------------------------------
# Section 4 — Suppression direction preservation
# ---------------------------------------------------------------------------

class TestSuppressionDirection:

    def test_canonical_signal_has_pre_suppression_action(self):
        src = (Path(__file__).parents[1] / "api" / "alpha" / "canonical_signal.py").read_text()
        assert "pre_suppression_action" in src

    def test_canonical_signal_has_suppression_active(self):
        src = (Path(__file__).parents[1] / "api" / "alpha" / "canonical_signal.py").read_text()
        assert "suppression_active" in src

    def test_build_signal_includes_suppression_fields(self):
        from api.alpha.canonical_signal import build_signal_from_features
        features = {
            "rsi14": 55.0, "mom_5": 0.05, "mom_21": 0.08, "mom_63": 0.12,
            "trend_sma20_50": 0.03, "volume_z20": 0.5, "sentiment_score": 0.1,
            "maxdd_63d": -0.05, "atr_pct": 0.015, "volatility_ann": 0.20,
            "event_overhang_score": 95.0, "earnings_window_flag": True,
            "implementation_fragility_score": 80.0, "market_stress_score": 80.0,
        }
        result = build_signal_from_features(features, symbol="AAPL")
        assert "pre_suppression_action" in result
        assert "suppression_active" in result
        assert "suppression_reason" in result


# ---------------------------------------------------------------------------
# Section 5 — Signal WAR Engine
# ---------------------------------------------------------------------------

class TestSignalWAREngine:

    def test_compute_stats_uses_full_war_formula(self):
        from api.intelligence.signal_memory import _compute_stats_from_pnl_rows
        rows = [
            {"horizon_days": 21, "return_pct": 0.05, "hit": True, "regime": "TRENDING"},
            {"horizon_days": 21, "return_pct": -0.02, "hit": False, "regime": "TRENDING"},
            {"horizon_days": 21, "return_pct": 0.03, "hit": True, "regime": "CHOPPY"},
            {"horizon_days": 21, "return_pct": 0.04, "hit": True, "regime": "TRENDING"},
            {"horizon_days": 21, "return_pct": 0.06, "hit": True, "regime": "TRENDING"},
        ]
        stats = _compute_stats_from_pnl_rows(rows)
        assert stats["signal_war"] is not None
        assert "war_ic_component" in stats
        assert "league_avg" in stats
        # With 5 samples at 4/5 = 0.80 BA and sqrt(5) ≈ 2.24
        # WAR should be larger than simple (BA - baseline)
        assert abs(stats["signal_war"]) > abs(0.80 - 0.52)

    def test_war_endpoint_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/signal-war/AAPL")
        assert r.status_code == 200

    def test_war_endpoint_has_signal_war_field(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/signal-war/AAPL")
        data = r.json()
        assert "signal_war" in data
        assert "sample_count" in data

    def test_war_leaderboard_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/signal-war/leaderboard")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Section 6 — Regime Transition Intelligence
# ---------------------------------------------------------------------------

class TestRegimeTransitions:

    def test_regime_transitions_module_exists(self):
        path = Path(__file__).parents[1] / "api" / "intelligence" / "regime_transitions.py"
        assert path.exists()

    def test_compute_regime_transition_probabilities_returns_dict(self):
        from api.intelligence.regime_transitions import compute_regime_transition_probabilities
        result = compute_regime_transition_probabilities("TRENDING")
        assert "current_regime" in result
        assert "transition_probabilities" in result
        assert "confidence" in result

    def test_identify_warning_signals_returns_list(self):
        from api.intelligence.regime_transitions import identify_warning_signals
        import datetime as dt
        warnings = identify_warning_signals(dt.date.today())
        assert isinstance(warnings, list)

    def test_regime_transitions_endpoint_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/regime-transitions/TRENDING")
        assert r.status_code == 200

    def test_regime_warnings_endpoint_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/regime-warnings")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Section 7 — Dossier IQ
# ---------------------------------------------------------------------------

class TestDossierIQ:

    def test_dossier_event_weights_defined(self):
        from api.intelligence.company_dossier import DOSSIER_EVENT_WEIGHTS
        assert "earnings_beat" in DOSSIER_EVENT_WEIGHTS
        assert "eis_deterioration" in DOSSIER_EVENT_WEIGHTS
        assert DOSSIER_EVENT_WEIGHTS["earnings_beat"] > DOSSIER_EVENT_WEIGHTS["sector_rotation"]

    def test_recency_half_life_defined(self):
        from api.intelligence.company_dossier import _RECENCY_HALF_LIFE_DAYS
        assert _RECENCY_HALF_LIFE_DAYS == 90.0

    def test_compute_iq_score_with_events(self):
        from api.intelligence.company_dossier import _compute_iq_score
        import datetime as dt
        recent_events = [
            {"event_type": "earnings_beat", "event_date": dt.date.today().isoformat()},
            {"event_type": "eis_deterioration", "event_date": dt.date.today().isoformat()},
        ]
        old_events = [
            {"event_type": "earnings_beat", "event_date": "2020-01-01"},
            {"event_type": "eis_deterioration", "event_date": "2020-01-01"},
        ]
        iq_recent = _compute_iq_score(252, 2, 0.6, recent_events)
        iq_old = _compute_iq_score(252, 2, 0.6, old_events)
        assert iq_recent > iq_old, "Recent events should yield higher IQ than old events"

    def test_dossier_endpoint_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/dossier/AAPL")
        assert r.status_code == 200

    def test_dossier_has_intelligence_quality_score(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/dossier/AAPL")
        data = r.json()
        assert "intelligence_quality_score" in data


# ---------------------------------------------------------------------------
# Section 9 — Ensemble Score
# ---------------------------------------------------------------------------

class TestEnsembleScore:

    def test_ensemble_module_exists(self):
        path = Path(__file__).parents[1] / "api" / "axiom" / "ml" / "ensemble.py"
        assert path.exists()

    def test_ensemble_result_dataclass(self):
        from api.axiom.ml.ensemble import EnsembleResult
        import datetime as dt
        result = EnsembleResult(
            symbol="AAPL",
            as_of_date=dt.date.today(),
            rule_dau=70.0,
            ml_probability=0.65,
            ensemble_dau=68.5,
            rule_weight=0.70,
            ml_weight=0.30,
            ic_composite=0.05,
            model_version="test_v1",
            blend_method="fixed_blend",
        )
        assert result.ensemble_dau == 68.5

    def test_compute_ensemble_dau_returns_result(self):
        from api.axiom.ml.ensemble import compute_ensemble_dau
        result = compute_ensemble_dau("AAPL", 70.0)
        assert result.symbol == "AAPL"
        assert result.rule_dau == 70.0
        assert 0.0 <= result.ensemble_dau <= 100.0

    def test_get_ensemble_status_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/axiom/ml/ensemble-status")
        assert r.status_code == 200

    def test_ensemble_status_has_model_version(self):
        with TestClient(app) as client:
            r = client.get("/axiom/ml/ensemble-status")
        data = r.json()
        assert "model_version" in data
        assert "blend_method" in data


# ---------------------------------------------------------------------------
# Section 10 — Morning Briefing narrative quality
# ---------------------------------------------------------------------------

class TestMorningBriefingNarrative:

    def test_build_text_has_four_sections(self):
        from api.jobs.morning_briefing import _build_text
        text = _build_text(
            regime="TRENDING",
            breadth_state="BROAD",
            n_favorable=15,
            top_symbol="AAPL",
            top_dau=75.0,
            top_driver="EIF",
            risk_symbol="GME",
            risk_type="elevated fragility",
            top_factor="EIF",
            sri=52.0,
            ic_state="MODERATE",
            sample_count=450,
        )
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        assert len(paragraphs) >= 4, "Morning briefing must have at least 4 narrative sections"

    def test_build_text_mentions_regime(self):
        from api.jobs.morning_briefing import _build_text
        text = _build_text(
            regime="TRENDING", breadth_state="BROAD", n_favorable=10,
            top_symbol=None, top_dau=0.0, top_driver="EIF",
            risk_symbol=None, risk_type="", top_factor="EIF",
            sri=50.0, ic_state="MODERATE", sample_count=100,
        )
        assert "Trending" in text or "TRENDING" in text or "trending" in text

    def test_briefing_endpoint_returns_briefing_text(self, monkeypatch):
        monkeypatch.setenv("FTIP_API_KEY", "test-key")
        with TestClient(app) as client:
            r = client.get(
                "/jobs/briefing/morning",
                headers={"X-FTIP-API-Key": "test-key"},
            )
        assert r.status_code == 200
        data = r.json()
        assert len(data.get("briefing_text", "")) > 20


# ---------------------------------------------------------------------------
# Section 11 — Moat Score
# ---------------------------------------------------------------------------

class TestMoatScore:

    def test_moat_score_endpoint_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/moat-score")
        assert r.status_code == 200

    def test_moat_score_has_components(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/moat-score")
        data = r.json()
        assert "components" in data
        components = data["components"]
        assert "signal_war_database" in components
        assert "regime_playbook_depth" in components
        assert "dossier_completeness" in components
        assert "connection_graph_density" in components

    def test_moat_score_has_irreproducibility(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/moat-score")
        data = r.json()
        assert "data_irreproducibility_estimate" in data

    def test_moat_score_has_moat_strength(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/moat-score")
        data = r.json()
        assert data["moat_strength"] in ("building", "established", "strong", "exceptional")


# ---------------------------------------------------------------------------
# Migration files
# ---------------------------------------------------------------------------

class TestMigrations:

    def test_migration_096_exists(self):
        assert (MIGRATIONS / "096_lookahead_bias_fix.sql").exists()

    def test_migration_097_exists(self):
        assert (MIGRATIONS / "097_ml_signal_predictions.sql").exists()

    def test_migration_097_has_ml_predictions_table(self):
        sql = (MIGRATIONS / "097_ml_signal_predictions.sql").read_text()
        assert "ml_signal_predictions" in sql
        assert "prediction_score" in sql
