"""Phase 41: Intelligence Memory and Compounding Data Moat tests."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# TestSignalMemory
# ---------------------------------------------------------------------------

class TestSignalMemory:
    def test_compute_stats_empty_returns_nones(self):
        from api.intelligence.signal_memory import _compute_stats_from_pnl_rows
        result = _compute_stats_from_pnl_rows([])
        assert result["batting_average_5d"] is None
        assert result["signal_war"] is None
        assert result["sample_count"] == 0

    def test_compute_stats_below_min_samples(self):
        from api.intelligence.signal_memory import _compute_stats_from_pnl_rows
        rows = [{"horizon_days": 5, "return_pct": 0.02, "hit": True, "regime": "TRENDING"}] * 4
        result = _compute_stats_from_pnl_rows(rows)
        assert result["batting_average_5d"] is None

    def test_compute_stats_batting_average(self):
        from api.intelligence.signal_memory import _compute_stats_from_pnl_rows
        rows = [
            {"horizon_days": 5, "return_pct": 0.03, "hit": True, "regime": "TRENDING"},
            {"horizon_days": 5, "return_pct": -0.01, "hit": False, "regime": "TRENDING"},
            {"horizon_days": 5, "return_pct": 0.02, "hit": True, "regime": "TRENDING"},
            {"horizon_days": 5, "return_pct": 0.01, "hit": True, "regime": "TRENDING"},
            {"horizon_days": 5, "return_pct": -0.02, "hit": False, "regime": "CHOPPY"},
        ]
        result = _compute_stats_from_pnl_rows(rows)
        assert result["batting_average_5d"] == pytest.approx(0.6, abs=0.01)

    def test_signal_war_above_spy_baseline(self):
        from api.intelligence.signal_memory import _compute_stats_from_pnl_rows
        rows = [
            {"horizon_days": 5, "return_pct": 0.03, "hit": True, "regime": "TRENDING"},
        ] * 8
        result = _compute_stats_from_pnl_rows(rows)
        # 100% batting average — war should be +0.48
        assert result["signal_war"] == pytest.approx(0.48, abs=0.01)

    def test_regime_breakdown_populated(self):
        from api.intelligence.signal_memory import _compute_stats_from_pnl_rows
        rows = [
            {"horizon_days": 5, "return_pct": 0.02, "hit": True, "regime": "TRENDING"},
            {"horizon_days": 5, "return_pct": 0.02, "hit": True, "regime": "TRENDING"},
            {"horizon_days": 5, "return_pct": -0.01, "hit": False, "regime": "CHOPPY"},
            {"horizon_days": 5, "return_pct": -0.01, "hit": False, "regime": "CHOPPY"},
            {"horizon_days": 5, "return_pct": 0.01, "hit": True, "regime": "TRENDING"},
        ]
        result = _compute_stats_from_pnl_rows(rows)
        assert "TRENDING" in result["regime_breakdown"]
        assert "CHOPPY" in result["regime_breakdown"]

    def test_compute_batting_no_db(self):
        from api.intelligence.signal_memory import compute_signal_batting_average
        with patch("api.intelligence.signal_memory.db.db_read_enabled", return_value=False):
            result = compute_signal_batting_average("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["batting_average_5d"] is None

    def test_leaderboard_no_db(self):
        from api.intelligence.signal_memory import get_signal_leaderboard
        with patch("api.intelligence.signal_memory.db.db_read_enabled", return_value=False):
            result = get_signal_leaderboard()
        assert result == []

    def test_update_archive_no_db(self):
        from api.intelligence.signal_memory import update_signal_performance_archive
        with patch("api.intelligence.signal_memory.db.db_read_enabled", return_value=False):
            result = update_signal_performance_archive(dt.date.today())
        assert result["updated"] == 0


# ---------------------------------------------------------------------------
# TestRegimePlaybook
# ---------------------------------------------------------------------------

class TestRegimePlaybook:
    def test_build_playbook_no_db(self):
        from api.intelligence.regime_playbook import build_regime_playbook
        with patch("api.intelligence.regime_playbook.db.db_read_enabled", return_value=False):
            result = build_regime_playbook()
        assert result == {}

    def test_compute_transition_probabilities_empty(self):
        from api.intelligence.regime_playbook import _compute_transition_probabilities
        result = _compute_transition_probabilities("TRENDING", [])
        assert result == {}

    def test_compute_transition_probabilities(self):
        from api.intelligence.regime_playbook import _compute_transition_probabilities
        transitions = [
            {"from_regime": "TRENDING", "to_regime": "CHOPPY"},
            {"from_regime": "TRENDING", "to_regime": "CHOPPY"},
            {"from_regime": "TRENDING", "to_regime": "HIGH_VOL"},
        ]
        result = _compute_transition_probabilities("TRENDING", transitions)
        assert result["CHOPPY"] == pytest.approx(0.6667, abs=0.001)
        assert result["HIGH_VOL"] == pytest.approx(0.3333, abs=0.001)

    def test_build_entry_insufficient_samples(self):
        from api.intelligence.regime_playbook import _build_entry_from_signal_rows
        result = _build_entry_from_signal_rows("TRENDING", [], [], 0.0)
        assert result is None

    def test_build_entry_accuracy(self):
        from api.intelligence.regime_playbook import _build_entry_from_signal_rows
        # Each row: (sym, date, label, dau, regime, driver, h5, h21, batting=None, slug=None, war=None)
        row_hit = ("AAPL", dt.date.today(), "BUY", 70.0, "TRENDING", "momentum",
                   0.03, 0.05, 0.8, 0.03, 0.28)
        row_miss = ("AAPL", dt.date.today(), "BUY", 70.0, "TRENDING", "value",
                    -0.01, -0.02, 0.0, 0.0, -0.52)
        rows = [row_hit] * 4 + [row_miss]
        entry = _build_entry_from_signal_rows("TRENDING", rows, [], 30.0)
        assert entry is not None
        assert entry.historical_accuracy == pytest.approx(0.8, abs=0.01)

    def test_get_regime_recommendation_unknown(self):
        from api.intelligence.regime_playbook import get_regime_recommendation
        result = get_regime_recommendation("UNKNOWN_REGIME", {})
        assert result["regime"] == "UNKNOWN_REGIME"
        assert result["expected_batting_average"] == 0.5
        assert result["most_likely_next_regime"] == "unknown"

    def test_update_playbook_no_db(self):
        from api.intelligence.regime_playbook import update_regime_playbook
        with patch("api.intelligence.regime_playbook.db.db_read_enabled", return_value=False):
            result = update_regime_playbook(dt.date.today())
        assert result["regimes_updated"] == 0


# ---------------------------------------------------------------------------
# TestCompanyDossier
# ---------------------------------------------------------------------------

class TestCompanyDossier:
    def _make_payload(self, **overrides) -> Dict:
        base = {
            "deployable_alpha_utility": 72.0,
            "regime_label": "TRENDING",
            "engine_scores": {
                "fundamental_reality": {
                    "components": {
                        "earnings_quality_component": 65.0,
                        "caps_component": 70.0,
                    }
                },
                "critical_fragility": {
                    "components": {
                        "scps_component": 30.0,
                        "pess_component": 40.0,
                    }
                },
            },
            "engine_inputs": {
                "fundamental": {
                    "days_to_next_earnings": 90,
                }
            },
        }
        base.update(overrides)
        return base

    def test_eis_deterioration_detected(self):
        from api.intelligence.company_dossier import _detect_events_from_payloads
        current = self._make_payload()
        prior = self._make_payload()
        prior["engine_scores"]["fundamental_reality"]["components"]["earnings_quality_component"] = 80.0
        events = _detect_events_from_payloads("AAPL", dt.date.today(), current, prior)
        types = [e.event_type for e in events]
        assert "eis_deterioration" in types

    def test_eis_recovery_detected(self):
        from api.intelligence.company_dossier import _detect_events_from_payloads
        current = self._make_payload()
        prior = self._make_payload()
        prior["engine_scores"]["fundamental_reality"]["components"]["earnings_quality_component"] = 50.0
        events = _detect_events_from_payloads("AAPL", dt.date.today(), current, prior)
        types = [e.event_type for e in events]
        assert "eis_recovery" in types

    def test_caps_downgrade_detected(self):
        from api.intelligence.company_dossier import _detect_events_from_payloads
        current = self._make_payload()
        prior = self._make_payload()
        prior["engine_scores"]["fundamental_reality"]["components"]["caps_component"] = 82.0
        events = _detect_events_from_payloads("AAPL", dt.date.today(), current, prior)
        types = [e.event_type for e in events]
        assert "caps_downgrade" in types

    def test_scps_spike_detected(self):
        from api.intelligence.company_dossier import _detect_events_from_payloads
        current = self._make_payload()
        prior = self._make_payload()
        current["engine_scores"]["critical_fragility"]["components"]["scps_component"] = 75.0
        prior["engine_scores"]["critical_fragility"]["components"]["scps_component"] = 55.0
        events = _detect_events_from_payloads("AAPL", dt.date.today(), current, prior)
        types = [e.event_type for e in events]
        assert "scps_spike" in types

    def test_scps_normalization_detected(self):
        from api.intelligence.company_dossier import _detect_events_from_payloads
        current = self._make_payload()
        prior = self._make_payload()
        current["engine_scores"]["critical_fragility"]["components"]["scps_component"] = 40.0
        prior["engine_scores"]["critical_fragility"]["components"]["scps_component"] = 75.0
        events = _detect_events_from_payloads("AAPL", dt.date.today(), current, prior)
        types = [e.event_type for e in events]
        assert "scps_normalization" in types

    def test_regime_transition_detected(self):
        from api.intelligence.company_dossier import _detect_events_from_payloads
        current = self._make_payload()
        prior = self._make_payload()
        prior["regime_label"] = "CHOPPY"
        events = _detect_events_from_payloads("AAPL", dt.date.today(), current, prior)
        types = [e.event_type for e in events]
        assert "regime_transition" in types

    def test_earnings_stress_detected(self):
        from api.intelligence.company_dossier import _detect_events_from_payloads
        current = self._make_payload()
        prior = self._make_payload()
        current["engine_scores"]["critical_fragility"]["components"]["pess_component"] = 70.0
        current["engine_inputs"]["fundamental"]["days_to_next_earnings"] = 30
        events = _detect_events_from_payloads("AAPL", dt.date.today(), current, prior)
        types = [e.event_type for e in events]
        assert "earnings_stress" in types

    def test_compute_iq_score(self):
        from api.intelligence.company_dossier import _compute_iq_score
        # Full year of data, 50+ events, perfect accuracy
        score = _compute_iq_score(252, 50, 1.0)
        assert score == pytest.approx(100.0, abs=0.1)

    def test_compute_iq_score_zero(self):
        from api.intelligence.company_dossier import _compute_iq_score
        score = _compute_iq_score(0, 0, 0.0)
        assert score == 0.0

    def test_get_dossier_no_db(self):
        from api.intelligence.company_dossier import get_company_dossier
        with patch("api.intelligence.company_dossier.db.db_read_enabled", return_value=False):
            result = get_company_dossier("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["event_count"] == 0
        assert result["intelligence_quality_score"] == 0.0


# ---------------------------------------------------------------------------
# TestIntelligenceGraph
# ---------------------------------------------------------------------------

class TestIntelligenceGraph:
    def test_build_symbol_sector_graph_nodes(self):
        from api.intelligence.intelligence_graph import build_symbol_sector_graph
        symbols = ["AAPL", "MSFT", "GOOGL"]
        sector_map = {"AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology"}
        result = build_symbol_sector_graph(symbols, sector_map)
        nodes = result["nodes"]
        assert "sym_AAPL" in nodes
        assert "sym_MSFT" in nodes
        assert "sec_Technology" in nodes

    def test_build_sector_member_edges(self):
        from api.intelligence.intelligence_graph import build_symbol_sector_graph
        symbols = ["AAPL", "MSFT"]
        sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
        result = build_symbol_sector_graph(symbols, sector_map)
        edge_types = [e.edge_type for e in result["edges"]]
        assert "sector_member" in edge_types

    def test_competes_with_intra_sector(self):
        from api.intelligence.intelligence_graph import build_symbol_sector_graph
        symbols = ["AAPL", "MSFT"]
        sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
        result = build_symbol_sector_graph(symbols, sector_map)
        edge_types = [e.edge_type for e in result["edges"]]
        assert "competes_with" in edge_types

    def test_propagate_stress_basic(self):
        from api.intelligence.intelligence_graph import (
            GraphEdge,
            propagate_stress_through_graph,
        )
        edges = [
            GraphEdge("e1", "A", "B", "competes_with", weight=1.0, direction="bidirectional"),
            GraphEdge("e2", "B", "C", "competes_with", weight=1.0, direction="bidirectional"),
        ]
        result = propagate_stress_through_graph("A", 100.0, edges, propagation_depth=2)
        assert "B" in result
        assert result["B"] == pytest.approx(60.0, abs=0.1)
        assert "C" in result
        assert result["C"] == pytest.approx(36.0, abs=0.1)

    def test_propagate_stress_no_edges(self):
        from api.intelligence.intelligence_graph import propagate_stress_through_graph
        result = propagate_stress_through_graph("A", 100.0, [], propagation_depth=2)
        assert result == {}

    def test_find_hidden_connections(self):
        from api.intelligence.intelligence_graph import GraphEdge, find_hidden_connections
        edges = [
            GraphEdge("e1", "sym_AAPL", "sym_MSFT", "competes_with", weight=0.7, direction="bidirectional"),
            GraphEdge("e2", "sym_MSFT", "sym_GOOGL", "competes_with", weight=0.7, direction="bidirectional"),
        ]
        result = find_hidden_connections("AAPL", edges, max_hops=3)
        assert "sym_MSFT" in result["direct_connections"]
        assert "sym_GOOGL" in result["indirect_connections"]

    def test_compute_network_centrality(self):
        from api.intelligence.intelligence_graph import GraphEdge, compute_network_centrality
        edges = [
            GraphEdge("e1", "A", "B", "competes_with", weight=1.0, direction="bidirectional"),
            GraphEdge("e2", "A", "C", "competes_with", weight=1.0, direction="bidirectional"),
        ]
        centrality = compute_network_centrality(edges)
        # A connects to both B and C → degree 2; n=3 nodes → centrality = 2/2 = 1.0
        assert centrality["A"] == pytest.approx(1.0, abs=0.01)
        assert centrality["B"] == pytest.approx(0.5, abs=0.01)

    def test_seed_symbol_graph_no_db(self):
        from api.intelligence.intelligence_graph import seed_symbol_graph
        with patch("api.intelligence.intelligence_graph.db.db_read_enabled", return_value=False):
            result = seed_symbol_graph(["AAPL", "MSFT"])
        assert result["nodes_created"] == 0
        assert result["edges_created"] == 0


# ---------------------------------------------------------------------------
# TestMoatScore
# ---------------------------------------------------------------------------

class TestMoatScore:
    def test_moat_score_no_db(self):
        from api.intelligence.intelligence_routes import compute_moat_score
        with patch("api.intelligence.intelligence_routes.db.db_read_enabled", return_value=False):
            result = compute_moat_score()
        assert result["moat_strength"] == "building"
        assert result["total_signals_archived"] == 0

    def test_moat_strength_building(self):
        from api.intelligence.intelligence_routes import _moat_strength
        assert _moat_strength(0) == "building"
        assert _moat_strength(89) == "building"

    def test_moat_strength_established(self):
        from api.intelligence.intelligence_routes import _moat_strength
        assert _moat_strength(90) == "established"
        assert _moat_strength(364) == "established"

    def test_moat_strength_strong(self):
        from api.intelligence.intelligence_routes import _moat_strength
        assert _moat_strength(365) == "strong"

    def test_moat_strength_exceptional(self):
        from api.intelligence.intelligence_routes import _moat_strength
        assert _moat_strength(730) == "exceptional"
        assert _moat_strength(1000) == "exceptional"
