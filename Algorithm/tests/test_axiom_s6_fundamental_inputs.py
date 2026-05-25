"""Session 6: earnings intelligence wired into FundamentalCandidateInputs and engine."""
from __future__ import annotations

import pytest

from api.axiom.contracts import AxiomEngineInput, FundamentalCandidateInputs
from api.axiom.engines.fundamental import score_fundamental_reality
from api.axiom.mappers.from_normalized_bundle import build_axiom_engine_input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_input(**fundamental_overrides) -> AxiomEngineInput:
    base = dict(
        latest_close=100.0,
        analyst_target_price=120.0,
        pe_ratio=15.0,
        peg_ratio=1.2,
        gross_margin=0.45,
        operating_margin=0.18,
        net_margin=0.12,
        return_on_assets=0.08,
        return_on_equity=0.18,
        positive_fcf_ratio=0.75,
        free_cash_flow_margin=0.12,
        current_ratio=1.8,
        cash_ratio=0.5,
        debt_to_equity=0.6,
        liabilities_to_assets=0.4,
        profitability_strength=0.7,
        balance_sheet_resilience=0.72,
        cash_flow_durability=0.68,
        filing_recency_days=45.0,
        reporting_completeness_score=0.8,
        reporting_quality_proxy=0.75,
        coverage_score=70.0,
        provider_confidence=65.0,
        statement_coverage_flags={"income": True, "balance": True, "cashflow": True},
    )
    base.update(fundamental_overrides)
    candidate = FundamentalCandidateInputs(**base)
    return AxiomEngineInput(
        framework_version="test",
        symbol="TEST",
        as_of="2024-01-01",
        fundamental=candidate,
    )


def _bundle_with_earnings(
    earnings_intel: dict | None = None,
    alpha_overview_extras: dict | None = None,
    event_estimate_revision: float | None = None,
) -> dict:
    provider_snapshot: dict = {}
    if earnings_intel is not None:
        provider_snapshot["alphavantage_earnings_intel"] = earnings_intel
    if alpha_overview_extras:
        provider_snapshot["alphavantage_overview"] = alpha_overview_extras

    event: dict = {}
    if event_estimate_revision is not None:
        event["estimate_revision_support"] = event_estimate_revision

    return {
        "fundamental_filing": {
            "provider_snapshot": provider_snapshot,
            "normalized_metrics": {},
            "quality_proxies": {},
            "durability_proxies": {},
        },
        "event_catalyst_risk": event,
        "market_price_volume": {},
        "liquidity_execution_fragility": {},
        "market_breadth_internals": {},
        "cross_asset_confirmation": {},
        "stress_spillover_conditions": {},
        "quality_provenance": {},
        "canonical_alpha_core": {},
        "raw_supporting_fields": {},
    }


# ---------------------------------------------------------------------------
# Mapper: extraction from provider_snapshot
# ---------------------------------------------------------------------------

def test_mapper_extracts_quarterly_earnings_growth_yoy():
    bundle = _bundle_with_earnings(alpha_overview_extras={"quarterly_earnings_growth_yoy": 0.18})
    result = build_axiom_engine_input(bundle)
    assert result.fundamental.quarterly_earnings_growth_yoy == pytest.approx(0.18)


def test_mapper_extracts_beat_rate_from_earnings_intel():
    bundle = _bundle_with_earnings(earnings_intel={"beat_rate_4q": 0.75, "miss_rate_4q": 0.25})
    result = build_axiom_engine_input(bundle)
    assert result.fundamental.earnings_beat_rate_4q == pytest.approx(0.75)
    assert result.fundamental.earnings_miss_rate_4q == pytest.approx(0.25)


def test_mapper_extracts_avg_surprise_pct():
    bundle = _bundle_with_earnings(earnings_intel={"average_surprise_pct_4q": 4.2})
    result = build_axiom_engine_input(bundle)
    assert result.fundamental.earnings_avg_surprise_pct == pytest.approx(4.2)


def test_mapper_extracts_estimate_revision_support_from_earnings_intel():
    bundle = _bundle_with_earnings(earnings_intel={"estimate_revision_support": 0.6})
    result = build_axiom_engine_input(bundle)
    assert result.fundamental.earnings_estimate_revision_support == pytest.approx(0.6)


def test_mapper_falls_back_estimate_revision_to_event_domain():
    bundle = _bundle_with_earnings(event_estimate_revision=0.45)
    result = build_axiom_engine_input(bundle)
    assert result.fundamental.earnings_estimate_revision_support == pytest.approx(0.45)


def test_mapper_earnings_intel_takes_priority_over_event_domain():
    bundle = _bundle_with_earnings(
        earnings_intel={"estimate_revision_support": 0.8},
        event_estimate_revision=0.3,
    )
    result = build_axiom_engine_input(bundle)
    assert result.fundamental.earnings_estimate_revision_support == pytest.approx(0.8)


def test_mapper_extracts_freshness_status():
    bundle = _bundle_with_earnings(earnings_intel={"freshness_status": "fresh"})
    result = build_axiom_engine_input(bundle)
    assert result.fundamental.earnings_freshness == "fresh"


def test_mapper_earnings_fields_none_when_absent():
    bundle = _bundle_with_earnings()
    result = build_axiom_engine_input(bundle)
    assert result.fundamental.quarterly_earnings_growth_yoy is None
    assert result.fundamental.earnings_beat_rate_4q is None
    assert result.fundamental.earnings_miss_rate_4q is None
    assert result.fundamental.earnings_avg_surprise_pct is None
    assert result.fundamental.earnings_estimate_revision_support is None
    assert result.fundamental.earnings_freshness is None


# ---------------------------------------------------------------------------
# Engine: earnings_quality_component influences score
# ---------------------------------------------------------------------------

def test_engine_score_higher_with_strong_earnings():
    strong = _make_engine_input(
        earnings_beat_rate_4q=0.85,
        earnings_avg_surprise_pct=8.0,
        earnings_estimate_revision_support=0.9,
        quarterly_earnings_growth_yoy=0.35,
    )
    weak = _make_engine_input(
        earnings_beat_rate_4q=0.25,
        earnings_avg_surprise_pct=-4.0,
        earnings_estimate_revision_support=0.1,
        quarterly_earnings_growth_yoy=-0.1,
    )
    strong_result = score_fundamental_reality(strong)
    weak_result = score_fundamental_reality(weak)
    assert strong_result.score is not None
    assert weak_result.score is not None
    assert strong_result.score > weak_result.score


def test_engine_earnings_component_present_in_components():
    inp = _make_engine_input(
        earnings_beat_rate_4q=0.6,
        earnings_avg_surprise_pct=3.0,
        earnings_estimate_revision_support=0.5,
    )
    result = score_fundamental_reality(inp)
    assert "earnings_quality_component" in result.components
    assert result.components["earnings_quality_component"] is not None


def test_engine_score_computed_without_earnings_fields():
    inp = _make_engine_input()
    result = score_fundamental_reality(inp)
    assert result.score is not None
    assert result.status in ("available", "partial")
    assert "earnings_quality_component" not in result.components


def test_engine_earnings_component_in_summary_when_present():
    inp = _make_engine_input(
        earnings_beat_rate_4q=0.7,
        earnings_estimate_revision_support=0.6,
    )
    result = score_fundamental_reality(inp)
    assert "earnings quality" in result.summary.lower()


def test_engine_summary_omits_earnings_when_absent():
    inp = _make_engine_input()
    result = score_fundamental_reality(inp)
    assert "earnings quality" not in result.summary.lower()
