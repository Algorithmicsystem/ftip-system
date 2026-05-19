from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from api.axiom.common import confidence_band
from api.axiom.contracts import AxiomArtifact, AxiomEngineInput, AxiomLineageBlock


AXIOM_LINEAGE_VERSION = "axiom50_phase4_lineage_v1"
AXIOM_LINEAGE_ARTIFACT_KIND = "assistant_axiom_lineage_artifact"

_ENGINE_COMPONENT_SOURCES: Mapping[str, Mapping[str, List[str]]] = {
    "fundamental_reality": {
        "valuation_gap_component": [
            "market_price_volume.latest_close",
            "fundamental_filing.provider_snapshot.alphavantage_overview.analyst_target_price",
            "fundamental_filing.provider_snapshot.alphavantage_overview.peg_ratio",
            "fundamental_filing.normalized_metrics.revenue_growth_yoy",
        ],
        "profitability_quality_component": [
            "fundamental_filing.quality_proxies.profitability_strength",
            "fundamental_filing.normalized_metrics.gross_margin",
            "fundamental_filing.normalized_metrics.operating_margin",
            "fundamental_filing.normalized_metrics.net_margin",
            "fundamental_filing.normalized_metrics.return_on_assets",
            "fundamental_filing.normalized_metrics.return_on_equity",
        ],
        "cashflow_quality_component": [
            "fundamental_filing.quality_proxies.cash_flow_durability",
            "fundamental_filing.normalized_metrics.positive_fcf_ratio",
            "fundamental_filing.normalized_metrics.free_cash_flow_margin",
        ],
        "balance_sheet_resilience_component": [
            "fundamental_filing.quality_proxies.balance_sheet_resilience",
            "fundamental_filing.normalized_metrics.current_ratio",
            "fundamental_filing.normalized_metrics.cash_ratio",
            "fundamental_filing.normalized_metrics.debt_to_equity",
            "fundamental_filing.normalized_metrics.liabilities_to_assets",
        ],
        "data_completeness_component": [
            "fundamental_filing.quality_proxies.reporting_completeness_score",
            "fundamental_filing.quality_proxies.reporting_quality_proxy",
            "fundamental_filing.meta.coverage_score",
            "fundamental_filing.provenance.confidence",
            "fundamental_filing.filing_recency_days",
        ],
    },
    "state_pricing": {
        "macro_alignment_component": [
            "macro_cross_asset",
            "feature_factor_bundle.macro_alignment",
            "feature_factor_bundle.composite_intelligence.Macro Alignment Score",
            "stress_spillover_conditions.market_stress_score",
        ],
        "factor_compensation_component": [
            "feature_factor_bundle.composite_intelligence.Opportunity Quality Score",
            "feature_factor_bundle.composite_intelligence.Cross-Domain Conviction Score",
            "feature_factor_bundle.composite_intelligence.Fundamental Durability Score",
            "feature_factor_bundle.domain_agreement",
            "cross_asset_relative_context",
            "market_price_volume.ret_21d",
        ],
        "discount_rate_regime_component": [
            "feature_factor_bundle.regime_intelligence",
            "feature_factor_bundle.macro_alignment",
            "stress_spillover_conditions.market_stress_score",
        ],
        "cross_asset_confirmation_component": [
            "cross_asset_confirmation",
            "market_breadth_internals",
            "feature_factor_bundle.cross_asset_relative_context",
        ],
        "bad_state_exposure_component": [
            "stress_spillover_conditions",
            "feature_factor_bundle.regime_intelligence",
            "event_catalyst_risk",
            "feature_factor_bundle.proprietary_scores.Signal Fragility Index",
        ],
        "state_pricing_conflict_component": [
            "feature_factor_bundle.domain_agreement",
            "cross_asset_confirmation.cross_asset_conflict_score",
            "feature_factor_bundle.sentiment_narrative_intelligence.contradiction_score",
        ],
    },
    "behavioral_distortion": {
        "narrative_intensity_component": [
            "sentiment_narrative_flow",
            "feature_factor_bundle.sentiment_narrative_intelligence.attention_intensity_score",
            "feature_factor_bundle.sentiment_narrative_intelligence.novelty_score",
            "event_catalyst_risk",
        ],
        "crowding_component": [
            "feature_factor_bundle.proprietary_scores.Narrative Crowding Index",
            "feature_factor_bundle.sentiment_narrative_intelligence.narrative_concentration_score",
            "event_catalyst_risk.event_overhang_score",
        ],
        "extrapolation_stretch_component": [
            "feature_factor_bundle.market_structure.trend_exhaustion_score",
            "feature_factor_bundle.sentiment_narrative_intelligence.hype_to_price_divergence_score",
            "event_catalyst_risk",
        ],
        "underreaction_continuation_component": [
            "feature_factor_bundle.sentiment_narrative_intelligence",
            "feature_factor_bundle.market_structure",
            "feature_factor_bundle.domain_agreement",
        ],
        "reversal_setup_component": [
            "market_price_volume.maxdd_63d",
            "market_price_volume.maxdd_126d",
            "feature_factor_bundle.sentiment_narrative_intelligence.negative_news_resilient_price_divergence",
            "fundamental_filing",
            "liquidity_execution_fragility",
        ],
        "contradiction_penalty_component": [
            "feature_factor_bundle.sentiment_narrative_intelligence.contradiction_score",
            "feature_factor_bundle.sentiment_narrative_intelligence.hype_to_price_divergence_score",
            "feature_factor_bundle.domain_agreement",
            "cross_asset_confirmation",
        ],
    },
    "flow_transmission": {
        "trend_quality_component": [
            "feature_factor_bundle.market_structure",
            "feature_factor_bundle.proprietary_scores.Market Structure Integrity Score",
        ],
        "transmission_strength_component": [
            "market_price_volume.ret_21d",
            "key_features.mom_vol_adj_21d",
            "feature_factor_bundle.market_structure",
            "feature_factor_bundle.composite_intelligence",
        ],
        "breadth_confirmation_component": [
            "market_breadth_internals",
            "cross_asset_confirmation",
            "feature_factor_bundle.cross_asset_relative_context",
        ],
        "market_structure_component": [
            "feature_factor_bundle.market_structure",
            "feature_factor_bundle.regime_intelligence",
            "feature_factor_bundle.domain_agreement",
        ],
        "flow_persistence_component": [
            "feature_factor_bundle.market_structure.directional_persistence_score",
            "feature_factor_bundle.cross_asset_relative_context",
            "feature_factor_bundle.market_structure.reversal_pressure_score",
        ],
        "conflict_penalty_component": [
            "feature_factor_bundle.domain_agreement",
            "feature_factor_bundle.macro_alignment",
            "cross_asset_confirmation",
            "stress_spillover_conditions",
        ],
    },
    "liquidity_convexity": {
        "liquidity_integrity_component": [
            "liquidity_execution_fragility",
            "deployment_readiness.risk_budgeting",
            "portfolio_construction.execution_quality_score",
        ],
        "liquidation_feasibility_component": [
            "liquidity_execution_fragility",
            "stress_spillover_conditions.market_stress_score",
            "deployment_readiness.model_readiness.live_readiness_score",
        ],
        "hedge_feasibility_component": [
            "cross_asset_confirmation",
            "macro_cross_asset",
            "feature_factor_bundle.cross_asset_relative_context",
            "stress_spillover_conditions",
        ],
        "convexity_value_component": [
            "liquidity_execution_fragility",
            "event_catalyst_risk",
            "fundamental_filing",
            "source_context.option_surface_available",
        ],
        "implied_vs_fundamental_risk_component": [
            "fundamental_filing",
            "event_catalyst_risk",
            "stress_spillover_conditions",
            "feature_factor_bundle.proprietary_scores.Signal Fragility Index",
        ],
        "execution_penalty_component": [
            "liquidity_execution_fragility",
            "portfolio_construction.execution_quality_score",
            "stress_spillover_conditions",
        ],
    },
    "critical_fragility": {
        "volatility_instability_component": [
            "market_price_volume.realized_vol_21d",
            "market_price_volume.realized_vol_63d",
            "market_price_volume.vol_of_vol_proxy",
            "feature_factor_bundle.fragility_intelligence.instability_score",
        ],
        "gap_jump_risk_component": [
            "market_price_volume.gap_instability_10d",
            "market_price_volume.abs_gap_mean_10d",
            "market_price_volume.gap_pct",
            "liquidity_execution_fragility.overnight_gap_risk_score",
        ],
        "drawdown_fragility_component": [
            "market_price_volume.maxdd_21d",
            "market_price_volume.maxdd_63d",
            "market_price_volume.maxdd_126d",
            "feature_factor_bundle.fragility_intelligence.drawdown_sensitivity_score",
        ],
        "crowding_fragility_component": [
            "feature_factor_bundle.proprietary_scores.Narrative Crowding Index",
            "event_catalyst_risk",
        ],
        "liquidity_fragility_component": [
            "liquidity_execution_fragility",
            "feature_factor_bundle.fragility_intelligence.anomaly_pressure_score",
        ],
        "regime_transition_risk_component": [
            "feature_factor_bundle.regime_intelligence",
            "stress_spillover_conditions",
            "market_breadth_internals",
            "cross_asset_confirmation",
        ],
    },
    "research_integrity": {
        "evidence_quality_component": [
            "evaluation.signal_scorecard",
            "evaluation.calibration_summary",
            "deployment_readiness.model_readiness",
            "quality_provenance",
        ],
        "out_of_sample_reliability_component": [
            "canonical_validation.net_return_summary",
            "canonical_validation.walkforward_summary",
            "canonical_validation.readiness_scorecard",
            "canonical_validation.suppression_effect_summary",
        ],
        "calibration_component": [
            "evaluation.calibration_summary",
            "evaluation.ranking_scorecard",
            "canonical_validation.readiness_scorecard",
        ],
        "coverage_integrity_component": [
            "quality_provenance",
            "fundamental_filing.meta.coverage_score",
            "source_governance.commercialization_readiness",
            "deployment_readiness.model_readiness",
        ],
        "drift_penalty_component": [
            "operational_guardrails.drift_monitoring",
            "operational_guardrails.system_health",
            "operational_guardrails.control_state",
            "stress_spillover_conditions",
        ],
        "source_governance_component": [
            "source_governance.commercialization_readiness",
            "source_governance.source_profile",
            "deployment_readiness.model_readiness",
            "operational_guardrails.control_state",
        ],
    },
}


def _evidence_type_for_sources(sources: Iterable[str]) -> str:
    source_list = list(sources)
    if not source_list:
        return "unavailable"
    joined = " ".join(source_list)
    if any(
        marker in joined
        for marker in (
            "axiom_history_record",
            "canonical_validation",
            "evaluation.",
            "deployment_readiness",
            "operational_guardrails",
            "source_governance",
        )
    ):
        return "historical_replay_estimate"
    if any(
        marker in joined
        for marker in (
            "feature_factor_bundle",
            "proprietary_scores",
            "domain_agreement",
            "cross_asset_relative_context",
        )
    ):
        return "derived_signal"
    if any(marker in joined for marker in ("option_surface_available", "quality_provenance", "stress_spillover_conditions")):
        return "partial_proxy"
    return "direct_source"


def _confidence_lineage(score_payload: Mapping[str, Any]) -> str:
    return confidence_band(
        min(
            float(score_payload.get("coverage") or 0.0),
            float(score_payload.get("confidence") or 0.0),
        )
    )


def _component_coverage_status(
    engine_status: str,
    component_name: str,
    component_value: Any,
    *,
    option_surface_available: bool,
) -> str:
    if component_value is None:
        return "unavailable"
    if component_name == "convexity_value_component" and not option_surface_available:
        return "partial"
    if engine_status == "partial":
        return "partial"
    return "available"


def _component_notes(
    engine_name: str,
    component_name: str,
    *,
    evidence_type: str,
    coverage_status: str,
    option_surface_available: bool,
) -> List[str]:
    notes: List[str] = []
    if coverage_status != "available":
        notes.append("Coverage is incomplete, so this component should be interpreted cautiously.")
    if evidence_type == "derived_signal":
        notes.append("This component is derived from existing FTIP factor and domain intelligence rather than a raw provider field.")
    if evidence_type == "historical_replay_estimate":
        notes.append("This component depends on validation, readiness, or governance evidence accumulated from historical or operational artifacts.")
    if component_name == "convexity_value_component" and not option_surface_available:
        notes.append("No direct option-surface data is available, so convexity is represented through execution and event-risk proxies.")
    if engine_name == "research_integrity":
        notes.append("Research Integrity components inherit quality from evaluation, calibration, readiness, and governance layers.")
    return notes


def build_axiom_lineage(
    *,
    engine_input: Dict[str, Any] | AxiomEngineInput,
    axiom_artifact: Dict[str, Any] | AxiomArtifact,
    workspace_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    input_model = (
        engine_input
        if isinstance(engine_input, AxiomEngineInput)
        else AxiomEngineInput.model_validate(engine_input)
    )
    axiom = (
        axiom_artifact
        if isinstance(axiom_artifact, AxiomArtifact)
        else AxiomArtifact.model_validate(axiom_artifact)
    )
    option_surface_available = bool(input_model.source_context.get("option_surface_available"))

    engine_lineage: Dict[str, Any] = {}
    weakest_areas: List[Dict[str, Any]] = []
    direct_count = 0
    derived_count = 0
    historical_count = 0
    partial_count = 0

    for engine_name, component_sources in _ENGINE_COMPONENT_SOURCES.items():
        engine_score = axiom.engine_scores.get(engine_name)
        engine_status = engine_score.status if engine_score else "unavailable"
        blocks: List[Dict[str, Any]] = []
        component_values = dict((engine_score.components or {})) if engine_score else {}
        for component_name, derived_from in component_sources.items():
            component_value = component_values.get(component_name)
            coverage_status = _component_coverage_status(
                engine_status,
                component_name,
                component_value,
                option_surface_available=option_surface_available,
            )
            evidence_type = (
                "unavailable"
                if coverage_status == "unavailable"
                else _evidence_type_for_sources(derived_from)
            )
            if evidence_type == "direct_source":
                direct_count += 1
            elif evidence_type == "derived_signal":
                derived_count += 1
            elif evidence_type == "historical_replay_estimate":
                historical_count += 1
            else:
                partial_count += 1
            block = AxiomLineageBlock(
                engine=engine_name,
                component=component_name,
                derived_from=derived_from,
                evidence_type=evidence_type,
                confidence_lineage=_confidence_lineage(engine_score.model_dump(mode="python") if engine_score else {}),
                coverage_status=coverage_status,
                notes=_component_notes(
                    engine_name,
                    component_name,
                    evidence_type=evidence_type,
                    coverage_status=coverage_status,
                    option_surface_available=option_surface_available,
                ),
            )
            block_payload = block.model_dump(mode="python")
            block_payload["component_value"] = component_value
            blocks.append(block_payload)
            if coverage_status != "available":
                weakest_areas.append(
                    {
                        "engine": engine_name,
                        "component": component_name,
                        "coverage_status": coverage_status,
                        "evidence_type": evidence_type,
                    }
                )

        engine_lineage[engine_name] = {
            "engine": engine_name,
            "engine_status": engine_status,
            "engine_score": engine_score.score if engine_score else None,
            "coverage": engine_score.coverage if engine_score else 0.0,
            "confidence": engine_score.confidence if engine_score else 0.0,
            "summary": engine_score.summary if engine_score else "No AXIOM engine score is available.",
            "blocks": blocks,
        }

    weakest_areas = sorted(
        weakest_areas,
        key=lambda item: (
            0 if item.get("coverage_status") == "unavailable" else 1,
            str(item.get("engine") or ""),
            str(item.get("component") or ""),
        ),
    )[:6]

    lineage_summary = (
        f"AXIOM evidence lineage combines {direct_count} direct-source components, "
        f"{derived_count} derived-signal components, {historical_count} historical/governance components, "
        f"and {partial_count} proxy or incomplete components."
    )
    return {
        "lineage_version": AXIOM_LINEAGE_VERSION,
        "framework_version": axiom.framework_version,
        "symbol": axiom.symbol,
        "as_of": axiom.as_of,
        "workspace_profile": workspace_profile or {},
        "source_context": dict(axiom.source_context or {}),
        "engine_lineage": engine_lineage,
        "historical_evidence_provenance": {
            "derived_from": [
                "axiom_history_record.forward_outcomes",
                "axiom_calibration_summary",
                "canonical_validation",
            ],
            "evidence_type": "historical_replay_estimate",
            "confidence_lineage": confidence_band(
                (axiom.calibration_summary or {}).get("matured_count")
            ),
            "coverage_status": str((axiom.historical_evidence or {}).get("status") or "partial"),
        },
        "coverage_confidence_provenance": {
            "derived_from": [
                "axiom.coverage_summary",
                "quality_provenance",
                "source_governance",
            ],
            "overall_coverage": (axiom.coverage_summary or {}).get("overall_coverage"),
            "overall_confidence": (axiom.coverage_summary or {}).get("overall_confidence"),
        },
        "weakest_evidence_areas": weakest_areas,
        "lineage_summary": lineage_summary,
    }
