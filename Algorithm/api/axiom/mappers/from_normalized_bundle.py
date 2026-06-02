from __future__ import annotations

from typing import Any, Dict, Optional

from api.assistant.phase3.common import clamp, first_available, mean, safe_float
from api.axiom.contracts import (
    AxiomEngineInput,
    AxiomSupportContext,
    FragilityCandidateInputs,
    FundamentalCandidateInputs,
)


def _coverage_percent(payload: Optional[Dict[str, Any]]) -> float:
    meta = (payload or {}).get("meta") or {}
    value = safe_float(meta.get("coverage_score"))
    if value is None:
        return 0.0
    if value <= 1.0:
        return round(clamp(value * 100.0, 0.0, 100.0), 2)
    return round(clamp(value, 0.0, 100.0), 2)


def _provider_confidence_percent(payload: Optional[Dict[str, Any]]) -> float:
    meta = (payload or {}).get("meta") or {}
    provenance = (payload or {}).get("provenance") or {}
    value = safe_float(meta.get("confidence"))
    if value is None:
        value = safe_float(provenance.get("confidence"))
    return round(clamp(value or 0.0, 0.0, 100.0), 2)


def _score_from_named_payload(bundle: Dict[str, Any], key: str) -> Optional[float]:
    payload = bundle.get("proprietary_scores") or {}
    score_payload = payload.get(key) or {}
    return safe_float(score_payload.get("score"))


def build_axiom_engine_input(
    normalized_bundle: Dict[str, Any],
    *,
    job_context: Optional[Dict[str, Any]] = None,
    feature_factor_bundle: Optional[Dict[str, Any]] = None,
    strategy_bundle: Optional[Dict[str, Any]] = None,
    report_context: Optional[Dict[str, Any]] = None,
) -> AxiomEngineInput:
    job_context = job_context or {}
    feature_factor_bundle = feature_factor_bundle or {}
    report_context = report_context or {}
    raw = normalized_bundle.get("raw_supporting_fields") or {}
    signal = raw.get("signal") or {}
    key_features = raw.get("key_features") or {}
    quality = raw.get("quality") or {}
    strategy = strategy_bundle or (report_context.get("strategy") or {})
    canonical_core = normalized_bundle.get("canonical_alpha_core") or {}
    lineage = canonical_core.get("lineage") or {}
    market = normalized_bundle.get("market_price_volume") or {}
    fundamentals = normalized_bundle.get("fundamental_filing") or {}
    liquidity = normalized_bundle.get("liquidity_execution_fragility") or {}
    event = normalized_bundle.get("event_catalyst_risk") or {}
    breadth = normalized_bundle.get("market_breadth_internals") or {}
    cross_asset = normalized_bundle.get("cross_asset_confirmation") or {}
    stress = normalized_bundle.get("stress_spillover_conditions") or {}
    quality_domain = normalized_bundle.get("quality_provenance") or {}
    provider_snapshot = fundamentals.get("provider_snapshot") or {}
    alpha_overview = provider_snapshot.get("alphavantage_overview") or {}
    earnings_intel = provider_snapshot.get("alphavantage_earnings_intel") or {}
    latest_quarter = (
        fundamentals.get("latest_quarter")
        or ((fundamentals.get("statement_snapshot") or {}).get("latest_quarter"))
        or {}
    )
    normalized_metrics = fundamentals.get("normalized_metrics") or {}
    quality_proxies = fundamentals.get("quality_proxies") or {}
    durability_proxies = fundamentals.get("durability_proxies") or {}
    market_structure = (
        feature_factor_bundle.get("market_structure")
        or feature_factor_bundle.get("multi_horizon_price_momentum")
        or {}
    )
    fragility_intelligence = feature_factor_bundle.get("fragility_intelligence") or {}
    regime_intelligence = feature_factor_bundle.get("regime_intelligence") or {}
    sentiment_intelligence = (
        feature_factor_bundle.get("sentiment_narrative_intelligence")
        or feature_factor_bundle.get("sentiment_intelligence")
        or {}
    )
    macro_intelligence = (
        feature_factor_bundle.get("macro_alignment")
        or feature_factor_bundle.get("macro_sensitivity")
        or {}
    )
    relative_context = (
        feature_factor_bundle.get("cross_asset_relative_context")
        or feature_factor_bundle.get("relative_peer")
        or {}
    )
    domain_agreement = feature_factor_bundle.get("domain_agreement") or {}
    composites = feature_factor_bundle.get("composite_intelligence") or {}
    domain_coverage = normalized_bundle.get("domain_availability") or {}
    evaluation = report_context.get("evaluation") or {}
    deployment = report_context.get("deployment_readiness") or {}
    validation = report_context.get("canonical_validation") or {}
    operational = report_context.get("operational_guardrails") or {}
    source_governance = report_context.get("source_governance") or {}
    commercialization = source_governance.get("commercialization_readiness") or {}
    portfolio = report_context.get("portfolio_construction") or {}
    current_candidate = portfolio.get("current_candidate") or {}
    calibration = evaluation.get("calibration_summary") or {}
    signal_scorecard = evaluation.get("signal_scorecard") or {}
    strategy_scorecard = evaluation.get("strategy_scorecard") or {}
    ranking_scorecard = evaluation.get("ranking_scorecard") or {}
    model_readiness = deployment.get("model_readiness") or {}
    deployment_permission = deployment.get("deployment_permission") or {}
    risk_budgeting = deployment.get("risk_budgeting") or {}
    operational_drift = operational.get("drift_monitoring") or {}
    operational_health = operational.get("system_health") or {}
    operational_controls = operational.get("control_state") or {}
    walkforward = validation.get("walkforward_summary") or {}
    net_returns = validation.get("net_return_summary") or {}
    readiness_scorecard = validation.get("readiness_scorecard") or {}
    suppression_effect = validation.get("suppression_effect_summary") or {}
    execution_posture = strategy.get("execution_posture") or {}
    invalidators = strategy.get("invalidators") or {}
    fragility_vetoes = strategy.get("fragility_vetoes") or []
    signal_overall = (signal_scorecard.get("final_signal_overall") or {})
    evaluation_consistency = model_readiness.get("evaluation_consistency") or {}
    operational_actions = [
        str(item.get("recommended_action"))
        for item in (operational.get("operational_alerts") or [])
        if isinstance(item, dict) and item.get("recommended_action")
    ]

    fundamental_inputs = FundamentalCandidateInputs(
        latest_close=safe_float(market.get("latest_close")),
        analyst_target_price=safe_float(alpha_overview.get("analyst_target_price")),
        pe_ratio=safe_float(alpha_overview.get("pe_ratio")),
        peg_ratio=safe_float(alpha_overview.get("peg_ratio")),
        market_cap=safe_float((fundamentals.get("company_profile") or {}).get("market_cap")),
        revenue_growth_yoy=safe_float(
            first_available(
                normalized_metrics.get("revenue_growth_yoy"),
                fundamentals.get("revenue_growth_yoy"),
            )
        ),
        gross_margin=safe_float(
            first_available(latest_quarter.get("gross_margin"), normalized_metrics.get("gross_margin"))
        ),
        operating_margin=safe_float(
            first_available(
                latest_quarter.get("op_margin"),
                latest_quarter.get("operating_margin"),
                normalized_metrics.get("operating_margin"),
            )
        ),
        net_margin=safe_float(normalized_metrics.get("net_margin")),
        return_on_assets=safe_float(normalized_metrics.get("return_on_assets")),
        return_on_equity=safe_float(normalized_metrics.get("return_on_equity")),
        positive_fcf_ratio=safe_float(
            first_available(
                normalized_metrics.get("positive_fcf_ratio"),
                fundamentals.get("positive_fcf_ratio"),
            )
        ),
        free_cash_flow=safe_float(normalized_metrics.get("free_cash_flow")),
        free_cash_flow_margin=safe_float(normalized_metrics.get("free_cash_flow_margin")),
        current_ratio=safe_float(normalized_metrics.get("current_ratio")),
        cash_ratio=safe_float(normalized_metrics.get("cash_ratio")),
        debt_to_equity=safe_float(normalized_metrics.get("debt_to_equity")),
        liabilities_to_assets=safe_float(normalized_metrics.get("liabilities_to_assets")),
        profitability_strength=safe_float(quality_proxies.get("profitability_strength")),
        balance_sheet_resilience=safe_float(
            first_available(
                quality_proxies.get("balance_sheet_resilience"),
                durability_proxies.get("balance_sheet_resilience"),
            )
        ),
        cash_flow_durability=safe_float(
            first_available(
                quality_proxies.get("cash_flow_durability"),
                durability_proxies.get("cash_flow_durability"),
            )
        ),
        filing_recency_days=safe_float(fundamentals.get("filing_recency_days")),
        reporting_completeness_score=safe_float(
            quality_proxies.get("reporting_completeness_score")
        ),
        reporting_quality_proxy=safe_float(quality_proxies.get("reporting_quality_proxy")),
        quarterly_earnings_growth_yoy=safe_float(alpha_overview.get("quarterly_earnings_growth_yoy")),
        earnings_beat_rate_4q=safe_float(earnings_intel.get("beat_rate_4q")),
        earnings_miss_rate_4q=safe_float(earnings_intel.get("miss_rate_4q")),
        earnings_avg_surprise_pct=safe_float(earnings_intel.get("average_surprise_pct_4q")),
        earnings_estimate_revision_support=safe_float(
            first_available(
                earnings_intel.get("estimate_revision_support"),
                event.get("estimate_revision_support"),
            )
        ),
        earnings_freshness=str(earnings_intel.get("freshness_status") or "") or None,
        coverage_score=_coverage_percent(fundamentals),
        provider_confidence=_provider_confidence_percent(fundamentals),
        statement_coverage_flags=dict(fundamentals.get("coverage_flags") or {}),
        strengths=list(fundamentals.get("strength_summary") or []),
        weaknesses=list(fundamentals.get("weakness_summary") or []),
        coverage_caveats=list(fundamentals.get("coverage_caveats") or []),
        fundamental_durability_score=safe_float(
            first_available(
                composites.get("Fundamental Durability Score"),
                (feature_factor_bundle.get("fundamental_durability") or {}).get(
                    "profitability_quality_score"
                ),
            )
        ),
    )
    fragility_inputs = FragilityCandidateInputs(
        realized_vol_21d=safe_float(market.get("realized_vol_21d")),
        realized_vol_63d=safe_float(market.get("realized_vol_63d")),
        vol_of_vol_proxy=safe_float(market.get("vol_of_vol_proxy")),
        gap_pct=safe_float(market.get("gap_pct")),
        gap_instability_10d=safe_float(market.get("gap_instability_10d")),
        abs_gap_mean_10d=safe_float(market.get("abs_gap_mean_10d")),
        return_dispersion_21d=safe_float(market.get("return_dispersion_21d")),
        return_dispersion_63d=safe_float(market.get("return_dispersion_63d")),
        downside_asymmetry_21d=safe_float(market.get("downside_asymmetry_21d")),
        downside_asymmetry_63d=safe_float(market.get("downside_asymmetry_63d")),
        maxdd_21d=safe_float(market.get("maxdd_21d")),
        maxdd_63d=safe_float(market.get("maxdd_63d")),
        maxdd_126d=safe_float(market.get("maxdd_126d")),
        event_overhang_score=safe_float(event.get("event_overhang_score")),
        event_uncertainty_score=safe_float(event.get("event_uncertainty_score")),
        event_risk_classification=str(event.get("event_risk_classification") or "")
        or None,
        implementation_fragility_score=safe_float(
            liquidity.get("implementation_fragility_score")
        ),
        liquidity_quality_score=safe_float(liquidity.get("liquidity_quality_score")),
        tradability_caution_score=safe_float(liquidity.get("tradability_caution_score")),
        overnight_gap_risk_score=safe_float(liquidity.get("overnight_gap_risk_score")),
        friction_proxy_score=safe_float(liquidity.get("friction_proxy_score")),
        execution_cleanliness_score=safe_float(
            liquidity.get("execution_cleanliness_score")
        ),
        breadth_confirmation_score=safe_float(
            breadth.get("breadth_confirmation_score")
        ),
        cross_asset_conflict_score=safe_float(
            cross_asset.get("cross_asset_conflict_score")
        ),
        market_stress_score=safe_float(stress.get("market_stress_score")),
        instability_score=safe_float(fragility_intelligence.get("instability_score")),
        volatility_stress_score=safe_float(
            fragility_intelligence.get("volatility_stress_score")
        ),
        drawdown_sensitivity_score=safe_float(
            fragility_intelligence.get("drawdown_sensitivity_score")
        ),
        anomaly_pressure_score=safe_float(
            fragility_intelligence.get("anomaly_pressure_score")
        ),
        clean_setup_score=safe_float(fragility_intelligence.get("clean_setup_score")),
        noisy_setup_score=safe_float(fragility_intelligence.get("noisy_setup_score")),
        narrative_crowding_score=safe_float(
            first_available(
                _score_from_named_payload(feature_factor_bundle, "Narrative Crowding Index"),
                (feature_factor_bundle.get("sentiment_narrative_intelligence") or {}).get(
                    "crowding_proxy_score"
                ),
            )
        ),
        signal_fragility_score=safe_float(
            _score_from_named_payload(feature_factor_bundle, "Signal Fragility Index")
        ),
        regime_transition_score=safe_float(regime_intelligence.get("transition_risk")),
        regime_instability_score=safe_float(regime_intelligence.get("regime_instability")),
        coverage_score=round(
            mean(
                [
                    _coverage_percent(normalized_bundle.get("market_price_volume")),
                    _coverage_percent(liquidity),
                    _coverage_percent(event),
                    _coverage_percent(stress),
                ]
            )
            or 0.0,
            2,
        ),
        provider_confidence=round(
            mean(
                [
                    _provider_confidence_percent(quality_domain),
                    _provider_confidence_percent(fundamentals),
                ]
            )
            or 0.0,
            2,
        ),
        suppression_flags=list(
            (canonical_core.get("signal_payload") or {}).get("suppression_flags") or []
        ),
        notes=list(
            (canonical_core.get("signal_payload") or {}).get("adjusted_confidence_notes")
            or []
        ),
    )
    support_context = AxiomSupportContext(
        signal_action=str(signal.get("action") or "").upper() or None,
        signal_score=safe_float(signal.get("score")),
        signal_confidence=safe_float(signal.get("confidence")),
        confidence_score=safe_float(
            first_available(strategy.get("confidence_score"), report_context.get("confidence_score"))
        ),
        actionability_score=safe_float(
            first_available(strategy.get("actionability_score"), report_context.get("actionability_score"))
        ),
        ret_21d=safe_float(first_available(key_features.get("ret_21d"), market.get("ret_21d"))),
        mom_vol_adj_21d=safe_float(key_features.get("mom_vol_adj_21d")),
        regime_label=str(
            first_available(
                key_features.get("regime_label"),
                (feature_factor_bundle.get("regime_intelligence") or {}).get("regime_label"),
            )
            or ""
        )
        or None,
        opportunity_quality_score=safe_float(composites.get("Opportunity Quality Score")),
        cross_domain_conviction_score=safe_float(
            composites.get("Cross-Domain Conviction Score")
        ),
        market_structure_integrity_score=safe_float(
            composites.get("Market Structure Integrity Score")
        ),
        macro_alignment_score=safe_float(composites.get("Macro Alignment Score")),
        regime_stability_score=safe_float(composites.get("Regime Stability Score")),
        fundamental_durability_score=safe_float(
            composites.get("Fundamental Durability Score")
        ),
        narrative_crowding_index=safe_float(composites.get("Narrative Crowding Index")),
        signal_fragility_index=safe_float(composites.get("Signal Fragility Index")),
        domain_agreement_score=safe_float(domain_agreement.get("domain_agreement_score")),
        domain_conflict_score=safe_float(domain_agreement.get("domain_conflict_score")),
        trend_quality_score=safe_float(market_structure.get("trend_quality_score")),
        momentum_consistency_score=safe_float(
            market_structure.get("momentum_consistency_score")
        ),
        breakout_follow_through_score=safe_float(
            market_structure.get("breakout_follow_through_score")
        ),
        price_volume_alignment_score=safe_float(
            market_structure.get("price_volume_alignment_score")
        ),
        directional_persistence_score=safe_float(
            market_structure.get("directional_persistence_score")
        ),
        reversal_pressure_score=safe_float(
            market_structure.get("reversal_pressure_score")
        ),
        trend_exhaustion_score=safe_float(
            market_structure.get("trend_exhaustion_score")
        ),
        benchmark_relative_strength_score=safe_float(
            relative_context.get("benchmark_relative_strength")
        ),
        sector_relative_strength_score=safe_float(
            relative_context.get("sector_relative_strength")
        ),
        sector_confirmation_score=safe_float(
            relative_context.get("sector_confirmation_score")
        ),
        relative_context_quality_score=safe_float(
            relative_context.get("relative_context_quality")
        ),
        idiosyncratic_strength_score=safe_float(
            relative_context.get("idiosyncratic_strength_vs_market")
        ),
        idiosyncratic_weakness_score=safe_float(
            relative_context.get("idiosyncratic_weakness_vs_market")
        ),
        macro_growth_alignment_score=safe_float(
            macro_intelligence.get("growth_alignment_score")
        ),
        risk_on_alignment_score=safe_float(
            macro_intelligence.get("risk_on_risk_off_alignment")
        ),
        macro_regime_consistency_score=safe_float(
            macro_intelligence.get("macro_regime_consistency")
        ),
        macro_conflict_score=safe_float(macro_intelligence.get("macro_conflict_score")),
        macro_fragility_score=safe_float(
            first_available(
                macro_intelligence.get("macro_fragility_score"),
                macro_intelligence.get("macro_stress_fragility"),
            )
        ),
        rates_sensitivity_proxy=safe_float(
            macro_intelligence.get("rates_sensitivity_proxy")
        ),
        inflation_stress_proxy=safe_float(
            macro_intelligence.get("inflation_stress_proxy")
        ),
        sentiment_direction_score=safe_float(
            sentiment_intelligence.get("sentiment_direction_score")
        ),
        sentiment_level_score=safe_float(
            sentiment_intelligence.get("sentiment_level_score")
        ),
        sentiment_trend_score=safe_float(
            sentiment_intelligence.get("sentiment_trend_score")
        ),
        attention_intensity_score=safe_float(
            sentiment_intelligence.get("attention_intensity_score")
        ),
        novelty_score=safe_float(sentiment_intelligence.get("novelty_score")),
        repetition_score=safe_float(sentiment_intelligence.get("repetition_score")),
        narrative_concentration_score=safe_float(
            sentiment_intelligence.get("narrative_concentration_score")
        ),
        contradiction_score=safe_float(sentiment_intelligence.get("contradiction_score")),
        hype_to_price_divergence_score=safe_float(
            sentiment_intelligence.get("hype_to_price_divergence_score")
        ),
        positive_news_weak_price_divergence=safe_float(
            sentiment_intelligence.get("positive_news_weak_price_divergence")
        ),
        negative_news_resilient_price_divergence=safe_float(
            sentiment_intelligence.get("negative_news_resilient_price_divergence")
        ),
        event_pressure_score=safe_float(sentiment_intelligence.get("event_pressure_score")),
        event_overhang_support_or_penalty=safe_float(
            first_available(
                event.get("event_overhang_support_or_penalty"),
                event.get("event_overhang_score"),
            )
        ),
        filings_change_signal=safe_float(event.get("filings_change_signal")),
        catalyst_quality=safe_float(
            first_available(
                event.get("catalyst_quality"),
                event.get("catalyst_burst_score"),
            )
        ),
        estimate_revision_support=safe_float(event.get("estimate_revision_support")),
        source_strength_support=safe_float(
            first_available(
                event.get("source_strength_support"),
                quality_domain.get("quality_score"),
            )
        ),
        source_strength_penalty=safe_float(event.get("source_strength_penalty")),
        premium_evidence_bonus=safe_float(event.get("premium_evidence_bonus")),
        evidence_recency_quality=safe_float(event.get("evidence_recency_quality")),
        strategy_posture=str(
            first_available(strategy.get("strategy_posture"), report_context.get("strategy_posture"))
            or ""
        )
        or None,
        conviction_tier=str(strategy.get("conviction_tier") or "") or None,
        fragility_tier=str(strategy.get("fragility_tier") or "") or None,
        preferred_posture=str(execution_posture.get("preferred_posture") or "") or None,
        signal_cleanliness=str(execution_posture.get("signal_cleanliness") or "") or None,
        urgency_level=str(execution_posture.get("urgency_level") or "") or None,
        patience_level=str(execution_posture.get("patience_level") or "") or None,
        deployment_permission=str(
            first_available(
                report_context.get("deployment_permission"),
                deployment_permission.get("deployment_permission"),
            )
            or ""
        )
        or None,
        trust_tier=str(
            first_available(report_context.get("trust_tier"), deployment_permission.get("trust_tier"))
            or ""
        )
        or None,
        live_readiness_score=safe_float(
            first_available(report_context.get("live_readiness_score"), model_readiness.get("live_readiness_score"))
        ),
        model_readiness_status=str(
            first_available(
                report_context.get("model_readiness_status"),
                model_readiness.get("model_readiness_status"),
            )
            or ""
        )
        or None,
        evaluation_consistency_score=safe_float(
            first_available(
                evaluation_consistency.get("consistency_score"),
                ((evaluation.get("calibration_summary") or {}).get("confidence_reliability_score")),
            )
        ),
        confidence_reliability_score=safe_float(
            calibration.get("confidence_reliability_score")
        ),
        ranking_monotonicity=str(
            first_available(
                calibration.get("confidence_monotonicity"),
                ranking_scorecard.get("confidence_monotonicity"),
            )
            or ""
        )
        or None,
        calibration_health_status=str(report_context.get("calibration_health_status") or "") or None,
        matured_prediction_count=safe_float(
            first_available(
                evaluation_consistency.get("matured_prediction_count"),
                signal_overall.get("matured_count"),
            )
        ),
        hit_rate=safe_float(
            first_available(
                signal_overall.get("hit_rate"),
                evaluation_consistency.get("hit_rate"),
            )
        ),
        actionable_vs_watchlist_return_spread=safe_float(
            strategy_scorecard.get("actionable_vs_watchlist_return_spread")
        ),
        validation_net_edge=safe_float(net_returns.get("average_edge_return")),
        walkforward_window_count=safe_float(walkforward.get("window_count")),
        readiness_bucket_quality=safe_float(
            readiness_scorecard.get("paper_vs_live_candidate_quality_summary")
        ),
        suppression_effect_edge_spread=safe_float(
            suppression_effect.get("suppression_effect_edge_spread")
        ),
        model_drift_score=safe_float(
            first_available(
                report_context.get("model_drift_score"),
                operational_drift.get("model_drift_score"),
            )
        ),
        system_health_status=str(
            first_available(
                report_context.get("system_health_status"),
                operational_health.get("system_health_status"),
            )
            or ""
        )
        or None,
        current_operating_mode=str(
            first_available(
                report_context.get("current_operating_mode"),
                operational_controls.get("current_operating_mode"),
            )
            or ""
        )
        or None,
        pause_required=bool(
            first_available(
                report_context.get("pause_required"),
                operational_controls.get("pause_required"),
            )
        )
        if first_available(
            report_context.get("pause_required"),
            operational_controls.get("pause_required"),
        )
        is not None
        else None,
        source_profile=str(report_context.get("source_profile") or "") or None,
        buyer_demo_suitability=str(
            first_available(
                report_context.get("buyer_demo_suitability"),
                commercialization.get("buyer_demo_suitability"),
            )
            or ""
        )
        or None,
        commercialization_risk_score=safe_float(
            first_available(
                report_context.get("commercialization_risk_score"),
                commercialization.get("commercialization_risk_score"),
            )
        ),
        portfolio_candidate_score=safe_float(
            first_available(
                report_context.get("portfolio_candidate_score"),
                current_candidate.get("portfolio_candidate_score"),
            )
        ),
        portfolio_fit_quality=safe_float(
            first_available(
                report_context.get("portfolio_fit_quality"),
                current_candidate.get("portfolio_fit_quality"),
            )
        ),
        execution_quality_score=safe_float(
            first_available(
                report_context.get("execution_quality_score"),
                current_candidate.get("execution_quality_score"),
            )
        ),
        size_band=str(
            first_available(report_context.get("size_band"), current_candidate.get("size_band"))
            or ""
        )
        or None,
        weight_band=str(
            first_available(report_context.get("weight_band"), current_candidate.get("weight_band"))
            or ""
        )
        or None,
        risk_budget_band=str(
            first_available(report_context.get("risk_budget_band"), current_candidate.get("risk_budget_band"))
            or ""
        )
        or None,
        quality_score=safe_float(quality.get("quality_score")),
        warnings=list(quality.get("warnings") or []),
        uncertainty_notes=list(
            (canonical_core.get("signal_payload") or {}).get("adjusted_confidence_notes")
            or []
        )
        + list(strategy.get("uncertainty_notes") or []),
        confirmation_triggers=list(
            report_context.get("confirmation_triggers")
            or strategy.get("confirmation_triggers")
            or []
        ),
        deterioration_triggers=list(
            report_context.get("deterioration_triggers")
            or strategy.get("deterioration_triggers")
            or []
        ),
        invalidators=list(invalidators.get("top_invalidators") or []),
        fragility_vetoes=[
            str(item.get("name") or item.get("label") or item)
            for item in fragility_vetoes
            if item not in (None, "")
        ],
        deployment_blockers=list(
            report_context.get("deployment_blockers")
            or deployment_permission.get("deployment_blockers")
            or model_readiness.get("live_readiness_blockers")
            or []
        ),
        monitoring_triggers=list(
            dict.fromkeys(
                list(report_context.get("operator_attention_items") or [])
                + list(strategy.get("deterioration_triggers") or [])
                + list(strategy.get("confirmation_triggers") or [])
                + operational_actions
            )
        ),
    )
    partial_engine_hints = {
        "state_pricing": round(
            mean(
                [
                    _coverage_percent(normalized_bundle.get("market_price_volume")),
                    _coverage_percent(normalized_bundle.get("macro_cross_asset")),
                    _coverage_percent(cross_asset),
                ]
            )
            or 0.0,
            2,
        ),
        "behavioral_distortion": round(
            mean(
                [
                    _coverage_percent(normalized_bundle.get("sentiment_narrative_flow")),
                    _coverage_percent(event),
                ]
            )
            or 0.0,
            2,
        ),
        "flow_transmission": round(
            mean(
                [
                    _coverage_percent(normalized_bundle.get("market_price_volume")),
                    _coverage_percent(breadth),
                    _coverage_percent(cross_asset),
                ]
            )
            or 0.0,
            2,
        ),
        "liquidity_convexity": round(
            mean(
                [
                    _coverage_percent(liquidity),
                    _coverage_percent(stress),
                ]
            )
            or 0.0,
            2,
        ),
        "research_integrity": round(
            mean(
                [
                    _coverage_percent(fundamentals),
                    _coverage_percent(quality_domain),
                    100.0 if evaluation else 0.0,
                    100.0 if deployment else 0.0,
                    100.0 if validation else 0.0,
                    100.0 if operational else 0.0,
                    100.0 if source_governance else 0.0,
                ]
            )
            or 0.0,
            2,
        ),
    }
    _macro_raw = normalized_bundle.get("macro_cross_asset") or {}
    _risk_on = safe_float(_macro_raw.get("risk_on_score"))
    _macro_cardi_inputs = {
        "carry_score": None,
        "value_score": None,
        "momentum_score": round(_risk_on * 100.0, 2) if _risk_on is not None else None,
        "defensive_score": None,
    }
    _mkt_bubble_ctx = {
        "cape_z_score": None,
        "kindleberger_stage": "normal",
        "narrative_intensity": 50.0,
    }
    return AxiomEngineInput(
        framework_version="axiom50_phase2_v1",
        symbol=str(job_context.get("symbol") or raw.get("signal", {}).get("symbol") or "").upper(),
        as_of=str(job_context.get("as_of_date") or ""),
        source_context={
            "snapshot_id": lineage.get("snapshot_id"),
            "snapshot_version": lineage.get("snapshot_version"),
            "feature_version": lineage.get("feature_version"),
            "signal_version": lineage.get("signal_version"),
            "symbol_meta": normalized_bundle.get("symbol_meta") or {},
            "domain_availability": domain_coverage,
            "price_series": market.get("price_series") or [],
            "return_series": market.get("return_series") or [],
            "price_returns_series": market.get("return_series") or [],
            "volume_series": market.get("volume_series") or [],
            "avg_volume_21d": safe_float(market.get("avg_volume_21d")),
            "market_bubble_context": _mkt_bubble_ctx,
            "macro_cardi_inputs": _macro_cardi_inputs,
        },
        fundamental=fundamental_inputs,
        fragility=fragility_inputs,
        support=support_context,
        domain_coverage={
            "market": _coverage_percent(normalized_bundle.get("market_price_volume")),
            "fundamentals": _coverage_percent(fundamentals),
            "sentiment": _coverage_percent(
                normalized_bundle.get("sentiment_narrative_flow")
            ),
            "macro": _coverage_percent(normalized_bundle.get("macro_cross_asset")),
            "cross_asset": _coverage_percent(cross_asset),
            "event": _coverage_percent(event),
            "liquidity": _coverage_percent(liquidity),
            "breadth": _coverage_percent(breadth),
            "stress": _coverage_percent(stress),
            "quality": _coverage_percent(quality_domain),
        },
        partial_engine_hints=partial_engine_hints,
        warnings=list(quality.get("warnings") or []) + list(fundamental_inputs.coverage_caveats),
    )
