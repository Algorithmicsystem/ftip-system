from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


ENGINE_KEYS = (
    "fundamental_reality",
    "state_pricing",
    "behavioral_distortion",
    "flow_transmission",
    "liquidity_convexity",
    "critical_fragility",
    "research_integrity",
)


class FundamentalCandidateInputs(BaseModel):
    latest_close: Optional[float] = None
    analyst_target_price: Optional[float] = None
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_equity: Optional[float] = None
    positive_fcf_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None
    free_cash_flow_margin: Optional[float] = None
    current_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    liabilities_to_assets: Optional[float] = None
    profitability_strength: Optional[float] = None
    balance_sheet_resilience: Optional[float] = None
    cash_flow_durability: Optional[float] = None
    filing_recency_days: Optional[float] = None
    reporting_completeness_score: Optional[float] = None
    reporting_quality_proxy: Optional[float] = None
    coverage_score: float = 0.0
    provider_confidence: float = 0.0
    statement_coverage_flags: Dict[str, bool] = Field(default_factory=dict)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    coverage_caveats: List[str] = Field(default_factory=list)
    fundamental_durability_score: Optional[float] = None


class FragilityCandidateInputs(BaseModel):
    realized_vol_21d: Optional[float] = None
    realized_vol_63d: Optional[float] = None
    vol_of_vol_proxy: Optional[float] = None
    gap_pct: Optional[float] = None
    gap_instability_10d: Optional[float] = None
    abs_gap_mean_10d: Optional[float] = None
    return_dispersion_21d: Optional[float] = None
    return_dispersion_63d: Optional[float] = None
    downside_asymmetry_21d: Optional[float] = None
    downside_asymmetry_63d: Optional[float] = None
    maxdd_21d: Optional[float] = None
    maxdd_63d: Optional[float] = None
    maxdd_126d: Optional[float] = None
    event_overhang_score: Optional[float] = None
    event_uncertainty_score: Optional[float] = None
    event_risk_classification: Optional[str] = None
    implementation_fragility_score: Optional[float] = None
    liquidity_quality_score: Optional[float] = None
    tradability_caution_score: Optional[float] = None
    overnight_gap_risk_score: Optional[float] = None
    friction_proxy_score: Optional[float] = None
    execution_cleanliness_score: Optional[float] = None
    breadth_confirmation_score: Optional[float] = None
    cross_asset_conflict_score: Optional[float] = None
    market_stress_score: Optional[float] = None
    instability_score: Optional[float] = None
    volatility_stress_score: Optional[float] = None
    drawdown_sensitivity_score: Optional[float] = None
    anomaly_pressure_score: Optional[float] = None
    clean_setup_score: Optional[float] = None
    noisy_setup_score: Optional[float] = None
    narrative_crowding_score: Optional[float] = None
    signal_fragility_score: Optional[float] = None
    regime_transition_score: Optional[float] = None
    regime_instability_score: Optional[float] = None
    coverage_score: float = 0.0
    provider_confidence: float = 0.0
    suppression_flags: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class AxiomSupportContext(BaseModel):
    signal_action: Optional[str] = None
    signal_score: Optional[float] = None
    signal_confidence: Optional[float] = None
    confidence_score: Optional[float] = None
    actionability_score: Optional[float] = None
    ret_21d: Optional[float] = None
    mom_vol_adj_21d: Optional[float] = None
    regime_label: Optional[str] = None
    opportunity_quality_score: Optional[float] = None
    cross_domain_conviction_score: Optional[float] = None
    market_structure_integrity_score: Optional[float] = None
    macro_alignment_score: Optional[float] = None
    regime_stability_score: Optional[float] = None
    fundamental_durability_score: Optional[float] = None
    narrative_crowding_index: Optional[float] = None
    signal_fragility_index: Optional[float] = None
    domain_agreement_score: Optional[float] = None
    domain_conflict_score: Optional[float] = None
    trend_quality_score: Optional[float] = None
    momentum_consistency_score: Optional[float] = None
    breakout_follow_through_score: Optional[float] = None
    price_volume_alignment_score: Optional[float] = None
    directional_persistence_score: Optional[float] = None
    reversal_pressure_score: Optional[float] = None
    trend_exhaustion_score: Optional[float] = None
    benchmark_relative_strength_score: Optional[float] = None
    sector_relative_strength_score: Optional[float] = None
    sector_confirmation_score: Optional[float] = None
    relative_context_quality_score: Optional[float] = None
    idiosyncratic_strength_score: Optional[float] = None
    idiosyncratic_weakness_score: Optional[float] = None
    macro_growth_alignment_score: Optional[float] = None
    risk_on_alignment_score: Optional[float] = None
    macro_regime_consistency_score: Optional[float] = None
    macro_conflict_score: Optional[float] = None
    macro_fragility_score: Optional[float] = None
    rates_sensitivity_proxy: Optional[float] = None
    inflation_stress_proxy: Optional[float] = None
    sentiment_direction_score: Optional[float] = None
    sentiment_level_score: Optional[float] = None
    sentiment_trend_score: Optional[float] = None
    attention_intensity_score: Optional[float] = None
    novelty_score: Optional[float] = None
    repetition_score: Optional[float] = None
    narrative_concentration_score: Optional[float] = None
    contradiction_score: Optional[float] = None
    hype_to_price_divergence_score: Optional[float] = None
    positive_news_weak_price_divergence: Optional[float] = None
    negative_news_resilient_price_divergence: Optional[float] = None
    event_pressure_score: Optional[float] = None
    event_overhang_support_or_penalty: Optional[float] = None
    filings_change_signal: Optional[float] = None
    catalyst_quality: Optional[float] = None
    estimate_revision_support: Optional[float] = None
    source_strength_support: Optional[float] = None
    source_strength_penalty: Optional[float] = None
    premium_evidence_bonus: Optional[float] = None
    evidence_recency_quality: Optional[float] = None
    strategy_posture: Optional[str] = None
    conviction_tier: Optional[str] = None
    fragility_tier: Optional[str] = None
    preferred_posture: Optional[str] = None
    signal_cleanliness: Optional[str] = None
    urgency_level: Optional[str] = None
    patience_level: Optional[str] = None
    deployment_permission: Optional[str] = None
    trust_tier: Optional[str] = None
    live_readiness_score: Optional[float] = None
    model_readiness_status: Optional[str] = None
    evaluation_consistency_score: Optional[float] = None
    confidence_reliability_score: Optional[float] = None
    ranking_monotonicity: Optional[str] = None
    calibration_health_status: Optional[str] = None
    matured_prediction_count: Optional[float] = None
    hit_rate: Optional[float] = None
    actionable_vs_watchlist_return_spread: Optional[float] = None
    validation_net_edge: Optional[float] = None
    walkforward_window_count: Optional[float] = None
    readiness_bucket_quality: Optional[float] = None
    suppression_effect_edge_spread: Optional[float] = None
    model_drift_score: Optional[float] = None
    system_health_status: Optional[str] = None
    current_operating_mode: Optional[str] = None
    pause_required: Optional[bool] = None
    source_profile: Optional[str] = None
    buyer_demo_suitability: Optional[str] = None
    commercialization_risk_score: Optional[float] = None
    portfolio_candidate_score: Optional[float] = None
    portfolio_fit_quality: Optional[float] = None
    execution_quality_score: Optional[float] = None
    size_band: Optional[str] = None
    weight_band: Optional[str] = None
    risk_budget_band: Optional[str] = None
    quality_score: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    uncertainty_notes: List[str] = Field(default_factory=list)
    confirmation_triggers: List[str] = Field(default_factory=list)
    deterioration_triggers: List[str] = Field(default_factory=list)
    invalidators: List[str] = Field(default_factory=list)
    fragility_vetoes: List[str] = Field(default_factory=list)
    deployment_blockers: List[str] = Field(default_factory=list)
    monitoring_triggers: List[str] = Field(default_factory=list)


class AxiomEngineInput(BaseModel):
    framework_version: str
    symbol: str
    as_of: str
    source_context: Dict[str, Any] = Field(default_factory=dict)
    fundamental: FundamentalCandidateInputs = Field(
        default_factory=FundamentalCandidateInputs
    )
    fragility: FragilityCandidateInputs = Field(
        default_factory=FragilityCandidateInputs
    )
    support: AxiomSupportContext = Field(default_factory=AxiomSupportContext)
    domain_coverage: Dict[str, float] = Field(default_factory=dict)
    partial_engine_hints: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class EngineScore(BaseModel):
    score: Optional[float] = None
    confidence: float = 0.0
    coverage: float = 0.0
    status: str
    components: Dict[str, float] = Field(default_factory=dict)
    flags: List[str] = Field(default_factory=list)
    summary: str


class AxiomScorecard(BaseModel):
    gross_opportunity: float
    friction_burden: float
    validated_edge: float
    deployable_alpha_utility: float
    cross_engine_alignment: float = 0.0
    timing_support: float = 0.0
    setup_maturity: float = 0.0
    mispricing_readiness: float = 0.0
    evidence_readiness: float = 0.0
    path_survivability: float = 0.0
    false_positive_penalty: float = 0.0
    exceptional_opportunity: float = 0.0
    support_drag_spread: float = 0.0
    event_overhang_support: float = 0.0
    filings_change_signal: float = 0.0
    catalyst_quality: float = 0.0
    estimate_revision_support: float = 0.0
    source_strength_support: float = 0.0
    source_strength_penalty: float = 0.0
    premium_evidence_bonus: float = 0.0
    evidence_recency_quality: float = 0.0
    regime_weighting_profile: str = "base_balance"
    overall_coverage: float
    overall_confidence: float
    component_support: Dict[str, float] = Field(default_factory=dict)
    summary: str


class AxiomRegimeDecision(BaseModel):
    regime_label: str
    trade_family: str
    confidence: float
    rationale: str
    flags: List[str] = Field(default_factory=list)


class AxiomDeployabilityDecision(BaseModel):
    deployability_tier: str
    confidence: float
    rationale: str
    flags: List[str] = Field(default_factory=list)
    review_required: bool = True
    invalidation_flags: List[str] = Field(default_factory=list)
    size_band_recommendation: str = "none"
    monitoring_triggers: List[str] = Field(default_factory=list)


class AxiomWorkspaceProfile(BaseModel):
    workspace_id: str = "default"
    workspace_name: str = "FTIP Default Workspace"
    workspace_status: str = "default"
    audience_type: str = "general"
    report_profile: str = "trading_focused"
    workflow_profile: str = "default"
    emphasis_domains: List[str] = Field(default_factory=list)
    preferred_sections: List[str] = Field(default_factory=list)
    profile_notes: List[str] = Field(default_factory=list)


class AxiomLineageBlock(BaseModel):
    engine: str
    component: str
    derived_from: List[str] = Field(default_factory=list)
    evidence_type: str = "unavailable"
    confidence_lineage: str = "low"
    coverage_status: str = "unavailable"
    notes: List[str] = Field(default_factory=list)


class AxiomInstitutionalReportPack(BaseModel):
    reporting_version: str
    framework_version: str
    symbol: str
    as_of: str
    workspace_profile: AxiomWorkspaceProfile
    summary_card: Dict[str, Any] = Field(default_factory=dict)
    institutional_one_pager: Dict[str, Any] = Field(default_factory=dict)
    ic_memo: Dict[str, Any] = Field(default_factory=dict)
    risk_deployability_memo: Dict[str, Any] = Field(default_factory=dict)
    historical_evidence_summary: Dict[str, Any] = Field(default_factory=dict)
    lineage_summary: Dict[str, Any] = Field(default_factory=dict)


class AxiomArtifact(BaseModel):
    framework_version: str
    symbol: str
    as_of: str
    source_context: Dict[str, Any] = Field(default_factory=dict)
    engine_scores: Dict[str, EngineScore] = Field(default_factory=dict)
    scorecard: AxiomScorecard
    regime_decision: AxiomRegimeDecision
    deployability_decision: AxiomDeployabilityDecision
    gross_opportunity: float
    friction_burden: float
    validated_edge: float
    deployable_alpha_utility: float
    regime_label: str
    trade_family: str
    deployability_tier: str
    invalidation_flags: List[str] = Field(default_factory=list)
    explanation: Dict[str, Any] = Field(default_factory=dict)
    coverage_summary: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    historical_evidence: Dict[str, Any] = Field(default_factory=dict)
    calibration_summary: Dict[str, Any] = Field(default_factory=dict)
    portfolio_governance: Dict[str, Any] = Field(default_factory=dict)
    evidence_backed_deployability: Dict[str, Any] = Field(default_factory=dict)
    workspace_profile: Dict[str, Any] = Field(default_factory=dict)
    lineage: Dict[str, Any] = Field(default_factory=dict)
    institutional_reports: Dict[str, Any] = Field(default_factory=dict)


class AxiomHistoricalOutcome(BaseModel):
    horizon_label: str
    horizon_days: int
    matured: bool = False
    outcome_status: Optional[str] = None
    entry_date: Optional[str] = None
    exit_date: Optional[str] = None
    gross_edge_return: Optional[float] = None
    net_edge_return: Optional[float] = None
    gross_trade_return: Optional[float] = None
    net_trade_return: Optional[float] = None
    final_signal_correct: Optional[bool] = None
    estimated_cost_bps: Optional[float] = None
    mae: Optional[float] = None
    mfe: Optional[float] = None
    invalidation_triggered: Optional[bool] = None
    signal_half_life_days: Optional[int] = None
    continuation_decay_score: Optional[float] = None


class AxiomHistoryRecord(BaseModel):
    history_version: str
    framework_version: str
    symbol: str
    as_of_date: str
    signal_action: str
    signal_score: Optional[float] = None
    signal_confidence: Optional[float] = None
    strategy_posture: Optional[str] = None
    deployment_permission: Optional[str] = None
    trust_tier: Optional[str] = None
    snapshot_id: Optional[str] = None
    snapshot_version: Optional[str] = None
    feature_version: Optional[str] = None
    signal_version: Optional[str] = None
    regime_label: str
    trade_family: str
    deployability_tier: str
    size_band_recommendation: Optional[str] = None
    gross_opportunity: float
    friction_burden: float
    validated_edge: float
    deployable_alpha_utility: float
    overall_coverage: float
    overall_confidence: float
    sector: Optional[str] = None
    benchmark_proxy: Optional[str] = None
    theme_tag: Optional[str] = None
    engine_scores: Dict[str, EngineScore] = Field(default_factory=dict)
    invalidation_flags: List[str] = Field(default_factory=list)
    top_positive_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    top_negative_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    explanation_summary: Optional[str] = None
    coverage_summary: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    source_context: Dict[str, Any] = Field(default_factory=dict)
    build_metadata: Dict[str, Any] = Field(default_factory=dict)
    forward_outcomes: Dict[str, AxiomHistoricalOutcome] = Field(default_factory=dict)
    evidence_backed_deployability: Dict[str, Any] = Field(default_factory=dict)


class AxiomCalibrationArtifact(BaseModel):
    calibration_version: str
    framework_version: str
    as_of_date: Optional[str] = None
    horizon_label: str
    status: str
    sample_count: int = 0
    matured_count: int = 0
    dau_bucket_summary: Dict[str, Any] = Field(default_factory=dict)
    validated_edge_curve: Dict[str, Any] = Field(default_factory=dict)
    regime_outcome_summary: List[Dict[str, Any]] = Field(default_factory=list)
    trade_family_outcome_summary: List[Dict[str, Any]] = Field(default_factory=list)
    deployability_tier_outcome_summary: List[Dict[str, Any]] = Field(default_factory=list)
    engine_conditioned_outcome_summary: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_supportive_for_live: bool = False
    evidence_supportive_for_paper: bool = False
    summary: str
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class AxiomPortfolioGovernanceArtifact(BaseModel):
    governance_version: str
    symbol: str
    as_of_date: str
    status: str
    base_deployability_tier: str
    evidence_backed_deployability_tier: str
    portfolio_rank_score: float
    overlap_penalty: float
    fragility_penalty: float
    liquidity_penalty: float
    research_penalty: float
    final_size_band: str
    portfolio_fit_label: str
    monitoring_triggers: List[str] = Field(default_factory=list)
    downgrade_triggers: List[str] = Field(default_factory=list)
    evidence_summary: str
    rationale: str
