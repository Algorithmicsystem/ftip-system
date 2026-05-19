from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    context: Optional[Dict[str, Any]] = None


class AnalysisReference(BaseModel):
    report_id: Optional[str] = None
    session_id: Optional[str] = None
    symbol: Optional[str] = None
    as_of_date: Optional[str] = None
    horizon: Optional[str] = None
    risk_mode: Optional[str] = None
    scenario: Optional[str] = None
    analysis_depth: Optional[str] = None
    refresh_mode: Optional[str] = None
    market_regime: Optional[str] = None
    signal: Optional[str] = None
    conviction_tier: Optional[str] = None
    strategy_posture: Optional[str] = None
    actionability_score: Optional[float] = None
    freshness_status: Optional[str] = None
    report_version: Optional[str] = None
    strategy_version: Optional[str] = None
    snapshot_id: Optional[str] = None
    snapshot_version: Optional[str] = None
    feature_version: Optional[str] = None
    signal_version: Optional[str] = None
    deployment_mode: Optional[str] = None
    deployment_permission: Optional[str] = None
    trust_tier: Optional[str] = None
    live_readiness_status: Optional[str] = None
    live_readiness_score: Optional[float] = None
    rollout_stage: Optional[str] = None
    candidate_classification: Optional[str] = None
    ranked_opportunity_score: Optional[float] = None
    portfolio_fit_quality: Optional[float] = None
    size_band: Optional[str] = None
    setup_archetype: Optional[str] = None
    research_version: Optional[str] = None
    learning_priority: Optional[str] = None
    axiom_framework_version: Optional[str] = None
    axiom_regime_label: Optional[str] = None
    axiom_trade_family: Optional[str] = None
    axiom_deployability_tier: Optional[str] = None
    axiom_validated_edge: Optional[float] = None
    axiom_deployable_alpha_utility: Optional[float] = None
    axiom_evidence_backed_deployability_tier: Optional[str] = None
    axiom_portfolio_fit_label: Optional[str] = None
    axiom_portfolio_rank_score: Optional[float] = None
    axiom_calibration_status: Optional[str] = None
    axiom_audience_type: Optional[str] = None
    axiom_report_profile: Optional[str] = None
    platform_profile: Optional[str] = None
    platform_workspace_id: Optional[str] = None
    platform_workflow_id: Optional[str] = None
    platform_workflow_stage: Optional[str] = None
    platform_dossier_id: Optional[str] = None
    platform_dossier_type: Optional[str] = None
    platform_dossier_status: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    citations: Optional[List[str]] = None
    active_analysis: Optional[AnalysisReference] = None
    report_found: bool = False


class SessionResponse(BaseModel):
    session_id: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class ExplainSignalRequest(BaseModel):
    symbol: str
    as_of: str
    lookback: int = 252


class ExplainBacktestRequest(BaseModel):
    symbols: List[str]
    from_date: str
    to_date: str
    lookback: int = 252
    rebalance_every: int = 21
    trading_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    max_weight: Optional[float] = None
    min_trade_delta: float = 0.0005
    max_turnover_per_rebalance: float = 0.25
    allow_shorts: bool = False


class TitleSessionRequest(BaseModel):
    session_id: str
    hint: Optional[str] = None


class TitleSessionResponse(BaseModel):
    session_id: str
    title: str


class AnalyzeRequest(BaseModel):
    session_id: Optional[str] = None
    symbol: str
    horizon: str
    risk_mode: str
    scenario_mode: str = "base"
    analysis_depth: str = "standard"
    refresh_mode: str = "refresh_stale"
    market_regime: str = "auto"
    audience_type: str = "general"
    report_profile: str = "trading_focused"
    platform_profile: Optional[str] = None
    workspace_id: Optional[str] = None
    workflow_id: Optional[str] = None
    workflow_template_id: Optional[str] = None
    dossier_id: Optional[str] = None
    create_dossier: bool = False
    dossier_type: str = "coverage"


class TopPicksRequest(BaseModel):
    universe: str
    horizon: str
    risk_mode: str
    limit: int = 10


class AxiomReplayRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    lookback: int = 252
    persist: bool = True
    session_id: Optional[str] = None


class AxiomCalibrationRequest(BaseModel):
    symbols: List[str]
    as_of_date: str
    horizon_label: str = "21d"
    start_date: Optional[str] = None
    session_id: Optional[str] = None


class AxiomRankedCandidatesRequest(BaseModel):
    symbols: Optional[List[str]] = None
    as_of_date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    current_holdings: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None


class NarrateRequest(BaseModel):
    payload: Dict[str, Any]
    user_message: str


class NarrateResponse(BaseModel):
    headline: str
    summary: str
    bullets: List[str]
    disclaimer: str
    followups: List[str]
