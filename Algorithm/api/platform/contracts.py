from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OrganizationProfile(BaseModel):
    organization_id: str
    name: str
    organization_type: str = "research_team"
    settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class WorkspaceRecord(BaseModel):
    workspace_id: str
    organization_id: str
    name: str
    audience_type: str = "general"
    report_profile: str = "trading_focused"
    default_workflow_template: str = "research_watchlist"
    platform_profile: str = "research_core"
    settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class WorkflowTemplate(BaseModel):
    template_id: str
    audience_type: str
    title: str
    description: str
    default_sections: List[str] = Field(default_factory=list)
    stage_sequence: List[str] = Field(default_factory=list)
    preferred_report_profile: str = "trading_focused"
    preferred_report_pack_emphasis: List[str] = Field(default_factory=list)
    expected_axiom_emphasis: List[str] = Field(default_factory=list)
    orientation: str = "research"


class WorkflowStageState(BaseModel):
    stage: str
    status: str = "active"
    completed_stages: List[str] = Field(default_factory=list)
    next_stage: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    updated_at: Optional[str] = None


class WorkflowInstance(BaseModel):
    workflow_id: str
    workspace_id: str
    workflow_template_id: str
    title: str
    status: str = "active"
    stage: str = "intake"
    priority: str = "normal"
    owner_placeholder: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    stage_state: WorkflowStageState
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CoverageEntity(BaseModel):
    entity_id: str
    symbol: Optional[str] = None
    external_identifier: Optional[str] = None
    entity_type: str = "public_equity"
    display_name: str
    sector: Optional[str] = None
    strategy: Optional[str] = None
    theme: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AnalysisLink(BaseModel):
    link_id: str
    report_id: Optional[str] = None
    session_id: Optional[str] = None
    axiom_artifact_id: Optional[str] = None
    axiom_report_pack_artifact_id: Optional[str] = None
    axiom_lineage_artifact_id: Optional[str] = None
    axiom_history_artifact_id: Optional[str] = None
    axiom_calibration_artifact_id: Optional[str] = None
    linked_at: Optional[str] = None
    source_summary: Dict[str, Any] = Field(default_factory=dict)


class DossierSection(BaseModel):
    section_key: str
    title: str
    summary: str
    status: str = "available"
    payload: Dict[str, Any] = Field(default_factory=dict)


class DossierRecord(BaseModel):
    dossier_id: str
    workflow_id: str
    entity_id: str
    dossier_type: str = "coverage"
    title: str
    current_summary: Dict[str, Any] = Field(default_factory=dict)
    latest_analysis_link: Optional[AnalysisLink] = None
    latest_axiom_analysis_id: Optional[str] = None
    latest_deployability_tier: Optional[str] = None
    latest_regime_label: Optional[str] = None
    latest_trade_family: Optional[str] = None
    latest_size_band: Optional[str] = None
    evidence_status: str = "partial"
    workflow_stage_state: Optional[WorkflowStageState] = None
    sections: List[DossierSection] = Field(default_factory=list)
    monitoring_state: Dict[str, Any] = Field(default_factory=dict)
    historical_evidence_summary: Dict[str, Any] = Field(default_factory=dict)
    lineage_summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PlatformProfile(BaseModel):
    profile_id: str
    audience_type: str
    default_workflow_template: str
    default_report_profile: str
    default_memo_emphasis: List[str] = Field(default_factory=list)
    preferred_axiom_sections: List[str] = Field(default_factory=list)
    preferred_dossier_sections: List[str] = Field(default_factory=list)


class PlatformSummaryView(BaseModel):
    platform_version: str
    workspace_count: int = 0
    dossier_count: int = 0
    workflow_count: int = 0
    latest_axiom_linked_dossiers: List[Dict[str, Any]] = Field(default_factory=list)
    dossiers_by_deployability_tier: Dict[str, int] = Field(default_factory=dict)
    dossiers_by_regime: Dict[str, int] = Field(default_factory=dict)
    dossiers_by_workflow_stage: Dict[str, int] = Field(default_factory=dict)
    current_workspace: Optional[Dict[str, Any]] = None
    current_workflow: Optional[Dict[str, Any]] = None
    current_dossier: Optional[Dict[str, Any]] = None


class CreateOrganizationRequest(BaseModel):
    name: str
    organization_type: str = "research_team"
    settings: Dict[str, Any] = Field(default_factory=dict)


class CreateWorkspaceRequest(BaseModel):
    organization_id: Optional[str] = None
    name: str
    audience_type: str = "general"
    report_profile: str = "trading_focused"
    default_workflow_template: Optional[str] = None
    platform_profile: str = "research_core"
    settings: Dict[str, Any] = Field(default_factory=dict)


class CreateWorkflowRequest(BaseModel):
    workspace_id: str
    workflow_template_id: str
    title: str
    status: str = "active"
    stage: Optional[str] = None
    priority: str = "normal"
    owner_placeholder: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateDossierRequest(BaseModel):
    workflow_id: str
    entity_id: Optional[str] = None
    symbol: Optional[str] = None
    display_name: Optional[str] = None
    entity_type: str = "public_equity"
    sector: Optional[str] = None
    strategy: Optional[str] = None
    theme: Optional[str] = None
    dossier_type: str = "coverage"
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AttachAnalysisRequest(BaseModel):
    report: Dict[str, Any]
    report_id: Optional[str] = None
    session_id: Optional[str] = None
    axiom_artifact_id: Optional[str] = None
    axiom_report_pack_artifact_id: Optional[str] = None
    axiom_lineage_artifact_id: Optional[str] = None
    axiom_history_artifact_id: Optional[str] = None
    axiom_calibration_artifact_id: Optional[str] = None

