from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

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


class ResourceRef(BaseModel):
    resource_type: str
    resource_id: Optional[str] = None
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScopedResourceRef(ResourceRef):
    required_organization_id: Optional[str] = None
    required_workspace_id: Optional[str] = None
    tenant_scope_required: bool = True


class PermissionSet(BaseModel):
    permissions: List[str] = Field(default_factory=list)


class RoleDefinition(BaseModel):
    role_id: str
    description: str
    scope: str = "workspace"
    permissions: PermissionSet


class UserContext(BaseModel):
    user_id: str
    user_name: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    organization_ids: List[str] = Field(default_factory=list)
    workspace_ids: List[str] = Field(default_factory=list)
    role: str = "service_account"
    permissions: List[str] = Field(default_factory=list)
    auth_mode: str = "development"
    is_system: bool = False
    session_id: Optional[str] = None
    role_bindings: List[Dict[str, Any]] = Field(default_factory=list)
    request_metadata: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MembershipRecord(BaseModel):
    membership_id: str
    user_id: str
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    role: str
    permissions: List[str] = Field(default_factory=list)
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AuthenticatedUser(BaseModel):
    user_id: str
    user_name: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None
    auth_mode: str = "development"
    is_system: bool = False


class TenantContext(BaseModel):
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    organization_ids: List[str] = Field(default_factory=list)
    workspace_ids: List[str] = Field(default_factory=list)
    role_bindings: List[Dict[str, Any]] = Field(default_factory=list)
    is_scoped: bool = False
    fallback_used: bool = False
    scope_summary: str = "unscoped"


class SessionContext(BaseModel):
    session_id: Optional[str] = None
    actor: AuthenticatedUser
    tenant: TenantContext
    role: str = "service_account"
    permissions: List[str] = Field(default_factory=list)
    auth_mode: str = "development"
    is_system: bool = False
    request_metadata: Dict[str, Any] = Field(default_factory=dict)


class AuthResolution(BaseModel):
    session: SessionContext
    user_context: UserContext
    auth_headers_present: bool = False
    fallback_used: bool = False
    enforcement_mode: str = "development_fallback"
    effective_membership: Optional[MembershipRecord] = None


class AccessDecision(BaseModel):
    allowed: bool
    permission: str
    role: str
    reason: str
    enforcement_mode: str = "default"
    permissions: List[str] = Field(default_factory=list)
    missing_permissions: List[str] = Field(default_factory=list)
    membership: Optional[MembershipRecord] = None
    resource: Optional[ResourceRef] = None
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    auth_mode: Optional[str] = None
    actor_user_id: Optional[str] = None
    tenant_scope_summary: Optional[str] = None


class AuditEvent(BaseModel):
    event_id: str
    event_type: str
    resource_type: str
    resource_id: Optional[str] = None
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    session_id: Optional[str] = None
    auth_mode: Optional[str] = None
    actor: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskState(BaseModel):
    task_id: str
    label: str
    status: str = "pending"
    owner_placeholder: Optional[str] = None
    due_at: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowAction(BaseModel):
    action_id: str
    workflow_id: str
    dossier_id: Optional[str] = None
    action_type: str
    requested_stage: Optional[str] = None
    requested_status: Optional[str] = None
    rationale: Optional[str] = None
    actor: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class StageTransition(BaseModel):
    template_id: str
    from_stage: str
    to_stage: str
    allowed: bool = True
    requires_role: Optional[str] = None
    requires_approval: bool = False
    notes: List[str] = Field(default_factory=list)


class WorkflowTimelineEvent(BaseModel):
    event_id: str
    workflow_id: str
    dossier_id: Optional[str] = None
    event_type: str
    title: str
    summary: str
    stage: Optional[str] = None
    status: Optional[str] = None
    actor_label: Optional[str] = None
    created_at: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    approval_id: str
    workflow_id: str
    dossier_id: Optional[str] = None
    requested_role: str
    requested_by: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"
    stage: Optional[str] = None
    rationale: Optional[str] = None
    required_permissions: List[str] = Field(default_factory=list)
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ApprovalDecision(BaseModel):
    decision_id: str
    approval_id: str
    decision_type: str
    decided_by: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExportSection(BaseModel):
    section_key: str
    title: str
    content: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    status: str = "available"


class ExportManifest(BaseModel):
    export_id: str
    dossier_id: str
    workflow_id: Optional[str] = None
    workspace_id: Optional[str] = None
    pack_type: str
    title: str
    subtitle: Optional[str] = None
    generated_at: Optional[str] = None
    framework_version: Optional[str] = None
    organization_context: Dict[str, Any] = Field(default_factory=dict)
    workspace_context: Dict[str, Any] = Field(default_factory=dict)
    entity_context: Dict[str, Any] = Field(default_factory=dict)
    approval_status: Optional[str] = None
    evidence_summary: Optional[str] = None
    ordered_sections: List[ExportSection] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_hash: Optional[str] = None
    status: str = "generated"


class RenderedExportResult(BaseModel):
    render_id: str
    export_id: str
    export_format: str
    content_type: str
    rendered_content: str
    file_name_hint: str
    section_count: int = 0
    checksum: Optional[str] = None
    generated_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExportStorageRef(BaseModel):
    storage_backend: str
    storage_key: str
    retrieval_hint: Optional[str] = None
    local_path: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None


class ExportVersionRecord(BaseModel):
    stored_export_id: str
    export_id: str
    render_id: str
    pack_type: str
    export_format: str
    version_group_key: str
    version_number: int
    version_label: str
    status: str
    approval_status: Optional[str] = None
    evidence_status: Optional[str] = None
    checksum: Optional[str] = None
    storage_backend: str
    storage_key: str
    file_name_hint: str
    created_at: Optional[str] = None


class StoredExportRecord(BaseModel):
    stored_export_id: str
    export_id: str
    render_id: str
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    dossier_id: Optional[str] = None
    workflow_id: Optional[str] = None
    pack_type: str
    export_format: str
    framework_version: Optional[str] = None
    approval_status: Optional[str] = None
    evidence_status: Optional[str] = None
    checksum: Optional[str] = None
    source_manifest_hash: Optional[str] = None
    content_hash: Optional[str] = None
    manifest_hash: Optional[str] = None
    section_count: int = 0
    file_name_hint: str
    content_type: Optional[str] = None
    storage_backend: str
    storage_key: str
    storage_ref: ExportStorageRef
    version_group_key: str
    version_number: int = 1
    version_label: str = "v1"
    status: str = "stored"
    document_identity: Dict[str, Any] = Field(default_factory=dict)
    source_context: Dict[str, Any] = Field(default_factory=dict)
    approval_context: Dict[str, Any] = Field(default_factory=dict)
    axiom_context: Dict[str, Any] = Field(default_factory=dict)
    evidence_context: Dict[str, Any] = Field(default_factory=dict)
    export_context: Dict[str, Any] = Field(default_factory=dict)
    lineage_summary: Dict[str, Any] = Field(default_factory=dict)
    created_by: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ExportIntegrityResult(BaseModel):
    stored_export_id: str
    export_id: str
    render_id: str
    version_number: int = 1
    version_label: str = "v1"
    status: str = "valid"
    checksum_expected: Optional[str] = None
    checksum_actual: Optional[str] = None
    section_count_expected: int = 0
    section_count_actual: int = 0
    manifest_hash_expected: Optional[str] = None
    manifest_hash_actual: Optional[str] = None
    tenant_scope_consistent: bool = True
    approval_context_consistent: bool = True
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    verified_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExportRetrievalResult(BaseModel):
    stored_export_id: str
    export_id: str
    render_id: str
    export_format: str
    content_type: str
    rendered_content: str
    file_name_hint: str
    storage_ref: ExportStorageRef
    retrieved_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExportFormatCapabilities(BaseModel):
    html_supported: bool = True
    markdown_supported: bool = True
    json_supported: bool = True
    pdf_ready: bool = False
    docx_ready: bool = False
    print_ready_html: bool = False


class ConnectorCapability(BaseModel):
    capability_id: str
    title: str
    description: str


class ConnectorHealthState(BaseModel):
    status: str = "unknown"
    checked_at: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class IntegrationDefinition(BaseModel):
    integration_type: str
    title: str
    description: str
    scope: str = "workspace"
    capabilities: List[ConnectorCapability] = Field(default_factory=list)
    config_schema: Dict[str, Any] = Field(default_factory=dict)


class IntegrationBinding(BaseModel):
    binding_id: str
    integration_type: str
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    status: str = "configured"
    config: Dict[str, Any] = Field(default_factory=dict)
    health: ConnectorHealthState = Field(default_factory=ConnectorHealthState)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class IntegrationExecutionRecord(BaseModel):
    execution_id: str
    binding_id: str
    integration_type: str
    action_type: str
    status: str = "completed"
    workspace_id: Optional[str] = None
    organization_id: Optional[str] = None
    dossier_id: Optional[str] = None
    export_id: Optional[str] = None
    render_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    payload_summary: Dict[str, Any] = Field(default_factory=dict)
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    error_summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlatformHealthSummary(BaseModel):
    platform_version: str
    workspace_id: Optional[str] = None
    organization_id: Optional[str] = None
    access_summary: Dict[str, Any] = Field(default_factory=dict)
    pending_approval_count: int = 0
    export_count: int = 0
    audit_event_count: int = 0
    integration_health_summary: Dict[str, Any] = Field(default_factory=dict)
    workflow_integrity_checks: List[str] = Field(default_factory=list)
    dossier_integrity_checks: List[str] = Field(default_factory=list)
    export_integrity_checks: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class WorkspaceAnalyticsView(BaseModel):
    workspace_id: str
    organization_id: Optional[str] = None
    workspace_name: Optional[str] = None
    audience_type: Optional[str] = None
    workflow_count: int = 0
    dossier_count: int = 0
    pending_approval_count: int = 0
    export_count: int = 0
    integration_binding_count: int = 0
    dossiers_by_deployability_tier: Dict[str, int] = Field(default_factory=dict)
    dossiers_by_regime: Dict[str, int] = Field(default_factory=dict)
    dossiers_by_stage: Dict[str, int] = Field(default_factory=dict)
    evidence_status_distribution: Dict[str, int] = Field(default_factory=dict)
    workflow_template_distribution: Dict[str, int] = Field(default_factory=dict)
    average_dau: Optional[float] = None
    live_candidate_ratio: Optional[float] = None
    size_band_distribution: Dict[str, int] = Field(default_factory=dict)
    high_dau_dossiers: List[Dict[str, Any]] = Field(default_factory=list)
    recent_exports: List[Dict[str, Any]] = Field(default_factory=list)
    recent_approvals: List[Dict[str, Any]] = Field(default_factory=list)
    dossier_records: List[Dict[str, Any]] = Field(default_factory=list)


class PlatformAnalyticsView(BaseModel):
    platform_version: str
    workspace_analytics: List[WorkspaceAnalyticsView] = Field(default_factory=list)
    counts_by_audience_type: Dict[str, int] = Field(default_factory=dict)
    counts_by_workflow_template: Dict[str, int] = Field(default_factory=dict)
    deployability_distribution: Dict[str, int] = Field(default_factory=dict)
    regime_distribution: Dict[str, int] = Field(default_factory=dict)
    trade_family_distribution: Dict[str, int] = Field(default_factory=dict)
    evidence_status_distribution: Dict[str, int] = Field(default_factory=dict)
    approval_throughput: Dict[str, int] = Field(default_factory=dict)
    export_throughput: Dict[str, int] = Field(default_factory=dict)
    live_candidate_ratio: Optional[float] = None
    supportive_evidence_ratio: Optional[float] = None
    average_dau_across_workspaces: Optional[float] = None
    size_band_distribution: Dict[str, int] = Field(default_factory=dict)
    recent_high_dau_dossiers: List[Dict[str, Any]] = Field(default_factory=list)


class DemoWorkspaceSnapshot(BaseModel):
    platform_version: str
    workspace_id: Optional[str] = None
    workspace_name: Optional[str] = None
    top_dossiers: List[Dict[str, Any]] = Field(default_factory=list)
    pending_approvals: List[Dict[str, Any]] = Field(default_factory=list)
    recent_exports: List[Dict[str, Any]] = Field(default_factory=list)
    integration_summary: Dict[str, Any] = Field(default_factory=dict)
    health_summary: Dict[str, Any] = Field(default_factory=dict)
    pilot_ready: bool = False
    warnings: List[str] = Field(default_factory=list)


class PlatformReadinessSnapshot(BaseModel):
    platform_version: str
    workspace_id: Optional[str] = None
    analysis_readiness: str = "partial"
    workflow_readiness: str = "partial"
    export_readiness: str = "partial"
    integration_readiness: str = "partial"
    health_warnings: List[str] = Field(default_factory=list)
    missing_enterprise_items: List[str] = Field(default_factory=list)
    pilot_ready: bool = False
    rationale: Optional[str] = None


class ReviewComment(BaseModel):
    comment_id: str
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    workflow_id: str
    dossier_id: str
    stage: Optional[str] = None
    author: Dict[str, Any] = Field(default_factory=dict)
    comment_type: str = "general"
    body: str
    severity: str = "info"
    created_at: Optional[str] = None
    resolved_at: Optional[str] = None
    status: str = "open"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConcernFlag(BaseModel):
    concern_id: str
    workflow_id: Optional[str] = None
    dossier_id: Optional[str] = None
    concern_type: str
    severity: str = "watch"
    summary: str
    source_comment_id: Optional[str] = None
    status: str = "open"


class ReviewThreadSummary(BaseModel):
    workflow_id: str
    dossier_id: str
    total_comments: int = 0
    unresolved_comments: int = 0
    resolved_comments: int = 0
    comments_by_severity: Dict[str, int] = Field(default_factory=dict)
    comments_by_type: Dict[str, int] = Field(default_factory=dict)
    latest_comment: Optional[Dict[str, Any]] = None


class DecisionRationale(BaseModel):
    summary: Optional[str] = None
    key_risks: List[str] = Field(default_factory=list)
    key_evidence_strengths: List[str] = Field(default_factory=list)
    key_evidence_gaps: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ReviewSummary(BaseModel):
    workflow_id: str
    dossier_id: str
    thread_summary: ReviewThreadSummary
    unresolved_concern_count: int = 0
    concern_flags: List[ConcernFlag] = Field(default_factory=list)
    latest_comments: List[Dict[str, Any]] = Field(default_factory=list)
    decision_rationale: Optional[DecisionRationale] = None


class ReviewerSlot(BaseModel):
    slot_type: str
    assignee_placeholder: Optional[str] = None
    status: str = "open"
    notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AssignmentRecord(BaseModel):
    assignment_id: str
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    workflow_id: str
    dossier_id: Optional[str] = None
    slot_type: str
    assignee_placeholder: Optional[str] = None
    assigned_by: Dict[str, Any] = Field(default_factory=dict)
    status: str = "assigned"
    notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RoleAssignmentSummary(BaseModel):
    workflow_id: str
    dossier_id: Optional[str] = None
    owner: Optional[ReviewerSlot] = None
    primary_reviewer: Optional[ReviewerSlot] = None
    risk_reviewer: Optional[ReviewerSlot] = None
    committee_reviewer: Optional[ReviewerSlot] = None
    observers: List[ReviewerSlot] = Field(default_factory=list)
    assignments: List[AssignmentRecord] = Field(default_factory=list)


class DecisionCondition(BaseModel):
    condition_id: str
    label: str
    status: str = "required"
    notes: List[str] = Field(default_factory=list)


class DecisionOutcome(BaseModel):
    outcome: str
    recommendation_state: Optional[str] = None
    approved: bool = False


class CommitteeDecisionSnapshot(BaseModel):
    decision_id: str
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    workflow_id: str
    dossier_id: str
    stage: Optional[str] = None
    decision_status: str
    recommendation_state: str
    summary: str
    conditions: List[DecisionCondition] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)
    key_evidence_strengths: List[str] = Field(default_factory=list)
    key_evidence_gaps: List[str] = Field(default_factory=list)
    actor_context: Dict[str, Any] = Field(default_factory=dict)
    reviewer_context: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EscalationRecord(BaseModel):
    escalation_id: str
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    workflow_id: str
    dossier_id: Optional[str] = None
    action_type: str
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    rationale: Optional[str] = None
    actor: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendationState(BaseModel):
    state: str = "draft"
    locked: bool = False
    summary: Optional[str] = None
    rationale: Optional[str] = None
    locked_at: Optional[str] = None
    locked_by: Dict[str, Any] = Field(default_factory=dict)
    last_changed_at: Optional[str] = None
    last_changed_by: Dict[str, Any] = Field(default_factory=dict)
    source_decision_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendationLockRecord(BaseModel):
    lock_id: str
    workflow_id: str
    dossier_id: str
    recommendation_state: RecommendationState
    reason: Optional[str] = None
    actor: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    unlocked_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendationChangeRecord(BaseModel):
    change_id: str
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    workflow_id: str
    dossier_id: str
    previous_state: Optional[str] = None
    new_state: str
    action_type: str
    locked: bool = False
    snapshot: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None
    actor: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


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


class WorkflowActionRequest(BaseModel):
    dossier_id: Optional[str] = None
    action_type: str
    requested_stage: Optional[str] = None
    requested_status: Optional[str] = None
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowApprovalRequestPayload(BaseModel):
    approval_id: Optional[str] = None
    dossier_id: Optional[str] = None
    mode: str = "request"
    requested_role: str = "committee"
    decision_type: Optional[str] = None
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DossierExportRequest(BaseModel):
    pack_type: str = "dossier_pack"
    report_profile: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RenderExportRequest(BaseModel):
    pack_type: str = "dossier_pack"
    export_format: str = "html"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StoreExportRequest(BaseModel):
    pack_type: str = "dossier_pack"
    export_format: str = "html"
    render_id: Optional[str] = None
    storage_backend: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReviewCommentRequest(BaseModel):
    stage: Optional[str] = None
    comment_type: str = "general"
    body: str
    severity: str = "info"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResolveCommentRequest(BaseModel):
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowAssignmentRequest(BaseModel):
    dossier_id: Optional[str] = None
    slot_type: str
    assignee_placeholder: Optional[str] = None
    status: str = "assigned"
    notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CommitteeDecisionRequest(BaseModel):
    dossier_id: Optional[str] = None
    decision_status: str
    recommendation_state: str
    summary: str
    conditions: List[Union[Dict[str, Any], str]] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)
    key_evidence_strengths: List[str] = Field(default_factory=list)
    key_evidence_gaps: List[str] = Field(default_factory=list)
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendationStateRequest(BaseModel):
    dossier_id: Optional[str] = None
    recommendation_state: str
    action_type: str = "revise_recommendation"
    lock_recommendation: Optional[bool] = None
    summary: Optional[str] = None
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateIntegrationBindingRequest(BaseModel):
    integration_type: str
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    status: str = "configured"
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntegrationExecutionRequest(BaseModel):
    action_type: str = "sync_export"
    dossier_id: Optional[str] = None
    export_id: Optional[str] = None
    render_id: Optional[str] = None
    pack_type: str = "dossier_pack"
    export_format: str = "html"
    event_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
