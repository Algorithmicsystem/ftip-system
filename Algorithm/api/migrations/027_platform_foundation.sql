CREATE TABLE IF NOT EXISTS organizations (
    organization_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    organization_type TEXT NOT NULL,
    settings JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(organization_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    audience_type TEXT NOT NULL,
    report_profile TEXT NOT NULL,
    default_workflow_template TEXT NOT NULL,
    platform_profile TEXT NOT NULL DEFAULT 'research_core',
    settings JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_workspaces_org
    ON workspaces(organization_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS coverage_entities (
    entity_id TEXT PRIMARY KEY,
    symbol TEXT,
    external_identifier TEXT,
    entity_type TEXT NOT NULL,
    display_name TEXT NOT NULL,
    sector TEXT,
    strategy TEXT,
    theme TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_coverage_entities_symbol
    ON coverage_entities(symbol, updated_at DESC);

CREATE TABLE IF NOT EXISTS workflow_instances (
    workflow_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    workflow_template_id TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'normal',
    owner_placeholder TEXT,
    metadata JSONB,
    stage_state JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_workflow_instances_workspace
    ON workflow_instances(workspace_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS dossiers (
    dossier_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflow_instances(workflow_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES coverage_entities(entity_id) ON DELETE CASCADE,
    dossier_type TEXT NOT NULL,
    title TEXT NOT NULL,
    current_summary JSONB,
    latest_axiom_analysis_id TEXT,
    latest_deployability_tier TEXT,
    latest_regime_label TEXT,
    latest_trade_family TEXT,
    latest_size_band TEXT,
    evidence_status TEXT,
    workflow_stage_state JSONB,
    sections JSONB,
    monitoring_state JSONB,
    historical_evidence_summary JSONB,
    lineage_summary JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dossiers_workflow
    ON dossiers(workflow_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_dossiers_regime_tier
    ON dossiers(latest_regime_label, latest_deployability_tier, updated_at DESC);

CREATE TABLE IF NOT EXISTS dossier_analysis_links (
    link_id TEXT PRIMARY KEY,
    dossier_id TEXT NOT NULL REFERENCES dossiers(dossier_id) ON DELETE CASCADE,
    report_id TEXT,
    session_id TEXT,
    axiom_artifact_id TEXT,
    axiom_report_pack_artifact_id TEXT,
    axiom_lineage_artifact_id TEXT,
    axiom_history_artifact_id TEXT,
    axiom_calibration_artifact_id TEXT,
    linked_payload JSONB,
    linked_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dossier_analysis_links_dossier
    ON dossier_analysis_links(dossier_id, linked_at DESC);

