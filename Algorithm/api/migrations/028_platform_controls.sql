CREATE TABLE IF NOT EXISTS platform_memberships (
    membership_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    organization_id TEXT REFERENCES organizations(organization_id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    permissions JSONB,
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_platform_memberships_scope
    ON platform_memberships(user_id, organization_id, workspace_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS platform_approval_requests (
    approval_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflow_instances(workflow_id) ON DELETE CASCADE,
    dossier_id TEXT REFERENCES dossiers(dossier_id) ON DELETE CASCADE,
    requested_role TEXT NOT NULL,
    requested_by JSONB,
    status TEXT NOT NULL DEFAULT 'pending',
    stage TEXT,
    rationale TEXT,
    required_permissions JSONB,
    decisions JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_platform_approval_requests_workflow
    ON platform_approval_requests(workflow_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS platform_audit_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    organization_id TEXT,
    workspace_id TEXT,
    actor JSONB,
    event_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload JSONB,
    rationale TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_platform_audit_events_workspace
    ON platform_audit_events(workspace_id, event_ts DESC);

CREATE INDEX IF NOT EXISTS idx_platform_audit_events_resource
    ON platform_audit_events(resource_type, resource_id, event_ts DESC);

CREATE TABLE IF NOT EXISTS platform_export_packs (
    export_id TEXT PRIMARY KEY,
    dossier_id TEXT NOT NULL REFERENCES dossiers(dossier_id) ON DELETE CASCADE,
    workflow_id TEXT REFERENCES workflow_instances(workflow_id) ON DELETE SET NULL,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE SET NULL,
    pack_type TEXT NOT NULL,
    title TEXT NOT NULL,
    subtitle TEXT,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    framework_version TEXT,
    organization_context JSONB,
    workspace_context JSONB,
    entity_context JSONB,
    approval_status TEXT,
    evidence_summary TEXT,
    ordered_sections JSONB,
    metadata JSONB,
    content_hash TEXT,
    status TEXT NOT NULL DEFAULT 'generated'
);

CREATE INDEX IF NOT EXISTS idx_platform_export_packs_dossier
    ON platform_export_packs(dossier_id, generated_at DESC);

CREATE TABLE IF NOT EXISTS platform_integration_bindings (
    binding_id TEXT PRIMARY KEY,
    integration_type TEXT NOT NULL,
    organization_id TEXT REFERENCES organizations(organization_id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'configured',
    config JSONB,
    health JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_platform_integration_bindings_scope
    ON platform_integration_bindings(organization_id, workspace_id, updated_at DESC);
