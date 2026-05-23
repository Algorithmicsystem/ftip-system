CREATE TABLE IF NOT EXISTS platform_review_comments (
    comment_id TEXT PRIMARY KEY,
    organization_id TEXT REFERENCES organizations(organization_id) ON DELETE SET NULL,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE SET NULL,
    workflow_id TEXT NOT NULL REFERENCES workflow_instances(workflow_id) ON DELETE CASCADE,
    dossier_id TEXT NOT NULL REFERENCES dossiers(dossier_id) ON DELETE CASCADE,
    stage TEXT,
    author JSONB,
    comment_type TEXT NOT NULL,
    body TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'open',
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_platform_review_comments_scope
    ON platform_review_comments(organization_id, workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_platform_review_comments_workflow
    ON platform_review_comments(workflow_id, dossier_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS platform_assignments (
    assignment_id TEXT PRIMARY KEY,
    organization_id TEXT REFERENCES organizations(organization_id) ON DELETE SET NULL,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE SET NULL,
    workflow_id TEXT NOT NULL REFERENCES workflow_instances(workflow_id) ON DELETE CASCADE,
    dossier_id TEXT REFERENCES dossiers(dossier_id) ON DELETE SET NULL,
    slot_type TEXT NOT NULL,
    assignee_placeholder TEXT,
    assigned_by JSONB,
    status TEXT NOT NULL DEFAULT 'assigned',
    notes JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_platform_assignments_scope
    ON platform_assignments(organization_id, workspace_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_platform_assignments_workflow
    ON platform_assignments(workflow_id, dossier_id, slot_type, updated_at DESC);

CREATE TABLE IF NOT EXISTS platform_committee_decisions (
    decision_id TEXT PRIMARY KEY,
    organization_id TEXT REFERENCES organizations(organization_id) ON DELETE SET NULL,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE SET NULL,
    workflow_id TEXT NOT NULL REFERENCES workflow_instances(workflow_id) ON DELETE CASCADE,
    dossier_id TEXT NOT NULL REFERENCES dossiers(dossier_id) ON DELETE CASCADE,
    stage TEXT,
    decision_status TEXT NOT NULL,
    recommendation_state TEXT NOT NULL,
    summary TEXT NOT NULL,
    conditions JSONB,
    key_risks JSONB,
    key_evidence_strengths JSONB,
    key_evidence_gaps JSONB,
    actor_context JSONB,
    reviewer_context JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_platform_committee_decisions_scope
    ON platform_committee_decisions(organization_id, workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_platform_committee_decisions_workflow
    ON platform_committee_decisions(workflow_id, dossier_id, created_at DESC);

CREATE TABLE IF NOT EXISTS platform_recommendation_changes (
    change_id TEXT PRIMARY KEY,
    organization_id TEXT REFERENCES organizations(organization_id) ON DELETE SET NULL,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE SET NULL,
    workflow_id TEXT NOT NULL REFERENCES workflow_instances(workflow_id) ON DELETE CASCADE,
    dossier_id TEXT NOT NULL REFERENCES dossiers(dossier_id) ON DELETE CASCADE,
    previous_state TEXT,
    new_state TEXT NOT NULL,
    action_type TEXT NOT NULL,
    locked BOOLEAN NOT NULL DEFAULT FALSE,
    snapshot JSONB,
    rationale TEXT,
    actor JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_platform_recommendation_changes_scope
    ON platform_recommendation_changes(organization_id, workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_platform_recommendation_changes_workflow
    ON platform_recommendation_changes(workflow_id, dossier_id, created_at DESC);
