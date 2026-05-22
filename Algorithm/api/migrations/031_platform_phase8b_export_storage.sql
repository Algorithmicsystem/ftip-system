CREATE TABLE IF NOT EXISTS platform_stored_exports (
    stored_export_id TEXT PRIMARY KEY,
    export_id TEXT NOT NULL REFERENCES platform_export_packs(export_id) ON DELETE CASCADE,
    render_id TEXT NOT NULL REFERENCES platform_rendered_exports(render_id) ON DELETE CASCADE,
    organization_id TEXT REFERENCES organizations(organization_id) ON DELETE SET NULL,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE SET NULL,
    dossier_id TEXT REFERENCES dossiers(dossier_id) ON DELETE SET NULL,
    workflow_id TEXT REFERENCES workflow_instances(workflow_id) ON DELETE SET NULL,
    pack_type TEXT NOT NULL,
    export_format TEXT NOT NULL,
    framework_version TEXT,
    approval_status TEXT,
    evidence_status TEXT,
    checksum TEXT,
    source_manifest_hash TEXT,
    content_hash TEXT,
    manifest_hash TEXT,
    section_count INT NOT NULL DEFAULT 0,
    file_name_hint TEXT NOT NULL,
    content_type TEXT,
    storage_backend TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    storage_ref JSONB,
    version_group_key TEXT NOT NULL,
    version_number INT NOT NULL DEFAULT 1,
    version_label TEXT NOT NULL DEFAULT 'v1',
    status TEXT NOT NULL DEFAULT 'stored',
    document_identity JSONB,
    source_context JSONB,
    approval_context JSONB,
    axiom_context JSONB,
    evidence_context JSONB,
    export_context JSONB,
    lineage_summary JSONB,
    created_by JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_platform_stored_exports_scope
    ON platform_stored_exports(organization_id, workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_platform_stored_exports_dossier
    ON platform_stored_exports(dossier_id, pack_type, export_format, version_number DESC);

CREATE INDEX IF NOT EXISTS idx_platform_stored_exports_version_group
    ON platform_stored_exports(version_group_key, version_number DESC);
