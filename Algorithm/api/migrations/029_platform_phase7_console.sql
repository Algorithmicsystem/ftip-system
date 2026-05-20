CREATE TABLE IF NOT EXISTS platform_rendered_exports (
    render_id TEXT PRIMARY KEY,
    export_id TEXT NOT NULL REFERENCES platform_export_packs(export_id) ON DELETE CASCADE,
    export_format TEXT NOT NULL,
    content_type TEXT NOT NULL,
    rendered_content TEXT NOT NULL,
    file_name_hint TEXT NOT NULL,
    section_count INT NOT NULL DEFAULT 0,
    checksum TEXT,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_platform_rendered_exports_export
    ON platform_rendered_exports(export_id, generated_at DESC);

CREATE TABLE IF NOT EXISTS platform_integration_executions (
    execution_id TEXT PRIMARY KEY,
    binding_id TEXT NOT NULL REFERENCES platform_integration_bindings(binding_id) ON DELETE CASCADE,
    integration_type TEXT NOT NULL,
    action_type TEXT NOT NULL,
    status TEXT NOT NULL,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE SET NULL,
    organization_id TEXT REFERENCES organizations(organization_id) ON DELETE SET NULL,
    dossier_id TEXT REFERENCES dossiers(dossier_id) ON DELETE SET NULL,
    export_id TEXT REFERENCES platform_export_packs(export_id) ON DELETE SET NULL,
    render_id TEXT REFERENCES platform_rendered_exports(render_id) ON DELETE SET NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload_summary JSONB,
    output_summary JSONB,
    error_summary TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_platform_integration_executions_binding
    ON platform_integration_executions(binding_id, completed_at DESC);
