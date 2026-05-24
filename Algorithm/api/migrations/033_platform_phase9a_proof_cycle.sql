CREATE TABLE IF NOT EXISTS platform_recommendation_tracks (
    track_id TEXT PRIMARY KEY,
    organization_id TEXT REFERENCES organizations(organization_id) ON DELETE SET NULL,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE SET NULL,
    workflow_id TEXT NOT NULL REFERENCES workflow_instances(workflow_id) ON DELETE CASCADE,
    dossier_id TEXT NOT NULL REFERENCES dossiers(dossier_id) ON DELETE CASCADE,
    entity_id TEXT REFERENCES coverage_entities(entity_id) ON DELETE SET NULL,
    symbol TEXT NOT NULL,
    axiom_artifact_id TEXT,
    axiom_history_artifact_id TEXT,
    report_id TEXT,
    session_id TEXT,
    recommendation_state_at_start TEXT NOT NULL,
    deployability_tier_at_start TEXT,
    size_band_at_start TEXT,
    regime_label TEXT,
    trade_family TEXT,
    strongest_engine_at_start TEXT,
    weakest_engine_at_start TEXT,
    signal_action_at_start TEXT,
    evidence_status_at_start TEXT,
    start_deployable_alpha_utility DOUBLE PRECISION,
    start_validated_edge DOUBLE PRECISION,
    start_overall_coverage DOUBLE PRECISION,
    start_overall_confidence DOUBLE PRECISION,
    start_engine_scores JSONB,
    start_source_context JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    tracking_start_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    tracking_end_at TIMESTAMPTZ,
    tracking_status JSONB,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_platform_recommendation_tracks_scope
    ON platform_recommendation_tracks(organization_id, workspace_id, tracking_start_at DESC);

CREATE INDEX IF NOT EXISTS idx_platform_recommendation_tracks_workflow
    ON platform_recommendation_tracks(workflow_id, dossier_id, tracking_start_at DESC);

CREATE TABLE IF NOT EXISTS platform_paper_trades (
    paper_trade_id TEXT PRIMARY KEY,
    track_id TEXT NOT NULL REFERENCES platform_recommendation_tracks(track_id) ON DELETE CASCADE,
    organization_id TEXT REFERENCES organizations(organization_id) ON DELETE SET NULL,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE SET NULL,
    workflow_id TEXT NOT NULL REFERENCES workflow_instances(workflow_id) ON DELETE CASCADE,
    dossier_id TEXT NOT NULL REFERENCES dossiers(dossier_id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    entry_reference_date DATE NOT NULL,
    entry_price DOUBLE PRECISION,
    thesis_state_at_entry JSONB,
    tracked_horizons JSONB,
    current_status TEXT NOT NULL DEFAULT 'active',
    outcome_summary JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_platform_paper_trades_scope
    ON platform_paper_trades(organization_id, workspace_id, entry_reference_date DESC);

CREATE INDEX IF NOT EXISTS idx_platform_paper_trades_track
    ON platform_paper_trades(track_id, dossier_id, entry_reference_date DESC);

CREATE TABLE IF NOT EXISTS platform_outcome_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    organization_id TEXT REFERENCES organizations(organization_id) ON DELETE SET NULL,
    workspace_id TEXT REFERENCES workspaces(workspace_id) ON DELETE SET NULL,
    workflow_id TEXT NOT NULL REFERENCES workflow_instances(workflow_id) ON DELETE CASCADE,
    dossier_id TEXT NOT NULL REFERENCES dossiers(dossier_id) ON DELETE CASCADE,
    track_id TEXT NOT NULL REFERENCES platform_recommendation_tracks(track_id) ON DELETE CASCADE,
    paper_trade_id TEXT NOT NULL REFERENCES platform_paper_trades(paper_trade_id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    evidence_mode TEXT NOT NULL DEFAULT 'paper_tracked',
    tracking_status JSONB,
    windows JSONB,
    assessment JSONB,
    evidence_status JSONB,
    benchmark_comparison JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_platform_outcome_snapshots_scope
    ON platform_outcome_snapshots(organization_id, workspace_id, snapshot_date DESC);

CREATE INDEX IF NOT EXISTS idx_platform_outcome_snapshots_track
    ON platform_outcome_snapshots(track_id, paper_trade_id, dossier_id, snapshot_date DESC);
