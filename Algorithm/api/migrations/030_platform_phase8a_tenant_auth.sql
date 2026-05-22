ALTER TABLE platform_audit_events
ADD COLUMN IF NOT EXISTS session_id TEXT;

ALTER TABLE platform_audit_events
ADD COLUMN IF NOT EXISTS auth_mode TEXT;

CREATE INDEX IF NOT EXISTS idx_platform_audit_events_session_id
ON platform_audit_events(session_id);

CREATE INDEX IF NOT EXISTS idx_platform_audit_events_org_workspace
ON platform_audit_events(organization_id, workspace_id, event_ts DESC);
