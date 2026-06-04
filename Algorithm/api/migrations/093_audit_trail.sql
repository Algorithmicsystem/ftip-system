CREATE TABLE IF NOT EXISTS audit_trail (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    tenant_id TEXT,
    actor_type TEXT NOT NULL DEFAULT 'system',
    resource_type TEXT NOT NULL DEFAULT '',
    resource_id TEXT NOT NULL DEFAULT '',
    symbol TEXT,
    as_of_date DATE,
    output_hash TEXT NOT NULL DEFAULT '',
    output_summary TEXT NOT NULL DEFAULT '',
    previous_event_hash TEXT NOT NULL DEFAULT '',
    event_hash TEXT NOT NULL DEFAULT '',
    ip_address TEXT,
    api_version TEXT NOT NULL DEFAULT 'v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_trail_tenant_created ON audit_trail(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trail_event_type ON audit_trail(event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trail_symbol ON audit_trail(symbol, created_at DESC);

-- Enforce append-only: no updates or deletes permitted via application
-- (enforced at application layer; DB-level rule below as extra safety)
CREATE OR REPLACE RULE no_update_audit AS ON UPDATE TO audit_trail DO INSTEAD NOTHING;
CREATE OR REPLACE RULE no_delete_audit AS ON DELETE TO audit_trail DO INSTEAD NOTHING;
