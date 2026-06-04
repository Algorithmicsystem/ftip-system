CREATE TABLE IF NOT EXISTS ips_constraints (
    portfolio_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    constraint_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ips_constraints_tenant ON ips_constraints(tenant_id);
