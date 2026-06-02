-- Migration 056: Multi-Tenant API Access Control
-- Maps API keys to tenants with pricing tier and sector access restrictions.

CREATE TABLE IF NOT EXISTS api_tenants (
    tenant_id       TEXT        NOT NULL PRIMARY KEY,
    org_name        TEXT        NOT NULL,
    api_key_hash    TEXT        NOT NULL UNIQUE,  -- SHA-256 hex of raw key
    tier            TEXT        NOT NULL DEFAULT 'free'
                                CHECK (tier IN ('free', 'pro', 'enterprise')),
    allowed_sectors JSONB,                        -- NULL = all sectors; array of sector strings
    rpm_limit       INT         NOT NULL DEFAULT 30,
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ                   -- NULL = no expiry
);

CREATE INDEX IF NOT EXISTS idx_api_tenants_key
    ON api_tenants (api_key_hash)
    WHERE is_active = TRUE;

COMMENT ON TABLE api_tenants IS
    'Multi-tenant API access control. Each row maps a hashed API key to an org '
    'with a pricing tier (free/pro/enterprise) and optional sector restrictions. '
    'Tier mapping: free → /prosperity + /axiom; pro → + /linkage + /ops; '
    'enterprise → full platform including /pe and /smb.';

COMMENT ON COLUMN api_tenants.allowed_sectors IS
    'JSON array of allowed sector names, e.g. ["Technology","Healthcare"]. '
    'NULL means unrestricted sector access within the tier.';
