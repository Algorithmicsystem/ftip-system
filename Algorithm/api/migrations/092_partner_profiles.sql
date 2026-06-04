CREATE TABLE IF NOT EXISTS partner_profiles (
    partner_id TEXT PRIMARY KEY,
    org_name TEXT NOT NULL,
    partner_tier TEXT NOT NULL DEFAULT 'integration_partner',
    contact_email TEXT NOT NULL,
    agreement_signed BOOLEAN NOT NULL DEFAULT FALSE,
    revenue_share_pct DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    custom_branding JSONB NOT NULL DEFAULT '{}',
    rate_limit_multiplier DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    endpoints_allowed JSONB NOT NULL DEFAULT '[]',
    api_key_prefix TEXT NOT NULL DEFAULT 'ax_',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
