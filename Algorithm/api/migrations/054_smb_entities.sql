-- Migration 054: SMB Entity Registry
-- Small and medium business entities with cash flow and supplier risk tracking.

CREATE TABLE IF NOT EXISTS smb_entities (
    entity_id       TEXT        NOT NULL PRIMARY KEY,
    owner_id        TEXT        NOT NULL,   -- user / org that owns this SMB record
    business_name   TEXT        NOT NULL,
    industry        TEXT,
    geography       TEXT,
    founded_year    INT,
    employee_count  INT,
    annual_revenue  NUMERIC,               -- most recent annual revenue ($K)
    primary_bank    TEXT,
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
    meta            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_smb_entities_owner
    ON smb_entities (owner_id, is_active);

COMMENT ON TABLE smb_entities IS
    'SMB entity registry. owner_id ties each business to a platform user/org. '
    'Annual_revenue is a snapshot for context, not the primary time-series source.';
