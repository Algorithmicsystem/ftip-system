-- Migration 052: Private Entity Registry
-- Tracks PE portfolio companies with entry metadata and public peer linkage.

CREATE TABLE IF NOT EXISTS private_entities (
    entity_id           TEXT        NOT NULL PRIMARY KEY,
    org_id              TEXT        NOT NULL,   -- owning PE fund / org
    entity_name         TEXT        NOT NULL,
    entity_type         TEXT        NOT NULL DEFAULT 'portfolio_company',
                                                -- portfolio_company | gp | lp | co_invest
    sector              TEXT,
    geography           TEXT,
    public_peer_symbol  TEXT,                   -- linked public market ticker
    entry_date          DATE,
    entry_ev            NUMERIC,                -- entry enterprise value ($M)
    target_exit_date    DATE,
    target_exit_multiple NUMERIC,               -- target EV / entry EV
    is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
    meta                JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_private_entities_org
    ON private_entities (org_id, is_active);

CREATE INDEX IF NOT EXISTS idx_private_entities_peer
    ON private_entities (public_peer_symbol)
    WHERE public_peer_symbol IS NOT NULL;

COMMENT ON TABLE private_entities IS
    'PE portfolio company registry with entry metadata and public peer linkage. '
    'org_id identifies the owning PE fund. public_peer_symbol links to market_symbols.';
