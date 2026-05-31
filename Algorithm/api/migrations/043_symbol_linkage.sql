CREATE TABLE IF NOT EXISTS symbol_linkage (
    symbol          TEXT        NOT NULL,
    linked_symbol   TEXT        NOT NULL,
    link_type       TEXT        NOT NULL,   -- sector_peer | etf_member | competitor | sector_proxy | benchmark
    weight          NUMERIC,               -- optional: ETF weight, correlation coefficient, etc.
    source          TEXT,                  -- sector_auto | manual | etf_holdings | external
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
    meta            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, linked_symbol, link_type)
);

CREATE INDEX IF NOT EXISTS idx_symbol_linkage_symbol
    ON symbol_linkage (symbol, link_type, is_active);

CREATE INDEX IF NOT EXISTS idx_symbol_linkage_linked
    ON symbol_linkage (linked_symbol, link_type, is_active);
