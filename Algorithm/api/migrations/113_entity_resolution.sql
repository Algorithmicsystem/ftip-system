CREATE TABLE IF NOT EXISTS entity_resolution (
    id             SERIAL      PRIMARY KEY,
    ticker         TEXT        NOT NULL UNIQUE,
    canonical_name TEXT        NOT NULL,
    aliases        JSONB       NOT NULL DEFAULT '[]'::jsonb,
    exchange       TEXT,
    sector         TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_entity_resolution_canonical
    ON entity_resolution(lower(canonical_name));

CREATE INDEX IF NOT EXISTS idx_entity_resolution_aliases
    ON entity_resolution USING GIN(aliases);
