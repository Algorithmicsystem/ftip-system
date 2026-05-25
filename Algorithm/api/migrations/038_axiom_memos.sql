-- Structured IC memos with lineage hash — one row per unique computation
CREATE TABLE IF NOT EXISTS axiom_memos (
    memo_id          TEXT        PRIMARY KEY,
    symbol           TEXT        NOT NULL,
    as_of_date       DATE        NOT NULL,
    lineage_hash     TEXT        NOT NULL,
    schema_version   TEXT        NOT NULL DEFAULT '1.0',
    signal_label     TEXT,
    dau              NUMERIC,
    conviction_score NUMERIC,
    suggested_weight NUMERIC,
    regime_label     TEXT,
    ic_state         TEXT,
    memo_body        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    canonical_inputs JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Unique hash ensures same inputs never produce duplicate rows
CREATE UNIQUE INDEX IF NOT EXISTS idx_axiom_memos_lineage_hash
    ON axiom_memos (lineage_hash);

CREATE INDEX IF NOT EXISTS idx_axiom_memos_symbol_date
    ON axiom_memos (symbol, as_of_date DESC);
