-- Phase 12.3: Company intelligence archive
CREATE TABLE IF NOT EXISTS company_intelligence_archive (
    event_id            UUID        NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol              TEXT        NOT NULL,
    event_date          DATE        NOT NULL,
    event_type          TEXT        NOT NULL,
    event_summary       TEXT,
    event_payload       JSONB       DEFAULT '{}'::JSONB,
    impact_score        NUMERIC,
    axiom_score_before  NUMERIC,
    axiom_score_after   NUMERIC,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cia_symbol_date
    ON company_intelligence_archive (symbol, event_date DESC);

CREATE INDEX IF NOT EXISTS idx_cia_type_date
    ON company_intelligence_archive (event_type, event_date DESC);
