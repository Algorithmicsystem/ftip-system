CREATE TABLE IF NOT EXISTS research_reports (
    report_id           TEXT PRIMARY KEY,
    symbol              TEXT NOT NULL,
    report_date         DATE NOT NULL,
    analyst_rating      TEXT NOT NULL,
    dau                 NUMERIC(8, 4),
    conviction          TEXT,
    executive_summary   TEXT,
    report_json         JSONB NOT NULL DEFAULT '{}',
    use_llm             BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rr_symbol_date
    ON research_reports (symbol, report_date DESC);
