CREATE TABLE IF NOT EXISTS job_postings_daily (
    symbol       TEXT        NOT NULL,
    as_of_date   DATE        NOT NULL,
    job_count    INTEGER,
    company_name TEXT,
    source       TEXT        NOT NULL DEFAULT 'indeed_rss',
    raw          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_job_postings_symbol_date
    ON job_postings_daily(symbol, as_of_date DESC);

CREATE INDEX IF NOT EXISTS idx_job_postings_date
    ON job_postings_daily(as_of_date DESC);
