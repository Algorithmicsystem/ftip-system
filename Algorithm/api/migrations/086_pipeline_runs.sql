CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id              TEXT PRIMARY KEY,
    as_of_date          DATE,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ,
    overall_status      TEXT,
    symbols_processed   INT DEFAULT 0,
    total_errors        INT DEFAULT 0,
    stages              JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started
    ON pipeline_runs (started_at DESC);
