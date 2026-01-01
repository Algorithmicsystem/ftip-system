CREATE TABLE IF NOT EXISTS prosperity_symbol_coverage (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    job_name TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('OK','FAILED','SKIPPED')),
    reason_code TEXT NULL,
    reason_detail TEXT NULL,
    bars_returned INT NULL,
    bars_required INT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_prosperity_symbol_coverage_run_id ON prosperity_symbol_coverage(run_id);
CREATE INDEX IF NOT EXISTS idx_prosperity_symbol_coverage_asof_symbol ON prosperity_symbol_coverage(as_of_date, symbol);
CREATE INDEX IF NOT EXISTS idx_prosperity_symbol_coverage_asof_reason ON prosperity_symbol_coverage(as_of_date, reason_code);
