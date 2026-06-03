-- Phase 13.5: LP Reports
CREATE TABLE IF NOT EXISTS lp_reports (
    report_id       TEXT        NOT NULL PRIMARY KEY,
    org_id          TEXT        NOT NULL,
    report_quarter  TEXT        NOT NULL,
    report          JSONB       DEFAULT '{}'::JSONB,
    generated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lp_reports_org_quarter
    ON lp_reports (org_id, report_quarter DESC);
