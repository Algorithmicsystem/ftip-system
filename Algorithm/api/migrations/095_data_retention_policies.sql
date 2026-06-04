CREATE TABLE IF NOT EXISTS data_retention_policies (
    tenant_id TEXT PRIMARY KEY,
    retain_trading_signals_days INT NOT NULL DEFAULT 2555,
    retain_audit_records_days INT NOT NULL DEFAULT 3650,
    retain_api_logs_days INT NOT NULL DEFAULT 365,
    retain_research_reports_days INT NOT NULL DEFAULT 1825,
    data_residency TEXT NOT NULL DEFAULT 'us',
    gdpr_applicable BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
