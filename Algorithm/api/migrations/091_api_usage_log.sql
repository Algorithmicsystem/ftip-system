CREATE TABLE IF NOT EXISTS api_usage_log (
    log_id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL DEFAULT 'GET',
    status_code INT,
    response_time_ms DOUBLE PRECISION,
    symbol TEXT,
    as_of_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_api_usage_log_tenant ON api_usage_log(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_log_endpoint ON api_usage_log(endpoint, created_at);
