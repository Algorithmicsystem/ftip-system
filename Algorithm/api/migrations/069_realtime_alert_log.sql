CREATE TABLE IF NOT EXISTS realtime_alert_log (
    alert_id           TEXT PRIMARY KEY,
    alert_type         TEXT NOT NULL,
    symbol             TEXT,
    severity           TEXT NOT NULL,
    payload            JSONB NOT NULL DEFAULT '{}'::jsonb,
    sent_to_connections INT NOT NULL DEFAULT 0,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_realtime_alert_log_created
    ON realtime_alert_log (created_at DESC, alert_type);
