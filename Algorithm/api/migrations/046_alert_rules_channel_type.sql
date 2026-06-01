-- Phase 23: add channel_type to signal_alert_rules
-- Distinguishes generic JSON webhooks from Slack Incoming Webhooks so the
-- payload can be formatted with Slack blocks when appropriate.

ALTER TABLE signal_alert_rules
    ADD COLUMN IF NOT EXISTS channel_type TEXT NOT NULL DEFAULT 'generic'
        CHECK (channel_type IN ('generic', 'slack'));
