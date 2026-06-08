-- Migration 105: Performance indexes for sub-200ms p95 SLA
-- Non-blocking in production; applied via standard migration runner on startup.

CREATE INDEX IF NOT EXISTS idx_axiom_scores_daily_symbol_date
    ON axiom_scores_daily (symbol, as_of_date DESC);

CREATE INDEX IF NOT EXISTS idx_signal_pnl_daily_symbol_date
    ON signal_pnl_daily (symbol, signal_date DESC, horizon_days);

CREATE INDEX IF NOT EXISTS idx_signal_ic_daily_field_horizon
    ON signal_ic_daily (score_field, horizon_label, as_of_date DESC);

CREATE INDEX IF NOT EXISTS idx_signal_performance_archive_war
    ON signal_performance_archive (signal_war DESC);

CREATE INDEX IF NOT EXISTS idx_company_intelligence_archive_symbol_date
    ON company_intelligence_archive (symbol, as_of_date DESC);

CREATE INDEX IF NOT EXISTS idx_axiom_intraday_session_symbol
    ON axiom_intraday_updates (session_date, symbol, update_time DESC);

CREATE INDEX IF NOT EXISTS idx_morning_briefings_date
    ON morning_briefings (briefing_date DESC);

CREATE INDEX IF NOT EXISTS idx_audit_trail_event_created
    ON audit_trail (event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_deal_flow_scores_acquisition
    ON deal_flow_scores (acquisition_score DESC, as_of_date DESC);
