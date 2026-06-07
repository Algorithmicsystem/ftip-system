-- Migration 098: Deal flow scores table for daily acquisition screening
CREATE TABLE IF NOT EXISTS deal_flow_scores (
    screen_date     DATE PRIMARY KEY,
    candidates      JSONB NOT NULL DEFAULT '[]'::jsonb,
    universe_screened INTEGER NOT NULL DEFAULT 0,
    candidates_found  INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_deal_flow_screen_date ON deal_flow_scores (screen_date DESC);
