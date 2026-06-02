-- Migration 051: Symbol Linkage Strength and Validation Metadata
-- Upgrades symbol_linkage to capture linkage quality and staleness.

ALTER TABLE symbol_linkage
    ADD COLUMN IF NOT EXISTS linkage_strength  NUMERIC,    -- 0–1: correlation / conviction
    ADD COLUMN IF NOT EXISTS last_validated    DATE,       -- date linkage was last confirmed
    ADD COLUMN IF NOT EXISTS validation_method TEXT;       -- rolling_correlation | manual | etf_holdings | fundamental_peer

COMMENT ON COLUMN symbol_linkage.linkage_strength IS
    '0–1 strength of the linkage (e.g. rolling 63d return correlation for sector_peer)';
COMMENT ON COLUMN symbol_linkage.last_validated IS
    'Date this linkage was last validated or recalculated';
COMMENT ON COLUMN symbol_linkage.validation_method IS
    'How the linkage was validated: rolling_correlation, manual, etf_holdings, fundamental_peer';
