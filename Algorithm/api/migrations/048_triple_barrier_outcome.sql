-- Migration 048: Add triple_barrier_outcome column to signal_pnl_daily
-- Implements De Prado Triple Barrier Labeling (Advances in Financial Machine Learning)
-- Values: 1 = profit target hit, -1 = stop hit, 0 = time stop, NULL = not computed

ALTER TABLE signal_pnl_daily
    ADD COLUMN IF NOT EXISTS triple_barrier_outcome INT
        CHECK (triple_barrier_outcome IN (-1, 0, 1));

COMMENT ON COLUMN signal_pnl_daily.triple_barrier_outcome IS
    'De Prado triple barrier label: 1=profit target, -1=stop loss, 0=time stop';
