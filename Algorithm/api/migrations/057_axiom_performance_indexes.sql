-- Performance indexes for axiom_scores_daily screener queries
CREATE INDEX IF NOT EXISTS idx_axiom_scores_dau_ranking
    ON axiom_scores_daily (as_of_date DESC, deployable_alpha_utility DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_axiom_scores_high_dau
    ON axiom_scores_daily (as_of_date DESC, symbol)
    WHERE deployable_alpha_utility >= 60;

-- Partial index on signal_pnl_daily for 5-day PnL lookups
CREATE INDEX IF NOT EXISTS idx_signal_pnl_5d
    ON signal_pnl_daily (signal_date DESC, symbol, horizon_days)
    WHERE horizon_days = 5;
