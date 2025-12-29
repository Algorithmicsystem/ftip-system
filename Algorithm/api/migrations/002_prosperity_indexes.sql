CREATE INDEX IF NOT EXISTS idx_prosperity_daily_bars_symbol_date ON prosperity_daily_bars(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_prosperity_features_symbol_asof ON prosperity_features_daily(symbol, as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_prosperity_signals_symbol_asof ON prosperity_signals_daily(symbol, as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_prosperity_backtests_created ON prosperity_backtest_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_prosperity_events_created ON prosperity_events(created_at DESC);
