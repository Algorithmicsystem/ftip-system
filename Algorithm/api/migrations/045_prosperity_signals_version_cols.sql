ALTER TABLE prosperity_signals_daily
    ADD COLUMN IF NOT EXISTS signal_version TEXT,
    ADD COLUMN IF NOT EXISTS feature_version TEXT;
