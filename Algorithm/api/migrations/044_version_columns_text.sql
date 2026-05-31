-- Standardize all version columns to TEXT.
-- Migrations 015-018 created signal_version/feature_version as INT (DEFAULT 1).
-- All newer tables (axiom_scores_daily, prosperity_signals_daily,
-- feature_provenance_daily) use TEXT. Converting INT → TEXT so the
-- canonical version strings (e.g. "phase9_canonical_signal_v1") can be
-- written to every table without type errors.

ALTER TABLE signals_daily
    ALTER COLUMN signal_version TYPE TEXT USING signal_version::TEXT;

ALTER TABLE signals_intraday
    ALTER COLUMN signal_version TYPE TEXT USING signal_version::TEXT;

ALTER TABLE features_daily
    ALTER COLUMN feature_version TYPE TEXT USING feature_version::TEXT;

ALTER TABLE features_intraday
    ALTER COLUMN feature_version TYPE TEXT USING feature_version::TEXT;
