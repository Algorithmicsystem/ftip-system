-- Migration 096: Document look-ahead bias fix in fundamentals_quarterly queries.
-- The bug: when report_date IS NULL, the query only filtered by fiscal_period_end,
-- allowing data that hadn't been publicly reported yet to pass through.
-- Fix applied in api/research/snapshot.py _load_fundamentals():
--   report_date <= as_of_date
--   OR (report_date IS NULL AND fiscal_period_end <= as_of_date - INTERVAL '45 days')
-- The 45-day lag approximates the typical SEC reporting delay for quarterly filings.
-- No schema changes required — this is a query-level fix.
SELECT 1;
