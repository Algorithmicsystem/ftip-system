ALTER TABLE signal_ic_daily
    ADD COLUMN IF NOT EXISTS effective_breadth INT;

COMMENT ON COLUMN signal_ic_daily.effective_breadth IS
    'Grinold-Kahn effective breadth: count of symbols whose per-symbol rolling IC exceeds 0.02 over the 63-day window ending on as_of_date';
