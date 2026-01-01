-- Drop legacy unique/partial indexes that cannot be used for ON CONFLICT
DROP INDEX IF EXISTS ftip_job_runs_active_idx;
DROP INDEX IF EXISTS ftip_job_runs_active_job;
DROP INDEX IF EXISTS ftip_job_runs_active_uniq;
DROP INDEX IF EXISTS idx_ftip_job_runs_active_idx;

-- Drop any unique constraints on job_name to prevent ON CONFLICT reliance
DO $$
DECLARE
    r record;
BEGIN
    FOR r IN (
        SELECT conname
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE t.relname = 'ftip_job_runs'
          AND c.contype = 'u'
          AND pg_get_constraintdef(c.oid) LIKE '%job_name%'
    )
    LOOP
        EXECUTE format('ALTER TABLE ftip_job_runs DROP CONSTRAINT %I', r.conname);
    END LOOP;
END $$;

-- Ensure a non-unique partial index exists for active job lookups
CREATE INDEX IF NOT EXISTS idx_ftip_job_runs_active
    ON ftip_job_runs(job_name)
    WHERE finished_at IS NULL;
