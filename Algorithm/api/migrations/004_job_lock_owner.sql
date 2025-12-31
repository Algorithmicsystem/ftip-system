DO $$
BEGIN
    IF to_regclass('ftip_job_runs') IS NOT NULL THEN
        ALTER TABLE IF EXISTS ftip_job_runs
            ADD COLUMN IF NOT EXISTS lock_owner TEXT;
        ALTER TABLE IF EXISTS ftip_job_runs
            ALTER COLUMN lock_owner SET DEFAULT 'unknown';
    END IF;

    IF to_regclass('prosperity_job_runs') IS NOT NULL THEN
        ALTER TABLE IF EXISTS prosperity_job_runs
            ADD COLUMN IF NOT EXISTS lock_owner TEXT;
        ALTER TABLE IF EXISTS prosperity_job_runs
            ALTER COLUMN lock_owner SET DEFAULT 'unknown';
    END IF;
END;
$$;
