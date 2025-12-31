DO $$
BEGIN
    CREATE TABLE IF NOT EXISTS ftip_job_runs (
        run_id UUID PRIMARY KEY,
        job_name TEXT NOT NULL,
        as_of_date DATE,
        started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        finished_at TIMESTAMPTZ,
        status TEXT,
        requested JSONB NOT NULL DEFAULT '{}'::jsonb,
        result JSONB,
        error TEXT,
        lock_owner TEXT NOT NULL DEFAULT 'unknown',
        lock_acquired_at TIMESTAMPTZ,
        lock_expires_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS job_name TEXT;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS as_of_date DATE;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS status TEXT;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS requested JSONB;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS result JSONB;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS error TEXT;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS lock_owner TEXT;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS lock_acquired_at TIMESTAMPTZ;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS lock_expires_at TIMESTAMPTZ;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

    UPDATE ftip_job_runs SET created_at = now() WHERE created_at IS NULL;
    UPDATE ftip_job_runs SET updated_at = now() WHERE updated_at IS NULL;
    UPDATE ftip_job_runs SET requested = '{}'::jsonb WHERE requested IS NULL;
    UPDATE ftip_job_runs SET as_of_date = CURRENT_DATE WHERE as_of_date IS NULL;
    UPDATE ftip_job_runs SET lock_owner = 'unknown' WHERE lock_owner IS NULL;
    UPDATE ftip_job_runs SET started_at = now() WHERE started_at IS NULL;

    ALTER TABLE IF EXISTS ftip_job_runs
        ALTER COLUMN job_name SET NOT NULL,
        ALTER COLUMN started_at SET DEFAULT now(),
        ALTER COLUMN started_at SET NOT NULL,
        ALTER COLUMN requested SET DEFAULT '{}'::jsonb,
        ALTER COLUMN requested SET NOT NULL,
        ALTER COLUMN lock_owner SET DEFAULT 'unknown',
        ALTER COLUMN lock_owner SET NOT NULL,
        ALTER COLUMN created_at SET DEFAULT now(),
        ALTER COLUMN created_at SET NOT NULL,
        ALTER COLUMN updated_at SET DEFAULT now(),
        ALTER COLUMN updated_at SET NOT NULL;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = ANY (current_schemas(false))
          AND tablename = 'ftip_job_runs'
          AND indexname = 'ftip_job_runs_active_idx'
    ) THEN
        CREATE UNIQUE INDEX ftip_job_runs_active_idx
            ON ftip_job_runs(job_name)
            WHERE status = 'IN_PROGRESS';
    END IF;
END;
$$;
