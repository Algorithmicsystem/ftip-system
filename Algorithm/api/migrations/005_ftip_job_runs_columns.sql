DO $$
DECLARE
    has_pkey BOOLEAN;
BEGIN
    -- Ensure base table exists with full schema for new deployments
    CREATE TABLE IF NOT EXISTS ftip_job_runs (
        run_id UUID PRIMARY KEY,
        job_name TEXT,
        as_of_date DATE,
        started_at TIMESTAMPTZ DEFAULT now(),
        finished_at TIMESTAMPTZ,
        status TEXT,
        requested JSONB DEFAULT '{}'::jsonb,
        result JSONB,
        error TEXT,
        lock_owner TEXT,
        lock_acquired_at TIMESTAMPTZ,
        lock_expires_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    );

    -- Add any missing columns as nullable first to keep legacy data intact
    ALTER TABLE IF EXISTS ftip_job_runs
        ADD COLUMN IF NOT EXISTS run_id UUID,
        ADD COLUMN IF NOT EXISTS job_name TEXT,
        ADD COLUMN IF NOT EXISTS as_of_date DATE,
        ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ,
        ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ,
        ADD COLUMN IF NOT EXISTS status TEXT,
        ADD COLUMN IF NOT EXISTS requested JSONB,
        ADD COLUMN IF NOT EXISTS result JSONB,
        ADD COLUMN IF NOT EXISTS error TEXT,
        ADD COLUMN IF NOT EXISTS lock_owner TEXT,
        ADD COLUMN IF NOT EXISTS lock_acquired_at TIMESTAMPTZ,
        ADD COLUMN IF NOT EXISTS lock_expires_at TIMESTAMPTZ,
        ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ,
        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

    -- Backfill legacy NULLs before applying constraints/defaults
    UPDATE ftip_job_runs SET created_at = COALESCE(created_at, now()) WHERE created_at IS NULL;
    UPDATE ftip_job_runs SET updated_at = COALESCE(updated_at, now()) WHERE updated_at IS NULL;
    UPDATE ftip_job_runs SET started_at = COALESCE(started_at, created_at, now()) WHERE started_at IS NULL;
    UPDATE ftip_job_runs SET requested = COALESCE(requested, '{}'::jsonb) WHERE requested IS NULL;

    -- Apply defaults and NOT NULL constraints after backfill
    ALTER TABLE IF EXISTS ftip_job_runs
        ALTER COLUMN started_at SET DEFAULT now(),
        ALTER COLUMN started_at SET NOT NULL,
        ALTER COLUMN requested SET DEFAULT '{}'::jsonb,
        ALTER COLUMN requested SET NOT NULL,
        ALTER COLUMN created_at SET DEFAULT now(),
        ALTER COLUMN created_at SET NOT NULL,
        ALTER COLUMN updated_at SET DEFAULT now(),
        ALTER COLUMN updated_at SET NOT NULL,
        ALTER COLUMN run_id SET NOT NULL;

    -- Ensure primary key exists on run_id
    SELECT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'ftip_job_runs'::regclass
          AND contype = 'p'
    ) INTO has_pkey;

    IF NOT has_pkey THEN
        ALTER TABLE ftip_job_runs ADD CONSTRAINT ftip_job_runs_pkey PRIMARY KEY (run_id);
    END IF;

    -- Ensure a non-unique partial index exists for active job lookups
    CREATE INDEX IF NOT EXISTS idx_ftip_job_runs_active
        ON ftip_job_runs (job_name)
        WHERE finished_at IS NULL;
END;
$$;
