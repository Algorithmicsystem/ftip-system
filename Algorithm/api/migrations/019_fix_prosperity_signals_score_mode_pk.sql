DO $$
DECLARE
    v_constraint_name text;
    v_as_of_column text;
BEGIN
    SELECT c.column_name
    INTO v_as_of_column
    FROM information_schema.columns c
    WHERE c.table_schema = current_schema()
      AND c.table_name = 'prosperity_signals_daily'
      AND c.column_name IN ('as_of', 'as_of_date')
    ORDER BY CASE WHEN c.column_name = 'as_of' THEN 1 ELSE 2 END
    LIMIT 1;

    IF v_as_of_column IS NULL THEN
        RETURN;
    END IF;

    SELECT tc.constraint_name
    INTO v_constraint_name
    FROM information_schema.table_constraints tc
    WHERE tc.table_schema = current_schema()
      AND tc.table_name = 'prosperity_signals_daily'
      AND tc.constraint_type = 'PRIMARY KEY'
    LIMIT 1;

    IF v_constraint_name IS NOT NULL THEN
        EXECUTE format(
            'ALTER TABLE %I.%I DROP CONSTRAINT IF EXISTS %I',
            current_schema(),
            'prosperity_signals_daily',
            v_constraint_name
        );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = current_schema()
          AND t.relname = 'prosperity_signals_daily'
          AND c.contype = 'p'
          AND c.conname = 'prosperity_signals_daily_pkey'
    ) THEN
        EXECUTE format(
            'ALTER TABLE %I.%I ADD CONSTRAINT prosperity_signals_daily_pkey PRIMARY KEY (symbol, %I, lookback, score_mode)',
            current_schema(),
            'prosperity_signals_daily',
            v_as_of_column
        );
    END IF;
END $$;
