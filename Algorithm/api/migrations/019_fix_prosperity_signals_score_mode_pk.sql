DO $$
DECLARE
    v_existing_pk text;
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

    SELECT c.conname
    INTO v_existing_pk
    FROM pg_constraint c
    JOIN pg_class t ON t.oid = c.conrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = current_schema()
      AND t.relname = 'prosperity_signals_daily'
      AND c.contype = 'p'
    LIMIT 1;

    IF v_existing_pk IS NOT NULL THEN
        EXECUTE format(
            'ALTER TABLE %I.%I DROP CONSTRAINT IF EXISTS %I',
            current_schema(),
            'prosperity_signals_daily',
            v_existing_pk
        );
    END IF;

    EXECUTE format(
        'ALTER TABLE %I.%I ADD CONSTRAINT prosperity_signals_daily_pkey PRIMARY KEY (symbol, %I, lookback, score_mode)',
        current_schema(),
        'prosperity_signals_daily',
        v_as_of_column
    );

    EXECUTE format(
        'CREATE UNIQUE INDEX IF NOT EXISTS idx_prosperity_signals_daily_symbol_asof_lookback ON %I.%I (symbol, %I, lookback)',
        current_schema(),
        'prosperity_signals_daily',
        v_as_of_column
    );
END $$;
