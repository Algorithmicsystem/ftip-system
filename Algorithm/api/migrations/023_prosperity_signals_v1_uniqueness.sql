DO $$
DECLARE
    v_as_of_column text;
    v_pk_name text;
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

    EXECUTE format(
        $sql$
        DELETE FROM %I.%I t
        USING (
            SELECT ctid,
                   row_number() OVER (
                       PARTITION BY symbol, %I, lookback
                       ORDER BY
                           COALESCE(updated_at, created_at) DESC NULLS LAST,
                           created_at DESC NULLS LAST,
                           CASE
                               WHEN lower(COALESCE(score_mode, '')) = 'stacked' THEN 3
                               WHEN lower(COALESCE(score_mode, '')) = 'base' THEN 2
                               WHEN lower(COALESCE(score_mode, '')) = 'single' THEN 1
                               ELSE 0
                           END DESC,
                           ctid DESC
                   ) AS rn
            FROM %I.%I
        ) ranked
        WHERE t.ctid = ranked.ctid
          AND ranked.rn > 1
        $sql$,
        current_schema(),
        'prosperity_signals_daily',
        v_as_of_column,
        current_schema(),
        'prosperity_signals_daily'
    );

    SELECT c.conname
    INTO v_pk_name
    FROM pg_constraint c
    JOIN pg_class t ON t.oid = c.conrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = current_schema()
      AND t.relname = 'prosperity_signals_daily'
      AND c.contype = 'p'
    LIMIT 1;

    IF v_pk_name IS NOT NULL THEN
        EXECUTE format(
            'ALTER TABLE %I.%I DROP CONSTRAINT IF EXISTS %I',
            current_schema(),
            'prosperity_signals_daily',
            v_pk_name
        );
    END IF;

    EXECUTE format(
        'ALTER TABLE %I.%I ADD CONSTRAINT prosperity_signals_daily_pkey PRIMARY KEY (symbol, %I, lookback)',
        current_schema(),
        'prosperity_signals_daily',
        v_as_of_column
    );

    EXECUTE format(
        'DROP INDEX IF EXISTS %I.%I',
        current_schema(),
        'idx_prosperity_signals_daily_symbol_asof_lookback'
    );

    EXECUTE format(
        'CREATE UNIQUE INDEX IF NOT EXISTS idx_prosperity_signals_daily_symbol_asof_lookback ON %I.%I (symbol, %I, lookback)',
        current_schema(),
        'prosperity_signals_daily',
        v_as_of_column
    );
END $$;
