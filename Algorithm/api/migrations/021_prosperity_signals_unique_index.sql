DO $$
DECLARE
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

    EXECUTE format(
        'CREATE UNIQUE INDEX IF NOT EXISTS idx_prosperity_signals_daily_symbol_asof_lookback ON %I.%I (symbol, %I, lookback)',
        current_schema(),
        'prosperity_signals_daily',
        v_as_of_column
    );
END $$;
