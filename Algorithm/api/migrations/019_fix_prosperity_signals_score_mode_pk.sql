DO $$
DECLARE
    constraint_name text;
    as_of_column text;
BEGIN
    SELECT column_name
    INTO as_of_column
    FROM information_schema.columns
    WHERE table_name = 'prosperity_signals_daily'
      AND column_name IN ('as_of', 'as_of_date')
    ORDER BY CASE WHEN column_name = 'as_of' THEN 1 ELSE 2 END
    LIMIT 1;

    IF as_of_column IS NULL THEN
        RETURN;
    END IF;

    SELECT constraint_name
    INTO constraint_name
    FROM information_schema.table_constraints
    WHERE table_name = 'prosperity_signals_daily'
      AND constraint_type = 'PRIMARY KEY';

    IF constraint_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE prosperity_signals_daily DROP CONSTRAINT %I', constraint_name);
    END IF;

    EXECUTE format(
        'ALTER TABLE prosperity_signals_daily ADD CONSTRAINT prosperity_signals_daily_pkey PRIMARY KEY (symbol, %s, lookback, score_mode)',
        as_of_column
    );
END $$;
