# FTIP Runbook

## Milestone E: Assistant Orchestrator + UI

### Endpoints

- `POST /assistant/analyze`
  - Input:
    ```json
    {"symbol":"NVDA","horizon":"swing","risk_mode":"balanced"}
    ```
  - Response includes `symbol`, `as_of_date`, `signal`, `key_features`, `quality`, `evidence`.
- `POST /assistant/top-picks`
  - Input:
    ```json
    {"universe":"sp500","horizon":"swing","risk_mode":"balanced","limit":10}
    ```
- `POST /assistant/narrate`
  - Input:
    ```json
    {"payload":{...},"user_message":"Why this signal?"}
    ```

### UI

Visit `/app` to use the minimal UI. The signal card, confidence gauge, and reason codes are always visible in the pinned chat panel.

### Environment variables

- `FTIP_DB_ENABLED=1` (required for data access)
- `FTIP_DB_READ_ENABLED=1`
- `FTIP_DB_WRITE_ENABLED=1`
- `FTIP_LLM_ENABLED=1` (required for narration)
- `OPENAI_API_KEY=...` (required for narration)

## Milestone F: Backtesting + Scorecards

### Endpoints

- `POST /backtest/run`
  - Input:
    ```json
    {
      "symbol": "NVDA",
      "universe": "sp500",
      "date_start": "2024-01-01",
      "date_end": "2024-03-01",
      "horizon": "swing",
      "risk_mode": "balanced",
      "signal_version_hash": "auto",
      "cost_model": {"fee_bps": 1, "slippage_bps": 5}
    }
    ```
- `GET /backtest/results?run_id=...`
- `GET /backtest/equity-curve?run_id=...`

## SQL Issue Fix Note

**What broke:** Inserts into `prosperity_signals_daily` used `ON CONFLICT (symbol, as_of, lookback, score_mode)` but the primary key omitted `score_mode`, so Postgres raised `no unique or exclusion constraint matching the ON CONFLICT specification`.

**Why:** The migration that ensured the primary key for `prosperity_signals_daily` only covered `(symbol, as_of, lookback)`.

**Fix:** Updated the migration to include `score_mode` in the primary key and added a targeted migration to repair existing databases.

**Preventing recurrence:** Add regression tests that verify primary key/unique constraints include every column referenced in `ON CONFLICT` clauses.
