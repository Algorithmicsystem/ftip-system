# FTIP Runbook

> Official v1 (Prosperity, strict DB-backed) now has a focused runbook at `docs/official_v1_runbook.md`. Use that as the primary path for developer/operator v1 work and signoff.

## Milestone E: Assistant Orchestrator + UI

### Endpoints

- `POST /assistant/analyze`
  - Input:
    ```json
    {
      "symbol": "NVDA",
      "horizon": "swing",
      "risk_mode": "balanced",
      "scenario_mode": "base",
      "analysis_depth": "standard",
      "refresh_mode": "refresh_stale",
      "market_regime": "auto"
    }
    ```
  - Response includes canonical assistant artifacts:
    - `analysis_job`
    - `freshness_summary`
    - `data_bundle`
    - `feature_factor_bundle`
    - `strategy`
    - presentation sections such as `signal_summary`, `technical_analysis`, `fundamental_analysis`, `statistical_analysis`, `sentiment_analysis`, `macro_geopolitical_analysis`, `strategy_view`, `risks_weaknesses_invalidators`, and `evidence_provenance`
- `POST /assistant/chat`
  - Input:
    ```json
    {
      "session_id": "uuid",
      "message": "What drove the HOLD signal?",
      "context": {
        "active_analysis": {
          "report_id": "uuid",
          "symbol": "NVDA",
          "as_of_date": "2026-04-05",
          "horizon": "swing",
          "risk_mode": "balanced"
        }
      }
    }
    ```
  - Chat loads the active report first and answers as the narrator of the stored analysis artifact instead of as a generic assistant.
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

Visit `/app` for the combined console.

- The official Prosperity v1 controls remain the primary surface and are unchanged.
- The non-v1 assistant area now acts as an advanced research console with:
  - Dashboard
  - Analyze
  - Signal
  - Strategy
  - Evidence
  - Chat / Narrator
  - Advanced / Research
  - System Health
- The active analysis banner, structured report sections, why-this-signal drilldown, and grounded chat all attach to the same persisted assistant report artifact.

### Environment variables

- `FTIP_DB_ENABLED=1` (required for data access)
- `FTIP_DB_READ_ENABLED=1`
- `FTIP_DB_WRITE_ENABLED=1`
- `FTIP_LLM_ENABLED=1` (required for narration)
- `OPENAI_API_KEY=...` (required for narration)
- `FTIP_DATA_FABRIC_ENABLED=1` (enables live multi-source enrichment on assistant analysis)
- `MASSIVE_API_KEY` or `POLYGON_API_KEY` (primary market data path)
- `FINNHUB_API_KEY`
- `FRED_API_KEY`
- `ALPHAVANTAGE_API_KEY`
- `GNEWS_API_KEY`
- `NEWS_API_KEY`
- `SEC_USER_AGENT`
- `GDELT_ENABLED=1`
- `WORLD_BANK_ENABLED=1`
- `STOOQ_ENABLED=1`

### Assistant data fabric scope

The assistant analysis bundle now layers external enrichment on top of the existing DB-backed signal/feature pipeline:

- Market verification and benchmark context via Massive/Polygon, Alpha Vantage, and Stooq fallback.
- Filing-aware overlays via SEC EDGAR, Finnhub, and Alpha Vantage overview/basic metrics.
- News and narrative aggregation via GNews, NewsAPI, Finnhub news, and GDELT.
- Macro normalization via FRED and World Bank.
- Geopolitical/event tagging via GDELT-backed headline buckets.

All enriched domains carry source/freshness/coverage/provider-status metadata and degrade gracefully when a provider is unavailable or data is thin.

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
