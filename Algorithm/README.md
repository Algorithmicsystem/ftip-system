# FTIP System Notes

## DB smoke tests

Set the environment for database access (for local testing you can point `DATABASE_URL` at a Postgres instance) and enable DB endpoints:

```bash
export FTIP_DB_ENABLED=1
export DATABASE_URL=postgresql://user:pass@host:5432/ftip
```

Then run a few smoke checks against a running server (default port 8000):

```bash
curl http://localhost:8000/db/health
curl -X POST http://localhost:8000/db/universe/load_default
curl -X POST http://localhost:8000/db/run_snapshot \
  -H "Content-Type: application/json" \
  -d '{"as_of":"2024-01-31","lookback":252,"active_only":true,"limit":25}'
curl -X POST http://localhost:8000/db/save_signal \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","as_of":"2024-01-31","lookback":252}'
```

Each call should return HTTP 200 with JSON payloads when the database is reachable.

## Strategy Graph Engine v1

The Strategy Graph Engine computes multi-strategy signals, ensembles them, and records audit metadata.

Run a daily computation across symbols:

```bash
curl -X POST http://localhost:8000/prosperity/strategy_graph/run \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL","MSFT"],"from_date":"2024-01-01","to_date":"2024-01-31","as_of_date":"2024-01-31","lookback":252}'
```

Fetch the latest ensemble or per-strategy breakdown:

```bash
curl "http://localhost:8000/prosperity/strategy_graph/latest/ensemble?symbol=AAPL&lookback=252"
curl "http://localhost:8000/prosperity/strategy_graph/latest/strategies?symbol=AAPL&lookback=252"
```

To smoke test locally, run:

```bash
bash tools/smoke_strategy_graph.sh
```

## Copy/paste safe zsh commands

When using zsh, enable `setopt interactivecomments` before copy/pasting any of the sample commands so inline `#` comments are ignored correctly:

```bash
setopt interactivecomments
```

If you see `command not found: #`, re-run the command above and try again.

## Prosperity DB v1

Prosperity DB adds a durable research warehouse (Postgres) for caching market data, features, and signals. Enable it with:

```bash
export FTIP_DB_ENABLED=1
export DATABASE_URL=postgresql://user:pass@host:5432/ftip
export FTIP_DB_WRITE_ENABLED=1
export FTIP_DB_READ_ENABLED=1
export FTIP_MIGRATIONS_AUTO=1
```

Example calls:

```bash
curl http://localhost:8000/prosperity/health
curl -X POST http://localhost:8000/prosperity/snapshot/run \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL","MSFT","GOOGL","AMZN","NVDA"],"from_date":"2024-01-01","to_date":"2024-01-31","as_of_date":"2024-01-31","lookback":252}'
curl "http://localhost:8000/prosperity/latest/signal?symbol=AAPL&lookback=252"
```

To smoke test the snapshot->latest round trip on a fresh database, run:

```bash
curl -X POST http://localhost:8000/prosperity/bootstrap
curl -X POST http://localhost:8000/prosperity/snapshot/run \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL"],"from_date":"2024-01-01","to_date":"2024-01-05","as_of_date":"2024-01-05","lookback":252}'
curl "http://localhost:8000/prosperity/latest/signal?symbol=AAPL&lookback=252"
```

Each call should return HTTP 200 with the latest signal containing `score_mode`.

To run a local end-to-end smoke check that exercises the prosperity endpoints:

```bash
bash tools/smoke_prosperity.sh
```

## Assistant / Narrator (Phase 5)

Environment variables:

```bash
export FTIP_LLM_ENABLED=1          # enable the assistant endpoints
export OPENAI_API_KEY=sk-...       # required when LLM is enabled
export FTIP_MIGRATIONS_AUTO=1      # run migrations automatically at startup
```

Health check:

```bash
curl http://localhost:8000/assistant/health
```

Chat example (persists a session when FTIP_DB_ENABLED=1):

```bash
curl -X POST http://localhost:8000/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How was the signal computed for AAPL?"}'
```

Explain a signal (reuses the internal /signal logic):

```bash
curl -X POST http://localhost:8000/assistant/explain/signal \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","as_of":"2024-01-31","lookback":252}'
```

The narrator is safety-first and should never provide financial advice. It explains what the model computed, which thresholds were used, and what calibration metadata was present.

## FTIP Narrator (OpenAI)

Environment variables (LLM feature-flagged by default):

```bash
export FTIP_LLM_ENABLED=1               # enable the narrator endpoints (default is 0)
export OPENAI_API_KEY=sk-...            # or set OpenAI_ftip-system
export FTIP_LLM_MODEL=gpt-4o-mini       # optional override
export FTIP_LLM_MAX_TOKENS=700          # optional override
export FTIP_LLM_TEMPERATURE=0.2         # optional override
```

Health check:

```bash
curl "http://localhost:8000/prosperity/narrator/health"
```

Explain a symbol using DB-grounded context (strategy graph ensemble + strategies + features + latest signal):

```bash
curl "http://localhost:8000/prosperity/narrator/explain?symbol=AAPL&as_of_date=2024-12-31&lookback=252"
```

Ask system or comparative symbol questions (returns 404 if DB context is missing):

```bash
curl -X POST http://localhost:8000/prosperity/narrator/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Why is AAPL a BUY?","symbols":["AAPL","MSFT"],"as_of_date":"2024-12-31","lookback":252}'
```

Recommended flow: bootstrap the DB, run `/prosperity/strategy_graph/run` to populate ensembles, then call `/prosperity/narrator/explain` for narration. The narrator refuses to answer when DB context is missing and always returns a trace ID for auditing.
