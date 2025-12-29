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
