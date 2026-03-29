# V1 Product Path Audit (2026-03-29)

Scope audited: official v1 path only.

1. `POST /prosperity/bootstrap`
2. `POST /prosperity/snapshot/run` OR `POST /jobs/prosperity/daily-snapshot`
3. `GET /prosperity/latest/signal`
4. `GET /prosperity/latest/features`
5. Optional strategy/coverage reads only where coupled to this path.

## Key findings

- The official read endpoints (`/prosperity/latest/signal`, `/prosperity/latest/features`) are strictly DB-read endpoints and do not provide synthetic fallback responses.
- The snapshot execution path can run in a degraded non-persistent mode when DB flags are disabled (`FTIP_DB_ENABLED=0`), returning computed results but writing nothing.
- The daily job wrapper enforces DB read+write and lock-backed run tracking, and it calls the same `snapshot_run` function.
- Core persistence currently depends on a schema/migration contract: writes use `ON CONFLICT(symbol, as_of, lookback)` while table PK includes `score_mode`; the unique index migration must exist.
- Backtest uses a no-op correlation guard (`correlation_guard_stub`) and synthetic in-memory bars fallback, but this is outside the official v1 endpoint sequence.

## Release interpretation (path-only)

- Path is usable as MVP with caveats, not fully production-like under all runtime configurations, because the same snapshot endpoint supports non-DB degraded execution and because confidence tests are heavily monkeypatched.
