# Strict V1 Release-Blocker Implementation Plan (Official V1 Path Only)

## Scope
Official v1 path only:
1. `POST /prosperity/bootstrap`
2. `POST /prosperity/snapshot/run` OR `POST /jobs/prosperity/daily-snapshot`
3. `GET /prosperity/latest/signal`
4. `GET /prosperity/latest/features`

This plan only covers the four already-identified must-fix items.

## Release Blockers

### RB-1) Fail-closed DB gating for `/prosperity/snapshot/run`
**Current issue**
- `snapshot_run` currently computes `db_enabled_for_run = db.db_enabled() and db.db_write_enabled() and db.db_read_enabled()` and only calls `_require_db_enabled(write=True, read=True)` when this is true.
- When any DB flag is disabled, route still runs compute path and returns payload with no persistence.

**Minimum safe change**
- In `api/prosperity/routes.py::snapshot_run`, remove conditional gating and call `_require_db_enabled(write=True, read=True)` unconditionally at route start.
- Keep rest of flow unchanged.

**Correct final behavior**
- `/prosperity/snapshot/run` returns HTTP 503 when DB is disabled, write-disabled, or read-disabled.
- No partial/degraded in-memory snapshot mode for official v1 endpoint.

**If left unfixed**
- Official endpoint may report "ok/partial" while writing nothing.
- `/prosperity/latest/*` may then return 404 despite prior successful snapshot response, violating operator expectations.

### RB-2) Unify signal persistence uniqueness contract (`score_mode` vs conflict target)
**Current issue**
- Persistence SQL writes into `prosperity_signals_daily` with `ON CONFLICT(symbol, as_of, lookback)` and updates `score_mode`.
- Table primary key includes `score_mode` (`(symbol, as_of, lookback, score_mode)`) in migration schema.
- Current behavior relies on a separate unique index migration on `(symbol, as_of, lookback)`, creating a fragile dual-contract state.

**Minimum safe change**
- Decide one canonical contract for v1 and align schema + write SQL + read SQL + tests to that contract.
- Recommended shortest safe v1 contract: one row per `(symbol, as_of, lookback)`.
  - Update migration/ensure-schema path to make `(symbol, as_of, lookback)` the only unique conflict target.
  - Keep `score_mode` as a data column (not key).
  - Ensure `_persist_symbol_outputs` conflict target remains `(symbol, as_of, lookback)`.
- Add a dedicated migration that:
  - resolves duplicates deterministically,
  - drops PK/indexes that include `score_mode`,
  - creates/keeps unique index (or PK) on `(symbol, as_of, lookback)`.

**Correct final behavior**
- Upsert path and DB constraints are structurally identical; no dependency on "extra" index for correctness.
- Re-running snapshot for same symbol/date/lookback updates exactly one row deterministically.

**If left unfixed**
- Environments missing the unique index migration can fail inserts/upserts.
- Different environments may behave differently depending on migration history.

### RB-3) True integration test path (real DB) for bootstrap -> snapshot/job -> latest readback
**Current issue**
- Existing tests for these routes are heavily monkeypatched and do not prove production DB contract end-to-end.

**Minimum safe change**
- Add new integration tests (guarded by `DATABASE_URL`) that run against real DB with minimal patching only for external data providers:
  1. `POST /prosperity/bootstrap`
  2. `POST /prosperity/snapshot/run` (or `/jobs/prosperity/daily-snapshot`)
  3. `GET /prosperity/latest/signal`
  4. `GET /prosperity/latest/features`
- Acceptable minimal stubs: deterministic bar fetch + deterministic signal compute, while preserving DB writes/reads and route stack.
- Assert persisted rows exist in `prosperity_features_daily` and `prosperity_signals_daily` with expected keys.

**Correct final behavior**
- CI/dev environments with DB enabled can prove official v1 path produces and reads real persisted rows.

**If left unfixed**
- Regressions in SQL/migrations/route wiring can ship undetected.
- v1 reliability claims remain unproven.

## Strongly Recommended Before Release

### SR-1) Retention target alignment for strategy graph reliability claims
**Current issue**
- Retention cleanup currently targets `prosperity_strategy_graph_daily`, but writes go to:
  - `prosperity_strategy_signals_daily`
  - `prosperity_ensemble_signals_daily`
- If strategy graph is part of reliability claims, retention policy is not applied to actual write tables.

**Minimum safe change**
- In `api/jobs/prosperity.py::cleanup_retention`, replace `prosperity_strategy_graph_daily` with both real strategy graph tables.
- Add integration/unit test for retention cleanup table list and delete behavior.

**Correct final behavior**
- Daily snapshot retention cleanup applies to all tables actually written by snapshot when `compute_strategy_graph=True`.

**If left unfixed**
- Strategy graph tables can grow indefinitely despite retention config.
- Operational mismatch between documented behavior and actual storage lifecycle.

## Can Defer Until After V1

- None from the four must-fix items should be deferred if "official reliability" is a release claim.
- If strictly minimizing scope and strategy graph is **not** in v1 claims, SR-1 can be deferred with explicit documentation that retention currently covers only features/signals.

## File-by-File Change Map

### `api/prosperity/routes.py`
- `snapshot_run(...)`: enforce unconditional `_require_db_enabled(write=True, read=True)`.
- `_persist_symbol_outputs(...)`: keep conflict target consistent with canonical uniqueness decision.

### `api/migrations/*.sql` (new migration)
- Add migration to normalize `prosperity_signals_daily` uniqueness contract.
- Include deterministic dedupe step before unique constraint creation.

### `api/migrations/__init__.py`
- Ensure schema helper creates/maintains the same uniqueness contract selected for `prosperity_signals_daily`.

### `api/prosperity/query.py`
- Verify/adjust `latest_signal(...)` query ordering/selection if uniqueness semantics change.
- Ensure behavior remains deterministic for "latest" readback.

### `api/jobs/prosperity.py`
- `cleanup_retention(...)`: align retention table list to actual strategy graph write tables.

### `tests/test_prosperity_endpoints.py` (or new dedicated integration file)
- Add non-monkeypatched (DB-real) integration tests for official v1 path.
- Keep unit tests that monkeypatch internals, but do not rely on them as release evidence.

### `tests/test_jobs_phase8.py` (or new dedicated integration file)
- Add daily snapshot job integration coverage with lock + write + latest readback assertions against real DB.

## Recommended Fix Order

1. **RB-2 (uniqueness contract) first**
   - Prevents writing against unstable schema semantics.
2. **RB-1 (fail-closed snapshot route)**
   - Enforces official runtime contract immediately.
3. **SR-1 (retention alignment)**
   - Quick consistency fix once write tables are finalized.
4. **RB-3 (integration tests)**
   - Lock in behavior with DB-backed regression tests after schema/route changes settle.

## Expected Final V1 Behavior

- `POST /prosperity/bootstrap` always prepares a schema consistent with snapshot upsert contracts.
- `POST /prosperity/snapshot/run` fails closed (503) unless DB read+write are enabled.
- `POST /jobs/prosperity/daily-snapshot` remains DB-gated and writes lock/run metadata plus snapshot outputs.
- `GET /prosperity/latest/signal` and `GET /prosperity/latest/features` return persisted DB-backed latest rows with deterministic uniqueness behavior.
- If strategy graph is enabled in the job path, retention cleanup applies to the actual strategy graph persistence tables.

## Remaining Risks After These Fixes

- External provider instability can still cause per-symbol partial results (expected), but path integrity remains DB-backed.
- Existing non-v1 endpoints may still have different fallback/quality guarantees.
- Migration rollout risk remains if historical duplicates in `prosperity_signals_daily` are not cleaned deterministically during deploy.
