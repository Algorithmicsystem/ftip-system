# Final Release-Readiness Checklist (Official V1 Product Path)

## Official V1 Scope

V1 includes only this product path:

1. `POST /prosperity/bootstrap`
2. `POST /prosperity/snapshot/run`
3. `GET /prosperity/latest/signal`
4. `GET /prosperity/latest/features`
5. Optional scheduler wrapper over the same snapshot path: `POST /jobs/prosperity/daily-snapshot`

Plain-language v1 promise:
- Operators can initialize schema, run a DB-backed snapshot for a symbol universe, and read latest persisted signal/features for a symbol+lookback.
- Snapshot execution is fail-closed if DB read/write are not enabled.
- Latest reads are DB-backed only and return 404 when data is absent (no synthetic fallback).

Anything outside the path above is not part of v1 release claims.

## Must Pass Before Release

All items below must be true simultaneously for v1 to be releasable.

1. **DB-gated execution contract is enforced for the full write path**
   - `POST /prosperity/snapshot/run` must return 503 when DB is disabled, DB writes are disabled, or DB reads are disabled.
   - `POST /jobs/prosperity/daily-snapshot` must enforce the same DB read+write requirement before run execution.

2. **Signal uniqueness contract is canonical and migration-safe**
   - `prosperity_signals_daily` canonical uniqueness must remain exactly `(symbol, as_of, lookback)`.
   - Snapshot persistence upserts must target the same canonical key.
   - Schema bootstrap/migration must converge existing environments to that key deterministically.

3. **Official v1 round-trip is proven in DB-backed integration tests**
   - Evidence test path must cover: bootstrap -> snapshot (or job wrapper) -> latest signal/features readback.
   - Tests must assert persisted rows exist in `prosperity_signals_daily` and `prosperity_features_daily`.

4. **Retention covers actual v1 strategy write tables (when strategy graph is enabled by wrapper)**
   - Retention cleanup must target `prosperity_strategy_signals_daily` and `prosperity_ensemble_signals_daily` in addition to core v1 tables.

5. **Release configuration is explicit and reproducible**
   - Required env flags and secrets for v1 runtime are documented and set in target environment (`FTIP_DB_ENABLED=1`, `FTIP_DB_WRITE_ENABLED=1`, `FTIP_DB_READ_ENABLED=1`, DB URL, API key requirements, admin token policy).
   - `/prosperity/bootstrap` is executed successfully in target environment before first snapshot run.

## Strongly Recommended Before Release

1. **Lock in a release runbook for only the official v1 path**
   - Include exact call order, expected success payload shape, and required failure handling (`404` on missing latest rows, `503` on DB gating).

2. **Add one production-like smoke script for the official path only**
   - Run bootstrap -> snapshot/job -> latest readback against deployment environment and store artifacts/logs.

3. **Operational guardrails for scheduler usage**
   - Confirm lock/TTL behavior and stale-lock cleanup are monitored for `prosperity_daily_snapshot` runs.

4. **Access-control tightening for bootstrap**
   - Ensure `PROSPERITY_ADMIN_TOKEN` is set in release environments so schema bootstrap is explicitly protected.

## Acceptable Known Limitations for V1

These are acceptable in v1 if explicitly documented:

1. **Partial snapshot status is allowed**
   - Snapshot/job may return `status="partial"` when some symbols fail (for example, missing/insufficient bars), while still persisting successful symbols.

2. **Latest endpoints are read-only views over persisted data**
   - `GET /prosperity/latest/signal` and `GET /prosperity/latest/features` return 404 when no row exists; they do not auto-compute.

3. **External data/provider dependency remains a runtime risk**
   - Symbol-level failures due to upstream data availability are treated as expected operational behavior, not v1 correctness failure.

4. **Wrapper date behavior is opinionated**
   - `POST /jobs/prosperity/daily-snapshot` defaults to `as_of_date = (UTC today - 1 day)`; this is acceptable if documented for operators.

## Non-Core / Legacy Areas to Hide or De-Emphasize

To avoid scope confusion in launch messaging, hide from v1 product claims and de-emphasize in external docs:

1. **Legacy `/db/*` surfaces in `api.main`**
   - Examples: `/db/save_signal`, `/db/save_signals`, `/db/run_snapshot`, `/db/universe/*`.
   - These are not the official v1 product path and should not be presented as primary interfaces.

2. **Legacy direct signal compute surfaces**
   - `/signal` and `/signals` compute paths in `api.main` are outside official v1 persistence/readback contract.

3. **Other feature domains not required for v1 claim**
   - Assistant, LLM, narrator, friction, backtest, generic data/provider routes, and non-v1 job families should be excluded from v1 readiness claims.

4. **Optional v1 wrapper should be framed correctly**
   - `/jobs/prosperity/daily-snapshot` is an operational wrapper around the same core snapshot path; it is not a separate product capability.

## Final Release Verdict

**Ready for MVP release with caveats**.

## Short Rationale

The official v1 path now has the core hardening expected at final gate: fail-closed DB behavior for snapshot execution, canonical signal uniqueness on `(symbol, as_of, lookback)`, retention aligned to real write tables, and DB-backed integration coverage for bootstrap -> snapshot/job -> latest readback.

Remaining risk is primarily operational (provider availability and partial symbol outcomes), plus product-surface clarity because numerous legacy/non-core endpoints still exist in the same service. Those risks are acceptable for MVP if explicitly documented and scoped out of v1 claims.
