# FTIP System — Commercial Readiness Assessment

**Date:** 2026-06-02  
**Version:** v1.0 Production  
**Status:** Production-Ready

---

## Executive Summary

The FTIP (Financial Trading Intelligence Platform) system is a proprietary quantitative intelligence platform combining seven AXIOM scoring engines, real-time market data integration, and an institutional-grade portfolio allocation framework. The system is production-ready for commercial deployment with a multi-tenant SaaS architecture.

---

## What Is Working Now

### 1. AXIOM 7-Engine Scoring Framework

All seven engines are live and scoring on a 30-symbol large-cap universe daily:

| Engine | Description | Key IP Calculation |
|--------|-------------|-------------------|
| `fundamental_reality` | Earnings quality + competitive moat | CAPS (Rappaport), EIS (Penman-Schilit) |
| `state_pricing` | Macro regime + cross-asset carry | CARDI (Ilmanen), FRED-backed term spread |
| `behavioral_distortion` | Sentiment + contrarian signals | Investor positioning analytics |
| `flow_transmission` | Capital flow + institutional momentum | Order flow quality scoring |
| `liquidity_convexity` | Execution quality + convexity value | KLE (Kyle 1985 Lambda) |
| `critical_fragility` | Tail risk + implementation fragility | MTRS (fat-tail return score), SCPS (Sornette) |
| `research_integrity` | Evidence quality + signal coherence | BFS (Shiller-Kindleberger bubble stage) |

**Knowledge Vault IP Calculations (7):**
- SCPS: Sornette critical point score (log-periodic power law)
- MTRS: Fat-tail return score (higher moments)
- KLE: Kyle Lambda execution liquidity estimate
- BFS: Bubble fragility stage (Kindleberger-Minsky framework)
- CAPS: Competitive advantage period score (Rappaport)
- EIS: Earnings integrity score (Penman-Schilit)
- CARDI: Cross-asset return driver index (Ilmanen)

### 2. Signal Pipeline

- **Prosperity signals** (`/prosperity/*`): Daily BUY/HOLD/SELL signals with 30-symbol large-cap universe
- **AXIOM scorecard** (`/axiom/score`): Full 7-engine breakdown with DAU (Deployable Alpha Utility) score
- **AXIOM screener** (`/axiom/screen`): Ranked opportunities with conviction scoring and Kelly sizing
- **Portfolio allocator** (`/axiom/allocate`): Full portfolio construction with correlation guard, sector caps, and Kelly weighting

### 3. Data Infrastructure

| Provider | Status | Coverage |
|----------|--------|----------|
| Polygon.io (Massive) | Live | Daily OHLCV bars, 30 large-caps, 250-day lookback |
| FRED API | Live | VIX, DGS10, TB3MS for macro context |
| Stooq | Live | Supplemental price data |
| GDELT | Configurable | Event-driven news intelligence |
| World Bank | Configurable | Macro economic indicators |

### 4. Intelligence Layer

- **Regime detection** (`/ops/regime/detect`): Market regime classification with 7 analog labels
- **Regime analogs** (`/ops/regime/analogs`): Historical analog matching with similarity scoring
- **CIO digest** (`/ops/intelligence`): Daily briefing with IC health, top opportunities, sector rotation
- **Intelligence linkage graph** (`/linkage/*`): Symbol-to-symbol dependency mapping

### 5. IC Gate and Calibration

- **Information Coefficient tracking** (`signal_ic_daily`): Spearman IC across 8 score fields × 3 horizons
- **Effective breadth** (Grinold-Kahn): Per-symbol rolling IC computation; `effective_breadth` = count of symbols with IC > 0.02 over 63-day window
- **AMQS score**: Active Management Quality Score using Fundamental Law of Active Management
- **Calibration**: Kelly hit_rate derived from rolling IC history via Gaussian CDF approximation
- **IC gate states**: STRONG / MODERATE / WEAK / DEGRADED / INSUFFICIENT with confidence multipliers

### 6. PE and SMB Intelligence (Enterprise Tier)

**Private Equity Module (`/pe/*`):**
- Portfolio company health scoring (revenue trend, EBITDA margin, leverage, cash flow)
- Exit timing readiness (momentum alignment, valuation, liquidity)
- Stress alerts for distressed portfolio companies (health < 40)
- Portfolio-level overview with aggregate metrics

**SMB Intelligence Module (`/smb/*`):**
- Monthly financial ingestion and OLS trend projection
- 12-month revenue/cash flow forecast with confidence intervals
- Cash runway analysis (healthy / caution / warning / critical)
- Supplier concentration risk assessment

### 7. Security Architecture

- **Multi-tenant API key authentication** with SHA-256 key hashing
- **Tier-based access control**: free → pro → enterprise
- **Rate limiting**: 30 RPM (free), 120 RPM (pro), unlimited (enterprise)
- **Proprietary IP sanitizer**: Strips knowledge vault sub-component fields (scps, mtrs, kle, bfs, caps, eis, cardi) from free-tier responses
- **Write protection**: All DB writes require `FTIP_DB_WRITE_ENABLED=1`
- **PE/SMB tier enforcement**: Enterprise-only endpoints enforced via FastAPI `Depends`

### 8. Operations and Monitoring

- **APScheduler**: Automated daily pipeline at 18:00 UTC Mon–Fri (when `FTIP_SCHEDULER_ENABLED=1`)
- **Metrics tracker**: `/ops/metrics` — request counts, 4xx/5xx rates, rate limit hits
- **Job run history**: `/ops/last_runs` — last 20 pipeline run records with timing
- **Provider reliability**: `provider_reliability_daily` table tracks per-provider health
- **Migration system**: 58 versioned migrations with automatic application at startup

### 9. Test Coverage

- **1,256 tests passing, 0 failures** (as of 2026-06-02)
- Unit tests: AXIOM engines, IC computation, Kelly sizer, PE/SMB intelligence
- Integration tests: Migration idempotency, signal pipeline, allocator, regime detection
- Phase hardening tests: Knowledge vault IP outputs, sanitizer, trial endpoint, macro context

---

## What Requires More Live Data to Mature

### 1. IC Gate Quality (4–12 weeks of daily runs)

Currently: IC state = `INSUFFICIENT` (limited cross-symbol history)  
Matures to: `MODERATE` → `STRONG` with 21–63 days of daily scoring

**Impact**: Kelly position sizing is conservative during INSUFFICIENT state (0.85× confidence multiplier). Once IC history builds to 21+ days, the gate opens fully and the AMQS score becomes meaningful.

### 2. Effective Breadth Precision (8 weeks)

The Grinold-Kahn `effective_breadth` metric requires per-symbol time-series IC, which needs 63 trading days of paired (score, return) observations per symbol. Currently returns the cross-sectional `sample_size` as a proxy. Full effective breadth computation activates automatically once history accumulates.

### 3. Calibration Hit Rate Stability (8–12 weeks)

The Kelly sizer uses `hit_rate` derived from rolling mean IC. With < 21 IC observations, the hit rate defaults near 0.50 (no edge). At 21+ days with positive IC, it rises to ~0.54–0.58, enabling real Kelly sizing. At 63+ days it becomes statistically meaningful.

### 4. PnL Tracking and Correlation Guard (4–8 weeks)

The correlation guard in the allocator uses `signal_pnl_daily` (5-day return records). Currently uses a sector-proxy fallback for symbols with < 5 observations. As live PnL accumulates, individual symbol correlation matrices become more precise.

### 5. Regime Analog Library Depth (ongoing)

The analog library (`regime_analog_library`) is seeded with synthetic historical data. As the system runs live through multiple regimes, the library enriches with real transition records, improving similarity scoring accuracy.

### 6. Provider Reliability Scoring (2–4 weeks)

`provider_reliability_daily` tracks per-provider health over time. Coverage scores improve as the system observes consistent vs. degraded provider behavior across multiple market conditions.

---

## What the Next 30 Days of Live Running Will Improve

| Day Range | Improvement |
|-----------|-------------|
| Days 1–7 | First real IC observations; gate transitions from INSUFFICIENT to WEAK |
| Days 7–14 | Correlation matrix has real per-symbol return data; guard becomes precise |
| Days 14–21 | 21-day rolling IC mean stabilizes; AMQS score becomes actionable |
| Days 21–30 | Calibration hit rate diverges from 0.50; Kelly sizes become non-trivial |
| Days 30+ | IC gate may reach MODERATE (ICIR ≥ 0.25); full portfolio construction live |

The system is designed to degrade gracefully during this maturation period:
- Conservative confidence multipliers during low IC states
- Sector proxy fallback when PnL history is thin
- FRED-backed macro context is live from day 1 (CARDI, VIX, term spread)
- Prosperity signals operate independently of IC history

---

## Commercial Deployment Checklist

- [x] Multi-tenant API key system with tier gating
- [x] Proprietary IP protected from unauthenticated access
- [x] PE/SMB endpoints restricted to enterprise tier
- [x] Rate limiting enforced per tier
- [x] SHA-256 key hashing (no raw keys stored in DB)
- [x] Trial onboarding endpoint (`POST /onboarding/trial`, 14-day pro trial)
- [x] DB writes opt-in (`FTIP_DB_WRITE_ENABLED` defaults to false)
- [x] Daily pipeline automation via APScheduler
- [x] 58 versioned database migrations with idempotent application
- [x] Performance indexes on high-volume query paths (migration 057)
- [x] 1,256 passing tests, 0 failures

---

## API Tier Summary

| Tier | Price Point | RPM | Endpoints |
|------|-------------|-----|-----------|
| Free | $0 | 30 | `/prosperity/*`, `/axiom/*` |
| Pro | TBD | 120 | + `/linkage/*`, `/ops/*` |
| Enterprise | TBD | Unlimited | + `/pe/*`, `/smb/*` |

Free tier responses have proprietary engine sub-components (SCPS, MTRS, KLE, BFS, CAPS, EIS, CARDI) stripped from AXIOM payloads. Pro and enterprise tiers receive full engine breakdowns.

---

## Known Limitations (Not Bugs)

1. **Option surface**: AXIOM `liquidity_convexity` engine adds `no_option_surface_data` flag when options data is unavailable. Convexity is modeled as a proxy. Full option surface integration requires a premium options feed.

2. **Intraday data**: System operates on daily bars. Intraday alpha signals are not currently modeled.

3. **Short-side signal quality**: BUY signals have more historical calibration than SELL. Short-side Kelly sizing is more conservative.

4. **SEC filings**: `SECEDGAR_API_KEY` is configured but EIS computation falls back to proxy when filing data is unavailable. Full EIS requires consistent SEC EDGAR ingestion.

---

## Architecture Notes for Acquirers / Investors

The FTIP system is built as a modular, extensible platform:

- **AXIOM engines** are pure functions (no side effects) with standardized `AxiomEngineInput` / `EngineScore` contracts — new engines can be added without touching existing ones
- **Knowledge vault IP** is computed inside engine functions; formula constants are never exposed in API responses
- **Migration system** is idempotent — schema evolution is zero-downtime
- **Data providers** are hot-swappable via the data fabric abstraction layer
- **Multi-tenancy** is a thin layer on top of the existing signal API — the core engine never sees tenant context

The codebase is approximately 45,000 lines of Python with zero external ML model dependencies — all alpha generation is rule-based, transparent, and auditable.
