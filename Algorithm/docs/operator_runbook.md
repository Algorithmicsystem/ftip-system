# FTIP Operator Runbook

This document is engineering and operating guidance for disciplined use of the platform. It is not trading advice, legal advice, or a substitute for separate risk, compliance, or legal review.

## Daily Workflow

1. Open `/app` and start on `Dashboard`.
2. Review:
   - changed signals
   - today’s candidate triage
   - new warnings or downgrades
   - active trust and deployment state
3. Check `Trust / Health` before relying on any higher-trust interpretation:
   - system health
   - provider degradation
   - shadow status
   - active alerts
   - source profile
4. Record shadow interpretation first when the platform remains in `research_only`, `paper_shadow`, or `shadow_only` operating states.
5. Do not treat a strong signal as actionable if:
   - deployment permission is still `paper_shadow_only` or `analysis_only`
   - pause or downgrade conditions are active
   - event, liquidity, or stress suppressors are elevated

## Weekly Workflow

1. Review the weekly operating summary.
2. Check:
   - net-of-friction validation behavior
   - confidence and readiness calibration
   - suppression usefulness
   - strongest and weakest weekly cohorts
   - drift, provider, and operational incidents
3. Decide whether trust should:
   - stay unchanged
   - tighten
   - remain shadow-only
   - move into formal promotion review
4. Update operator attention items and note any repeated failure modes.

## Monthly Workflow

1. Review the monthly refinement summary.
2. Focus on:
   - strongest and weakest setup families
   - persistent failure modes
   - research priority queue
   - open commercialization cleanup
   - trust promotion and demotion candidates
3. Turn observations into governed follow-up work, not ad hoc rule changes.
4. Keep threshold, gating, and weighting changes versioned and reviewable.

## Shadow Mode Discipline

- Shadow mode is for evidence collection, not implied live trust.
- Preserve:
  - what the system said
  - what it allowed or blocked
  - what later happened
  - whether the trust gate helped or hurt
- Promotion out of shadow should require:
  - enough matured outcomes
  - stable walk-forward behavior
  - acceptable operational health
  - acceptable drift and calibration status

## When Drift or Degradation Appears

1. Check whether the issue is:
   - data freshness
   - fallback overuse
   - provider degradation
   - calibration drift
   - ranking drift
   - regime shift
2. If downgrade or pause conditions are active:
   - stop using higher-trust deployment support
   - stay in shadow-first interpretation
   - review recovery criteria before restoring trust
3. Record the incident and add it to the next weekly or monthly review queue.

## Buyer / Commercial Use Note

Commercial-readiness and source-governance summaries are separate from alpha quality. Before external buyer deployment or commercialization:

1. Review `FTIP_SOURCE_PROFILE`.
2. Check commercialization blockers and gated domains.
3. Replace or disable higher-risk sources where required.
4. Confirm the intended stack is suitable for the target profile.
