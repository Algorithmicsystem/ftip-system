# Source Governance Profiles

This document is engineering and product-governance guidance. It is not legal advice and does not replace contract review or counsel.

## Purpose

Phase 13 adds a canonical source-governance layer so the platform can:

- inventory external sources
- classify them by commercialization risk
- enforce cleaner deployment profiles
- explain which domains degrade if a source is removed or blocked
- make buyer diligence easier

## Active environment control

Set `FTIP_SOURCE_PROFILE` to one of:

- `dev_experimental`
  - Broadest technical flexibility. Experimental and fallback feeds are allowed.
- `internal_research`
  - Default profile. Broad source usage is allowed, but commercialization warnings remain active.
- `buyer_demo`
  - Blocks obviously experimental or internal-only sources and keeps review-required sources visible.
- `commercial_candidate`
  - Similar to `buyer_demo`, but intended for a cleaner pre-production stack that still expects commercial-review signoff.
- `restricted_cleanroom`
  - Only the cleanest internal-policy source set is allowed.

## What the profiles do

- Provider fetch paths consult the active source profile before using gated sources.
- Assistant commercialization artifacts summarize:
  - active profile
  - active external sources
  - blocked or disallowed sources
  - gated domains
  - cleanup queue
  - buyer-demo suitability
- Health and audit surfaces show commercialization blockers alongside operational state.

## Practical cleanup guidance

If you want the cleanest buyer-facing stack, start by moving toward:

- paid or contract-backed market data instead of convenience fallbacks
- SEC-backed fundamentals plus explicitly reviewed commercial enrichment
- FRED / World Bank for macro where acceptable under your deployment model
- removal or substitution of RSS, yfinance, Stooq, and other convenience feeds from sensitive profiles

## Expected degradation when sources are gated

- Market-data fallback removal weakens canonical verification and benchmark context first.
- News-source cleanup weakens narrative breadth, crowding, catalyst density, and geopolitical overlays.
- Fundamental-source cleanup weakens secondary enrichment before it weakens the SEC-anchored backbone.
- Cleanroom profiles are intentionally more conservative; some report sections should degrade rather than silently use higher-risk feeds.
