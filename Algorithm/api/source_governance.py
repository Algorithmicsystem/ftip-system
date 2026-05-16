from __future__ import annotations

import importlib.util
from typing import Any, Dict, Iterable, List, Optional, Sequence

from api import config


GOVERNANCE_DEV_TEST_ONLY = "dev_test_only"
GOVERNANCE_INTERNAL_RESEARCH_ONLY = "internal_research_only"
GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW = "requires_commercial_review"
GOVERNANCE_COMMERCIAL_CANDIDATE = "commercial_candidate"
GOVERNANCE_UNKNOWN_REVIEW_REQUIRED = "unknown_review_required"

PROFILE_DEV_EXPERIMENTAL = "dev_experimental"
PROFILE_INTERNAL_RESEARCH = "internal_research"
PROFILE_BUYER_DEMO = "buyer_demo"
PROFILE_COMMERCIAL_CANDIDATE = "commercial_candidate"
PROFILE_RESTRICTED_CLEANROOM = "restricted_cleanroom"

SOURCE_GOVERNANCE_VERSION = "phase13_source_governance_v1"


_PROFILE_RULES: Dict[str, Dict[str, Any]] = {
    PROFILE_DEV_EXPERIMENTAL: {
        "allowed_statuses": {
            GOVERNANCE_DEV_TEST_ONLY,
            GOVERNANCE_INTERNAL_RESEARCH_ONLY,
            GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW,
            GOVERNANCE_COMMERCIAL_CANDIDATE,
            GOVERNANCE_UNKNOWN_REVIEW_REQUIRED,
        },
        "summary": "Development profile: all sources are technically allowed, including experimental and higher-risk feeds.",
    },
    PROFILE_INTERNAL_RESEARCH: {
        "allowed_statuses": {
            GOVERNANCE_DEV_TEST_ONLY,
            GOVERNANCE_INTERNAL_RESEARCH_ONLY,
            GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW,
            GOVERNANCE_COMMERCIAL_CANDIDATE,
            GOVERNANCE_UNKNOWN_REVIEW_REQUIRED,
        },
        "summary": "Internal research profile: broad source usage is allowed, but commercialization warnings remain active.",
    },
    PROFILE_BUYER_DEMO: {
        "allowed_statuses": {
            GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW,
            GOVERNANCE_COMMERCIAL_CANDIDATE,
        },
        "summary": "Buyer-demo profile: blocks clearly experimental or internal-only feeds and keeps review-required sources visible.",
    },
    PROFILE_COMMERCIAL_CANDIDATE: {
        "allowed_statuses": {
            GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW,
            GOVERNANCE_COMMERCIAL_CANDIDATE,
        },
        "summary": "Commercial-candidate profile: excludes experimental feeds and expects contract or legal review for conditional sources.",
    },
    PROFILE_RESTRICTED_CLEANROOM: {
        "allowed_statuses": {
            GOVERNANCE_COMMERCIAL_CANDIDATE,
        },
        "summary": "Restricted cleanroom profile: only the cleanest internal-policy source set is allowed.",
    },
}


_SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "massive_polygon": {
        "display_name": "Massive / Polygon",
        "provider_type": "market_data",
        "domains_used_for": [
            "canonical_market_bars",
            "reference_bars",
            "prosperity_snapshot",
        ],
        "required_env_keys": ["MASSIVE_API_KEY", "POLYGON_API_KEY"],
        "governance_status": GOVERNANCE_COMMERCIAL_CANDIDATE,
        "fallback_role": "primary_candidate",
        "criticality": "high",
        "notes": [
            "Primary paid-style market-data candidate for canonical daily bars and benchmark context.",
            "Commercial use still requires provider contract review and deployment-specific rights confirmation.",
        ],
    },
    "alphavantage": {
        "display_name": "Alpha Vantage",
        "provider_type": "market_and_fundamentals",
        "domains_used_for": [
            "daily_bars_fallback",
            "quarterly_fundamentals",
            "company_overview",
            "reference_bars",
        ],
        "required_env_keys": ["ALPHAVANTAGE_API_KEY"],
        "governance_status": GOVERNANCE_INTERNAL_RESEARCH_ONLY,
        "fallback_role": "secondary_enrichment",
        "criticality": "medium",
        "notes": [
            "Used as a secondary enrichment and fallback source for bars and fundamentals.",
            "Treat as an internal-research or review-gated dependency until a clean commercial entitlement is confirmed.",
        ],
    },
    "finnhub": {
        "display_name": "Finnhub",
        "provider_type": "fundamentals_and_news",
        "domains_used_for": [
            "company_profile",
            "basic_financials",
            "company_news",
            "provider_health",
        ],
        "required_env_keys": ["FINNHUB_API_KEY"],
        "governance_status": GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW,
        "fallback_role": "secondary_enrichment",
        "criticality": "medium",
        "notes": [
            "Used for profile, basic-financial, and news enrichment.",
            "Can remain in buyer or commercial-candidate profiles only with explicit commercial review.",
        ],
    },
    "fred": {
        "display_name": "FRED",
        "provider_type": "macro_data",
        "domains_used_for": ["macro_cross_asset", "rates", "inflation", "labor"],
        "required_env_keys": ["FRED_API_KEY"],
        "governance_status": GOVERNANCE_COMMERCIAL_CANDIDATE,
        "fallback_role": "primary_macro_candidate",
        "criticality": "medium",
        "notes": [
            "Primary macro series source for the assistant data fabric.",
            "Operational use still needs normal diligence on redistribution and commercial packaging.",
        ],
    },
    "sec_edgar": {
        "display_name": "SEC EDGAR",
        "provider_type": "filings_and_fundamentals",
        "domains_used_for": ["fundamental_filing", "event_context", "filing_profile"],
        "required_env_keys": ["SEC_USER_AGENT"],
        "governance_status": GOVERNANCE_COMMERCIAL_CANDIDATE,
        "fallback_role": "primary_filing_candidate",
        "criticality": "high",
        "notes": [
            "Primary filing-aware backbone for fundamental and event context.",
            "Engineering profile treats it as a clean commercial candidate, subject to normal legal review.",
        ],
    },
    "gnews": {
        "display_name": "GNews",
        "provider_type": "news",
        "domains_used_for": ["sentiment_narrative_flow", "news_enrichment"],
        "required_env_keys": ["GNEWS_API_KEY"],
        "governance_status": GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW,
        "fallback_role": "secondary_news_source",
        "criticality": "medium",
        "notes": [
            "Used in multi-source news enrichment.",
            "Keep behind review-gated profiles before external demo or commercial deployment.",
        ],
    },
    "newsapi": {
        "display_name": "NewsAPI",
        "provider_type": "news",
        "domains_used_for": ["sentiment_narrative_flow", "news_enrichment"],
        "required_env_keys": ["NEWS_API_KEY"],
        "governance_status": GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW,
        "fallback_role": "secondary_news_source",
        "criticality": "medium",
        "notes": [
            "Used in multi-source news enrichment.",
            "Treat as review-gated for buyer or commercial use.",
        ],
    },
    "gdelt": {
        "display_name": "GDELT",
        "provider_type": "event_and_news",
        "domains_used_for": ["sentiment_narrative_flow", "geopolitical_policy", "event_overlay"],
        "required_env_keys": [],
        "governance_status": GOVERNANCE_UNKNOWN_REVIEW_REQUIRED,
        "fallback_role": "broad_event_overlay",
        "criticality": "medium",
        "notes": [
            "Adds broad event and policy coverage.",
            "Use in buyer or production profiles only after explicit review of commercial and redistribution constraints.",
        ],
    },
    "world_bank": {
        "display_name": "World Bank",
        "provider_type": "macro_data",
        "domains_used_for": ["macro_cross_asset", "growth_context"],
        "required_env_keys": [],
        "governance_status": GOVERNANCE_COMMERCIAL_CANDIDATE,
        "fallback_role": "supplemental_macro_candidate",
        "criticality": "low",
        "notes": [
            "Supplemental macro context source for growth, inflation, and labor snapshots.",
        ],
    },
    "stooq": {
        "display_name": "Stooq",
        "provider_type": "market_data",
        "domains_used_for": ["reference_bars", "daily_bars_fallback"],
        "required_env_keys": [],
        "governance_status": GOVERNANCE_INTERNAL_RESEARCH_ONLY,
        "fallback_role": "market_data_fallback",
        "criticality": "medium",
        "notes": [
            "No-key market-data fallback used in reference-bar and daily-bar recovery paths.",
            "Treat as an internal-research convenience feed unless separately cleared.",
        ],
    },
    "yfinance": {
        "display_name": "yfinance",
        "provider_type": "market_and_fundamentals",
        "domains_used_for": [
            "daily_bars_fallback",
            "intraday_bars",
            "fundamentals_fallback",
            "reference_bars",
        ],
        "required_env_keys": [],
        "governance_status": GOVERNANCE_DEV_TEST_ONLY,
        "fallback_role": "last_resort_fallback",
        "criticality": "medium",
        "notes": [
            "Convenience development fallback for bars, intraday, and coarse fundamentals.",
            "Do not treat as buyer-safe or production-commercial by default.",
        ],
    },
    "google_news_rss": {
        "display_name": "Google News RSS",
        "provider_type": "news",
        "domains_used_for": ["news_ingestion", "sentiment_narrative_flow"],
        "required_env_keys": [],
        "governance_status": GOVERNANCE_INTERNAL_RESEARCH_ONLY,
        "fallback_role": "broad_headline_fallback",
        "criticality": "low",
        "notes": [
            "Convenience headline fallback for research and resilience.",
            "Treat as internal-research only until a cleaner commercial substitute is chosen.",
        ],
    },
}


_SOURCE_ALIASES = {
    "polygon": "massive_polygon",
    "massive": "massive_polygon",
    "massive/polygon": "massive_polygon",
    "finnhub_profile": "finnhub",
    "finnhub_basic_financials": "finnhub",
    "finnhub_news": "finnhub",
    "alphavantage_overview": "alphavantage",
    "fred:rates": "fred",
    "fred:policy_rate": "fred",
    "fred:inflation": "fred",
    "fred:labor": "fred",
    "fred:growth": "fred",
    "fred:credit": "fred",
    "world_bank:gdp_growth": "world_bank",
    "world_bank:inflation": "world_bank",
    "world_bank:unemployment": "world_bank",
    "sec": "sec_edgar",
    "secedgar": "sec_edgar",
}


_FEATURE_SOURCE_MAP: Dict[str, Dict[str, Any]] = {
    "canonical_market_structure": {
        "domains": ["market_price_volume", "technical_market_structure"],
        "sources": ["massive_polygon", "alphavantage", "stooq", "yfinance"],
    },
    "filing_anchored_fundamentals": {
        "domains": ["fundamental_filing", "event_catalyst_risk"],
        "sources": ["sec_edgar", "alphavantage", "finnhub", "yfinance"],
    },
    "sentiment_narrative_flow": {
        "domains": ["sentiment_narrative_flow", "geopolitical_policy", "event_catalyst_risk"],
        "sources": ["google_news_rss", "gnews", "newsapi", "finnhub", "gdelt"],
    },
    "macro_cross_asset": {
        "domains": ["macro_cross_asset"],
        "sources": ["fred", "world_bank"],
    },
    "relative_context": {
        "domains": ["relative_context", "cross_asset_confirmation", "market_breadth_internals"],
        "sources": ["massive_polygon", "alphavantage", "stooq", "yfinance"],
    },
}


_DOMAIN_DEPENDENCY_MAP: Dict[str, Dict[str, Any]] = {
    "market_price_volume": {
        "sources": ["massive_polygon", "alphavantage", "stooq", "yfinance"],
        "impact_if_removed": "Canonical market bars and verification degrade first; trend, volatility, and breadth derivatives weaken materially.",
    },
    "fundamental_filing": {
        "sources": ["sec_edgar", "alphavantage", "finnhub", "yfinance"],
        "impact_if_removed": "Filing-backed durability, leverage, and event context thin out quickly.",
    },
    "sentiment_narrative_flow": {
        "sources": ["google_news_rss", "gnews", "newsapi", "finnhub", "gdelt"],
        "impact_if_removed": "Narrative, crowding, catalyst, and geopolitical overlays degrade to thinner internal coverage.",
    },
    "macro_cross_asset": {
        "sources": ["fred", "world_bank"],
        "impact_if_removed": "Rates, inflation, growth, and macro regime framing lose external confirmation.",
    },
    "relative_context": {
        "sources": ["massive_polygon", "alphavantage", "stooq", "yfinance"],
        "impact_if_removed": "Benchmark, sector, and cross-asset confirmation become thinner or unavailable.",
    },
    "event_catalyst_risk": {
        "sources": ["sec_edgar", "alphavantage", "finnhub", "google_news_rss", "gnews", "newsapi", "gdelt"],
        "impact_if_removed": "Earnings proximity, catalyst density, and post-event distortion checks weaken.",
    },
}


_REPORT_SECTION_SOURCE_MAP: Dict[str, List[str]] = {
    "fundamental_analysis": ["sec_edgar", "alphavantage", "finnhub", "yfinance"],
    "sentiment_analysis": ["google_news_rss", "gnews", "newsapi", "finnhub", "gdelt"],
    "macro_geopolitical_analysis": ["fred", "world_bank", "gdelt"],
    "event_catalyst_risk_analysis": ["sec_edgar", "finnhub", "gnews", "newsapi", "gdelt"],
    "liquidity_execution_fragility_analysis": ["massive_polygon", "alphavantage", "stooq", "yfinance"],
    "cross_asset_confirmation_analysis": ["massive_polygon", "alphavantage", "stooq", "yfinance", "fred"],
}


def _runtime_enabled(source_name: str) -> tuple[bool, str]:
    normalized = normalize_source_name(source_name)
    if normalized == "massive_polygon":
        enabled = bool(config.massive_api_key())
        return enabled, "env key present" if enabled else "MASSIVE_API_KEY / POLYGON_API_KEY missing"
    if normalized == "alphavantage":
        enabled = bool(config.alphavantage_api_key())
        return enabled, "env key present" if enabled else "ALPHAVANTAGE_API_KEY missing"
    if normalized == "finnhub":
        enabled = bool(config.finnhub_api_key())
        return enabled, "env key present" if enabled else "FINNHUB_API_KEY missing"
    if normalized == "fred":
        enabled = bool(config.fred_api_key())
        return enabled, "env key present" if enabled else "FRED_API_KEY missing"
    if normalized == "sec_edgar":
        enabled = bool(config.sec_user_agent())
        return enabled, "SEC user-agent configured" if enabled else "SEC_USER_AGENT missing"
    if normalized == "gnews":
        enabled = bool(config.gnews_api_key())
        return enabled, "env key present" if enabled else "GNEWS_API_KEY missing"
    if normalized == "newsapi":
        enabled = bool(config.news_api_key())
        return enabled, "env key present" if enabled else "NEWS_API_KEY missing"
    if normalized == "gdelt":
        enabled = bool(config.gdelt_enabled())
        return enabled, "enabled by config" if enabled else "GDELT disabled"
    if normalized == "world_bank":
        enabled = bool(config.world_bank_enabled())
        return enabled, "enabled by config" if enabled else "WORLD_BANK disabled"
    if normalized == "stooq":
        enabled = bool(config.stooq_enabled())
        return enabled, "enabled by config" if enabled else "STOOQ disabled"
    if normalized == "google_news_rss":
        return True, "no-key RSS path"
    if normalized == "yfinance":
        enabled = importlib.util.find_spec("yfinance") is not None
        return enabled, "optional package available" if enabled else "yfinance not installed"
    return False, "unknown source"


def available_source_profiles() -> List[str]:
    return list(_PROFILE_RULES)


def active_source_profile() -> str:
    requested = str(config.source_profile() or PROFILE_INTERNAL_RESEARCH).strip().lower()
    return requested if requested in _PROFILE_RULES else PROFILE_INTERNAL_RESEARCH


def normalize_source_name(source_name: Optional[str]) -> str:
    raw = str(source_name or "").strip().lower()
    if not raw:
        return ""
    if raw in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[raw]
    if raw in _SOURCE_REGISTRY:
        return raw
    if raw.startswith("fred:"):
        return "fred"
    if raw.startswith("world_bank:"):
        return "world_bank"
    if raw.startswith("finnhub_"):
        return "finnhub"
    if raw.startswith("alphavantage"):
        return "alphavantage"
    return raw


def source_definition(source_name: Optional[str]) -> Dict[str, Any]:
    normalized = normalize_source_name(source_name)
    if normalized in _SOURCE_REGISTRY:
        return {**_SOURCE_REGISTRY[normalized], "source_name": normalized}
    return {
        "source_name": normalized or "unknown",
        "display_name": str(source_name or "unknown"),
        "provider_type": "unknown",
        "domains_used_for": [],
        "required_env_keys": [],
        "governance_status": GOVERNANCE_UNKNOWN_REVIEW_REQUIRED,
        "fallback_role": "unknown",
        "criticality": "low",
        "notes": ["Source is not yet classified in the canonical governance registry."],
    }


def governance_risk_score(governance_status: Optional[str]) -> float:
    mapping = {
        GOVERNANCE_COMMERCIAL_CANDIDATE: 22.0,
        GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW: 52.0,
        GOVERNANCE_UNKNOWN_REVIEW_REQUIRED: 68.0,
        GOVERNANCE_INTERNAL_RESEARCH_ONLY: 82.0,
        GOVERNANCE_DEV_TEST_ONLY: 92.0,
    }
    return float(mapping.get(str(governance_status or ""), 70.0))


def _criticality_penalty(level: Optional[str]) -> float:
    mapping = {"high": 16.0, "medium": 10.0, "low": 4.0}
    return float(mapping.get(str(level or "").lower(), 6.0))


def risk_tier(score: Any) -> str:
    try:
        number = float(score)
    except Exception:
        return "unknown"
    if number >= 80:
        return "high"
    if number >= 55:
        return "elevated"
    if number >= 35:
        return "moderate"
    return "lower"


def _status_labels(governance_status: Optional[str]) -> Dict[str, str]:
    normalized = str(governance_status or "")
    if normalized == GOVERNANCE_COMMERCIAL_CANDIDATE:
        return {
            "dev_test_status": "allowed",
            "production_status": "candidate",
            "commercial_status": "candidate_with_review",
        }
    if normalized == GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW:
        return {
            "dev_test_status": "allowed",
            "production_status": "conditional",
            "commercial_status": "review_required",
        }
    if normalized == GOVERNANCE_UNKNOWN_REVIEW_REQUIRED:
        return {
            "dev_test_status": "allowed",
            "production_status": "unknown_review_required",
            "commercial_status": "unknown_review_required",
        }
    if normalized == GOVERNANCE_INTERNAL_RESEARCH_ONLY:
        return {
            "dev_test_status": "allowed",
            "production_status": "internal_research_only",
            "commercial_status": "not_clean",
        }
    return {
        "dev_test_status": "allowed",
        "production_status": "dev_test_only",
        "commercial_status": "not_clean",
    }


def profile_definition(profile: Optional[str] = None) -> Dict[str, Any]:
    resolved = str(profile or active_source_profile()).strip().lower()
    return {
        "source_profile": resolved,
        **(_PROFILE_RULES.get(resolved) or _PROFILE_RULES[PROFILE_INTERNAL_RESEARCH]),
    }


def source_allowed(source_name: Optional[str], *, profile: Optional[str] = None) -> bool:
    if not source_name:
        return True
    definition = source_definition(source_name)
    governance_status = definition.get("governance_status")
    rules = profile_definition(profile)
    return str(governance_status) in set(rules.get("allowed_statuses") or [])


def gating_decision(source_name: Optional[str], *, profile: Optional[str] = None) -> Dict[str, Any]:
    resolved_profile = str(profile or active_source_profile())
    definition = source_definition(source_name)
    allowed = source_allowed(source_name, profile=resolved_profile)
    runtime_enabled, runtime_note = _runtime_enabled(definition.get("source_name"))
    governance_status = str(definition.get("governance_status") or "")
    return {
        "source_name": definition.get("source_name"),
        "profile": resolved_profile,
        "allowed": allowed,
        "governance_status": governance_status,
        "runtime_enabled": runtime_enabled,
        "runtime_note": runtime_note,
        "note": (
            f"Blocked by source profile {resolved_profile} because {definition.get('display_name')} is classified as {governance_status}."
            if not allowed
            else runtime_note
        ),
    }


def build_source_inventory(*, profile: Optional[str] = None) -> List[Dict[str, Any]]:
    resolved_profile = str(profile or active_source_profile())
    inventory: List[Dict[str, Any]] = []
    for source_name in sorted(_SOURCE_REGISTRY):
        definition = source_definition(source_name)
        runtime_enabled, runtime_note = _runtime_enabled(source_name)
        base_score = governance_risk_score(definition.get("governance_status"))
        dependency_risk = min(base_score + _criticality_penalty(definition.get("criticality")), 100.0)
        allowed = source_allowed(source_name, profile=resolved_profile)
        status_labels = _status_labels(definition.get("governance_status"))
        inventory.append(
            {
                **definition,
                "source_profile": resolved_profile,
                "current_enabled_state": runtime_enabled,
                "enabled_reason": runtime_note,
                **status_labels,
                "licensing_risk_tier": risk_tier(base_score),
                "commercialization_risk_score": round(base_score, 1),
                "source_restriction_flag": not allowed,
                "requires_legal_review": definition.get("governance_status")
                in {
                    GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW,
                    GOVERNANCE_UNKNOWN_REVIEW_REQUIRED,
                },
                "production_eligible_flag": definition.get("governance_status")
                in {
                    GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW,
                    GOVERNANCE_COMMERCIAL_CANDIDATE,
                },
                "commercial_distribution_risk": risk_tier(base_score),
                "dependency_risk_score": round(dependency_risk, 1),
                "buyer_demo_allowed": source_allowed(source_name, profile=PROFILE_BUYER_DEMO),
                "commercial_candidate_allowed": source_allowed(
                    source_name, profile=PROFILE_COMMERCIAL_CANDIDATE
                ),
                "restricted_cleanroom_allowed": source_allowed(
                    source_name, profile=PROFILE_RESTRICTED_CLEANROOM
                ),
            }
        )
    return inventory


def dependency_maps() -> Dict[str, Any]:
    return {
        "feature_source_map": {key: dict(value) for key, value in _FEATURE_SOURCE_MAP.items()},
        "domain_dependency_map": {
            key: dict(value) for key, value in _DOMAIN_DEPENDENCY_MAP.items()
        },
        "report_section_source_map": {
            key: list(value) for key, value in _REPORT_SECTION_SOURCE_MAP.items()
        },
    }


def collect_sources(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str):
            items: Sequence[Any] = [value]
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            continue
        for item in items:
            normalized = normalize_source_name(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
    return output


def source_dependency_domains(source_name: str) -> List[str]:
    normalized = normalize_source_name(source_name)
    domains: List[str] = []
    for domain, payload in _DOMAIN_DEPENDENCY_MAP.items():
        if normalized in set(payload.get("sources") or []):
            domains.append(domain)
    return domains
