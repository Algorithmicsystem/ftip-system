from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List, Optional

from api.source_governance import (
    active_source_profile,
    governance_risk_score,
    source_definition,
)


_PROVIDER_CAPABILITY_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "massive_polygon": {
        "domain": "market_data",
        "asset_capability": "equity_bars",
        "capabilities": ["daily_bars", "reference_bars", "snapshot_bars"],
        "fallback_priority": 10,
        "connector_slot": "premium_market_data",
        "freshness_grade": "institutional_candidate",
        "quality_hint": "institutional_candidate",
        "latency_hint": "low_latency",
        "partial_result_behavior": "fail_closed",
    },
    "alphavantage": {
        "domain": "mixed_enrichment",
        "asset_capability": "bars_and_fundamentals",
        "capabilities": ["daily_bars", "reference_bars", "quarterly_fundamentals", "company_overview"],
        "fallback_priority": 30,
        "connector_slot": "filings_estimates",
        "freshness_grade": "daily_enrichment",
        "quality_hint": "secondary_enrichment",
        "latency_hint": "medium_latency",
        "partial_result_behavior": "allow_partial_with_warning",
    },
    "stooq": {
        "domain": "market_data",
        "asset_capability": "reference_bars",
        "capabilities": ["daily_bars", "reference_bars"],
        "fallback_priority": 40,
        "connector_slot": "public_market_backup",
        "freshness_grade": "public_delayed",
        "quality_hint": "research_fallback",
        "latency_hint": "medium_latency",
        "partial_result_behavior": "allow_partial_with_warning",
    },
    "yfinance": {
        "domain": "mixed_enrichment",
        "asset_capability": "bars_and_fundamentals",
        "capabilities": ["daily_bars", "reference_bars", "intraday_bars", "quarterly_fundamentals"],
        "fallback_priority": 5,
        "connector_slot": "primary_free_market_data",
        "freshness_grade": "daily_free",
        "quality_hint": "primary_free",
        "latency_hint": "medium_latency",
        "partial_result_behavior": "allow_partial_with_warning",
    },
    "sec_edgar": {
        "domain": "filings",
        "asset_capability": "filing_backbone",
        "capabilities": ["filing_profile", "fundamental_backbone", "event_context"],
        "fallback_priority": 10,
        "connector_slot": "filings_estimates",
        "freshness_grade": "filing_lagged",
        "quality_hint": "primary_backbone",
        "latency_hint": "medium_latency",
        "partial_result_behavior": "fail_closed",
    },
    "finnhub": {
        "domain": "enrichment",
        "asset_capability": "profile_news_fundamentals",
        "capabilities": ["company_profile", "basic_financials", "company_news"],
        "fallback_priority": 20,
        "connector_slot": "premium_news_intel",
        "freshness_grade": "daily_enrichment",
        "quality_hint": "secondary_enrichment",
        "latency_hint": "low_latency",
        "partial_result_behavior": "allow_partial_with_warning",
    },
    "gnews": {
        "domain": "news",
        "asset_capability": "headline_news",
        "capabilities": ["news_items", "headline_search"],
        "fallback_priority": 20,
        "connector_slot": "premium_news_intel",
        "freshness_grade": "headline_fast",
        "quality_hint": "secondary_news",
        "latency_hint": "low_latency",
        "partial_result_behavior": "allow_partial_with_warning",
    },
    "newsapi": {
        "domain": "news",
        "asset_capability": "headline_news",
        "capabilities": ["news_items", "headline_search"],
        "fallback_priority": 25,
        "connector_slot": "premium_news_intel",
        "freshness_grade": "headline_fast",
        "quality_hint": "secondary_news",
        "latency_hint": "low_latency",
        "partial_result_behavior": "allow_partial_with_warning",
    },
    "google_news_rss": {
        "domain": "news",
        "asset_capability": "headline_news",
        "capabilities": ["news_items", "headline_search"],
        "fallback_priority": 35,
        "connector_slot": "scraping_news_backup",
        "freshness_grade": "scraped_headline",
        "quality_hint": "headline_backup",
        "latency_hint": "low_latency",
        "partial_result_behavior": "allow_partial_with_warning",
    },
    "gdelt": {
        "domain": "events",
        "asset_capability": "event_overlay",
        "capabilities": ["news_items", "event_overlay", "policy_overlay"],
        "fallback_priority": 30,
        "connector_slot": "alt_data_events",
        "freshness_grade": "broad_event_overlay",
        "quality_hint": "broad_overlay",
        "latency_hint": "medium_latency",
        "partial_result_behavior": "allow_partial_with_warning",
    },
    "fred": {
        "domain": "macro",
        "asset_capability": "macro_series",
        "capabilities": ["macro_series", "rates", "inflation", "labor"],
        "fallback_priority": 10,
        "connector_slot": "premium_macro",
        "freshness_grade": "macro_primary",
        "quality_hint": "primary_macro",
        "latency_hint": "medium_latency",
        "partial_result_behavior": "allow_partial_with_warning",
    },
    "world_bank": {
        "domain": "macro",
        "asset_capability": "macro_series",
        "capabilities": ["macro_series", "growth_context"],
        "fallback_priority": 25,
        "connector_slot": "macro_backup",
        "freshness_grade": "macro_supplemental",
        "quality_hint": "supplemental_macro",
        "latency_hint": "high_latency",
        "partial_result_behavior": "allow_partial_with_warning",
    },
}


def provider_capability_profile(
    provider_name: Optional[str],
    *,
    capability: Optional[str] = None,
) -> Dict[str, Any]:
    definition = source_definition(provider_name)
    source_name = str(definition.get("source_name") or provider_name or "unknown")
    override = _PROVIDER_CAPABILITY_OVERRIDES.get(source_name, {})
    capabilities = list(override.get("capabilities") or definition.get("capabilities") or [])
    resolved_capability = capability or (capabilities[0] if capabilities else "data_fetch")
    profile = {
        "provider_name": source_name,
        "display_name": definition.get("display_name") or source_name,
        "domain": override.get("domain") or definition.get("provider_type") or "unknown",
        "asset_capability": override.get("asset_capability") or definition.get("provider_type") or "unknown",
        "capabilities": capabilities,
        "requested_capability": resolved_capability,
        "fallback_priority": int(override.get("fallback_priority") or 99),
        "connector_slot": override.get("connector_slot") or "unassigned",
        "freshness_grade": override.get("freshness_grade") or "unknown",
        "quality_hint": override.get("quality_hint") or "unclassified",
        "latency_hint": override.get("latency_hint") or "unknown",
        "partial_result_behavior": override.get("partial_result_behavior") or "allow_partial_with_warning",
        "governance_status": definition.get("governance_status"),
        "fallback_role": definition.get("fallback_role"),
        "criticality": definition.get("criticality"),
    }
    return profile


def provider_confidence_grade(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.62:
        return "medium"
    if confidence >= 0.45:
        return "watch"
    return "low"


def provider_freshness_grade(
    freshness_status: str,
    *,
    provider_name: Optional[str] = None,
) -> str:
    profile = provider_capability_profile(provider_name)
    configured_grade = str(profile.get("freshness_grade") or "unknown")
    if freshness_status == "fresh":
        return configured_grade
    if freshness_status == "recent":
        return f"{configured_grade}_recent"
    if freshness_status == "historical":
        return f"{configured_grade}_historical"
    return configured_grade


def provider_chain_used(attempts: Iterable[Dict[str, Any]]) -> List[str]:
    chain: List[str] = []
    for item in attempts:
        provider_name = str(item.get("provider_name") or "")
        if provider_name and provider_name not in chain:
            chain.append(provider_name)
    return chain


def ordered_provider_candidates(
    candidates: Iterable[tuple[str, Any]],
    *,
    capability: Optional[str] = None,
    preferred_provider: Optional[str] = None,
) -> List[tuple[str, Any]]:
    normalized_preferred = str(preferred_provider or "").strip().lower()
    ranked: List[tuple[int, int, tuple[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        provider_name = str(candidate[0] or "").strip().lower()
        profile = provider_capability_profile(
            provider_name,
            capability=capability,
        )
        priority = int(profile.get("fallback_priority") or 99)
        if normalized_preferred and provider_name == normalized_preferred:
            priority = -1
        ranked.append((priority, index, candidate))
    return [
        candidate
        for _priority, _index, candidate in sorted(
            ranked,
            key=lambda row: (row[0], row[1]),
        )
    ]


def _bounded_confidence(value: Any, *, default: float = 0.55) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(number, 1.0))


def provider_freshness_status(
    *, end_date: Optional[dt.date] = None, generated_at: Optional[dt.datetime] = None
) -> str:
    if end_date is None:
        return "unknown"
    now = (generated_at or dt.datetime.now(dt.timezone.utc)).date()
    age_days = max((now - end_date).days, 0)
    if age_days <= 2:
        return "fresh"
    if age_days <= 7:
        return "recent"
    return "historical"


def provider_confidence(
    provider_name: Optional[str],
    *,
    fallback_used: bool = False,
    response_quality: str = "complete",
    override: Optional[float] = None,
) -> float:
    if override is not None:
        return _bounded_confidence(override)
    definition = source_definition(provider_name)
    capability_profile = provider_capability_profile(provider_name)
    governance_penalty = governance_risk_score(definition.get("governance_status")) / 200.0
    role = str(definition.get("fallback_role") or "")
    role_bonus = 0.1 if role.startswith("primary") else 0.04 if "secondary" in role else -0.03
    fallback_penalty = 0.08 if fallback_used else 0.0
    quality_penalty = 0.12 if response_quality == "partial" else 0.18 if response_quality == "degraded" else 0.0
    quality_hint = str(capability_profile.get("quality_hint") or "")
    hint_bonus = (
        0.06
        if quality_hint in {"institutional_candidate", "primary_backbone", "primary_macro"}
        else 0.02
        if quality_hint in {"secondary_enrichment", "secondary_news", "supplemental_macro"}
        else -0.05
        if quality_hint in {"developer_fallback", "headline_backup"}
        else 0.0
    )
    return _bounded_confidence(
        0.78 + role_bonus + hint_bonus - governance_penalty - fallback_penalty - quality_penalty
    )


def provider_attempt(
    provider_name: str,
    *,
    status: str,
    source_type: Optional[str] = None,
    reason_code: Optional[str] = None,
    reason_detail: Optional[str] = None,
    response_quality: Optional[str] = None,
    fallback_used: bool = False,
) -> Dict[str, Any]:
    definition = source_definition(provider_name)
    capability_profile = provider_capability_profile(provider_name)
    resolved_source_type = source_type or definition.get("provider_type") or "unknown"
    return {
        "provider_name": definition.get("source_name") or provider_name,
        "display_name": definition.get("display_name") or provider_name,
        "source_type": resolved_source_type,
        "domain": capability_profile.get("domain") or resolved_source_type,
        "requested_capability": capability_profile.get("requested_capability"),
        "fallback_priority": capability_profile.get("fallback_priority"),
        "connector_slot": capability_profile.get("connector_slot"),
        "status": status,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "fallback_role": definition.get("fallback_role"),
        "response_quality": response_quality,
        "fallback_used": fallback_used,
        "source_profile": active_source_profile(),
    }


def provider_warning_flags(
    *,
    fallback_used: bool = False,
    response_quality: str = "complete",
    partial_result: bool = False,
    attempts: Optional[Iterable[Dict[str, Any]]] = None,
    error_summary: Optional[str] = None,
    confidence: Optional[float] = None,
    freshness_status: Optional[str] = None,
) -> List[str]:
    flags: List[str] = []
    attempts_list = list(attempts or [])
    if fallback_used:
        flags.append("fallback_chain_used")
    if partial_result:
        flags.append("partial_result")
    if response_quality == "partial":
        flags.append("response_partial")
    elif response_quality == "degraded":
        flags.append("response_degraded")
    if any(item.get("status") == "blocked" for item in attempts_list):
        flags.append("source_profile_blocked")
    if any(item.get("status") == "failed" for item in attempts_list):
        flags.append("provider_failures_present")
    if any(item.get("status") == "suppressed" for item in attempts_list):
        flags.append("provider_run_suppressed")
    if error_summary:
        flags.append("error_summary_present")
    if freshness_status in {"historical", "unknown"}:
        flags.append("freshness_not_ideal")
    if confidence is not None and confidence < 0.55:
        flags.append("confidence_below_comfort")
    return flags


def provider_strength_label(
    *,
    confidence: float,
    fallback_used: bool = False,
    response_quality: str = "complete",
    partial_result: bool = False,
) -> str:
    if response_quality == "degraded" or confidence < 0.45:
        return "weak"
    if partial_result or fallback_used or response_quality == "partial" or confidence < 0.68:
        return "mixed"
    return "strong"


def provider_strength_summary(
    provider_name: str,
    *,
    source_type: str,
    freshness_status: str,
    confidence: float,
    fallback_used: bool = False,
    response_quality: str = "complete",
    partial_result: bool = False,
) -> str:
    capability_profile = provider_capability_profile(provider_name)
    strength = provider_strength_label(
        confidence=confidence,
        fallback_used=fallback_used,
        response_quality=response_quality,
        partial_result=partial_result,
    )
    return (
        f"{provider_name} supplied {source_type} with {freshness_status} freshness, "
        f"{strength} source strength, confidence {round(confidence, 3)}, "
        f"response quality {response_quality}, fallback {'on' if fallback_used else 'off'}, "
        f"and connector slot {capability_profile.get('connector_slot') or 'unassigned'}."
    )


def provider_result_metadata(
    provider_name: str,
    *,
    source_type: Optional[str] = None,
    end_date: Optional[dt.date] = None,
    fallback_used: bool = False,
    response_quality: str = "complete",
    error_summary: Optional[str] = None,
    attempts: Optional[Iterable[Dict[str, Any]]] = None,
    capability: Optional[str] = None,
    retry_policy: Optional[str] = None,
    partial_result: bool = False,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    definition = source_definition(provider_name)
    capability_profile = provider_capability_profile(provider_name, capability=capability)
    attempts_list = list(attempts or [])
    freshness_status = provider_freshness_status(end_date=end_date)
    resolved_source_type = source_type or definition.get("provider_type") or "unknown"
    resolved_confidence = round(
        provider_confidence(
            provider_name,
            fallback_used=fallback_used,
            response_quality=response_quality,
            override=confidence,
        ),
        3,
    )
    confidence_grade = provider_confidence_grade(resolved_confidence)
    freshness_grade = provider_freshness_grade(
        freshness_status,
        provider_name=provider_name,
    )
    chain_used = provider_chain_used(attempts_list)
    warning_flags = provider_warning_flags(
        fallback_used=fallback_used,
        response_quality=response_quality,
        partial_result=bool(partial_result),
        attempts=attempts_list,
        error_summary=error_summary,
        confidence=resolved_confidence,
        freshness_status=freshness_status,
    )
    return {
        "provider_name": definition.get("source_name") or provider_name,
        "provider_display_name": definition.get("display_name") or provider_name,
        "provider_domain": capability_profile.get("domain") or resolved_source_type,
        "source_type": resolved_source_type,
        "asset_capability": capability_profile.get("asset_capability"),
        "freshness_status": freshness_status,
        "freshness_grade": freshness_grade,
        "confidence": resolved_confidence,
        "confidence_grade": confidence_grade,
        "fallback_used": fallback_used,
        "response_quality": response_quality,
        "error_summary": error_summary,
        "partial_result": bool(partial_result),
        "capability": capability_profile.get("requested_capability") or capability or "data_fetch",
        "capabilities": capability_profile.get("capabilities") or [],
        "retry_policy": retry_policy or "best_available_fallback_chain",
        "fallback_role": definition.get("fallback_role"),
        "criticality": definition.get("criticality"),
        "fallback_priority": capability_profile.get("fallback_priority"),
        "connector_slot": capability_profile.get("connector_slot"),
        "quality_hint": capability_profile.get("quality_hint"),
        "latency_hint": capability_profile.get("latency_hint"),
        "partial_result_behavior": capability_profile.get("partial_result_behavior"),
        "source_profile": active_source_profile(),
        "attempts": attempts_list,
        "attempt_count": len(attempts_list),
        "attempted_provider_count": len(chain_used),
        "best_source_used": definition.get("source_name") or provider_name,
        "fallback_chain_used": chain_used,
        "source_warning_flags": warning_flags,
        "strength_label": provider_strength_label(
            confidence=resolved_confidence,
            fallback_used=fallback_used,
            response_quality=response_quality,
            partial_result=bool(partial_result),
        ),
        "source_strength_summary": provider_strength_summary(
            definition.get("source_name") or provider_name,
            source_type=resolved_source_type,
            freshness_status=freshness_status,
            confidence=resolved_confidence,
            fallback_used=fallback_used,
            response_quality=response_quality,
            partial_result=bool(partial_result),
        ),
        "provider_definition": definition,
    }


def summarize_provider_failures(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_provider: Dict[str, Dict[str, Any]] = {}
    by_reason: Dict[str, int] = {}
    by_domain: Dict[str, int] = {}
    for item in items:
        provider_name = str(item.get("provider_name") or "unknown")
        bucket = by_provider.setdefault(
            provider_name,
            {
                "provider_name": provider_name,
                "count": 0,
                "reason_codes": {},
                "response_quality": {},
                "domains": {},
            },
        )
        bucket["count"] += 1
        reason_code = str(item.get("reason_code") or "UNKNOWN")
        reasons = bucket["reason_codes"]
        reasons[reason_code] = int(reasons.get(reason_code, 0)) + 1
        by_reason[reason_code] = int(by_reason.get(reason_code, 0)) + 1
        domain = str(item.get("domain") or item.get("source_type") or "unknown")
        bucket["domains"][domain] = int(bucket["domains"].get(domain, 0)) + 1
        by_domain[domain] = int(by_domain.get(domain, 0)) + 1
        quality = str(item.get("response_quality") or "unknown")
        response_quality = bucket["response_quality"]
        response_quality[quality] = int(response_quality.get(quality, 0)) + 1
    return {
        "provider_count": len(by_provider),
        "providers": sorted(by_provider.values(), key=lambda row: (-row["count"], row["provider_name"])),
        "top_reason_codes": [
            {"reason_code": reason_code, "count": count}
            for reason_code, count in sorted(by_reason.items(), key=lambda row: (-row[1], row[0]))
        ],
        "domains": [
            {"domain": domain, "count": count}
            for domain, count in sorted(by_domain.items(), key=lambda row: (-row[1], row[0]))
        ],
    }


def summarize_provider_usage(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    fallback_used_count = 0
    response_quality_counts: Dict[str, int] = {}
    domains: Dict[str, int] = {}
    strength_counts: Dict[str, int] = {}
    for item in items:
        provider_name = str(item.get("provider_name") or "unknown")
        counts[provider_name] = counts.get(provider_name, 0) + 1
        if item.get("fallback_used"):
            fallback_used_count += 1
        quality = str(item.get("response_quality") or "unknown")
        response_quality_counts[quality] = response_quality_counts.get(quality, 0) + 1
        domain = str(item.get("provider_domain") or item.get("domain") or item.get("source_type") or "unknown")
        domains[domain] = domains.get(domain, 0) + 1
        strength = str(item.get("strength_label") or "unknown")
        strength_counts[strength] = strength_counts.get(strength, 0) + 1
    return {
        "provider_count": len(counts),
        "providers": [
            {"provider_name": provider_name, "count": count}
            for provider_name, count in sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        ],
        "fallback_used_count": fallback_used_count,
        "response_quality_counts": response_quality_counts,
        "domains": [
            {"domain": domain, "count": count}
            for domain, count in sorted(domains.items(), key=lambda row: (-row[1], row[0]))
        ],
        "strength_counts": strength_counts,
    }
