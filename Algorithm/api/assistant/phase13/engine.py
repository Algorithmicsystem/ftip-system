from __future__ import annotations

from typing import Any, Dict, Iterable, List

from api import source_governance

from .common import (
    SOURCE_GOVERNANCE_ARTIFACT_KIND,
    SOURCE_GOVERNANCE_VERSION,
    compact_list,
    now_utc,
)


def _is_internal_source(source_name: str) -> bool:
    normalized = str(source_name or "").strip().lower()
    return normalized in {
        "",
        "market_bars_daily",
        "market_bars_intraday",
        "prices_daily_versioned",
        "prosperity_daily_bars",
        "fundamentals_quarterly",
        "fundamentals_pit",
        "news_raw",
        "news_items",
        "sentiment_daily",
        "quality_daily",
        "features_daily",
        "signals_daily",
        "provided_bars",
        "fundamentals+news_heuristic",
        "none",
    }


def _collect_active_sources(report: Dict[str, Any]) -> List[str]:
    data_bundle = report.get("data_bundle") or {}
    quality = (data_bundle.get("quality_provenance") or {})
    canonical = data_bundle.get("canonical_alpha_core") or {}
    feature_meta = canonical.get("feature_meta") or {}
    provider_status_map = quality.get("provider_status_map") or {}
    source_map = quality.get("source_map") or {}
    domain_availability = report.get("domain_availability") or {}
    raw_values: List[Any] = []
    raw_values.extend(source_map.values())
    raw_values.extend(
        provider_name
        for payload in provider_status_map.values()
        for provider_name in (payload or {}).keys()
    )
    raw_values.extend(
        [
            feature_meta.get("price_source"),
            feature_meta.get("event_source"),
            feature_meta.get("breadth_source"),
        ]
    )
    for entry in domain_availability.values():
        raw_values.append((entry or {}).get("fallback_source") or [])
    sources = [
        item
        for item in source_governance.collect_sources(raw_values)
        if not _is_internal_source(item)
    ]
    return sources


def _dependency_sources_for_domain(
    domain: str,
    dependency_map: Dict[str, Dict[str, Any]],
) -> List[str]:
    return list((dependency_map.get(domain) or {}).get("sources") or [])


def _commercial_blockers(
    *,
    active_sources: List[str],
    inventory_by_source: Dict[str, Dict[str, Any]],
    profile: str,
) -> List[str]:
    blockers: List[str] = []
    for source_name in active_sources:
        item = inventory_by_source.get(source_name) or {}
        governance_status = str(item.get("governance_status") or "unknown")
        if item.get("source_restriction_flag"):
            blockers.append(
                f"{source_name} is currently blocked by the {profile} source profile."
            )
        elif governance_status in {
            source_governance.GOVERNANCE_DEV_TEST_ONLY,
            source_governance.GOVERNANCE_INTERNAL_RESEARCH_ONLY,
            source_governance.GOVERNANCE_UNKNOWN_REVIEW_REQUIRED,
        }:
            blockers.append(
                f"{source_name} is active in the stack but remains classified as {governance_status}."
            )
        elif item.get("requires_legal_review"):
            blockers.append(
                f"{source_name} remains usable only with explicit commercial or legal review."
            )
    return blockers[:10]


def _mixed_mode_dependencies(
    *,
    active_sources: List[str],
    inventory_by_source: Dict[str, Dict[str, Any]],
    dependency_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    active_set = set(active_sources)
    for domain, payload in dependency_map.items():
        domain_sources = [
            source for source in payload.get("sources") or [] if source in active_set
        ]
        if len(domain_sources) < 2:
            continue
        governance_mix = sorted(
            {
                str((inventory_by_source.get(source) or {}).get("governance_status") or "unknown")
                for source in domain_sources
            }
        )
        if len(governance_mix) < 2:
            continue
        output.append(
            {
                "domain": domain,
                "sources": domain_sources,
                "governance_mix": governance_mix,
                "summary": (
                    f"{domain} is using a mixed governance stack across "
                    f"{', '.join(domain_sources)}."
                ),
            }
        )
    return output[:8]


def _gated_domains(
    *,
    inventory_by_source: Dict[str, Dict[str, Any]],
    dependency_map: Dict[str, Dict[str, Any]],
    active_sources: List[str],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    active_set = set(active_sources)
    for domain, payload in dependency_map.items():
        domain_sources = list(payload.get("sources") or [])
        blocked = [
            source
            for source in domain_sources
            if (inventory_by_source.get(source) or {}).get("source_restriction_flag")
        ]
        allowed = [
            source
            for source in domain_sources
            if not (inventory_by_source.get(source) or {}).get("source_restriction_flag")
        ]
        blocked_active = [source for source in blocked if source in active_set]
        if not blocked_active and allowed:
            continue
        output.append(
            {
                "domain": domain,
                "blocked_sources": blocked_active or blocked,
                "impact": payload.get("impact_if_removed"),
            }
        )
    return output


def _removable_source_impact(
    *,
    active_sources: List[str],
    dependency_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for source_name in active_sources:
        domains = source_governance.source_dependency_domains(source_name)
        if not domains:
            continue
        output.append(
            {
                "source_name": source_name,
                "affected_domains": domains,
                "impact_summary": compact_list(
                    (dependency_map.get(domain) or {}).get("impact_if_removed")
                    for domain in domains
                ),
            }
        )
    return output[:10]


def _buyer_demo_suitability(
    *,
    active_sources: List[str],
    inventory_by_source: Dict[str, Dict[str, Any]],
    profile: str,
) -> str:
    if profile in {
        source_governance.PROFILE_DEV_EXPERIMENTAL,
        source_governance.PROFILE_INTERNAL_RESEARCH,
    }:
        return "mixed_risk_profile"
    if any(
        (inventory_by_source.get(source) or {}).get("source_restriction_flag")
        for source in active_sources
    ):
        return "blocked_by_profile"
    if any(
        str((inventory_by_source.get(source) or {}).get("governance_status") or "")
        == source_governance.GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW
        for source in active_sources
    ):
        return "conditional_review_required"
    if profile == source_governance.PROFILE_RESTRICTED_CLEANROOM:
        return "restricted_cleanroom_candidate"
    return "cleaner_candidate"


def _commercialization_risk_score(
    *,
    active_sources: List[str],
    inventory_by_source: Dict[str, Dict[str, Any]],
    profile: str,
) -> float:
    if active_sources:
        scores = [
            float((inventory_by_source.get(source) or {}).get("commercialization_risk_score") or 70.0)
            for source in active_sources
        ]
    else:
        scores = [
            float(item.get("commercialization_risk_score") or 70.0)
            for item in inventory_by_source.values()
            if item.get("current_enabled_state")
        ]
    if not scores:
        scores = [55.0]
    profile_penalty = {
        source_governance.PROFILE_DEV_EXPERIMENTAL: 18.0,
        source_governance.PROFILE_INTERNAL_RESEARCH: 10.0,
        source_governance.PROFILE_BUYER_DEMO: 4.0,
        source_governance.PROFILE_COMMERCIAL_CANDIDATE: 2.0,
        source_governance.PROFILE_RESTRICTED_CLEANROOM: 0.0,
    }.get(profile, 8.0)
    return round(min(sum(scores) / len(scores) + profile_penalty, 100.0), 1)


def _commercial_cleanup_queue(
    *,
    active_sources: List[str],
    inventory_by_source: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for source_name in active_sources:
        item = inventory_by_source.get(source_name) or {}
        if float(item.get("commercialization_risk_score") or 0.0) < 60.0:
            continue
        output.append(
            {
                "source_name": source_name,
                "governance_status": item.get("governance_status"),
                "priority": (
                    "high"
                    if str(item.get("criticality") or "") == "high"
                    else "medium"
                ),
                "cleanup_reason": (
                    f"{source_name} remains classified as {item.get('governance_status')}."
                ),
                "affected_domains": source_governance.source_dependency_domains(source_name),
            }
        )
    output.sort(key=lambda item: (item.get("priority") != "high", item.get("source_name")))
    return output[:10]


def _commercialization_readiness_summary(
    *,
    profile: str,
    readiness: Dict[str, Any],
    active_sources: List[str],
) -> str:
    blockers = compact_list(readiness.get("commercial_blockers") or [], limit=4)
    sources = compact_list(active_sources, limit=6)
    return (
        f"Source profile is {profile}, buyer-demo suitability is "
        f"{readiness.get('buyer_demo_suitability')}, and commercialization risk is "
        f"{readiness.get('commercialization_risk_score')} / 100. "
        f"Active external sources are {sources or ['none']}, and top blockers are "
        f"{blockers or ['none']}."
    )


def _source_governance_summary(
    *,
    profile: str,
    disallowed_sources: List[str],
    gated_domains: List[Dict[str, Any]],
) -> str:
    return (
        f"Current source profile is {profile}. "
        f"Disallowed sources under this profile are {compact_list(disallowed_sources) or ['none']}, "
        f"and the domains most exposed to profile gating are "
        f"{compact_list(item.get('domain') for item in gated_domains) or ['none']}."
    )


def _buyer_diligence_summary(
    *,
    readiness: Dict[str, Any],
    clean_stack: List[Dict[str, Any]],
) -> str:
    candidate_sources = compact_list(
        item.get("source_name") for item in clean_stack[:6]
    )
    return (
        f"Buyer diligence view: current stack status is "
        f"{readiness.get('buyer_safe_profile_status')}, with cleanup queue "
        f"{compact_list(item.get('source_name') for item in (readiness.get('commercial_cleanup_queue') or [])) or ['none']}. "
        f"The cleanest currently configured source candidates are {candidate_sources or ['none']}."
    )


def build_source_governance_artifact(
    *,
    current_report: Dict[str, Any],
) -> Dict[str, Any]:
    profile = source_governance.active_source_profile()
    inventory = source_governance.build_source_inventory(profile=profile)
    inventory_by_source = {item["source_name"]: item for item in inventory}
    dependency_maps = source_governance.dependency_maps()
    domain_dependency_map = dependency_maps["domain_dependency_map"]
    active_sources = _collect_active_sources(current_report)
    active_source_details = [
        inventory_by_source.get(source) or source_governance.source_definition(source)
        for source in active_sources
    ]
    disallowed_sources = [
        item["source_name"]
        for item in inventory
        if item.get("source_restriction_flag") and item.get("current_enabled_state")
    ]
    gated_domains = _gated_domains(
        inventory_by_source=inventory_by_source,
        dependency_map=domain_dependency_map,
        active_sources=active_sources,
    )
    mixed_mode_dependencies = _mixed_mode_dependencies(
        active_sources=active_sources,
        inventory_by_source=inventory_by_source,
        dependency_map=domain_dependency_map,
    )
    commercialization_risk_score = _commercialization_risk_score(
        active_sources=active_sources,
        inventory_by_source=inventory_by_source,
        profile=profile,
    )
    buyer_demo_suitability = _buyer_demo_suitability(
        active_sources=active_sources,
        inventory_by_source=inventory_by_source,
        profile=profile,
    )
    commercial_blockers = _commercial_blockers(
        active_sources=active_sources,
        inventory_by_source=inventory_by_source,
        profile=profile,
    )
    cleanup_queue = _commercial_cleanup_queue(
        active_sources=active_sources,
        inventory_by_source=inventory_by_source,
    )
    clean_stack = [
        item
        for item in inventory
        if item.get("governance_status")
        == source_governance.GOVERNANCE_COMMERCIAL_CANDIDATE
        and item.get("current_enabled_state")
    ]
    clean_stack.sort(key=lambda item: item.get("source_name"))
    readiness = {
        "source_profile": profile,
        "buyer_safe_profile_status": (
            "restricted_cleanroom"
            if profile == source_governance.PROFILE_RESTRICTED_CLEANROOM
            else "conditional_review_required"
            if buyer_demo_suitability == "conditional_review_required"
            else "mixed_risk"
            if buyer_demo_suitability in {"mixed_risk_profile", "blocked_by_profile"}
            else "cleaner_candidate"
        ),
        "buyer_demo_suitability": buyer_demo_suitability,
        "commercialization_risk_score": commercialization_risk_score,
        "licensing_risk_tier": source_governance.risk_tier(
            commercialization_risk_score
        ),
        "active_external_sources": active_sources,
        "active_source_details": active_source_details,
        "high_risk_sources": [
            item["source_name"]
            for item in active_source_details
            if float(item.get("commercialization_risk_score") or 0.0) >= 70.0
        ],
        "mixed_mode_dependencies": mixed_mode_dependencies,
        "disallowed_sources": disallowed_sources,
        "gated_domains": gated_domains,
        "degraded_due_to_profile": [item.get("domain") for item in gated_domains],
        "commercial_blockers": commercial_blockers,
        "commercial_cleanup_queue": cleanup_queue,
        "commercial_clean_stack_candidate": clean_stack[:10],
        "commercialization_profile_notes": compact_list(
            [
                source_governance.profile_definition(profile).get("summary"),
                "This artifact is engineering and product governance guidance, not legal advice.",
            ]
        ),
    }
    critical_source_dependencies = [
        {
            "domain": domain,
            "sources": _dependency_sources_for_domain(domain, domain_dependency_map),
            "impact": (domain_dependency_map.get(domain) or {}).get("impact_if_removed"),
        }
        for domain in ("market_price_volume", "fundamental_filing", "macro_cross_asset", "sentiment_narrative_flow")
    ]
    removable_source_impact = _removable_source_impact(
        active_sources=active_sources,
        dependency_map=domain_dependency_map,
    )
    commercialization_readiness_summary = _commercialization_readiness_summary(
        profile=profile,
        readiness=readiness,
        active_sources=active_sources,
    )
    source_governance_summary = _source_governance_summary(
        profile=profile,
        disallowed_sources=disallowed_sources,
        gated_domains=gated_domains,
    )
    buyer_diligence_summary = _buyer_diligence_summary(
        readiness=readiness,
        clean_stack=clean_stack,
    )
    return {
        "source_governance_kind": SOURCE_GOVERNANCE_ARTIFACT_KIND,
        "source_governance_version": SOURCE_GOVERNANCE_VERSION,
        "generated_at": now_utc(),
        "source_profile": profile,
        "profile_definition": source_governance.profile_definition(profile),
        "source_inventory": inventory,
        "feature_source_map": dependency_maps["feature_source_map"],
        "domain_dependency_map": domain_dependency_map,
        "report_section_source_map": dependency_maps["report_section_source_map"],
        "critical_source_dependencies": critical_source_dependencies,
        "removable_source_impact": removable_source_impact,
        "commercialization_readiness": readiness,
        "source_inventory_export": {
            "source_profile": profile,
            "sources": [
                {
                    "source_name": item.get("source_name"),
                    "display_name": item.get("display_name"),
                    "governance_status": item.get("governance_status"),
                    "current_enabled_state": item.get("current_enabled_state"),
                    "source_restriction_flag": item.get("source_restriction_flag"),
                    "domains_used_for": item.get("domains_used_for"),
                    "required_env_keys": item.get("required_env_keys"),
                }
                for item in inventory
            ],
        },
        "source_governance_export": {
            "source_profile": profile,
            "disallowed_sources": disallowed_sources,
            "gated_domains": gated_domains,
            "commercial_cleanup_queue": cleanup_queue,
            "buyer_demo_suitability": buyer_demo_suitability,
        },
        "commercialization_readiness_summary": commercialization_readiness_summary,
        "source_governance_summary": source_governance_summary,
        "buyer_diligence_summary": buyer_diligence_summary,
    }
