from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from api.assistant.phase3.common import (
    clamp,
    coverage_score,
    coverage_status_from_score,
    first_available,
    mean,
    safe_float,
)


def centered_score(value: Optional[float]) -> float:
    number = safe_float(value)
    if number is None:
        return 0.0
    return clamp((number - 50.0) / 50.0, -1.0, 1.0)


def normalized_score(value: Optional[float]) -> Optional[float]:
    number = safe_float(value)
    if number is None:
        return None
    return clamp(50.0 + 50.0 * float(number), 0.0, 100.0)


def support_level(value: Optional[float], coverage: Optional[float]) -> Optional[float]:
    number = safe_float(value)
    if number is None:
        return None
    base = normalized_score(abs(number))
    if base is None:
        return None
    if coverage is None:
        return base
    return clamp(base * (0.55 + 0.45 * clamp(coverage, 0.0, 1.0)), 0.0, 100.0)


def coverage_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not payload:
        return None
    return coverage_score(payload)


def mean_defined(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [safe_float(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return None
    return mean(clean)


def count_statuses(payload: Optional[Dict[str, Any]], statuses: Sequence[str]) -> int:
    freshness = (payload or {}).get("freshness_summary") or {}
    wanted = set(statuses)
    return sum(1 for entry in freshness.values() if str(entry.get("status") or "") in wanted)


def freshness_penalty(quality_domain: Optional[Dict[str, Any]]) -> float:
    stale_but_usable = count_statuses(quality_domain, ["stale_but_usable"])
    stale = count_statuses(quality_domain, ["stale"])
    return stale_but_usable * 6.0 + stale * 11.0


def availability_penalty(domain_availability: Optional[Dict[str, Any]], label: str) -> float:
    entry = (domain_availability or {}).get(label) or {}
    status = str(entry.get("coverage_status") or "")
    if status in {"available", "fresh"}:
        return 0.0
    if status in {"partial", "mixed", "stale_but_usable"}:
        return 5.0
    if status in {"limited", "insufficient_history", "stale"}:
        return 10.0
    if status in {"unavailable", "missing"}:
        return 15.0
    return 0.0


def conviction_tier(score: Optional[float]) -> str:
    number = safe_float(score)
    if number is None:
        return "unknown"
    if number >= 80:
        return "very_high"
    if number >= 67:
        return "high"
    if number >= 52:
        return "moderate"
    if number >= 35:
        return "low"
    return "very_low"


def fragility_tier(score: Optional[float]) -> str:
    number = safe_float(score)
    if number is None:
        return "unknown"
    if number >= 75:
        return "elevated"
    if number >= 58:
        return "moderate"
    return "contained"


def confidence_quality_label(
    score: Optional[float],
    *,
    evidence_quality: Optional[float],
    fragility: Optional[float],
) -> str:
    confidence = safe_float(score) or 0.0
    evidence = safe_float(evidence_quality) or 0.0
    fragility_value = safe_float(fragility) or 0.0
    if evidence < 42:
        return "low_evidence"
    if fragility_value >= 68:
        return "fragile"
    if confidence >= 72 and evidence >= 62:
        return "well_supported"
    return "adequate"


def direction_label(raw_score: Optional[float], *, veto_effect: float = 0.0) -> str:
    number = safe_float(raw_score) or 0.0
    if veto_effect >= 0.55:
        return "veto"
    if number >= 0.12:
        return "supportive"
    if number <= -0.12:
        return "opposing"
    return "neutral"


def component_payload(
    *,
    name: str,
    raw_score: Optional[float],
    weight: float,
    coverage: Optional[float],
    rationale: Sequence[str],
    penalty_effect: float = 0.0,
    veto_effect: float = 0.0,
) -> Dict[str, Any]:
    normalized = normalized_score(raw_score)
    return {
        "name": name,
        "score": round(float(raw_score), 4) if raw_score is not None else None,
        "raw_component_score": round(float(raw_score), 4) if raw_score is not None else None,
        "normalized_score": round(float(normalized), 2) if normalized is not None else None,
        "weight": round(float(weight), 4),
        "support_level": round(float(support_level(raw_score, coverage)), 2)
        if support_level(raw_score, coverage) is not None
        else None,
        "coverage_score": round(float(coverage), 4) if coverage is not None else None,
        "coverage_status": coverage_status_from_score(coverage),
        "contribution_direction": direction_label(raw_score, veto_effect=veto_effect),
        "penalty_effect": round(float(penalty_effect), 4),
        "veto_effect": round(float(veto_effect), 4),
        "notes": [item for item in rationale if item],
    }


def compact_list(items: Iterable[Optional[str]]) -> List[str]:
    output: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def signal_bias(signal: Optional[Dict[str, Any]]) -> float:
    action = str((signal or {}).get("action") or "HOLD").upper()
    score = safe_float((signal or {}).get("score"))
    if score is not None:
        return clamp(score, -1.0, 1.0)
    if action == "BUY":
        return 0.25
    if action == "SELL":
        return -0.25
    return 0.0


def top_component_names(
    components: Dict[str, Dict[str, Any]],
    *,
    positive: bool = True,
    limit: int = 3,
) -> List[str]:
    sorted_items = sorted(
        components.values(),
        key=lambda item: (safe_float(item.get("score")) or 0.0) * (safe_float(item.get("weight")) or 0.0),
        reverse=positive,
    )
    output: List[str] = []
    for item in sorted_items:
        raw = safe_float(item.get("score")) or 0.0
        if positive and raw <= 0.08:
            continue
        if not positive and raw >= -0.08:
            continue
        output.append(str(item.get("name") or "").replace("_", " "))
        if len(output) >= limit:
            break
    return output


def first_text(*values: Any) -> Optional[str]:
    value = first_available(*values)
    if value in (None, ""):
        return None
    return str(value)
