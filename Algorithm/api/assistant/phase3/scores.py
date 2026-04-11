from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .common import clamp, component_support, coverage_status_from_score, mean, weighted_average


def compose_score(
    label: str,
    *,
    components: Sequence[Dict[str, Any]],
    penalties: Optional[Sequence[Dict[str, Any]]] = None,
    domain_support: Optional[Sequence[Optional[float]]] = None,
    staleness_penalty: float = 0.0,
    penalty_scale: float = 0.18,
    coverage_penalty_scale: float = 18.0,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    penalties = list(penalties or [])
    base_score = weighted_average(components)
    penalty_score = weighted_average(penalties)
    support = mean(
        [component_support(components, penalties)]
        + [value for value in (domain_support or []) if value is not None]
    )
    support = clamp(support or 0.0, 0.0, 1.0)

    final_score = None
    coverage_penalty = (1.0 - support) * coverage_penalty_scale
    if base_score is not None:
        final_score = float(base_score)
        if higher_is_better:
            if penalty_score is not None:
                final_score -= penalty_score * penalty_scale
            final_score -= coverage_penalty
            final_score -= staleness_penalty
        else:
            if penalty_score is not None:
                final_score += penalty_score * penalty_scale
            final_score += coverage_penalty * 0.75
            final_score += staleness_penalty * 0.5
        final_score = clamp(final_score, 0.0, 100.0)

    return {
        "label": label,
        "score": round(final_score, 2) if final_score is not None else None,
        "raw_base_score": round(float(base_score), 2) if base_score is not None else None,
        "penalty_score": round(float(penalty_score), 2) if penalty_score is not None else None,
        "staleness_penalty": round(float(staleness_penalty), 2),
        "coverage_penalty": round(float(coverage_penalty), 2),
        "coverage_score": round(float(support), 4),
        "coverage_status": coverage_status_from_score(support),
        "higher_is_better": higher_is_better,
        "formula": {
            "method": "weighted_average_with_penalties",
            "penalty_scale": penalty_scale,
            "coverage_penalty_scale": coverage_penalty_scale,
        },
        "components": list(components),
        "penalties": penalties,
    }

