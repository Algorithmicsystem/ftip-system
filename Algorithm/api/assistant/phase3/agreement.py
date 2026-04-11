from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .common import clamp, first_available, mean


def _signal_direction(signal: Optional[Dict[str, Any]]) -> int:
    action = str((signal or {}).get("action") or "").upper()
    if action == "BUY":
        return 1
    if action == "SELL":
        return -1
    return 0


def build_domain_agreement(
    *,
    signal: Optional[Dict[str, Any]],
    domain_views: Sequence[Dict[str, Any]],
    narrative_crowding_index: Optional[float],
    regime_label: str,
    regime_instability: Optional[float],
    signal_confidence: Optional[float],
) -> Dict[str, Any]:
    active = [dict(item) for item in domain_views if item.get("score") is not None]
    if not active:
        return {
            "domain_views": [],
            "domain_agreement_score": None,
            "domain_conflict_score": None,
            "confidence_penalty_from_conflict": None,
            "strongest_confirming_domains": [],
            "strongest_conflicting_domains": [],
            "agreement_flags": [],
            "consensus_direction": "neutral",
        }

    weighted_sum = sum(float(item["score"]) * float(item.get("coverage") or 1.0) for item in active)
    absolute_sum = sum(abs(float(item["score"])) * float(item.get("coverage") or 1.0) for item in active)
    agreement_ratio = abs(weighted_sum) / absolute_sum if absolute_sum else 0.0
    conflict_ratio = 1.0 - agreement_ratio

    target_direction = _signal_direction(signal)
    if target_direction == 0:
        target_direction = 1 if weighted_sum > 0 else -1 if weighted_sum < 0 else 0
    consensus_direction = "bullish" if target_direction > 0 else "bearish" if target_direction < 0 else "neutral"

    confirmers = [
        {
            "domain": item["domain"],
            "score": round(float(item["score"]), 4),
            "coverage": round(float(item.get("coverage") or 0.0), 4),
            "detail": item.get("detail"),
        }
        for item in sorted(active, key=lambda row: abs(float(row["score"])), reverse=True)
        if target_direction != 0 and float(item["score"]) * target_direction > 0.12
    ]
    conflicters = [
        {
            "domain": item["domain"],
            "score": round(float(item["score"]), 4),
            "coverage": round(float(item.get("coverage") or 0.0), 4),
            "detail": item.get("detail"),
        }
        for item in sorted(active, key=lambda row: abs(float(row["score"])), reverse=True)
        if target_direction != 0 and float(item["score"]) * target_direction < -0.12
    ]

    by_domain = {item["domain"]: float(item["score"]) for item in active}
    agreement_flags: List[str] = []
    if by_domain.get("market_structure", 0.0) > 0.2 and by_domain.get("fundamentals", 0.0) < -0.2:
        agreement_flags.append("technical strong / fundamentals weak")
    if by_domain.get("sentiment", 0.0) > 0.2 and by_domain.get("market_structure", 0.0) < -0.1:
        agreement_flags.append("sentiment strong / price weak")
    if by_domain.get("macro", 0.0) > 0.2 and by_domain.get("relative_context", 0.0) < -0.2:
        agreement_flags.append("macro supportive / stock-specific weak")
    if by_domain.get("fundamentals", 0.0) > 0.2 and (narrative_crowding_index or 0.0) >= 65:
        agreement_flags.append("fundamentals solid / narrative crowded")
    if regime_label in {"transition", "high_vol"} and (signal_confidence or 0.0) >= 0.65:
        agreement_flags.append("regime unstable / signal strength high")
    if agreement_ratio >= 0.7:
        agreement_flags.append("many domains agree")
    if conflict_ratio >= 0.55:
        agreement_flags.append("many domains conflict")

    disagreement_intensity = mean(
        [
            100.0 * conflict_ratio,
            regime_instability,
            narrative_crowding_index,
            100.0 * (len(conflicters) / max(len(active), 1)),
        ]
    )
    confidence_penalty = None
    if disagreement_intensity is not None:
        confidence_penalty = clamp(
            0.45 * disagreement_intensity
            + 10.0 * max(len(conflicters) - len(confirmers), 0),
            0.0,
            100.0,
        )

    agreement_score = None
    conflict_score = None
    if agreement_ratio is not None:
        agreement_score = clamp(
            100.0 * agreement_ratio * mean(item.get("coverage") for item in active) if active else 0.0,
            0.0,
            100.0,
        )
        conflict_score = clamp(
            first_available(disagreement_intensity, 100.0 * conflict_ratio),
            0.0,
            100.0,
        )

    return {
        "domain_views": [
            {
                "domain": item["domain"],
                "score": round(float(item["score"]), 4),
                "coverage": round(float(item.get("coverage") or 0.0), 4),
                "detail": item.get("detail"),
            }
            for item in active
        ],
        "consensus_direction": consensus_direction,
        "domain_agreement_score": round(float(agreement_score), 2) if agreement_score is not None else None,
        "domain_conflict_score": round(float(conflict_score), 2) if conflict_score is not None else None,
        "confidence_penalty_from_conflict": round(float(confidence_penalty), 2) if confidence_penalty is not None else None,
        "strongest_confirming_domains": confirmers[:4],
        "strongest_conflicting_domains": conflicters[:4],
        "agreement_flags": agreement_flags[:6],
    }

