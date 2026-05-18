from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from api.axiom.analytics import safe_float
from api.axiom.common import clamp, inverse_score, rounded


def _count_penalty(label: Optional[str], counts: Counter[str], *, weight: float) -> float:
    if not label:
        return 0.0
    extra = max(counts.get(label, 0) - 1, 0)
    return extra * weight


def rank_axiom_history_records(
    records: Sequence[Dict[str, Any]],
    *,
    current_holdings: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    prepared = [dict(record) for record in records]
    sector_counts = Counter(str(record.get("sector") or "unknown") for record in prepared)
    regime_counts = Counter(str(record.get("regime_label") or "unknown") for record in prepared)
    trade_family_counts = Counter(str(record.get("trade_family") or "none") for record in prepared)
    theme_counts = Counter(str(record.get("theme_tag") or "unknown") for record in prepared)
    holding_set = {str(symbol).upper() for symbol in (current_holdings or [])}
    ranked: List[Dict[str, Any]] = []

    for record in prepared:
        deployability = (record.get("evidence_backed_deployability") or {}).get(
            "deployability_tier"
        ) or record.get("deployability_tier")
        axiom_utility = safe_float(record.get("deployable_alpha_utility")) or 0.0
        fragility = safe_float(
            ((record.get("engine_scores") or {}).get("critical_fragility") or {}).get("score")
        ) or 100.0
        liquidity = safe_float(
            ((record.get("engine_scores") or {}).get("liquidity_convexity") or {}).get("score")
        ) or 50.0
        research = safe_float(
            ((record.get("engine_scores") or {}).get("research_integrity") or {}).get("score")
        ) or 50.0
        overlap_penalty = (
            _count_penalty(str(record.get("sector") or "unknown"), sector_counts, weight=6.0)
            + _count_penalty(str(record.get("regime_label") or "unknown"), regime_counts, weight=4.0)
            + _count_penalty(str(record.get("trade_family") or "none"), trade_family_counts, weight=5.0)
            + _count_penalty(str(record.get("theme_tag") or "unknown"), theme_counts, weight=3.0)
        )
        if str(record.get("symbol") or "").upper() in holding_set:
            overlap_penalty += 8.0
        fragility_penalty = max(0.0, fragility - 35.0) * 0.45
        liquidity_penalty = max(0.0, 58.0 - liquidity) * 0.52
        research_penalty = max(0.0, 62.0 - research) * 0.48
        final_score = clamp(
            axiom_utility
            - overlap_penalty
            - fragility_penalty
            - liquidity_penalty
            - research_penalty,
            0.0,
            100.0,
        )

        size_band = str(
            ((record.get("evidence_backed_deployability") or {}).get("size_band"))
            or record.get("size_band_recommendation")
            or "none"
        )
        if final_score < 35.0 or deployability == "not_actionable":
            size_band = "none"
        elif final_score < 52.0 or deployability in {"monitor_only", "paper_trade_only"}:
            size_band = "small"
        elif fragility > 40.0:
            size_band = "medium"
        elif size_band == "none":
            size_band = "large"

        fit_label = "watchlist_only"
        if deployability == "not_actionable" or final_score < 28.0:
            fit_label = "avoid"
        elif str(record.get("trade_family") or "none") == "convexity" and liquidity >= 52.0:
            fit_label = "convexity_overlay"
        elif deployability == "live_candidate" and final_score >= 72.0 and research >= 65.0:
            fit_label = "core_candidate"
        elif final_score >= 50.0:
            fit_label = "tactical_candidate"

        ranked.append(
            {
                **record,
                "portfolio_rank_score": rounded(final_score, digits=2),
                "overlap_penalty": rounded(overlap_penalty, digits=2),
                "fragility_penalty": rounded(fragility_penalty, digits=2),
                "liquidity_penalty": rounded(liquidity_penalty, digits=2),
                "research_penalty": rounded(research_penalty, digits=2),
                "final_size_band": size_band,
                "portfolio_fit_label": fit_label,
                "deployability_quality_score": rounded(
                    (axiom_utility * 0.45)
                    + (inverse_score(fragility) * 0.15)
                    + (liquidity * 0.2)
                    + (research * 0.2),
                    digits=2,
                ),
            }
        )

    ranked.sort(
        key=lambda item: (
            safe_float(item.get("portfolio_rank_score")) or 0.0,
            safe_float(item.get("deployability_quality_score")) or 0.0,
        ),
        reverse=True,
    )
    for index, item in enumerate(ranked, start=1):
        item["portfolio_rank"] = index
    return ranked
