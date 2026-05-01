from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .common import compact_list, sample_confidence, safe_float


def build_regime_conditioned_learnings(
    current_snapshot: Dict[str, Any],
    cohort: List[Dict[str, Any]],
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in cohort:
        grouped[str(row.get("regime_label") or "unknown")].append(row)

    learnings: List[Dict[str, Any]] = []
    for regime_label, rows in grouped.items():
        reliability = [safe_float(item.get("evaluation_reliability_score")) for item in rows]
        hit_rates = [safe_float(item.get("evaluation_hit_rate")) for item in rows]
        fragility = [safe_float(item.get("signal_fragility_index")) for item in rows]
        portfolio_scores = [safe_float(item.get("portfolio_candidate_score")) for item in rows]
        avg_reliability = (
            round(
                sum(value for value in reliability if value is not None)
                / max(1, len([value for value in reliability if value is not None])),
                2,
            )
            if any(value is not None for value in reliability)
            else None
        )
        avg_hit_rate = (
            round(
                sum(value for value in hit_rates if value is not None)
                / max(1, len([value for value in hit_rates if value is not None])),
                4,
            )
            if any(value is not None for value in hit_rates)
            else None
        )
        avg_fragility = (
            round(
                sum(value for value in fragility if value is not None)
                / max(1, len([value for value in fragility if value is not None])),
                2,
            )
            if any(value is not None for value in fragility)
            else None
        )
        avg_portfolio_score = (
            round(
                sum(value for value in portfolio_scores if value is not None)
                / max(1, len([value for value in portfolio_scores if value is not None])),
                2,
            )
            if any(value is not None for value in portfolio_scores)
            else None
        )
        learnings.append(
            {
                "regime_label": regime_label,
                "sample_size": len(rows),
                "sample_confidence": sample_confidence(len(rows)),
                "average_reliability": avg_reliability,
                "average_hit_rate": avg_hit_rate,
                "average_fragility": avg_fragility,
                "average_portfolio_score": avg_portfolio_score,
                "decision_quality_summary": (
                    "stronger decision quality"
                    if (avg_reliability or 0.0) >= 68.0 and (avg_hit_rate or 0.0) >= 0.58
                    else "mixed decision quality"
                    if (avg_reliability or 0.0) >= 55.0
                    else "weaker decision quality"
                ),
                "adaptation_suggestion": (
                    "allow cleaner trend-following and portfolio emphasis in this regime"
                    if (avg_reliability or 0.0) >= 68.0 and (avg_fragility or 100.0) <= 42.0
                    else "increase caution and favor confirmation-heavy logic in this regime"
                ),
            }
        )

    learnings.sort(
        key=lambda item: (
            float(item.get("average_reliability") or 0.0),
            float(item.get("average_hit_rate") or 0.0),
            item.get("sample_size") or 0,
        ),
        reverse=True,
    )

    strongest = learnings[:2]
    weakest = sorted(
        learnings,
        key=lambda item: (
            float(item.get("average_reliability") or 0.0),
            float(item.get("average_hit_rate") or 0.0),
        ),
    )[:2]
    current_regime = str(current_snapshot.get("regime_label") or "unknown")
    current_regime_note = next(
        (item for item in learnings if item.get("regime_label") == current_regime),
        None,
    )
    weakest_conditions = evaluation.get("weakest_conditions") or []
    current_regime_is_weak = any(
        item.get("dimension") == "regime_label" and item.get("label") == current_regime
        for item in weakest_conditions
    )
    summary = (
        f"Regime-conditioned learning currently sees the strongest cohort behavior in {compact_list((item.get('regime_label') for item in strongest), limit=2)}, "
        f"while the weakest conditions skew toward {compact_list((item.get('regime_label') for item in weakest), limit=2)}. "
        f"The active regime is {current_regime}, which is {'currently a weaker condition' if current_regime_is_weak else 'not currently flagged as a weak condition'}."
    )
    return {
        "regime_conditioned_learnings": learnings,
        "strongest_regimes": strongest,
        "weakest_regimes": weakest,
        "current_regime_note": current_regime_note,
        "current_regime_is_weak": current_regime_is_weak,
        "regime_learning_summary": summary,
    }
