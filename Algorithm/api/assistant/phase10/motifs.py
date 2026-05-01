from __future__ import annotations

from typing import Any, Dict, List

from .common import sample_confidence, safe_float


def _motif_templates() -> List[Dict[str, Any]]:
    return [
        {
            "motif_id": "strong_cross_domain_confirmation",
            "motif_summary": "strong cross-domain confirmation with low fragility",
            "conditions": "cross-domain conviction >= 65, domain agreement >= 65, fragility <= 40",
            "predicate": lambda row: (
                (safe_float(row.get("cross_domain_conviction_score")) or 0.0) >= 65.0
                and (safe_float(row.get("domain_agreement_score")) or 0.0) >= 65.0
                and (safe_float(row.get("signal_fragility_index")) or 100.0) <= 40.0
            ),
            "failure_modes": ["macro reversal", "late-cycle crowding", "regime transition"],
        },
        {
            "motif_id": "low_fragility_quality_drift",
            "motif_summary": "quality and structure drift higher without obvious fragility",
            "conditions": "opportunity quality >= 70, fundamentals >= 65, fragility <= 40",
            "predicate": lambda row: (
                (safe_float(row.get("opportunity_quality_score")) or 0.0) >= 70.0
                and (safe_float(row.get("fundamental_durability_score")) or 0.0) >= 65.0
                and (safe_float(row.get("signal_fragility_index")) or 100.0) <= 40.0
            ),
            "failure_modes": ["valuation crowding", "macro shock", "execution slippage during fast breakouts"],
        },
        {
            "motif_id": "crowding_divergence",
            "motif_summary": "crowding builds faster than structural quality or confirmation",
            "conditions": "crowding >= 60 and portfolio fit <= 55 or fragility >= 55",
            "predicate": lambda row: (
                (safe_float(row.get("narrative_crowding_index")) or 0.0) >= 60.0
                and (
                    (safe_float(row.get("portfolio_fit_quality")) or 100.0) <= 55.0
                    or (safe_float(row.get("signal_fragility_index")) or 0.0) >= 55.0
                )
            ),
            "failure_modes": ["attention collapse", "false breakout", "narrative exhaustion"],
        },
        {
            "motif_id": "macro_fragile_false_breakout",
            "motif_summary": "constructive structure is offset by macro conflict and rising fragility",
            "conditions": "market structure >= 58, macro alignment <= 42, fragility >= 50",
            "predicate": lambda row: (
                (safe_float(row.get("market_structure_integrity_score")) or 0.0) >= 58.0
                and (safe_float(row.get("macro_alignment_score")) or 100.0) <= 42.0
                and (safe_float(row.get("signal_fragility_index")) or 0.0) >= 50.0
            ),
            "failure_modes": ["macro spillover", "benchmark weakness", "regime instability"],
        },
    ]


def build_motif_library(
    current_snapshot: Dict[str, Any],
    cohort: List[Dict[str, Any]],
) -> Dict[str, Any]:
    library: List[Dict[str, Any]] = []
    active_motifs: List[Dict[str, Any]] = []
    for template in _motif_templates():
        matches = [row for row in cohort if template["predicate"](row)]
        sample_count = len(matches)
        if sample_count == 0:
            continue
        avg_reliability = (
            sum(safe_float(row.get("evaluation_reliability_score")) or 0.0 for row in matches)
            / sample_count
        )
        avg_hit_rate = (
            sum(safe_float(row.get("evaluation_hit_rate")) or 0.0 for row in matches)
            / sample_count
        )
        motif_strength = round(
            max(
                0.0,
                min(
                    100.0,
                    (avg_reliability * 0.45)
                    + (avg_hit_rate * 100.0 * 0.25)
                    + (sample_count * 5.0),
                ),
            ),
            2,
        )
        motif = {
            "motif_id": template["motif_id"],
            "motif_summary": template["motif_summary"],
            "motif_conditions": template["conditions"],
            "motif_strength": motif_strength,
            "motif_reliability": round(avg_reliability, 2),
            "motif_failure_modes": template["failure_modes"],
            "motif_sample_count": sample_count,
            "motif_validation_status": (
                "validated"
                if sample_count >= 8 and avg_reliability >= 60.0
                else "under_review"
                if sample_count >= 4
                else "exploratory"
            ),
            "sample_confidence": sample_confidence(sample_count),
        }
        library.append(motif)
        if template["predicate"](current_snapshot):
            active_motifs.append(motif)

    library.sort(key=lambda item: float(item.get("motif_strength") or 0.0), reverse=True)
    active_motifs.sort(key=lambda item: float(item.get("motif_strength") or 0.0), reverse=True)
    return {
        "active_motifs": active_motifs[:4],
        "motif_library": library[:10],
    }
