from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .common import clamp


def pairwise_overlap(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    sector_overlap = 100.0 if left.get("sector") == right.get("sector") else 24.0
    benchmark_overlap = 100.0 if left.get("benchmark_proxy") == right.get("benchmark_proxy") else 28.0
    regime_overlap = 100.0 if left.get("regime_label") == right.get("regime_label") else 34.0
    macro_overlap = clamp(
        100.0 - abs(float(left.get("macro_alignment_score") or 0.0) - float(right.get("macro_alignment_score") or 0.0)) * 1.2,
        0.0,
        100.0,
    )
    style_overlap = 84.0 if left.get("primary_participant_fit") == right.get("primary_participant_fit") else 36.0
    signal_overlap = 88.0 if left.get("final_signal") == right.get("final_signal") else 42.0
    theme_overlap = 82.0 if left.get("theme_tag") == right.get("theme_tag") else 28.0
    correlation_proxy = round(
        clamp(
            (sector_overlap * 0.24)
            + (benchmark_overlap * 0.22)
            + (regime_overlap * 0.12)
            + (macro_overlap * 0.15)
            + (style_overlap * 0.12)
            + (signal_overlap * 0.08)
            + (theme_overlap * 0.07),
            0.0,
            100.0,
        ),
        2,
    )
    redundancy_score = round(
        clamp(
            (correlation_proxy * 0.62)
            + ((100.0 - abs(float(left.get("opportunity_quality_score") or 0.0) - float(right.get("opportunity_quality_score") or 0.0))) * 0.12)
            + ((100.0 - abs(float(left.get("confidence_score") or 0.0) - float(right.get("confidence_score") or 0.0))) * 0.10)
            + (signal_overlap * 0.16),
            0.0,
            100.0,
        ),
        2,
    )
    diversification_contribution_score = round(
        clamp(
            100.0 - (correlation_proxy * 0.58) - (redundancy_score * 0.22)
            + (12.0 if left.get("final_signal") != right.get("final_signal") else 0.0),
            0.0,
            100.0,
        ),
        2,
    )
    return {
        "peer_symbol": right.get("symbol"),
        "sector_overlap": sector_overlap,
        "benchmark_overlap": benchmark_overlap,
        "regime_overlap": regime_overlap,
        "macro_overlap": macro_overlap,
        "style_overlap": style_overlap,
        "theme_overlap": theme_overlap,
        "correlation_proxy": correlation_proxy,
        "overlap_score": correlation_proxy,
        "redundancy_score": redundancy_score,
        "diversification_contribution_score": diversification_contribution_score,
        "cluster_membership": [
            f"sector:{left.get('sector')}",
            f"benchmark:{left.get('benchmark_proxy')}",
            f"regime:{left.get('regime_label')}",
            f"macro:{left.get('macro_state')}",
            f"participant:{left.get('primary_participant_fit')}",
        ],
        "exposure_family": "|".join(
            [
                str(left.get("sector") or "unknown"),
                str(left.get("benchmark_proxy") or "unknown"),
                str(left.get("macro_state") or "mixed"),
                str(left.get("final_signal") or "neutral"),
            ]
        ),
    }


def summarize_overlap(
    current: Dict[str, Any],
    peers: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    comparisons = [pairwise_overlap(current, peer) for peer in peers if peer.get("symbol") != current.get("symbol")]
    comparisons.sort(key=lambda item: (item.get("overlap_score") or 0.0), reverse=True)
    top = comparisons[:5]
    if not top:
        return (
            {
                "overlap_score": 18.0,
                "redundancy_score": 16.0,
                "diversification_contribution_score": 72.0,
                "cluster_membership": [f"sector:{current.get('sector')}", f"benchmark:{current.get('benchmark_proxy')}"],
                "exposure_family": "|".join(
                    [
                        str(current.get("sector") or "unknown"),
                        str(current.get("benchmark_proxy") or "unknown"),
                        str(current.get("macro_state") or "mixed"),
                        str(current.get("final_signal") or "neutral"),
                    ]
                ),
                "most_redundant_symbol": None,
            },
            [],
        )

    return (
        {
            "overlap_score": round(sum(item["overlap_score"] for item in top) / len(top), 2),
            "redundancy_score": round(max(item["redundancy_score"] for item in top), 2),
            "diversification_contribution_score": round(
                sum(item["diversification_contribution_score"] for item in top) / len(top),
                2,
            ),
            "cluster_membership": top[0]["cluster_membership"],
            "exposure_family": top[0]["exposure_family"],
            "most_redundant_symbol": top[0].get("peer_symbol"),
        },
        top,
    )
