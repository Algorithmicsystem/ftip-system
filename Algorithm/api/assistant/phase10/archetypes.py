from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .common import compact_list, safe_float


def classify_archetype(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    signal = str(snapshot.get("final_signal") or "HOLD").upper()
    posture = str(snapshot.get("strategy_posture") or "wait")
    fragility = float(snapshot.get("signal_fragility_index") or 100.0)
    crowding = float(snapshot.get("narrative_crowding_index") or 0.0)
    opportunity = float(snapshot.get("opportunity_quality_score") or 0.0)
    macro_alignment = float(snapshot.get("macro_alignment_score") or 0.0)
    fundamentals = float(snapshot.get("fundamental_durability_score") or 0.0)
    regime_stability = float(snapshot.get("regime_stability_score") or 0.0)
    structure = float(snapshot.get("market_structure_integrity_score") or 0.0)
    conviction = str(snapshot.get("conviction_tier") or "low")
    regime_label = str(snapshot.get("regime_label") or "unknown")
    candidate_classification = str(snapshot.get("candidate_classification") or "watchlist_candidate")

    if signal == "SELL" or posture == "actionable_short":
        archetype_id = "structurally_weak_fade_candidate"
        name = "Structurally Weak Fade Candidate"
        summary = "Bearish posture with weak structure or high deterioration risk."
        failures = ["short squeezes", "macro risk-on reversal", "crowding unwind against the thesis"]
        caution = "high"
    elif regime_label == "squeeze" and structure >= 60.0 and fragility <= 45.0:
        archetype_id = "squeeze_expansion"
        name = "Squeeze Expansion"
        summary = "Compressed structure with constructive breakout potential if confirmation arrives."
        failures = ["false breakout", "volume confirmation failure", "regime instability shock"]
        caution = "moderate"
    elif opportunity >= 74.0 and fragility <= 40.0 and posture in {"actionable_long", "trend_continuation_candidate"}:
        archetype_id = "clean_high_quality_continuation"
        name = "Clean High-Quality Continuation"
        summary = "High-quality trend continuation with relatively clean structure and limited fragility."
        failures = ["crowding acceleration", "macro reversal", "relative-strength stall"]
        caution = "measured"
    elif macro_alignment >= 65.0 and fundamentals >= 65.0 and regime_stability >= 60.0:
        archetype_id = "macro_aligned_quality_trend"
        name = "Macro-Aligned Quality Trend"
        summary = "Quality and macro support are aligned, creating a more durable trend-style thesis."
        failures = ["macro support deteriorates", "quality premium fades", "valuation crowding"]
        caution = "measured"
    elif crowding >= 60.0 and signal == "BUY":
        archetype_id = "crowded_narrative_long"
        name = "Crowded Narrative Long"
        summary = "Narrative and attention are elevated, creating upside interest but also crowding risk."
        failures = ["novelty collapse", "price fails to confirm narrative", "macro de-rating"]
        caution = "high"
    elif fragility >= 60.0 and signal == "BUY":
        archetype_id = "fragile_momentum"
        name = "Fragile Momentum"
        summary = "Momentum remains present, but the setup is structurally fragile and prone to breakdown."
        failures = ["high-volatility reversal", "gap risk", "crowding without follow-through"]
        caution = "high"
    elif regime_label in {"transition", "unstable", "high_vol"} or regime_stability < 45.0:
        archetype_id = "regime_transition_setup"
        name = "Regime-Transition Setup"
        summary = "The setup is being shaped by unstable regime conditions and needs extra caution."
        failures = ["rapid trend reversal", "macro shock", "confidence collapse"]
        caution = "high"
    elif posture == "opportunistic_reversal" or conviction in {"very_low", "low"}:
        archetype_id = "low_conviction_mean_reversion"
        name = "Low-Conviction Mean Reversion"
        summary = "The setup relies on reversal logic and needs stronger evidence before it earns higher trust."
        failures = ["trend persists against the reversal", "macro conflict", "weak confirmation"]
        caution = "high"
    elif candidate_classification in {"watchlist_candidate", "no_trade_candidate", "redundant_candidate"}:
        archetype_id = "watchlist_only_thesis"
        name = "Watchlist-Only Thesis"
        summary = "The setup is analytically relevant but remains better suited to watchlist monitoring than deployment."
        failures = ["never earns confirmation", "portfolio overlap remains elevated", "freshness degrades before entry"]
        caution = "moderate"
    else:
        archetype_id = "trend_continuation"
        name = "Trend Continuation"
        summary = "A constructive directional setup that still depends on confirmation and context staying supportive."
        failures = ["trend quality slips", "fragility rises", "macro alignment fades"]
        caution = "moderate"

    return {
        "archetype_id": archetype_id,
        "archetype_name": name,
        "summary": summary,
        "defining_characteristics": compact_list(
            [
                f"signal={signal}",
                f"posture={posture}",
                f"fragility={fragility:.1f}",
                f"crowding={crowding:.1f}",
                f"macro={macro_alignment:.1f}",
                f"structure={structure:.1f}",
            ],
            limit=6,
        ),
        "common_failure_modes": compact_list(failures, limit=4),
        "best_regimes": compact_list([regime_label if regime_label != "unknown" else "trend"], limit=2),
        "worst_regimes": compact_list(
            ["transition", "high_vol", "crowded"] if caution == "high" else ["unstable"], limit=3
        ),
        "strategy_fit": posture,
        "deployment_caution_level": caution,
    }


def summarize_archetype_library(archetyped_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cohorts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in archetyped_rows:
        archetype = (row.get("setup_archetype") or {}).get("archetype_id") or "unknown"
        cohorts[archetype].append(row)

    summaries: List[Dict[str, Any]] = []
    for archetype_id, rows in cohorts.items():
        first = rows[0].get("setup_archetype") or {}
        reliability = [
            safe_float(item.get("evaluation_reliability_score")) for item in rows
        ]
        hit_rates = [safe_float(item.get("evaluation_hit_rate")) for item in rows]
        portfolio_scores = [safe_float(item.get("portfolio_candidate_score")) for item in rows]
        summaries.append(
            {
                "archetype_id": archetype_id,
                "archetype_name": first.get("archetype_name") or archetype_id.replace("_", " "),
                "sample_count": len(rows),
                "average_reliability": round(
                    sum(value for value in reliability if value is not None) / max(1, len([v for v in reliability if v is not None])),
                    2,
                )
                if any(value is not None for value in reliability)
                else None,
                "average_hit_rate": round(
                    sum(value for value in hit_rates if value is not None) / max(1, len([v for v in hit_rates if v is not None])),
                    4,
                )
                if any(value is not None for value in hit_rates)
                else None,
                "average_portfolio_score": round(
                    sum(value for value in portfolio_scores if value is not None)
                    / max(1, len([v for v in portfolio_scores if v is not None])),
                    2,
                )
                if any(value is not None for value in portfolio_scores)
                else None,
                "common_failure_modes": first.get("common_failure_modes") or [],
                "deployment_caution_level": first.get("deployment_caution_level"),
            }
        )

    summaries.sort(
        key=lambda item: (
            float(item.get("average_reliability") or 0.0),
            float(item.get("average_portfolio_score") or 0.0),
            item.get("sample_count") or 0,
        ),
        reverse=True,
    )
    return {
        "archetype_cohorts": summaries,
    }
