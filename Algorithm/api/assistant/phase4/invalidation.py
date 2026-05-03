from __future__ import annotations

from typing import Any, Dict

from .common import compact_list, safe_float


def build_invalidation_map(
    *,
    job_context: Dict[str, Any],
    data_bundle: Dict[str, Any],
    feature_factor_bundle: Dict[str, Any],
    final_signal: str,
    strategy_posture: str,
    actionability_score: float,
) -> Dict[str, Any]:
    market = data_bundle.get("market_price_volume") or {}
    quality = data_bundle.get("quality_provenance") or {}
    market_structure = (
        feature_factor_bundle.get("market_structure")
        or feature_factor_bundle.get("multi_horizon_price_momentum")
        or {}
    )
    regime = feature_factor_bundle.get("regime_intelligence") or feature_factor_bundle.get("regime_engine") or {}
    sentiment = (
        feature_factor_bundle.get("sentiment_narrative_intelligence")
        or feature_factor_bundle.get("sentiment_intelligence")
        or {}
    )
    macro = feature_factor_bundle.get("macro_alignment") or feature_factor_bundle.get("macro_sensitivity") or {}
    relative = (
        feature_factor_bundle.get("cross_asset_relative_context")
        or feature_factor_bundle.get("relative_peer")
        or {}
    )
    event_risk = data_bundle.get("event_catalyst_risk") or {}
    liquidity = data_bundle.get("liquidity_execution_fragility") or {}
    breadth = data_bundle.get("market_breadth_internals") or {}
    cross_asset_depth = data_bundle.get("cross_asset_confirmation") or {}
    stress = data_bundle.get("stress_spillover_conditions") or {}
    agreement = feature_factor_bundle.get("domain_agreement") or {}
    composites = feature_factor_bundle.get("composite_intelligence") or {}

    support = safe_float(market.get("support_21d"))
    resistance = safe_float(market.get("resistance_21d"))
    regime_label = str(regime.get("regime_label") or "unknown")
    fragility_score = safe_float(composites.get("Signal Fragility Index")) or 50.0
    crowding_score = safe_float(composites.get("Narrative Crowding Index")) or 50.0
    macro_alignment = safe_float(composites.get("Macro Alignment Score")) or 50.0
    conflict_score = safe_float(agreement.get("domain_conflict_score")) or 0.0
    relative_strength = safe_float(relative.get("benchmark_relative_strength")) or 50.0
    breakout_follow_through = safe_float(market_structure.get("breakout_follow_through_score")) or 50.0
    regime_stability = safe_float(composites.get("Regime Stability Score")) or 50.0
    missingness = min(safe_float(quality.get("missingness")) or 0.0, 1.0)
    event_overhang = safe_float(event_risk.get("event_overhang_score")) or 0.0
    implementation_fragility = safe_float(liquidity.get("implementation_fragility_score")) or 0.0
    breadth_confirmation = safe_float(breadth.get("breadth_confirmation_score")) or 50.0
    cross_asset_conflict = safe_float(cross_asset_depth.get("cross_asset_conflict_score")) or 0.0
    market_stress = safe_float(stress.get("market_stress_score")) or 0.0

    regime_invalidators = compact_list(
        [
            f"A regime shift from {regime_label} into transition / high-volatility instability would invalidate the current posture."
            if regime_label in {"trend", "squeeze", "trending"}
            else "Failure to stabilize the current regime would keep the setup in a wait / no-trade state.",
            "A drop in regime stability below the current threshold would force a more defensive stance."
            if regime_stability >= 45
            else "Regime stability is already weak; another deterioration would void actionability.",
        ]
    )

    narrative_invalidators = compact_list(
        [
            "Narrative crowding rising without cleaner price confirmation would weaken the setup."
            if crowding_score >= 55
            else "A sudden jump in crowding or contradiction would reduce thesis quality.",
            "Positive narrative without price follow-through is already a risk and would invalidate further chase."
            if (safe_float(sentiment.get("positive_news_weak_price_divergence")) or 0.0) >= 50
            else None,
        ]
    )

    macro_invalidators = compact_list(
        [
            "Macro alignment flipping materially lower would invalidate the constructive read."
            if final_signal == "BUY"
            else "A renewed macro risk-on tailwind would undermine the current defensive read."
            if final_signal == "SELL"
            else "Macro alignment breaking lower would keep the setup from graduating out of HOLD / watchlist.",
            "Macro conflict or stress rising further would cut into confidence and actionability."
            if (safe_float(macro.get("macro_conflict_score")) or 0.0) >= 50
            else "A new macro conflict pulse would move the posture toward wait / no-trade.",
        ]
    )

    quality_invalidators = compact_list(
        [
            "Evidence freshness degrading to stale would invalidate the current confidence tier.",
            "Missingness rising materially above current levels would force the posture back toward observation only."
            if missingness >= 0.08
            else "A material increase in missingness would degrade the strategy layer quickly.",
            "A new event-overhang window would invalidate treating the setup as clean structural alpha."
            if event_overhang < 70
            else "Event overhang is already elevated; another catalyst pulse would keep the setup suppressed.",
            "Implementation fragility worsening would invalidate any attempt to treat the setup as fully deployable."
            if implementation_fragility >= 55
            else None,
        ]
    )

    top_invalidators = compact_list(
        [
            f"Loss of the near support zone around {support:.2f} would weaken the current constructive posture."
            if support is not None and final_signal == "BUY"
            else None,
            f"Recovery through the near resistance zone around {resistance:.2f} would undermine the current defensive posture."
            if resistance is not None and final_signal == "SELL"
            else None,
            regime_invalidators[0] if regime_invalidators else None,
            narrative_invalidators[0] if narrative_invalidators else None,
            macro_invalidators[0] if macro_invalidators else None,
            quality_invalidators[0] if quality_invalidators else None,
            "Breadth confirmation breaking lower would invalidate the current posture."
            if breadth_confirmation >= 45
            else "Breadth is already weak; another deterioration would reinforce the invalidation path.",
            "Cross-asset contradiction intensifying would invalidate the thesis."
            if cross_asset_conflict >= 55
            else None,
            "Market stress spilling higher would invalidate the setup even if stock-level structure still looks intact."
            if market_stress >= 55
            else None,
            "Relative strength turning decisively against the thesis would invalidate the current read."
            if relative_strength >= 50 and final_signal == "BUY"
            else "Relative weakness failing to persist would undercut the bearish read."
            if final_signal == "SELL"
            else "Relative context failing to improve would keep the posture capped.",
        ]
    )

    confirmation_triggers = compact_list(
        [
            "Cleaner breakout follow-through would improve the base case."
            if breakout_follow_through < 60
            else "Continuation with clean breakout follow-through would confirm the current posture.",
            "Cross-domain conflict falling would improve actionability."
            if conflict_score >= 45
            else "Cross-domain agreement staying firm would confirm the setup.",
            "Macro alignment staying constructive would support posture upgrade."
            if macro_alignment >= 50 and final_signal != "SELL"
            else "Macro drag fading would be required for any posture upgrade.",
            "Relative strength improving versus the benchmark would confirm idiosyncratic strength."
            if relative_strength < 58
            else "Relative strength staying above benchmark context would confirm the thesis.",
            "Event overhang needs to recede before the setup can be treated as cleaner actionability."
            if event_overhang >= 60
            else None,
        ]
    )

    deterioration_triggers = compact_list(
        [
            "Fragility crossing into a clearly elevated regime would force a downgrade."
            if fragility_score < 70
            else "Further fragility expansion would invalidate remaining actionability.",
            "Domain conflict climbing further would reduce confidence."
            if conflict_score >= 45
            else "A rise in domain conflict would shift the setup toward wait / watchlist.",
            "Relative momentum fading against the benchmark would weaken the setup.",
            "Freshness or coverage deterioration would lower the stance toward observation only.",
            "Breadth deterioration or narrow leadership would weaken the setup further."
            if breadth_confirmation < 60
            else None,
            "Stress and spillover rising would force a more defensive stance."
            if market_stress >= 55
            else None,
        ]
    )

    if actionability_score < 40:
        confirmation_triggers.append("Actionability must improve before the setup can leave observation mode.")

    if strategy_posture in {"watchlist_positive", "watchlist_negative", "wait"}:
        confirmation_triggers.append("A cleaner evidence-quality stack is needed before the setup becomes actionable.")

    return {
        "top_invalidators": top_invalidators[:5],
        "regime_invalidators": regime_invalidators,
        "narrative_invalidators": narrative_invalidators,
        "macro_invalidators": macro_invalidators,
        "quality_invalidators": quality_invalidators,
        "confirmation_triggers": confirmation_triggers[:6],
        "deterioration_triggers": deterioration_triggers[:6],
    }
