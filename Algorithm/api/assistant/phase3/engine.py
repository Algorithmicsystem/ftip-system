from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agreement import build_domain_agreement
from .common import (
    as_percentile_score,
    bounded_score,
    clamp,
    component,
    coverage_score,
    first_available,
    freshness_penalty,
    group_meta,
    inverse_metric,
    mean,
    ratio,
    safe_float,
    scaled_score,
)
from .regime import classify_regime
from .scores import compose_score


_SECTOR_RATE_SENSITIVITY = {
    "technology": 78.0,
    "communication services": 72.0,
    "consumer cyclical": 68.0,
    "real estate": 82.0,
    "utilities": 52.0,
    "financial services": 60.0,
    "financial": 60.0,
    "healthcare": 55.0,
    "industrials": 58.0,
    "energy": 42.0,
    "consumer defensive": 40.0,
    "consumer staples": 40.0,
    "materials": 48.0,
}


def _domain_support(*payloads: Optional[Dict[str, Any]]) -> List[Optional[float]]:
    return [coverage_score(payload) for payload in payloads]


def _return_direction_score(*values: Optional[float]) -> Optional[float]:
    clean = [safe_float(value) for value in values if safe_float(value) is not None]
    if not clean:
        return None
    return bounded_score(mean(clean), low=-0.2, high=0.2)


def _market_structure_group(
    market: Dict[str, Any],
    technical: Dict[str, Any],
    relative: Dict[str, Any],
) -> Dict[str, Any]:
    return_stack = {
        "1d": first_available(market.get("day_return"), market.get("ret_1d")),
        "3d": market.get("ret_3d"),
        "5d": market.get("ret_5d"),
        "10d": market.get("ret_10d"),
        "21d": market.get("ret_21d"),
        "63d": market.get("ret_63d"),
        "126d": market.get("ret_126d"),
        "252d": market.get("ret_252d"),
    }
    volatility_adjusted_return_stack = {
        "5d": ratio(return_stack.get("5d"), market.get("realized_vol_21d")),
        "21d": ratio(return_stack.get("21d"), market.get("realized_vol_21d")),
        "63d": ratio(return_stack.get("63d"), market.get("realized_vol_63d")),
        "126d": ratio(return_stack.get("126d"), market.get("realized_vol_126d")),
        "252d": ratio(return_stack.get("252d"), market.get("realized_vol_252d")),
    }

    horizon_alignment = [
        1.0 if value > 0 else 0.0
        for value in return_stack.values()
        if value is not None
    ]
    momentum_consistency_score = bounded_score(mean(horizon_alignment), low=0.0, high=1.0)
    trend_quality_score = mean(
        [
            bounded_score(abs(safe_float(technical.get("trend_slope_21d")) or 0.0), low=0.0, high=0.25),
            bounded_score(abs(safe_float(technical.get("trend_slope_63d")) or 0.0), low=0.0, high=0.15),
            bounded_score(abs(safe_float(technical.get("trend_curvature")) or 0.0), low=0.0, high=0.12, invert=True),
            bounded_score(abs(safe_float(technical.get("ma_stack_alignment")) or 0.0), low=0.0, high=1.0),
        ]
    )
    breakout_follow_through_score = mean(
        [
            bounded_score(market.get("breakout_distance_63d"), low=-0.15, high=0.05),
            bounded_score(technical.get("mean_reversion_gap"), low=-0.1, high=0.12),
            bounded_score(technical.get("volume_price_alignment"), low=-0.2, high=0.2),
            bounded_score(market.get("range_position_21d"), low=0.0, high=1.0),
        ]
    )

    reversal_pressure_score = mean(
        [
            bounded_score(abs((return_stack.get("5d") or 0.0) - (return_stack.get("21d") or 0.0)), low=0.0, high=0.18),
            bounded_score(abs((return_stack.get("3d") or 0.0) - (return_stack.get("10d") or 0.0)), low=0.0, high=0.12),
            bounded_score(abs(safe_float(technical.get("trend_curvature")) or 0.0), low=0.0, high=0.08),
        ]
    )
    trend_exhaustion_score = mean(
        [
            bounded_score(max((return_stack.get("5d") or 0.0) - (return_stack.get("21d") or 0.0), 0.0), low=0.0, high=0.12),
            bounded_score(max(technical.get("mean_reversion_gap") or 0.0, 0.0), low=0.0, high=0.12),
            bounded_score(market.get("volume_anomaly"), low=1.0, high=3.0),
        ]
    )
    price_volume_alignment_score = bounded_score(technical.get("volume_price_alignment"), low=-0.2, high=0.2)
    participation_quality_score = mean(
        [
            bounded_score(market.get("volume_anomaly"), low=0.5, high=2.5),
            bounded_score(market.get("up_down_volume_ratio_21d"), low=0.6, high=1.8),
            price_volume_alignment_score,
        ]
    )
    range_compression_score = bounded_score(market.get("compression_ratio"), low=0.02, high=0.18, invert=True)
    range_expansion_score = bounded_score(market.get("range_expansion_ratio"), low=0.0, high=1.5)
    support_resistance_pressure_score = bounded_score(market.get("range_position_21d"), low=0.0, high=1.0)
    benchmark_relative_strength_score = first_available(
        bounded_score(relative.get("benchmark_relative_strength"), low=-0.2, high=0.2),
        safe_float(relative.get("market_relative_behavior")),
        bounded_score(relative.get("relative_ret_21d"), low=-0.2, high=0.2),
    )
    sector_relative_strength_score = first_available(
        as_percentile_score(relative.get("relative_strength_percentile")),
        bounded_score(relative.get("relative_momentum"), low=-0.25, high=0.25),
    )
    directional_persistence_score = mean(
        [
            as_percentile_score(market.get("positive_day_ratio_10d")),
            as_percentile_score(market.get("positive_day_ratio_21d")),
            as_percentile_score(market.get("positive_day_ratio_63d")),
        ]
    )

    components = [
        component("momentum_consistency", momentum_consistency_score, 0.18, "Cross-horizon return stack agreement."),
        component("trend_quality", trend_quality_score, 0.22, "Slope quality and curvature discipline."),
        component("breakout_follow_through", breakout_follow_through_score, 0.16, "Breakout behavior and follow-through."),
        component("price_volume_alignment", price_volume_alignment_score, 0.12, "Price move quality versus participation."),
        component("participation_quality", participation_quality_score, 0.14, "Volume confirmation and participation depth."),
        component("support_resistance_pressure", support_resistance_pressure_score, 0.08, "Location within the active range."),
        component("benchmark_relative_strength", benchmark_relative_strength_score, 0.10, "Strength versus market context."),
    ]
    penalties = [
        component("reversal_pressure", reversal_pressure_score, 0.5, "Short and medium horizons are diverging."),
        component("trend_exhaustion", trend_exhaustion_score, 0.5, "Extension and exhaustion pressure."),
    ]

    return {
        "return_stack": return_stack,
        "volatility_adjusted_return_stack": volatility_adjusted_return_stack,
        "trend_slope_21d": technical.get("trend_slope_21d"),
        "trend_slope_63d": technical.get("trend_slope_63d"),
        "trend_curvature": technical.get("trend_curvature"),
        "trend_quality_score": round(trend_quality_score, 2) if trend_quality_score is not None else None,
        "momentum_consistency_score": round(momentum_consistency_score, 2) if momentum_consistency_score is not None else None,
        "breakout_follow_through_score": round(breakout_follow_through_score, 2) if breakout_follow_through_score is not None else None,
        "reversal_pressure_score": round(reversal_pressure_score, 2) if reversal_pressure_score is not None else None,
        "reversal_pressure_proxy": round(reversal_pressure_score, 2) if reversal_pressure_score is not None else None,
        "trend_exhaustion_score": round(trend_exhaustion_score, 2) if trend_exhaustion_score is not None else None,
        "exhaustion_score": round(trend_exhaustion_score, 2) if trend_exhaustion_score is not None else None,
        "price_volume_alignment_score": round(price_volume_alignment_score, 2) if price_volume_alignment_score is not None else None,
        "price_volume_alignment": technical.get("volume_price_alignment"),
        "participation_quality_score": round(participation_quality_score, 2) if participation_quality_score is not None else None,
        "range_compression_score": round(range_compression_score, 2) if range_compression_score is not None else None,
        "range_expansion_score": round(range_expansion_score, 2) if range_expansion_score is not None else None,
        "support_resistance_pressure_score": round(support_resistance_pressure_score, 2) if support_resistance_pressure_score is not None else None,
        "benchmark_relative_strength_score": round(benchmark_relative_strength_score, 2) if benchmark_relative_strength_score is not None else None,
        "sector_relative_strength_score": round(sector_relative_strength_score, 2) if sector_relative_strength_score is not None else None,
        "directional_persistence_score": round(directional_persistence_score, 2) if directional_persistence_score is not None else None,
        "components": components,
        "penalties": penalties,
        "meta": group_meta(
            coverage_inputs=_domain_support(market, technical, relative),
            components=components + penalties,
            note="Market structure factors are built from multi-horizon returns, price/volume behavior, and relative context.",
        ),
    }


def _fragility_group(
    market: Dict[str, Any],
    quality_domain: Dict[str, Any],
    domain_conflict_hint: Optional[float] = None,
) -> Dict[str, Any]:
    vol_ladder = {
        "5d": market.get("realized_vol_5d"),
        "10d": market.get("realized_vol_10d"),
        "21d": market.get("realized_vol_21d"),
        "63d": market.get("realized_vol_63d"),
        "126d": market.get("realized_vol_126d"),
        "252d": market.get("realized_vol_252d"),
    }
    volatility_stress_score = mean(
        [
            bounded_score(vol_ladder.get("21d"), low=0.12, high=0.6),
            bounded_score(vol_ladder.get("63d"), low=0.12, high=0.5),
            bounded_score(vol_ladder.get("126d"), low=0.12, high=0.45),
        ]
    )
    vol_of_vol_score = bounded_score(market.get("vol_of_vol_proxy"), low=0.0, high=1.0)
    downside_asymmetry_score = first_available(
        bounded_score(market.get("downside_asymmetry_21d"), low=0.0, high=2.0),
        bounded_score(market.get("downside_asymmetry_63d"), low=0.0, high=2.0),
    )
    drawdown_sensitivity_score = mean(
        [
            bounded_score(abs(market.get("maxdd_21d") or 0.0), low=0.0, high=0.18),
            bounded_score(abs(market.get("maxdd_63d") or 0.0), low=0.0, high=0.3),
            bounded_score(abs(market.get("maxdd_126d") or 0.0), low=0.0, high=0.4),
        ]
    )
    gap_instability_score = bounded_score(
        first_available(market.get("gap_instability_10d"), abs(market.get("gap_pct") or 0.0)),
        low=0.0,
        high=0.04,
    )
    return_dispersion_score = mean(
        [
            bounded_score(market.get("return_dispersion_10d"), low=0.0, high=0.05),
            bounded_score(market.get("return_dispersion_21d"), low=0.0, high=0.045),
            bounded_score(market.get("return_dispersion_63d"), low=0.0, high=0.04),
        ]
    )
    instability_score = mean(
        [
            volatility_stress_score,
            vol_of_vol_score,
            gap_instability_score,
            return_dispersion_score,
        ]
    )
    anomaly_pressure_score = mean(
        [
            instability_score,
            bounded_score(safe_float(quality_domain.get("missingness")), low=0.0, high=0.2),
            bounded_score(len(quality_domain.get("anomaly_flags") or []), low=0.0, high=4.0),
            domain_conflict_hint,
        ]
    )
    clean_setup_score = mean(
        [
            100.0 - (instability_score or 50.0),
            100.0 - (anomaly_pressure_score or 50.0),
            100.0 - (downside_asymmetry_score or 50.0),
        ]
    )
    noisy_setup_score = mean(
        [
            instability_score,
            anomaly_pressure_score,
            gap_instability_score,
        ]
    )

    degradation_triggers: List[str] = []
    if (instability_score or 0.0) >= 65:
        degradation_triggers.append("instability score is elevated")
    if (gap_instability_score or 0.0) >= 65:
        degradation_triggers.append("gap instability is elevated")
    if (drawdown_sensitivity_score or 0.0) >= 60:
        degradation_triggers.append("drawdown sensitivity is elevated")
    if safe_float(quality_domain.get("missingness")) is not None and float(quality_domain["missingness"]) >= 0.12:
        degradation_triggers.append("coverage missingness is degrading confidence")
    if domain_conflict_hint is not None and domain_conflict_hint >= 60:
        degradation_triggers.append("domain conflict is compounding setup fragility")

    components = [
        component("realized_volatility", volatility_stress_score, 0.24, "Realized volatility ladder."),
        component("vol_of_vol", vol_of_vol_score, 0.18, "Volatility instability."),
        component("downside_asymmetry", downside_asymmetry_score, 0.16, "Downside dominates upside behavior."),
        component("drawdown_sensitivity", drawdown_sensitivity_score, 0.16, "Drawdown sensitivity across horizons."),
        component("gap_instability", gap_instability_score, 0.14, "Gap behavior instability."),
        component("return_dispersion", return_dispersion_score, 0.12, "Dispersion of daily returns."),
    ]

    return {
        "realized_volatility_ladder": vol_ladder,
        "volatility_stress_score": round(volatility_stress_score, 2) if volatility_stress_score is not None else None,
        "vol_of_vol_score": round(vol_of_vol_score, 2) if vol_of_vol_score is not None else None,
        "vol_of_vol_proxy": market.get("vol_of_vol_proxy"),
        "downside_asymmetry_score": round(downside_asymmetry_score, 2) if downside_asymmetry_score is not None else None,
        "drawdown_sensitivity_score": round(drawdown_sensitivity_score, 2) if drawdown_sensitivity_score is not None else None,
        "gap_instability_score": round(gap_instability_score, 2) if gap_instability_score is not None else None,
        "return_dispersion_score": round(return_dispersion_score, 2) if return_dispersion_score is not None else None,
        "instability_score": round(instability_score, 2) if instability_score is not None else None,
        "fragility_score": round(instability_score, 2) if instability_score is not None else None,
        "anomaly_pressure_score": round(anomaly_pressure_score, 2) if anomaly_pressure_score is not None else None,
        "clean_setup_score": round(clean_setup_score, 2) if clean_setup_score is not None else None,
        "noisy_setup_score": round(noisy_setup_score, 2) if noisy_setup_score is not None else None,
        "confidence_degradation_triggers": degradation_triggers,
        "components": components,
        "meta": group_meta(
            coverage_inputs=_domain_support(market, quality_domain),
            components=components,
            note="Fragility factors are built from realized volatility, instability, drawdown, dispersion, and anomaly pressure.",
        ),
    }


def _sentiment_group(
    sentiment: Dict[str, Any],
    market: Dict[str, Any],
    geopolitical: Dict[str, Any],
) -> Dict[str, Any]:
    sentiment_level = first_available(
        safe_float(sentiment.get("sentiment_score")),
        safe_float((sentiment.get("sentiment_summary") or {}).get("bias")),
        safe_float(sentiment.get("aggregated_sentiment_bias")),
    )
    sentiment_level_score = bounded_score(sentiment_level, low=-0.45, high=0.45)
    sentiment_trend_score = bounded_score(sentiment.get("sentiment_trend"), low=-0.08, high=0.08)
    narrative_concentration_score = bounded_score(sentiment.get("narrative_concentration"), low=0.12, high=0.8)
    attention_intensity_score = first_available(
        safe_float(sentiment.get("attention_score")),
        bounded_score(sentiment.get("attention_crowding"), low=0.5, high=3.0),
    )
    novelty_score = first_available(
        safe_float(sentiment.get("novelty_score")),
        bounded_score(sentiment.get("novelty_ratio"), low=0.1, high=1.0),
    )
    repetition_score = first_available(
        safe_float(sentiment.get("repetition_score")),
        100.0 - novelty_score if novelty_score is not None else None,
    )
    persistence_score = first_available(
        safe_float(sentiment.get("persistence_score")),
        repetition_score,
    )
    contradiction_raw = safe_float(sentiment.get("contradiction_score"))
    contradiction_score = first_available(
        contradiction_raw if contradiction_raw is not None and contradiction_raw > 1.0 else None,
        bounded_score(
            contradiction_raw if contradiction_raw is not None and contradiction_raw <= 1.0 else sentiment.get("disagreement_score"),
            low=0.0,
            high=0.7,
        ),
    )
    hype_divergence_score = bounded_score(abs(sentiment.get("hype_price_divergence") or 0.0), low=0.0, high=0.45)
    event_pressure_score = first_available(
        bounded_score((sentiment.get("event_overlay") or {}).get("gdelt_article_count"), low=0.0, high=6.0),
        bounded_score(geopolitical.get("exogenous_event_score"), low=0.0, high=1.0),
    )
    crowding_proxy_score = mean(
        [
            attention_intensity_score,
            narrative_concentration_score,
            repetition_score,
            hype_divergence_score,
            contradiction_score,
        ]
    )
    positive_news_weak_price_divergence = bounded_score(
        max(sentiment_level or 0.0, 0.0) * max((0.03 - (market.get("ret_5d") or 0.0)), 0.0) * 15.0,
        low=0.0,
        high=0.25,
    )
    negative_news_resilient_price_divergence = bounded_score(
        max(-(sentiment_level or 0.0), 0.0) * max((market.get("ret_5d") or 0.0) + 0.02, 0.0) * 15.0,
        low=0.0,
        high=0.25,
    )
    sentiment_direction_score = mean(
        [
            sentiment_level_score,
            sentiment_trend_score,
            100.0 - (crowding_proxy_score or 50.0),
            negative_news_resilient_price_divergence,
        ]
    )

    components = [
        component("sentiment_level", sentiment_level_score, 0.2, "Aggregate sentiment level."),
        component("sentiment_trend", sentiment_trend_score, 0.1, "Trend in sentiment over recent history."),
        component("attention_intensity", attention_intensity_score, 0.18, "Attention intensity and amplification."),
        component("narrative_concentration", narrative_concentration_score, 0.14, "Concentration of the narrative tape."),
        component("repetition", repetition_score, 0.14, "Repetition and persistence of the same narrative."),
        component("contradiction", contradiction_score, 0.12, "Narrative disagreement across headlines."),
        component("hype_divergence", hype_divergence_score, 0.12, "Narrative intensity relative to price behavior."),
    ]

    return {
        "sentiment_level": sentiment_level,
        "sentiment_level_score": round(sentiment_level_score, 2) if sentiment_level_score is not None else None,
        "sentiment_trend": sentiment.get("sentiment_trend"),
        "sentiment_trend_score": round(sentiment_trend_score, 2) if sentiment_trend_score is not None else None,
        "sentiment_confidence": sentiment.get("sentiment_confidence"),
        "attention_intensity_score": round(attention_intensity_score, 2) if attention_intensity_score is not None else None,
        "attention_score": sentiment.get("attention_score"),
        "novelty_score": round(novelty_score, 2) if novelty_score is not None else None,
        "repetition_score": round(repetition_score, 2) if repetition_score is not None else None,
        "persistence_score": round(persistence_score, 2) if persistence_score is not None else None,
        "narrative_concentration_score": round(narrative_concentration_score, 2) if narrative_concentration_score is not None else None,
        "narrative_concentration": sentiment.get("narrative_concentration"),
        "contradiction_score": round(contradiction_score, 2) if contradiction_score is not None else None,
        "disagreement_score": sentiment.get("disagreement_score"),
        "hype_to_price_divergence_score": round(hype_divergence_score, 2) if hype_divergence_score is not None else None,
        "hype_to_price_divergence": sentiment.get("hype_price_divergence"),
        "crowding_proxy_score": round(crowding_proxy_score, 2) if crowding_proxy_score is not None else None,
        "event_pressure_score": round(event_pressure_score, 2) if event_pressure_score is not None else None,
        "positive_news_weak_price_divergence": round(positive_news_weak_price_divergence, 2) if positive_news_weak_price_divergence is not None else None,
        "negative_news_resilient_price_divergence": round(negative_news_resilient_price_divergence, 2) if negative_news_resilient_price_divergence is not None else None,
        "sentiment_direction_score": round(sentiment_direction_score, 2) if sentiment_direction_score is not None else None,
        "components": components,
        "meta": group_meta(
            coverage_inputs=_domain_support(sentiment, geopolitical),
            components=components,
            note="Narrative factors are built from sentiment level, attention, novelty, repetition, contradiction, and event pressure.",
        ),
    }


def _fundamental_group(
    fundamentals: Dict[str, Any],
    quality_domain: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = fundamentals.get("normalized_metrics") or {}
    quality = fundamentals.get("quality_proxies") or {}
    growth_quality_score = first_available(
        safe_float((fundamentals.get("durability_proxies") or {}).get("growth_quality")),
        bounded_score(first_available(metrics.get("revenue_growth_yoy"), fundamentals.get("revenue_growth_yoy")), low=-0.25, high=0.35),
    )
    profitability_quality_score = mean(
        [
            safe_float(quality.get("profitability_strength")),
            bounded_score(metrics.get("operating_margin"), low=-0.05, high=0.35),
            bounded_score(metrics.get("net_margin"), low=-0.05, high=0.3),
            bounded_score(metrics.get("return_on_equity"), low=0.0, high=0.35),
        ]
    )
    balance_sheet_resilience_score = first_available(
        safe_float(quality.get("balance_sheet_resilience")),
        mean(
            [
                bounded_score(metrics.get("current_ratio"), low=0.8, high=2.5),
                bounded_score(metrics.get("cash_ratio"), low=0.2, high=1.5),
                bounded_score(inverse_metric(metrics.get("debt_to_equity"), cap=3.0), low=0.0, high=1.0),
                bounded_score(inverse_metric(metrics.get("liabilities_to_assets"), cap=1.0), low=0.0, high=1.0),
            ]
        ),
    )
    liquidity_resilience_score = mean(
        [
            bounded_score(metrics.get("current_ratio"), low=0.8, high=2.5),
            bounded_score(metrics.get("cash_ratio"), low=0.2, high=1.5),
        ]
    )
    leverage_pressure_score = mean(
        [
            bounded_score(metrics.get("debt_to_equity"), low=0.0, high=3.0),
            bounded_score(metrics.get("liabilities_to_assets"), low=0.2, high=0.9),
        ]
    )
    cash_flow_durability_score = first_available(
        safe_float(quality.get("cash_flow_durability")),
        mean(
            [
                bounded_score(first_available(metrics.get("positive_fcf_ratio"), fundamentals.get("positive_fcf_ratio")), low=0.0, high=1.0),
                bounded_score(metrics.get("free_cash_flow_margin"), low=-0.05, high=0.3),
                bounded_score(metrics.get("free_cash_flow"), low=0.0, high=max(abs(metrics.get("free_cash_flow") or 1.0), 1.0)),
            ]
        ),
    )
    filing_recency_score = first_available(
        safe_float(quality.get("filing_recency_score")),
        bounded_score(365.0 - min(float(fundamentals.get("filing_recency_days") or 365), 365.0), low=0.0, high=365.0),
    )
    reporting_quality_score = first_available(
        safe_float(quality.get("reporting_quality_proxy")),
        safe_float(quality.get("reporting_completeness_score")),
        bounded_score((fundamentals.get("coverage_score") or 0.0), low=0.0, high=1.0),
    )

    components = [
        component("growth_quality", growth_quality_score, 0.16, "Growth quality and consistency."),
        component("profitability_quality", profitability_quality_score, 0.18, "Margin and profitability quality."),
        component("balance_sheet_resilience", balance_sheet_resilience_score, 0.18, "Balance-sheet resilience."),
        component("liquidity_resilience", liquidity_resilience_score, 0.12, "Near-term liquidity resilience."),
        component("cash_flow_durability", cash_flow_durability_score, 0.16, "Cash-flow durability."),
        component("filing_recency", filing_recency_score, 0.1, "Freshness of filing evidence."),
        component("reporting_quality", reporting_quality_score, 0.1, "Reporting quality and completeness."),
    ]

    return {
        "growth_quality_score": round(growth_quality_score, 2) if growth_quality_score is not None else None,
        "profitability_quality_score": round(profitability_quality_score, 2) if profitability_quality_score is not None else None,
        "balance_sheet_resilience_score": round(balance_sheet_resilience_score, 2) if balance_sheet_resilience_score is not None else None,
        "liquidity_resilience_score": round(liquidity_resilience_score, 2) if liquidity_resilience_score is not None else None,
        "leverage_pressure_score": round(leverage_pressure_score, 2) if leverage_pressure_score is not None else None,
        "cash_flow_durability_score": round(cash_flow_durability_score, 2) if cash_flow_durability_score is not None else None,
        "filing_recency_score": round(filing_recency_score, 2) if filing_recency_score is not None else None,
        "reporting_quality_score": round(reporting_quality_score, 2) if reporting_quality_score is not None else None,
        "components": components,
        "meta": group_meta(
            coverage_inputs=_domain_support(fundamentals, quality_domain),
            components=components,
            note="Fundamental durability factors are built from growth, profitability, balance-sheet, liquidity, cash-flow, and reporting quality evidence.",
        ),
    }


def _macro_group(
    symbol_meta: Dict[str, Any],
    macro: Dict[str, Any],
    geopolitical: Dict[str, Any],
    market: Dict[str, Any],
) -> Dict[str, Any]:
    sector = str(symbol_meta.get("sector") or "").strip().lower()
    rates_sensitivity_proxy = _SECTOR_RATE_SENSITIVITY.get(sector, 55.0)
    risk_on_risk_off_alignment = mean(
        [
            first_available(
                safe_float(macro.get("macro_alignment_score")),
                bounded_score((market.get("ret_21d") or 0.0) - (macro.get("benchmark_ret_21d") or 0.0), low=-0.15, high=0.15),
            ),
            bounded_score(macro.get("risk_on_score"), low=-0.08, high=0.08),
        ]
    )
    growth_alignment_score = first_available(
        safe_float(macro.get("growth_alignment_score")),
        bounded_score(macro.get("risk_on_score"), low=-0.08, high=0.08),
    )
    macro_regime_consistency = mean(
        [
            safe_float(macro.get("macro_alignment_score")),
            growth_alignment_score,
            risk_on_risk_off_alignment,
            bounded_score(-(macro.get("stress_overlay") or 0.0), low=-0.08, high=0.08),
        ]
    )
    macro_conflict_score = mean(
        [
            bounded_score(abs((macro.get("benchmark_ret_21d") or 0.0) - (market.get("ret_21d") or 0.0)), low=0.0, high=0.18),
            bounded_score(max(macro.get("stress_overlay") or 0.0, 0.0), low=0.0, high=0.12),
            bounded_score(geopolitical.get("exogenous_event_score"), low=0.0, high=1.0),
        ]
    )
    macro_fragility_score = mean(
        [
            bounded_score(max(macro.get("stress_overlay") or 0.0, 0.0), low=0.0, high=0.12),
            bounded_score(geopolitical.get("exogenous_event_score"), low=0.0, high=1.0),
            macro_conflict_score,
        ]
    )
    inflation_stress_proxy = first_available(
        safe_float(macro.get("inflation_stress_proxy")),
        mean(
            [
                bounded_score(rates_sensitivity_proxy, low=30.0, high=90.0),
                bounded_score(max(macro.get("stress_overlay") or 0.0, 0.0), low=0.0, high=0.12),
            ]
        ),
    )

    components = [
        component("macro_alignment", safe_float(macro.get("macro_alignment_score")), 0.34, "Primary macro alignment score."),
        component("growth_alignment", growth_alignment_score, 0.2, "Growth backdrop alignment."),
        component("risk_on_alignment", risk_on_risk_off_alignment, 0.22, "Risk-on/risk-off alignment."),
        component("macro_regime_consistency", macro_regime_consistency, 0.24, "Consistency across macro regime signals."),
    ]
    penalties = [
        component("macro_conflict", macro_conflict_score, 0.55, "Macro signals are internally conflicting."),
        component("macro_fragility", macro_fragility_score, 0.45, "Macro and event fragility are elevated."),
    ]

    return {
        "macro_alignment_score": safe_float(macro.get("macro_alignment_score")),
        "macro_fragility_score": round(macro_fragility_score, 2) if macro_fragility_score is not None else None,
        "rates_sensitivity_proxy": round(rates_sensitivity_proxy, 2),
        "inflation_stress_proxy": round(inflation_stress_proxy, 2) if inflation_stress_proxy is not None else None,
        "growth_alignment_score": round(growth_alignment_score, 2) if growth_alignment_score is not None else None,
        "risk_on_risk_off_alignment": round(risk_on_risk_off_alignment, 2) if risk_on_risk_off_alignment is not None else None,
        "macro_regime_consistency": round(macro_regime_consistency, 2) if macro_regime_consistency is not None else None,
        "macro_conflict_score": round(macro_conflict_score, 2) if macro_conflict_score is not None else None,
        "macro_stress_fragility": round(macro_fragility_score, 2) if macro_fragility_score is not None else None,
        "risk_on_score": macro.get("risk_on_score"),
        "components": components,
        "penalties": penalties,
        "meta": group_meta(
            coverage_inputs=_domain_support(macro, geopolitical),
            components=components + penalties,
            note="Macro alignment factors are probabilistic and coverage-aware rather than causal claims.",
        ),
    }


def _cross_asset_group(
    relative: Dict[str, Any],
    macro: Dict[str, Any],
    market: Dict[str, Any],
) -> Dict[str, Any]:
    benchmark_relative_strength = first_available(
        bounded_score(relative.get("benchmark_relative_strength"), low=-0.18, high=0.18),
        safe_float(relative.get("market_relative_behavior")),
        bounded_score((market.get("ret_21d") or 0.0) - (macro.get("benchmark_ret_21d") or 0.0), low=-0.18, high=0.18),
    )
    sector_relative_strength = first_available(
        as_percentile_score(relative.get("relative_strength_percentile")),
        bounded_score(relative.get("relative_momentum"), low=-0.2, high=0.2),
    )
    market_relative_momentum = first_available(
        bounded_score(relative.get("relative_momentum"), low=-0.25, high=0.25),
        benchmark_relative_strength,
    )
    cross_asset_divergence_score = mean(
        [
            bounded_score(abs((market.get("ret_21d") or 0.0) - (macro.get("benchmark_ret_21d") or 0.0)), low=0.0, high=0.18),
            bounded_score(abs(relative.get("relative_ret_21d") or 0.0), low=0.0, high=0.2),
        ]
    )
    sector_confirmation_score = mean(
        [
            sector_relative_strength,
            bounded_score(relative.get("sector_median_ret_21d"), low=-0.12, high=0.12),
        ]
    )
    relative_context_quality = mean(
        [
            safe_float((relative.get("meta") or {}).get("coverage_score")) * 100.0 if safe_float((relative.get("meta") or {}).get("coverage_score")) is not None else None,
            bounded_score(relative.get("peer_count"), low=0.0, high=12.0),
        ]
    )
    benchmark_relative_raw = first_available(
        relative.get("relative_ret_21d"),
        (market.get("ret_21d") or 0.0) - (macro.get("benchmark_ret_21d") or 0.0),
    )
    idiosyncratic_strength_vs_market = bounded_score(max(benchmark_relative_raw or 0.0, 0.0), low=0.0, high=0.18)
    idiosyncratic_weakness_vs_market = bounded_score(max(-(benchmark_relative_raw or 0.0), 0.0), low=0.0, high=0.18)

    components = [
        component("benchmark_relative_strength", benchmark_relative_strength, 0.28, "Strength relative to the benchmark."),
        component("sector_relative_strength", sector_relative_strength, 0.28, "Strength relative to the sector set."),
        component("market_relative_momentum", market_relative_momentum, 0.18, "Relative momentum versus the market."),
        component("sector_confirmation", sector_confirmation_score, 0.14, "Whether the sector is confirming the move."),
        component("relative_context_quality", relative_context_quality, 0.12, "Quality of the relative comparison set."),
    ]

    return {
        "benchmark_relative_strength": round(benchmark_relative_strength, 2) if benchmark_relative_strength is not None else None,
        "sector_relative_strength": round(sector_relative_strength, 2) if sector_relative_strength is not None else None,
        "market_relative_momentum": round(market_relative_momentum, 2) if market_relative_momentum is not None else None,
        "cross_asset_divergence_score": round(cross_asset_divergence_score, 2) if cross_asset_divergence_score is not None else None,
        "sector_confirmation_score": round(sector_confirmation_score, 2) if sector_confirmation_score is not None else None,
        "relative_context_quality": round(relative_context_quality, 2) if relative_context_quality is not None else None,
        "idiosyncratic_strength_vs_market": round(idiosyncratic_strength_vs_market, 2) if idiosyncratic_strength_vs_market is not None else None,
        "idiosyncratic_weakness_vs_market": round(idiosyncratic_weakness_vs_market, 2) if idiosyncratic_weakness_vs_market is not None else None,
        "relative_strength_percentile": sector_relative_strength,
        "components": components,
        "meta": group_meta(
            coverage_inputs=_domain_support(relative, macro),
            components=components,
            note="Cross-asset context separates beta, sector drift, and idiosyncratic strength or weakness.",
        ),
    }


def build_feature_factor_bundle(
    *,
    data_bundle: Dict[str, Any],
    signal: Dict[str, Any],
    key_features: Dict[str, Any],
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    market = data_bundle.get("market_price_volume") or {}
    technical = data_bundle.get("technical_market_structure") or {}
    fundamentals = data_bundle.get("fundamental_filing") or {}
    sentiment = data_bundle.get("sentiment_narrative_flow") or {}
    macro = data_bundle.get("macro_cross_asset") or {}
    geopolitical = data_bundle.get("geopolitical_policy") or {}
    relative = data_bundle.get("relative_context") or {}
    quality_domain = data_bundle.get("quality_provenance") or {}
    symbol_meta = data_bundle.get("symbol_meta") or {}

    market_structure = _market_structure_group(market, technical, relative)
    sentiment_group = _sentiment_group(sentiment, market, geopolitical)
    macro_group = _macro_group(symbol_meta, macro, geopolitical, market)
    cross_asset_group = _cross_asset_group(relative, macro, market)
    fundamental_group = _fundamental_group(fundamentals, quality_domain)
    initial_fragility = _fragility_group(market, quality_domain)

    regime_group = classify_regime(
        trend_quality_score=market_structure.get("trend_quality_score"),
        momentum_consistency_score=market_structure.get("momentum_consistency_score"),
        directional_persistence_score=market_structure.get("directional_persistence_score"),
        breakout_readiness_score=mean(
            [
                market_structure.get("range_compression_score"),
                market_structure.get("breakout_follow_through_score"),
                100.0 - (initial_fragility.get("instability_score") or 50.0),
            ]
        ),
        squeeze_score=market_structure.get("range_compression_score"),
        chop_score=mean(
            [
                market_structure.get("reversal_pressure_score"),
                100.0 - (market_structure.get("trend_quality_score") or 50.0),
                100.0 - (market_structure.get("momentum_consistency_score") or 50.0),
            ]
        ),
        volatility_stress_score=initial_fragility.get("volatility_stress_score"),
        instability_score=initial_fragility.get("instability_score"),
        transition_risk_score=mean(
            [
                market_structure.get("reversal_pressure_score"),
                initial_fragility.get("instability_score"),
                bounded_score(abs(technical.get("trend_curvature") or 0.0), low=0.0, high=0.08),
            ]
        ),
    )

    domain_views = [
        {
            "domain": "market_structure",
            "score": mean(
                [
                    scaled_score(_return_direction_score(market.get("ret_5d"), market.get("ret_21d"), market.get("ret_63d"))),
                    scaled_score(market_structure.get("trend_quality_score")),
                    scaled_score(market_structure.get("momentum_consistency_score")),
                ]
            ),
            "coverage": coverage_score({"meta": market_structure.get("meta")}),
            "detail": "Price structure, momentum persistence, and participation quality.",
        },
        {
            "domain": "fundamentals",
            "score": scaled_score(
                mean(
                    [
                        fundamental_group.get("growth_quality_score"),
                        fundamental_group.get("profitability_quality_score"),
                        100.0 - (fundamental_group.get("leverage_pressure_score") or 50.0),
                    ]
                )
            ),
            "coverage": coverage_score({"meta": fundamental_group.get("meta")}),
            "detail": "Growth, profitability, leverage, and reporting quality.",
        },
        {
            "domain": "sentiment",
            "score": scaled_score(sentiment_group.get("sentiment_direction_score")),
            "coverage": coverage_score({"meta": sentiment_group.get("meta")}),
            "detail": "Narrative tone adjusted for crowding and contradiction.",
        },
        {
            "domain": "macro",
            "score": scaled_score(
                mean(
                    [
                        macro_group.get("macro_alignment_score"),
                        macro_group.get("growth_alignment_score"),
                        100.0 - (macro_group.get("macro_conflict_score") or 50.0),
                    ]
                )
            ),
            "coverage": coverage_score({"meta": macro_group.get("meta")}),
            "detail": "Macro alignment, growth backdrop, and regime consistency.",
        },
        {
            "domain": "relative_context",
            "score": scaled_score(
                mean(
                    [
                        cross_asset_group.get("benchmark_relative_strength"),
                        cross_asset_group.get("sector_relative_strength"),
                        cross_asset_group.get("sector_confirmation_score"),
                    ]
                )
            ),
            "coverage": coverage_score({"meta": cross_asset_group.get("meta")}),
            "detail": "Benchmark-relative, sector-relative, and idiosyncratic context.",
        },
        {
            "domain": "fragility",
            "score": scaled_score(100.0 - (initial_fragility.get("instability_score") or 50.0)),
            "coverage": coverage_score({"meta": initial_fragility.get("meta")}),
            "detail": "Clean setup versus noisy or fragile setup.",
        },
    ]

    domain_agreement = build_domain_agreement(
        signal=signal,
        domain_views=domain_views,
        narrative_crowding_index=sentiment_group.get("crowding_proxy_score"),
        regime_label=str(regime_group.get("regime_label") or "unknown"),
        regime_instability=regime_group.get("regime_instability"),
        signal_confidence=safe_float(signal.get("confidence")),
    )

    fragility_group = _fragility_group(
        market,
        quality_domain,
        domain_conflict_hint=domain_agreement.get("domain_conflict_score"),
    )

    staleness = freshness_penalty(quality_domain)
    market_structure_score = compose_score(
        "Market Structure Integrity Score",
        components=market_structure.get("components") or [],
        penalties=market_structure.get("penalties") or [],
        domain_support=_domain_support(market, technical, relative),
        staleness_penalty=staleness * 0.35,
    )
    regime_stability_score = compose_score(
        "Regime Stability Score",
        components=[
            component("regime_confidence", regime_group.get("regime_confidence"), 0.34, "Confidence in the detected regime."),
            component("trend_quality", regime_group.get("trend_quality"), 0.22, "Trend quality within the regime."),
            component("directional_persistence", regime_group.get("directional_persistence"), 0.22, "Directional persistence."),
            component("breakout_readiness", regime_group.get("breakout_readiness"), 0.22, "Readiness for continuation."),
        ],
        penalties=[
            component("regime_instability", regime_group.get("regime_instability"), 0.6, "Instability inside the regime."),
            component("transition_risk", regime_group.get("transition_risk"), 0.4, "Risk of transition or break."),
        ],
        domain_support=_domain_support(market, technical),
        staleness_penalty=staleness * 0.25,
    )
    signal_fragility_index = compose_score(
        "Signal Fragility Index",
        components=[
            component("volatility_stress", fragility_group.get("volatility_stress_score"), 0.22, "Realized volatility stress."),
            component("vol_of_vol", fragility_group.get("vol_of_vol_score"), 0.16, "Volatility instability."),
            component("downside_asymmetry", fragility_group.get("downside_asymmetry_score"), 0.15, "Downside asymmetry."),
            component("drawdown_sensitivity", fragility_group.get("drawdown_sensitivity_score"), 0.15, "Drawdown sensitivity."),
            component("gap_instability", fragility_group.get("gap_instability_score"), 0.12, "Gap instability."),
            component("anomaly_pressure", fragility_group.get("anomaly_pressure_score"), 0.1, "Anomaly pressure."),
            component("domain_conflict", domain_agreement.get("domain_conflict_score"), 0.1, "Domain conflict compounds fragility."),
        ],
        domain_support=_domain_support(market, quality_domain),
        staleness_penalty=staleness * 0.2,
        higher_is_better=False,
        penalty_scale=0.0,
    )
    narrative_crowding_index = compose_score(
        "Narrative Crowding Index",
        components=[
            component("attention_intensity", sentiment_group.get("attention_intensity_score"), 0.22, "Attention intensity."),
            component("narrative_concentration", sentiment_group.get("narrative_concentration_score"), 0.16, "Narrative concentration."),
            component("repetition", sentiment_group.get("repetition_score"), 0.15, "Narrative repetition."),
            component("contradiction", sentiment_group.get("contradiction_score"), 0.12, "Narrative contradiction."),
            component("hype_divergence", sentiment_group.get("hype_to_price_divergence_score"), 0.2, "Narrative versus price divergence."),
            component("event_pressure", sentiment_group.get("event_pressure_score"), 0.15, "Event pressure."),
        ],
        domain_support=_domain_support(sentiment, geopolitical),
        higher_is_better=False,
        penalty_scale=0.0,
    )
    fundamental_durability_score = compose_score(
        "Fundamental Durability Score",
        components=fundamental_group.get("components") or [],
        penalties=[
            component("leverage_pressure", fundamental_group.get("leverage_pressure_score"), 1.0, "Leverage pressure."),
        ],
        domain_support=_domain_support(fundamentals, quality_domain),
        staleness_penalty=staleness * 0.25,
    )
    macro_alignment_score = compose_score(
        "Macro Alignment Score",
        components=macro_group.get("components") or [],
        penalties=macro_group.get("penalties") or [],
        domain_support=_domain_support(macro, geopolitical),
        staleness_penalty=staleness * 0.2,
    )
    cross_domain_conviction_score = compose_score(
        "Cross-Domain Conviction Score",
        components=[
            component("domain_agreement", domain_agreement.get("domain_agreement_score"), 0.26, "Agreement across domains."),
            component("market_structure_integrity", market_structure_score.get("score"), 0.18, "Market structure integrity."),
            component("regime_stability", regime_stability_score.get("score"), 0.14, "Regime stability."),
            component("fundamental_durability", fundamental_durability_score.get("score"), 0.14, "Fundamental durability."),
            component("macro_alignment", macro_alignment_score.get("score"), 0.14, "Macro alignment."),
            component("relative_context", cross_asset_group.get("relative_context_quality"), 0.14, "Relative context quality."),
        ],
        penalties=[
            component("domain_conflict", domain_agreement.get("domain_conflict_score"), 0.45, "Conflict across domains."),
            component("fragility", signal_fragility_index.get("score"), 0.35, "Fragility pressure."),
            component("crowding", narrative_crowding_index.get("score"), 0.2, "Narrative crowding."),
        ],
        domain_support=_domain_support(market, fundamentals, sentiment, macro, relative, quality_domain),
        staleness_penalty=staleness * 0.3,
    )
    opportunity_quality_score = compose_score(
        "Opportunity Quality Score",
        components=[
            component("market_structure_integrity", market_structure_score.get("score"), 0.2, "Structural quality."),
            component("regime_stability", regime_stability_score.get("score"), 0.12, "Regime stability."),
            component("fundamental_durability", fundamental_durability_score.get("score"), 0.14, "Fundamental durability."),
            component("macro_alignment", macro_alignment_score.get("score"), 0.1, "Macro alignment."),
            component("cross_domain_conviction", cross_domain_conviction_score.get("score"), 0.24, "Cross-domain conviction."),
            component("clean_setup", fragility_group.get("clean_setup_score"), 0.1, "Clean setup versus noise."),
            component("idiosyncratic_strength", cross_asset_group.get("idiosyncratic_strength_vs_market"), 0.1, "Idiosyncratic strength versus market."),
        ],
        penalties=[
            component("signal_fragility", signal_fragility_index.get("score"), 0.45, "Fragility and instability."),
            component("narrative_crowding", narrative_crowding_index.get("score"), 0.25, "Crowding and hype."),
            component("macro_fragility", macro_group.get("macro_fragility_score"), 0.15, "Macro fragility."),
            component("domain_conflict", domain_agreement.get("domain_conflict_score"), 0.15, "Domain conflict."),
        ],
        domain_support=_domain_support(market, fundamentals, sentiment, macro, relative, quality_domain),
        staleness_penalty=staleness * 0.4,
    )

    proprietary_scores = {
        market_structure_score["label"]: market_structure_score,
        regime_stability_score["label"]: regime_stability_score,
        signal_fragility_index["label"]: signal_fragility_index,
        narrative_crowding_index["label"]: narrative_crowding_index,
        fundamental_durability_score["label"]: fundamental_durability_score,
        macro_alignment_score["label"]: macro_alignment_score,
        cross_domain_conviction_score["label"]: cross_domain_conviction_score,
        opportunity_quality_score["label"]: opportunity_quality_score,
    }
    composite_intelligence = {
        label: detail.get("score")
        for label, detail in proprietary_scores.items()
    }

    factor_groups = {
        "market_structure": market_structure,
        "regime_intelligence": {**regime_group, "meta": group_meta(
            coverage_inputs=_domain_support(market, technical),
            components=regime_group.get("components") or [],
            note="Regime intelligence is explainable and built from persistence, compression, instability, and transition risk.",
        )},
        "fragility_intelligence": fragility_group,
        "sentiment_narrative_intelligence": sentiment_group,
        "fundamental_durability": fundamental_group,
        "macro_alignment": macro_group,
        "cross_asset_relative_context": cross_asset_group,
    }

    conviction_components = {
        "agreement": domain_agreement.get("domain_agreement_score"),
        "conflict": domain_agreement.get("domain_conflict_score"),
        "structure": market_structure_score.get("score"),
        "regime": regime_stability_score.get("score"),
        "fundamentals": fundamental_durability_score.get("score"),
        "macro": macro_alignment_score.get("score"),
        "relative": cross_asset_group.get("relative_context_quality"),
    }
    opportunity_quality_components = {
        "structure": market_structure_score.get("score"),
        "regime": regime_stability_score.get("score"),
        "fundamentals": fundamental_durability_score.get("score"),
        "macro": macro_alignment_score.get("score"),
        "conviction": cross_domain_conviction_score.get("score"),
        "clean_setup": fragility_group.get("clean_setup_score"),
        "idiosyncratic_strength": cross_asset_group.get("idiosyncratic_strength_vs_market"),
        "fragility_penalty": signal_fragility_index.get("score"),
        "crowding_penalty": narrative_crowding_index.get("score"),
        "conflict_penalty": domain_agreement.get("domain_conflict_score"),
    }

    return {
        "factor_groups": factor_groups,
        "market_structure": market_structure,
        "regime_intelligence": factor_groups["regime_intelligence"],
        "fragility_intelligence": fragility_group,
        "sentiment_narrative_intelligence": sentiment_group,
        "fundamental_durability": fundamental_group,
        "macro_alignment": macro_group,
        "cross_asset_relative_context": cross_asset_group,
        "domain_agreement": domain_agreement,
        "proprietary_scores": proprietary_scores,
        "conviction_components": conviction_components,
        "opportunity_quality_components": opportunity_quality_components,
        "composite_intelligence": composite_intelligence,
        "multi_horizon_price_momentum": market_structure,
        "volatility_risk_microstructure": fragility_group,
        "regime_engine": factor_groups["regime_intelligence"],
        "sentiment_intelligence": sentiment_group,
        "fundamental_intelligence": fundamental_group,
        "macro_sensitivity": macro_group,
        "relative_peer": cross_asset_group,
        "coverage_intelligence": data_bundle.get("domain_availability") or {},
    }
