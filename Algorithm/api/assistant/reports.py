from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from decimal import Decimal
from numbers import Real
from typing import Any, Dict, Iterable, Optional


ANALYSIS_REPORT_KIND = "analysis_report"


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: sanitize_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_payload(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc).isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Real):
        try:
            number = float(value)
            return number if math.isfinite(number) else None
        except (TypeError, ValueError, OverflowError):
            return value
    return value


def _iso_date(value: Any) -> str:
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value or "")


def _fmt_num(value: Any, *, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.{digits}f}"


def _fmt_pct(value: Any, *, digits: int = 1, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value) * 100.0
    except (TypeError, ValueError):
        return str(value)
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.{digits}f}%"


def _fmt_list(values: Iterable[Any]) -> str:
    cleaned = [str(value) for value in values if value not in (None, "", [], {})]
    return ", ".join(cleaned) if cleaned else "none"


def _signal_bias(action: str) -> str:
    action_upper = (action or "HOLD").upper()
    if action_upper == "BUY":
        return "bullish"
    if action_upper == "SELL":
        return "bearish"
    return "neutral"


def _entry_text(signal: Dict[str, Any]) -> str:
    entry_low = signal.get("entry_low")
    entry_high = signal.get("entry_high")
    if entry_low is None and entry_high is None:
        return "No explicit entry band is available in the stored signal."
    if entry_low is not None and entry_high is not None:
        return f"Entry band is {_fmt_num(entry_low, digits=2)} to {_fmt_num(entry_high, digits=2)}."
    if entry_low is not None:
        return f"Entry reference is {_fmt_num(entry_low, digits=2)}."
    return f"Entry reference is {_fmt_num(entry_high, digits=2)}."


def _first_available(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _freshness_status(data_bundle: Optional[Dict[str, Any]]) -> str:
    quality = (data_bundle or {}).get("quality_provenance") or {}
    freshness = quality.get("freshness_summary") or {}
    statuses = [item.get("status") for item in freshness.values() if item.get("status")]
    if not statuses:
        return "unknown"
    if all(status == "fresh" for status in statuses):
        return "fresh"
    if any(status == "stale" for status in statuses):
        return "mixed_stale"
    return "mixed"


def _fmt_driver_list(drivers: Iterable[Dict[str, Any]]) -> str:
    parts = []
    for item in drivers:
        label = item.get("label") or "driver"
        detail = item.get("detail") or ""
        score = item.get("score")
        if score is None:
            parts.append(f"{label}: {detail}")
        else:
            parts.append(f"{label} ({_fmt_num(score, digits=1)}): {detail}")
    return "; ".join(parts) if parts else "none"


def _why_this_signal(
    strategy: Dict[str, Any],
    data_bundle: Optional[Dict[str, Any]],
    feature_factor_bundle: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    quality = (data_bundle or {}).get("quality_provenance") or {}
    warnings = list(quality.get("warnings") or [])
    freshness = quality.get("freshness_summary") or {}
    freshness_warnings = []
    for domain, summary in freshness.items():
        status = summary.get("status")
        if status in {"stale", "stale_but_usable"}:
            freshness_warnings.append(f"{domain} is {status.replace('_', ' ')}.")
    return {
        "top_positive_drivers": strategy.get("top_contributors") or [],
        "top_negative_drivers": strategy.get("top_detractors") or [],
        "confidence_modifiers": strategy.get("confidence_degraders") or [],
        "missing_data_warnings": warnings,
        "freshness_warnings": freshness_warnings,
        "composite_scores": (
            (feature_factor_bundle or {}).get("composite_intelligence") or {}
        ),
    }


def build_active_analysis_reference(
    report: Dict[str, Any],
    *,
    session_id: Optional[str] = None,
    report_id: Optional[str] = None,
) -> Dict[str, Any]:
    return sanitize_payload(
        {
            "report_id": report_id,
            "session_id": session_id or report.get("session_id"),
            "symbol": report.get("symbol"),
            "as_of_date": report.get("as_of_date"),
            "horizon": report.get("horizon"),
            "risk_mode": report.get("risk_mode"),
            "scenario": report.get("scenario"),
            "analysis_depth": report.get("analysis_depth"),
            "refresh_mode": report.get("refresh_mode"),
            "market_regime": report.get("market_regime"),
            "signal": _first_available(
                (report.get("strategy") or {}).get("final_signal"),
                (report.get("signal") or {}).get("final_action"),
                (report.get("signal") or {}).get("action"),
            ),
            "freshness_status": _first_available(
                (report.get("freshness_summary") or {}).get("overall_status"),
                _freshness_status(report.get("data_bundle")),
            ),
            "report_version": report.get("report_version"),
        }
    )


def build_analysis_report(
    *,
    symbol: str,
    as_of_date: Any,
    horizon: str,
    risk_mode: str,
    signal: Dict[str, Any],
    key_features: Dict[str, Any],
    quality: Dict[str, Any],
    evidence: Dict[str, Any],
    job_context: Optional[Dict[str, Any]] = None,
    data_bundle: Optional[Dict[str, Any]] = None,
    feature_factor_bundle: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    as_of_text = _iso_date(as_of_date)
    action = (signal.get("action") or "HOLD").upper()
    score = signal.get("score")
    confidence = signal.get("confidence")
    strategy = strategy or {}
    job_context = job_context or {}
    data_bundle = data_bundle or {}
    feature_factor_bundle = feature_factor_bundle or {}

    strategy_signal = (strategy.get("final_signal") or action).upper()
    strategy_confidence = _first_available(strategy.get("confidence"), confidence)
    conviction_tier = strategy.get("conviction_tier") or "unknown"
    fragility_tier = strategy.get("fragility_tier") or "unknown"
    regime = _first_available(
        (feature_factor_bundle.get("regime_engine") or {}).get("regime_label"),
        key_features.get("regime_label"),
        "unknown",
    )
    quality_score = _first_available(
        (data_bundle.get("quality_provenance") or {}).get("quality_score"),
        quality.get("quality_score"),
    )
    warnings = quality.get("warnings") or []
    reason_codes = signal.get("reason_codes") or []
    why_signal = _why_this_signal(strategy, data_bundle, feature_factor_bundle)
    freshness_summary = {
        "overall_status": _freshness_status(data_bundle),
        "domains": ((data_bundle.get("quality_provenance") or {}).get("freshness_summary") or {}),
    }
    market = data_bundle.get("market_price_volume") or {}
    technical = data_bundle.get("technical_market_structure") or {}
    fundamentals = data_bundle.get("fundamental_filing") or {}
    sentiment = data_bundle.get("sentiment_narrative_flow") or {}
    macro = data_bundle.get("macro_cross_asset") or {}
    geopolitical = data_bundle.get("geopolitical_policy") or {}
    relative = data_bundle.get("relative_context") or {}
    composites = feature_factor_bundle.get("composite_intelligence") or {}
    strategy_component_scores = strategy.get("component_scores") or {}
    signal_view = {
        **signal,
        "horizon": horizon,
        "risk_mode": risk_mode,
        "final_action": strategy_signal,
        "strategy_confidence": strategy_confidence,
        "conviction_tier": conviction_tier,
        "fragility_tier": fragility_tier,
    }

    signal_summary = " ".join(
        [
            f"As of {as_of_text}, the assistant pipeline lands on a {strategy_signal} posture for {symbol}, framed on the {horizon} horizon under {risk_mode} risk mode.",
            f"The underlying signal engine prints {action} with score {_fmt_num(score)} and confidence {_fmt_num(confidence)}, while the strategy layer converts that into {strategy_signal} with {_fmt_num(strategy_confidence)} confidence, {conviction_tier} conviction, and {fragility_tier} fragility.",
            f"The dominant regime reads {regime}, freshness is {freshness_summary['overall_status']}, and the main positive drivers are {_fmt_driver_list(why_signal['top_positive_drivers'])}.",
            f"The main risks are {_fmt_driver_list(why_signal['top_negative_drivers'])}, with scenario framing set to {job_context.get('scenario') or 'base'}.",
        ]
    )

    technical_analysis = " ".join(
        [
            f"Price is carrying {_fmt_pct(market.get('day_return'))} over 1 day, {_fmt_pct(market.get('ret_5d'))} over 5 days, {_fmt_pct(market.get('ret_21d'))} over 21 days, and {_fmt_pct(market.get('ret_63d'))} over 63 days.",
            f"The structure stack shows moving averages at 10/21/63/126 days of {_fmt_num((technical.get('moving_averages') or {}).get('ma_10'), digits=2)}, {_fmt_num((technical.get('moving_averages') or {}).get('ma_21'), digits=2)}, {_fmt_num((technical.get('moving_averages') or {}).get('ma_63'), digits=2)}, and {_fmt_num((technical.get('moving_averages') or {}).get('ma_126'), digits=2)}.",
            f"Trend slope is {_fmt_num(technical.get('trend_slope_21d'), signed=True)} on 21 days and {_fmt_num(technical.get('trend_slope_63d'), signed=True)} on 63 days, with curvature {_fmt_num(technical.get('trend_curvature'), signed=True)} and breakout state {technical.get('breakout_state') or 'n/a'}.",
            f"Support and resistance cluster around {_fmt_num(market.get('support_21d'), digits=2)} and {_fmt_num(market.get('resistance_21d'), digits=2)}, while volume anomaly is {_fmt_num(market.get('volume_anomaly'))} and compression ratio is {_fmt_pct(market.get('compression_ratio'), digits=2)}.",
        ]
    )

    if fundamentals.get("latest_quarter"):
        fundamental_analysis = " ".join(
            [
                f"Fundamental coverage is live, with latest quarter revenue at {_fmt_num((fundamentals.get('latest_quarter') or {}).get('revenue'), digits=0)} and operating margin {_fmt_pct((fundamentals.get('latest_quarter') or {}).get('op_margin'))}.",
                f"Revenue growth versus the year-ago quarter is {_fmt_pct(fundamentals.get('revenue_growth_yoy'))}, gross-margin trend is {_fmt_pct(fundamentals.get('gross_margin_trend'))}, and filing recency is {fundamentals.get('filing_recency_days') or 'n/a'} days.",
                f"The proprietary durability read is {_fmt_num(composites.get('Fundamental Durability Score'), digits=1)} / 100, which suggests {conviction_tier} business-quality support rather than purely price-led conviction."
                if composites.get("Fundamental Durability Score") is not None
                else "Fundamental durability is not fully scored because the quarterly history is partial.",
            ]
        )
    elif quality.get("fundamentals_ok") is False:
        fundamental_analysis = (
            "Fundamental coverage is explicitly missing in the quality layer, so the current report is weighted toward market structure, sentiment, and cross-domain signal composition rather than a filing-rich fundamental thesis."
        )
    else:
        fundamental_analysis = (
            "Fundamental data is limited or incomplete. The assistant pipeline preserves the gap as a real coverage constraint and degrades conviction accordingly instead of pretending the filing layer is stronger than it is."
        )

    statistical_analysis = " ".join(
        [
            f"The quant read is built around realized volatility of {_fmt_pct(market.get('realized_vol_21d'))} on 21 days and {_fmt_pct(market.get('realized_vol_63d'))} on 63 days, with ATR percent {_fmt_pct(market.get('atr_pct'))} and gap behavior {_fmt_pct(market.get('gap_pct'))}.",
            f"Risk-adjusted momentum on 21 days is {_fmt_num(key_features.get('mom_vol_adj_21d'), signed=True)}, regime stability scores {_fmt_num(composites.get('Regime Stability Score'), digits=1)} / 100, and cross-domain conviction scores {_fmt_num(composites.get('Cross-Domain Conviction Score'), digits=1)} / 100.",
            f"The statistical posture therefore reads as {'cleaner than average' if (composites.get('Market Structure Integrity Score') or 0) >= 60 else 'mixed or noisy'}, while fragility scores {_fmt_num(composites.get('Signal Fragility Index'), digits=1)} / 100 and missingness sits at {_fmt_num(quality.get('missingness'))}.",
        ]
    )

    sentiment_analysis = " ".join(
        [
            f"Sentiment level is {_fmt_num(sentiment.get('sentiment_score'), signed=True)} with surprise {_fmt_num(sentiment.get('sentiment_surprise'), signed=True)} and short-run trend {_fmt_num(sentiment.get('sentiment_trend'), signed=True)}.",
            f"Headline flow counts {sentiment.get('headline_count') or 0} recent items, attention crowding is {_fmt_num(sentiment.get('attention_crowding'))}, novelty ratio is {_fmt_num(sentiment.get('novelty_ratio'))}, and disagreement score is {_fmt_num(sentiment.get('disagreement_score'))}.",
            f"Multi-source news coverage is drawing on {_fmt_list((sentiment.get('meta') or {}).get('sources') or sentiment.get('source_breakdown', {}).keys())}, with aggregated bias {_fmt_num(sentiment.get('aggregated_sentiment_bias'), signed=True)} and source breakdown {sentiment.get('source_breakdown') or {}}."
            if (sentiment.get("source_breakdown") or (sentiment.get("meta") or {}).get("sources"))
            else "",
            f"Top narratives currently cluster around {_fmt_list(item.get('topic') for item in sentiment.get('top_narratives') or [])}, which means narrative flow is {'supportive' if (sentiment.get('sentiment_score') or 0) > 0 else 'cautious or contradictory'} rather than detached from price."
            if sentiment.get("top_narratives")
            else "Narrative coverage is limited, so sentiment is treated as a low-confidence overlay rather than a dominant driver.",
        ]
    )

    macro_geopolitical_analysis = " ".join(
        [
            f"Cross-asset context is anchored to {macro.get('benchmark_proxy') or 'limited benchmark coverage'}, with benchmark 21-day return {_fmt_pct(macro.get('benchmark_ret_21d'))}, inferred regime {macro.get('inferred_market_regime') or 'n/a'}, and macro alignment score {_fmt_num(macro.get('macro_alignment_score'), digits=1)} / 100.",
            f"Normalized macro series point to {(macro.get('macro_regime_context') or {}).get('regime') or 'n/a'} conditions, with FRED and World Bank snapshots {(macro.get('fred_series') or {}) or (macro.get('world_bank_series') or {})}.",
            f"Geopolitical and policy sensitivity currently shows exogenous-event score {_fmt_num(geopolitical.get('exogenous_event_score') or geopolitical.get('event_intensity_score'), digits=2)} with category counts {geopolitical.get('category_counts') or geopolitical.get('event_buckets') or {}}.",
            f"Relative context versus {relative.get('sector') or 'available peers'} is {_fmt_num(relative.get('relative_strength_percentile') * 100 if relative.get('relative_strength_percentile') is not None else None, digits=0)} percentile strength, which means the setup is {'aligned with broader tape' if (macro.get('macro_alignment_score') or 0) >= 55 else 'not receiving much help from the broader tape'}."
            if relative.get("peer_count")
            else f"Cross-asset proxies are {(relative.get('benchmark_context') or {}).get('benchmark_symbol') or 'limited'}, with proxy coverage score {_fmt_num((relative.get('meta') or {}).get('coverage_score'))}.",
        ]
    )

    risk_quality_analysis = " ".join(
        [
            f"Quality flags are bars_ok={quality.get('bars_ok')}, fundamentals_ok={quality.get('fundamentals_ok')}, news_ok={quality.get('news_ok')}, sentiment_ok={quality.get('sentiment_ok')}, and intraday_ok={quality.get('intraday_ok')}.",
            _entry_text(signal),
            f"Stop loss is {_fmt_num(signal.get('stop_loss'), digits=2)}, take-profit levels are {_fmt_num(signal.get('take_profit_1'), digits=2)} and {_fmt_num(signal.get('take_profit_2'), digits=2)}.",
            f"Provider notes include {_fmt_list((data_bundle.get('quality_provenance') or {}).get('provider_notes') or [])}.",
            f"Warnings include {_fmt_list(warnings + why_signal['freshness_warnings'])}, anomaly flags {_fmt_list(quality.get('anomaly_flags') or [])}, and confidence degraders {_fmt_list(strategy.get('confidence_degraders') or [])}.",
        ]
    )

    overall_analysis = " ".join(
        [
            f"The unified system view on {symbol} is {strategy_signal}. That posture is not coming from one score alone; it is the result of trend, mean-reversion, sentiment, macro-alignment, quality/fundamental, and fragility-veto components being fused inside the strategy layer.",
            f"The strongest evidence for the thesis is {_fmt_driver_list(why_signal['top_positive_drivers'])}. The strongest evidence against it is {_fmt_driver_list(why_signal['top_negative_drivers'])}.",
            f"The final posture stays at {strategy_signal} because raw signal action {action}, regime {regime}, and opportunity-quality score {_fmt_num(composites.get('Opportunity Quality Score'), digits=1)} / 100 outweigh the current detractors, but the system is explicitly least certain where {strategy.get('where_least_certain') or 'cross-domain disagreement is highest'}.",
            "This remains a description of the platform's computed state, not personalized investment advice.",
        ]
    )
    strategy_view = " ".join(
        [
            strategy.get("base_case")
            or f"The base case is to carry a {strategy_signal} stance on the {horizon} horizon.",
            strategy.get("upside_case") or "",
            strategy.get("downside_case") or "",
            f"Participant fit is {_fmt_list(strategy.get('participant_fit') or [])}, and invalidation conditions are {_fmt_list(strategy.get('invalidation_conditions') or [])}.",
        ]
    ).strip()
    risks_weaknesses_invalidators = " ".join(
        [
            f"Key weaknesses are {_fmt_driver_list(why_signal['top_negative_drivers'])}.",
            f"Missing-data warnings are {_fmt_list(why_signal['missing_data_warnings'])}; freshness warnings are {_fmt_list(why_signal['freshness_warnings'])}.",
            f"Formal invalidators are {_fmt_list(strategy.get('invalidation_conditions') or [])}.",
        ]
    )
    evidence_provenance = " ".join(
        [
            f"Evidence provenance spans {_fmt_list(evidence.get('sources') or [])} plus normalized domains for market, technical, fundamentals, sentiment, macro/cross-asset, geopolitical, relative context, and quality.",
            f"Freshness status is {freshness_summary['overall_status']} with domain states {freshness_summary['domains']}.",
            f"Domain source map is {(data_bundle.get('quality_provenance') or {}).get('source_map') or {}}, and external data fabric status is {(data_bundle.get('external_data_fabric') or {}).get('status') or 'n/a'}.",
            f"Reason codes are {_fmt_list(reason_codes)}, strategy components are {strategy_component_scores}, and report version is 2.0.",
        ]
    )
    evidence_map = {
        "signal_summary": [
            "signal.action",
            "strategy.final_signal",
            "strategy.component_scores",
            "feature_factor_bundle.composite_intelligence",
        ],
        "technical_analysis": [
            "data_bundle.market_price_volume",
            "data_bundle.technical_market_structure",
            "key_features.ret_1d/ret_5d/ret_21d",
        ],
        "fundamental_analysis": [
            "data_bundle.fundamental_filing",
            "feature_factor_bundle.fundamental_intelligence",
        ],
        "statistical_analysis": [
            "feature_factor_bundle.multi_horizon_price_momentum",
            "feature_factor_bundle.volatility_risk_microstructure",
            "feature_factor_bundle.regime_engine",
        ],
        "sentiment_analysis": [
            "data_bundle.sentiment_narrative_flow",
            "feature_factor_bundle.sentiment_intelligence",
        ],
        "macro_geopolitical_analysis": [
            "data_bundle.macro_cross_asset",
            "data_bundle.geopolitical_policy",
            "data_bundle.relative_context",
        ],
        "strategy_view": [
            "strategy.base_case",
            "strategy.upside_case",
            "strategy.downside_case",
            "strategy.invalidation_conditions",
        ],
        "risks_weaknesses_invalidators": [
            "quality.warnings",
            "quality.anomaly_flags",
            "strategy.confidence_degraders",
        ],
    }

    report = {
        "report_kind": ANALYSIS_REPORT_KIND,
        "report_version": "2.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "symbol": symbol,
        "as_of_date": as_of_text,
        "horizon": horizon,
        "risk_mode": risk_mode,
        "scenario": job_context.get("scenario"),
        "analysis_depth": job_context.get("analysis_depth"),
        "refresh_mode": job_context.get("refresh_mode"),
        "market_regime": job_context.get("market_regime"),
        "analysis_job": job_context,
        "freshness_summary": freshness_summary,
        "signal": signal_view,
        "key_features": key_features,
        "quality": quality,
        "evidence": evidence,
        "data_bundle": data_bundle,
        "feature_factor_bundle": feature_factor_bundle,
        "strategy": strategy,
        "why_this_signal": why_signal,
        "top_positive_drivers": why_signal["top_positive_drivers"],
        "top_negative_drivers": why_signal["top_negative_drivers"],
        "confidence_modifiers": why_signal["confidence_modifiers"],
        "missing_data_warnings": why_signal["missing_data_warnings"],
        "freshness_warnings": why_signal["freshness_warnings"],
        "evidence_map": evidence_map,
        "signal_summary": signal_summary,
        "technical_analysis": technical_analysis,
        "fundamental_analysis": fundamental_analysis,
        "statistical_analysis": statistical_analysis,
        "sentiment_analysis": sentiment_analysis,
        "macro_geopolitical_analysis": macro_geopolitical_analysis,
        "risk_quality_analysis": risk_quality_analysis,
        "overall_analysis": overall_analysis,
        "strategy_view": strategy_view,
        "risks_weaknesses_invalidators": risks_weaknesses_invalidators,
        "evidence_provenance": evidence_provenance,
    }
    return sanitize_payload(report)
