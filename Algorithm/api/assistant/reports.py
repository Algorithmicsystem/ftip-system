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


def _join_sentences(parts: Iterable[Optional[str]]) -> str:
    return " ".join(str(part).strip() for part in parts if part and str(part).strip())


def _metric_phrase(
    label: str,
    value: Any,
    *,
    formatter: str = "num",
    digits: int = 1,
    signed: bool = True,
) -> Optional[str]:
    if value is None:
        return None
    text = _fmt_pct(value, digits=digits, signed=signed) if formatter == "pct" else _fmt_num(
        value,
        digits=digits,
        signed=signed,
    )
    return f"{label} {text}"


def _coverage_status_text(status: Optional[str]) -> str:
    mapping = {
        "available": "available",
        "partial": "partial",
        "insufficient_history": "constrained by limited usable history",
        "stale": "stale and should be read with caution",
        "unavailable": "currently unavailable",
        "not_relevant": "not currently a primary driver",
        "unknown": "unclear",
    }
    return mapping.get(str(status or "unknown"), str(status or "unknown").replace("_", " "))


def _domain_availability(data_bundle: Dict[str, Any], domain: str) -> Dict[str, Any]:
    availability = (data_bundle.get("domain_availability") or {}) or (
        (data_bundle.get("quality_provenance") or {}).get("domain_availability") or {}
    )
    return availability.get(domain) or {}


def _history_gap_sentence(entry: Dict[str, Any], label: str) -> Optional[str]:
    if entry.get("missing_reason") != "insufficient_history" or not entry.get("missing_horizons"):
        return None
    available = list(entry.get("available_horizons") or [])
    missing = _fmt_list(entry.get("missing_horizons") or [])
    if available:
        return (
            f"{label} is populated through {available[-1]}; longer-horizon context ({missing}) is currently unavailable because of limited usable history."
        )
    return f"{label} is currently unavailable because of limited usable history."


def _coverage_note(entry: Dict[str, Any], label: str) -> str:
    note = entry.get("data_quality_note")
    status_text = _coverage_status_text(entry.get("coverage_status"))
    if note:
        if entry.get("coverage_status") in {"not_relevant", "unavailable", "stale", "insufficient_history"}:
            return f"{note} {label} coverage is {status_text}."
        return note
    return f"{label} coverage is {status_text}."


def _tone_label(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "mixed to neutral"
    if number >= 0.15:
        return "constructive"
    if number <= -0.15:
        return "negative"
    return "mixed to neutral"


def _fundamental_analysis_text(
    fundamentals: Dict[str, Any],
    composites: Dict[str, Any],
    conviction_tier: str,
    quality: Dict[str, Any],
) -> str:
    metrics = fundamentals.get("normalized_metrics") or {}
    filing_backbone = fundamentals.get("filing_backbone") or {}
    meta = fundamentals.get("meta") or {}
    coverage_score = fundamentals.get("coverage_score")
    quality_proxies = fundamentals.get("quality_proxies") or {}
    strengths = list(fundamentals.get("strength_summary") or [])
    weaknesses = list(fundamentals.get("weakness_summary") or [])
    caveats = list(fundamentals.get("coverage_caveats") or [])
    latest_quarter = (
        ((fundamentals.get("statement_snapshot") or {}).get("latest_quarter"))
        or fundamentals.get("latest_quarter")
        or {}
    )
    latest_annual = ((fundamentals.get("statement_snapshot") or {}).get("latest_annual")) or {}
    latest_balance_sheet = (
        ((fundamentals.get("statement_snapshot") or {}).get("latest_balance_sheet")) or {}
    )
    sources = meta.get("sources") or []
    latest_10k = filing_backbone.get("latest_10k") or {}
    latest_10q = filing_backbone.get("latest_10q") or {}
    coverage_status = meta.get("coverage_status")

    has_real_fundamental_coverage = any(
        value is not None
        for value in [
            metrics.get("revenue_growth_yoy"),
            metrics.get("operating_margin"),
            metrics.get("net_margin"),
            metrics.get("current_ratio"),
            metrics.get("free_cash_flow"),
            coverage_score,
        ]
    )
    if not has_real_fundamental_coverage:
        if quality.get("fundamentals_ok") is False:
            return (
                "Fundamental coverage is explicitly missing in the quality layer, so the current report is weighted toward market structure, sentiment, and cross-domain signal composition rather than a filing-rich fundamental thesis."
            )
        missing_flags = _fmt_list(fundamentals.get("missingness_flags") or [])
        return (
            "Fundamental data is still thin. The filing-aware layer could not populate enough SEC-backed coverage to make a stronger statement, "
            f"and the missing areas are {missing_flags}. Secondary enrichment is preserved in provenance, but conviction is degraded rather than fabricated."
        )

    operating_margin = _first_available(metrics.get("operating_margin"), latest_quarter.get("op_margin"))
    balance_clauses = [
        _metric_phrase("current ratio", metrics.get("current_ratio"), formatter="num", digits=2, signed=False),
        _metric_phrase("cash ratio", metrics.get("cash_ratio"), formatter="num", digits=2, signed=False),
        _metric_phrase("debt-to-equity", metrics.get("debt_to_equity"), formatter="num", digits=2, signed=False),
        _metric_phrase("liabilities-to-assets", metrics.get("liabilities_to_assets"), formatter="num", digits=2, signed=False),
    ]
    performance_clauses = [
        _metric_phrase("quarterly revenue", latest_quarter.get("revenue"), formatter="num", digits=0, signed=False),
        _metric_phrase("year-on-year revenue growth", metrics.get("revenue_growth_yoy"), formatter="pct"),
        _metric_phrase("operating margin", operating_margin, formatter="pct"),
        _metric_phrase("net margin", metrics.get("net_margin"), formatter="pct"),
        _metric_phrase("free-cash-flow margin", metrics.get("free_cash_flow_margin"), formatter="pct"),
    ]
    quality_clauses = [
        _metric_phrase(
            "reporting-quality proxy",
            quality_proxies.get("reporting_quality_proxy"),
            formatter="num",
            digits=1,
            signed=False,
        ),
        _metric_phrase(
            "business-quality durability",
            quality_proxies.get("business_quality_durability"),
            formatter="num",
            digits=1,
            signed=False,
        ),
        _metric_phrase(
            "fundamental durability score",
            composites.get("Fundamental Durability Score"),
            formatter="num",
            digits=1,
            signed=False,
        ),
    ]
    summary_parts = [
        f"Fundamental coverage is {_coverage_status_text(coverage_status)} and anchored to {_fmt_list(sources)} with coverage score {_fmt_num(coverage_score, digits=2)}. Filing freshness is {meta.get('status') or 'unknown'}.",
        _join_sentences(
            [
                f"Latest periodic filing is {filing_backbone.get('latest_form')} dated {filing_backbone.get('latest_filing_date')}."
                if filing_backbone.get("latest_form") and filing_backbone.get("latest_filing_date")
                else None,
                f"Latest 10-K is {latest_10k.get('filing_date')} and latest 10-Q is {latest_10q.get('filing_date')}."
                if latest_10k.get("filing_date") or latest_10q.get("filing_date")
                else None,
            ]
        ),
        f"Statement-level evidence shows {_fmt_list(item for item in performance_clauses if item)}."
        if any(performance_clauses)
        else "Statement-level profitability and growth coverage is still partial, so the system is avoiding a stronger business-performance claim.",
        f"Balance-sheet and liquidity reads show {_fmt_list(item for item in balance_clauses if item)}."
        if any(balance_clauses)
        else "Balance-sheet and liquidity coverage is still thin, so leverage and resilience are being treated cautiously.",
        f"Quality and durability signals show {_fmt_list(item for item in quality_clauses if item)}, which supports a {conviction_tier} fundamental read."
        if any(quality_clauses)
        else None,
        f"Fundamental strengths are {_fmt_list(strengths)}. Weaknesses are {_fmt_list(weaknesses)}."
        if strengths or weaknesses
        else None,
        f"Coverage caveats are {_fmt_list(caveats)}."
        if caveats
        else None,
        "Missing fundamental coverage is explicitly reducing conviction in the overall stance."
        if coverage_status in {"partial", "stale", "unavailable"}
        else None,
    ]
    return _join_sentences(summary_parts)


def _technical_analysis_text(
    market: Dict[str, Any],
    technical: Dict[str, Any],
    data_bundle: Dict[str, Any],
) -> str:
    market_entry = _domain_availability(data_bundle, "market")
    technical_entry = _domain_availability(data_bundle, "technical")
    return_clauses = [
        _metric_phrase("1-day return", market.get("day_return"), formatter="pct"),
        _metric_phrase("5-day return", market.get("ret_5d"), formatter="pct"),
        _metric_phrase("21-day return", market.get("ret_21d"), formatter="pct"),
        _metric_phrase("63-day return", market.get("ret_63d"), formatter="pct"),
        _metric_phrase("126-day return", market.get("ret_126d"), formatter="pct"),
        _metric_phrase("252-day return", market.get("ret_252d"), formatter="pct"),
    ]
    ma = technical.get("moving_averages") or {}
    ma_clauses = [
        _metric_phrase("10-day average", ma.get("ma_10"), formatter="num", digits=2, signed=False),
        _metric_phrase("21-day average", ma.get("ma_21"), formatter="num", digits=2, signed=False),
        _metric_phrase("63-day average", ma.get("ma_63"), formatter="num", digits=2, signed=False),
        _metric_phrase("126-day average", ma.get("ma_126"), formatter="num", digits=2, signed=False),
    ]
    structure_clauses = [
        _metric_phrase("21-day trend slope", technical.get("trend_slope_21d"), formatter="num", digits=3),
        _metric_phrase("63-day trend slope", technical.get("trend_slope_63d"), formatter="num", digits=3),
        _metric_phrase("trend curvature", technical.get("trend_curvature"), formatter="num", digits=3),
    ]
    support_parts = [
        _metric_phrase("support", market.get("support_21d"), formatter="num", digits=2, signed=False),
        _metric_phrase("resistance", market.get("resistance_21d"), formatter="num", digits=2, signed=False),
        _metric_phrase("volume anomaly", market.get("volume_anomaly"), formatter="num", digits=2, signed=False),
        _metric_phrase("compression ratio", market.get("compression_ratio"), formatter="pct", digits=2),
    ]
    return _join_sentences(
        [
            f"Return context currently covers {_fmt_list(item for item in return_clauses if item)}."
            if any(return_clauses)
            else _coverage_note(market_entry, "Market"),
            _history_gap_sentence(market_entry, "Longer-horizon return context"),
            f"The moving-average stack is populated across {_fmt_list(item for item in ma_clauses if item)}."
            if any(ma_clauses)
            else _coverage_note(technical_entry, "Technical"),
            _history_gap_sentence(technical_entry, "Longer-horizon technical structure"),
            _join_sentences(
                [
                    f"Trend and structure show {_fmt_list(item for item in structure_clauses if item)}."
                    if any(structure_clauses)
                    else None,
                    f"Breakout state is {technical.get('breakout_state')}."
                    if technical.get("breakout_state")
                    else None,
                ]
            ),
            f"Support/resistance and participation currently show {_fmt_list(item for item in support_parts if item)}."
            if any(support_parts)
            else None,
            f"Structural levels are being derived from a shorter {market.get('support_window_days')}-day usable window."
            if market.get("support_window_days") and market.get("support_window_days") < 21
            else None,
            technical_entry.get("data_quality_note"),
        ]
    )


def _statistical_analysis_text(
    market: Dict[str, Any],
    key_features: Dict[str, Any],
    composites: Dict[str, Any],
    quality: Dict[str, Any],
    data_bundle: Dict[str, Any],
) -> str:
    market_entry = _domain_availability(data_bundle, "market")
    vol_clauses = [
        _metric_phrase("5-day realized vol", market.get("realized_vol_5d"), formatter="pct"),
        _metric_phrase("21-day realized vol", market.get("realized_vol_21d"), formatter="pct"),
        _metric_phrase("63-day realized vol", market.get("realized_vol_63d"), formatter="pct"),
        _metric_phrase("ATR percent", market.get("atr_pct"), formatter="pct"),
        _metric_phrase("gap behavior", market.get("gap_pct"), formatter="pct"),
    ]
    score_clauses = [
        _metric_phrase("risk-adjusted 21-day momentum", key_features.get("mom_vol_adj_21d"), formatter="num", digits=2),
        _metric_phrase("regime stability", composites.get("Regime Stability Score"), formatter="num", digits=1, signed=False),
        _metric_phrase("cross-domain conviction", composites.get("Cross-Domain Conviction Score"), formatter="num", digits=1, signed=False),
        _metric_phrase("signal fragility", composites.get("Signal Fragility Index"), formatter="num", digits=1, signed=False),
    ]
    structure_state = (
        "cleaner than average"
        if (composites.get("Market Structure Integrity Score") or 0) >= 60
        else "mixed or noisy"
    )
    return _join_sentences(
        [
            f"The quant read currently covers {_fmt_list(item for item in vol_clauses if item)}."
            if any(vol_clauses)
            else _coverage_note(market_entry, "Quant/market"),
            _history_gap_sentence(market_entry, "Longer-horizon volatility context"),
            f"Composite statistical overlays show {_fmt_list(item for item in score_clauses if item)}."
            if any(score_clauses)
            else None,
            f"The current distributional posture reads as {structure_state}, while reported missingness is {_fmt_num(quality.get('missingness'))}."
            if quality.get("missingness") is not None
            else f"The current distributional posture reads as {structure_state}.",
        ]
    )


def _sentiment_analysis_text(
    sentiment: Dict[str, Any],
    data_bundle: Dict[str, Any],
) -> str:
    sentiment_entry = _domain_availability(data_bundle, "sentiment")
    tone_source = _first_available(sentiment.get("sentiment_score"), sentiment.get("aggregated_sentiment_bias"))
    level_text = (
        f"Headline tone reads {_tone_label(tone_source)} at {_fmt_num(tone_source, digits=2, signed=True)}."
        if tone_source is not None
        else None
    )
    flow_clauses = [
        _metric_phrase("attention crowding", sentiment.get("attention_crowding"), formatter="num", digits=2, signed=False),
        _metric_phrase("novelty ratio", sentiment.get("novelty_ratio"), formatter="num", digits=2, signed=False),
        _metric_phrase("disagreement score", sentiment.get("disagreement_score"), formatter="num", digits=2, signed=False),
        _metric_phrase("sentiment surprise", sentiment.get("sentiment_surprise"), formatter="num", digits=2),
        _metric_phrase("sentiment trend", sentiment.get("sentiment_trend"), formatter="num", digits=2),
    ]
    return _join_sentences(
        [
            level_text or _coverage_note(sentiment_entry, "Sentiment"),
            f"Headline flow currently spans {sentiment.get('headline_count') or 0} recent items and shows {_fmt_list(item for item in flow_clauses if item)}."
            if any(flow_clauses)
            else f"Headline flow currently spans {sentiment.get('headline_count') or 0} recent items."
            if sentiment.get("headline_count")
            else None,
            f"News aggregation is drawing on {_fmt_list((sentiment.get('meta') or {}).get('sources') or sentiment.get('source_breakdown', {}).keys())}."
            if (sentiment.get("meta") or {}).get("sources") or sentiment.get("source_breakdown")
            else None,
            f"Top narratives currently cluster around {_fmt_list(item.get('topic') for item in sentiment.get('top_narratives') or [])}."
            if sentiment.get("top_narratives")
            else None,
            "Headline tone is available, but the system does not yet have enough density for a stable sentiment-level inference."
            if tone_source is None and sentiment.get("headline_count")
            else None,
            sentiment_entry.get("data_quality_note")
            if sentiment_entry.get("coverage_status") in {"partial", "insufficient_history", "unavailable"}
            else None,
        ]
    )


def _macro_geopolitical_analysis_text(
    macro: Dict[str, Any],
    geopolitical: Dict[str, Any],
    relative: Dict[str, Any],
    data_bundle: Dict[str, Any],
) -> str:
    macro_entry = _domain_availability(data_bundle, "macro")
    geopolitical_entry = _domain_availability(data_bundle, "geopolitical")
    relative_entry = _domain_availability(data_bundle, "cross_asset")
    benchmark_symbol = _first_available(
        macro.get("benchmark_proxy"),
        relative.get("benchmark_proxy"),
        (relative.get("benchmark_context") or {}).get("benchmark_symbol"),
    )
    benchmark_ret = _first_available(
        macro.get("benchmark_ret_21d"),
        relative.get("benchmark_ret_21d"),
        (relative.get("benchmark_context") or {}).get("benchmark_ret_21d"),
    )
    benchmark_vol = _first_available(
        macro.get("benchmark_vol_21d"),
        relative.get("benchmark_vol_21d"),
        (relative.get("benchmark_context") or {}).get("benchmark_vol_21d"),
    )
    macro_regime = _first_available(
        (macro.get("macro_regime_context") or {}).get("regime"),
        macro.get("inferred_market_regime"),
    )
    geo_score = _first_available(
        geopolitical.get("exogenous_event_score"),
        geopolitical.get("event_intensity_score"),
    )
    return _join_sentences(
        [
            f"Cross-asset context is anchored to {benchmark_symbol}, with benchmark 21-day return {_fmt_pct(benchmark_ret)} and benchmark volatility {_fmt_pct(benchmark_vol)}."
            if benchmark_symbol and (benchmark_ret is not None or benchmark_vol is not None)
            else _coverage_note(macro_entry, "Macro/cross-asset"),
            f"The broader macro backdrop currently reads as {macro_regime}, with macro alignment score {_fmt_num(macro.get('macro_alignment_score'), digits=1)} / 100."
            if macro_regime and macro.get("macro_alignment_score") is not None
            else f"Macro alignment score is {_fmt_num(macro.get('macro_alignment_score'), digits=1)} / 100."
            if macro.get("macro_alignment_score") is not None
            else f"The broader macro backdrop currently reads as {macro_regime}."
            if macro_regime
            else None,
            macro_entry.get("data_quality_note")
            if macro_entry.get("coverage_status") in {"partial", "stale", "unavailable"}
            else None,
            f"Geopolitical and policy sensitivity is registering at {_fmt_num(geo_score, digits=2, signed=False)} with category counts {geopolitical.get('category_counts') or geopolitical.get('event_buckets') or {}}."
            if geo_score is not None
            else _coverage_note(geopolitical_entry, "Geopolitical"),
            f"Peer-relative context versus {relative.get('sector') or 'the local comparison set'} is running at {_fmt_num(relative.get('relative_strength_percentile') * 100 if relative.get('relative_strength_percentile') is not None else None, digits=0, signed=False)} percentile strength."
            if relative.get("peer_count") and relative.get("relative_strength_percentile") is not None
            else _coverage_note(relative_entry, "Peer-relative"),
        ]
    )


def _risk_quality_analysis_text(
    signal: Dict[str, Any],
    quality: Dict[str, Any],
    warnings: Sequence[str],
    why_signal: Dict[str, Any],
    strategy: Dict[str, Any],
    data_bundle: Dict[str, Any],
) -> str:
    domain_availability = (data_bundle.get("domain_availability") or {}) or (
        (data_bundle.get("quality_provenance") or {}).get("domain_availability") or {}
    )
    domain_states = [
        f"{label}={entry.get('coverage_status')}"
        for label, entry in domain_availability.items()
        if entry.get("coverage_status")
    ]
    warning_text = list(warnings) + list(why_signal.get("freshness_warnings") or [])
    confidence_degraders = strategy.get("confidence_degraders") or []
    return _join_sentences(
        [
            f"Domain coverage currently reads {_fmt_list(domain_states)}."
            if domain_states
            else None,
            _entry_text(signal),
            f"Risk framing uses stop loss {_fmt_num(signal.get('stop_loss'), digits=2)} and take-profit levels {_fmt_num(signal.get('take_profit_1'), digits=2)} / {_fmt_num(signal.get('take_profit_2'), digits=2)}."
            if signal.get("stop_loss") is not None or signal.get("take_profit_1") is not None or signal.get("take_profit_2") is not None
            else None,
            f"Warnings currently include {_fmt_list(warning_text)}, anomaly flags {_fmt_list(quality.get('anomaly_flags') or [])}, and confidence degraders {_fmt_list(confidence_degraders)}."
            if warning_text or quality.get("anomaly_flags") or confidence_degraders
            else "No material provider-note, anomaly, or freshness escalation is currently attached to the active report.",
        ]
    )


def _strategy_view_text(
    strategy: Dict[str, Any],
    strategy_signal: str,
    horizon: str,
) -> str:
    participant_fit = strategy.get("participant_fit") or []
    invalidators = strategy.get("invalidation_conditions") or []
    return _join_sentences(
        [
            strategy.get("base_case")
            or f"The base case is to carry a {strategy_signal} stance on the {horizon} horizon.",
            strategy.get("upside_case"),
            strategy.get("downside_case"),
            f"Participant fit is {_fmt_list(participant_fit)}."
            if participant_fit
            else "Participant fit remains broad because the setup is not resolving into one narrow style bucket.",
            f"Invalidation conditions are {_fmt_list(invalidators)}."
            if invalidators
            else "No single hard invalidator dominates yet; the posture is more sensitive to incremental evidence decay.",
        ]
    )


def _risks_invalidators_text(
    why_signal: Dict[str, Any],
    strategy: Dict[str, Any],
) -> str:
    missing = why_signal.get("missing_data_warnings") or []
    freshness = why_signal.get("freshness_warnings") or []
    invalidators = strategy.get("invalidation_conditions") or []
    return _join_sentences(
        [
            f"Key weaknesses are {_fmt_driver_list(why_signal.get('top_negative_drivers') or [])}.",
            f"Missing-data warnings are {_fmt_list(missing)} and freshness warnings are {_fmt_list(freshness)}."
            if missing or freshness
            else "No major missing-data or freshness escalation is currently dominating the risk framing.",
            f"Formal invalidators are {_fmt_list(invalidators)}."
            if invalidators
            else "Formal invalidators remain scenario-dependent rather than singular.",
        ]
    )


def _evidence_provenance_text(
    evidence: Dict[str, Any],
    freshness_summary: Dict[str, Any],
    data_bundle: Dict[str, Any],
    reason_codes: Sequence[str],
    strategy_component_scores: Dict[str, Any],
) -> str:
    quality = data_bundle.get("quality_provenance") or {}
    domain_availability = quality.get("domain_availability") or data_bundle.get("domain_availability") or {}
    fallback_domains = [
        label
        for label, entry in domain_availability.items()
        if entry.get("fallback_used")
    ]
    return _join_sentences(
        [
            f"Evidence provenance spans {_fmt_list(evidence.get('sources') or [])} plus normalized domains for market, technical, fundamentals, sentiment, macro/cross-asset, geopolitical, relative context, and quality.",
            f"Freshness status is {freshness_summary['overall_status']} with domain states {freshness_summary['domains']}.",
            f"Domain source map is {quality.get('source_map') or {}}, and fallback logic was used in {_fmt_list(fallback_domains)}."
            if fallback_domains
            else f"Domain source map is {quality.get('source_map') or {}}.",
            f"external data fabric status is {(data_bundle.get('external_data_fabric') or {}).get('status') or 'disabled'}, reason codes are {_fmt_list(reason_codes)}, and strategy components are {strategy_component_scores}.",
        ]
    )


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
    domain_availability = (data_bundle.get("domain_availability") or {}) or (
        (data_bundle.get("quality_provenance") or {}).get("domain_availability") or {}
    )
    coverage_headwinds = [
        label.replace("_", " ")
        for label, entry in domain_availability.items()
        if entry.get("coverage_status") in {"partial", "insufficient_history", "stale", "unavailable"}
    ]
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
            f"Coverage headwinds are concentrated in {_fmt_list(coverage_headwinds)}, which is dampening conviction."
            if coverage_headwinds
            else "Cross-domain coverage is broadly in place, so the signal is not being driven by one thin data pocket alone.",
        ]
    )

    technical_analysis = _technical_analysis_text(market, technical, data_bundle)

    fundamental_analysis = _fundamental_analysis_text(
        fundamentals,
        composites,
        conviction_tier,
        quality,
    )

    statistical_analysis = _statistical_analysis_text(
        market,
        key_features,
        composites,
        quality,
        data_bundle,
    )

    sentiment_analysis = _sentiment_analysis_text(sentiment, data_bundle)

    macro_geopolitical_analysis = _macro_geopolitical_analysis_text(
        macro,
        geopolitical,
        relative,
        data_bundle,
    )

    risk_quality_analysis = _risk_quality_analysis_text(
        signal,
        quality,
        warnings,
        why_signal,
        strategy,
        data_bundle,
    )

    overall_analysis = " ".join(
        [
            f"The unified system view on {symbol} is {strategy_signal}. That posture is not coming from one score alone; it is the result of trend, mean-reversion, sentiment, macro-alignment, quality/fundamental, and fragility-veto components being fused inside the strategy layer.",
            f"The strongest evidence for the thesis is {_fmt_driver_list(why_signal['top_positive_drivers'])}. The strongest evidence against it is {_fmt_driver_list(why_signal['top_negative_drivers'])}.",
            f"The final posture stays at {strategy_signal} because raw signal action {action}, regime {regime}, and opportunity-quality score {_fmt_num(composites.get('Opportunity Quality Score'), digits=1)} / 100 outweigh the current detractors, but the system is explicitly least certain where {strategy.get('where_least_certain') or 'cross-domain disagreement is highest'}.",
            "This remains a description of the platform's computed state, not personalized investment advice.",
        ]
    )
    strategy_view = _strategy_view_text(strategy, strategy_signal, horizon)
    risks_weaknesses_invalidators = _risks_invalidators_text(why_signal, strategy)
    evidence_provenance = _evidence_provenance_text(
        evidence,
        freshness_summary,
        data_bundle,
        reason_codes,
        strategy_component_scores,
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
        "domain_availability": domain_availability,
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
