from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from decimal import Decimal
from numbers import Real
from typing import Any, Dict, Iterable, Optional, Sequence


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
        f"Quarterly revenue is {_fmt_num(latest_quarter.get('revenue'), digits=0, signed=False)}."
        if latest_quarter.get("revenue") is not None
        else None,
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
    fragility: Dict[str, Any],
    domain_agreement: Dict[str, Any],
    quality: Dict[str, Any],
    data_bundle: Dict[str, Any],
) -> str:
    market_entry = _domain_availability(data_bundle, "market")
    vol_clauses = [
        _metric_phrase("5-day realized vol", market.get("realized_vol_5d"), formatter="pct"),
        _metric_phrase("10-day realized vol", market.get("realized_vol_10d"), formatter="pct"),
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
        _metric_phrase("domain agreement", domain_agreement.get("domain_agreement_score"), formatter="num", digits=1, signed=False),
        _metric_phrase("domain conflict", domain_agreement.get("domain_conflict_score"), formatter="num", digits=1, signed=False),
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
            f"Fragility diagnostics show clean setup {_fmt_num(fragility.get('clean_setup_score'), digits=1, signed=False)} / 100, instability {_fmt_num(fragility.get('instability_score'), digits=1, signed=False)} / 100, and anomaly pressure {_fmt_num(fragility.get('anomaly_pressure_score'), digits=1, signed=False)} / 100."
            if fragility
            else None,
            f"The current distributional posture reads as {structure_state}, while reported missingness is {_fmt_num(quality.get('missingness'))}."
            if quality.get("missingness") is not None
            else f"The current distributional posture reads as {structure_state}.",
        ]
    )


def _sentiment_analysis_text(
    sentiment: Dict[str, Any],
    sentiment_factor: Dict[str, Any],
    composites: Dict[str, Any],
    data_bundle: Dict[str, Any],
) -> str:
    sentiment_entry = _domain_availability(data_bundle, "sentiment")
    sentiment_summary = sentiment.get("sentiment_summary") or {}
    tone_source = _first_available(
        sentiment.get("sentiment_score"),
        sentiment_summary.get("bias"),
        sentiment.get("aggregated_sentiment_bias"),
    )
    tone_label = _first_available(sentiment_summary.get("tone_label"), _tone_label(tone_source))
    level_text = (
        f"Headline tone reads {tone_label} at {_fmt_num(tone_source, digits=2, signed=True)}, with sentiment confidence {_fmt_num(sentiment.get('sentiment_confidence'), digits=1, signed=False)} / 100."
        if tone_source is not None
        else None
    )
    flow_clauses = [
        _metric_phrase("attention score", sentiment.get("attention_score"), formatter="num", digits=1, signed=False),
        _metric_phrase("attention crowding", sentiment.get("attention_crowding"), formatter="num", digits=2, signed=False),
        _metric_phrase("novelty score", sentiment.get("novelty_score"), formatter="num", digits=1, signed=False),
        _metric_phrase("persistence score", sentiment.get("persistence_score"), formatter="num", digits=1, signed=False),
        _metric_phrase("contradiction score", sentiment.get("contradiction_score"), formatter="num", digits=2, signed=False),
        _metric_phrase("sentiment surprise", sentiment.get("sentiment_surprise"), formatter="num", digits=2),
        _metric_phrase("sentiment trend", sentiment.get("sentiment_trend"), formatter="num", digits=2),
    ]
    source_mix = sentiment.get("source_mix") or []
    source_mix_text = ", ".join(
        f"{item.get('source')} ({item.get('count')})" for item in source_mix if item.get("source")
    )
    topic_clusters = sentiment.get("topic_clusters") or sentiment.get("top_narratives") or []
    return _join_sentences(
        [
            level_text or _coverage_note(sentiment_entry, "Sentiment"),
            f"Headline flow currently spans {sentiment.get('headline_count') or 0} recent items and shows {_fmt_list(item for item in flow_clauses if item)}."
            if any(flow_clauses)
            else f"Headline flow currently spans {sentiment.get('headline_count') or 0} recent items."
            if sentiment.get("headline_count")
            else None,
            f"News aggregation is drawing on {source_mix_text}."
            if source_mix_text
            else f"News aggregation is drawing on {_fmt_list((sentiment.get('meta') or {}).get('sources') or sentiment.get('source_breakdown', {}).keys())}."
            if (sentiment.get("meta") or {}).get("sources") or sentiment.get("source_breakdown")
            else None,
            f"Dominant narrative clusters are {_fmt_list(item.get('topic') for item in topic_clusters)}, with thematic bucket counts {sentiment.get('topic_buckets') or {}}."
            if topic_clusters or sentiment.get("topic_buckets")
            else None,
            f"Narrative crowding is {_fmt_num(composites.get('Narrative Crowding Index'), digits=1, signed=False)} / 100, with crowding proxy {_fmt_num(sentiment_factor.get('crowding_proxy_score'), digits=1, signed=False)} / 100, positive-news/weak-price divergence {_fmt_num(sentiment_factor.get('positive_news_weak_price_divergence'), digits=1, signed=False)}, and negative-news/resilient-price divergence {_fmt_num(sentiment_factor.get('negative_news_resilient_price_divergence'), digits=1, signed=False)}."
            if composites.get("Narrative Crowding Index") is not None or sentiment_factor
            else None,
            f"GDELT event overlay is contributing {((sentiment.get('event_overlay') or {}).get('gdelt_article_count') or 0)} broader event articles with average tone {_fmt_num(((sentiment.get('event_overlay') or {}).get('gdelt_tone_average')), digits=2, signed=True)}."
            if sentiment.get("event_overlay")
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
    macro_factor: Dict[str, Any],
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
        ((macro.get("macro_regime_summary") or {}).get("regime")),
        (macro.get("macro_regime_context") or {}).get("regime"),
        macro.get("inferred_market_regime"),
    )
    geo_score = _first_available(
        geopolitical.get("exogenous_event_score"),
        geopolitical.get("event_intensity_score"),
    )
    rates_context = macro.get("rates_context") or {}
    inflation_context = macro.get("inflation_context") or {}
    growth_context = macro.get("growth_context") or {}
    liquidity_context = macro.get("liquidity_context") or {}
    broad_market = relative.get("broad_market_context") or {}
    sector_context = relative.get("sector_context") or {}
    relative_summary = relative.get("relative_move_summary") or {}
    macro_notes = macro.get("macro_alignment_notes") or []
    return _join_sentences(
        [
            f"Cross-asset context is anchored to {benchmark_symbol}, with benchmark 21-day return {_fmt_pct(benchmark_ret)} and benchmark volatility {_fmt_pct(benchmark_vol)}."
            if benchmark_symbol and (benchmark_ret is not None or benchmark_vol is not None)
            else _coverage_note(macro_entry, "Macro/cross-asset"),
            ((macro.get("macro_regime_summary") or {}).get("summary"))
            if (macro.get("macro_regime_summary") or {}).get("summary")
            else f"The broader macro backdrop currently reads as {macro_regime}, with macro alignment score {_fmt_num(macro.get('macro_alignment_score'), digits=1)} / 100."
            if macro_regime and macro.get("macro_alignment_score") is not None
            else f"Macro alignment score is {_fmt_num(macro.get('macro_alignment_score'), digits=1)} / 100."
            if macro.get("macro_alignment_score") is not None
            else f"The broader macro backdrop currently reads as {macro_regime}."
            if macro_regime
            else None,
            f"Rates are {rates_context.get('direction') or 'unclear'}, inflation is {inflation_context.get('direction') or 'unclear'}, growth is {growth_context.get('interpretation') or growth_context.get('direction') or 'unclear'}, and liquidity conditions are {liquidity_context.get('regime') or 'unclear'}."
            if rates_context or inflation_context or growth_context or liquidity_context
            else None,
            f"Broad market tone is {broad_market.get('risk_tone')} across the major benchmark basket, and sector context says {sector_context.get('coverage_note')}."
            if broad_market or sector_context.get("coverage_note")
            else None,
            f"Macro fragility is {_fmt_num(macro_factor.get('macro_fragility_score'), digits=1, signed=False)} / 100, macro conflict is {_fmt_num(macro_factor.get('macro_conflict_score'), digits=1, signed=False)} / 100, and risk-on/risk-off alignment is {_fmt_num(macro_factor.get('risk_on_risk_off_alignment'), digits=1, signed=False)} / 100."
            if macro_factor
            else None,
            macro_entry.get("data_quality_note")
            if macro_entry.get("coverage_status") in {"partial", "stale", "unavailable"}
            else None,
            f"Geopolitical and policy sensitivity is registering at {_fmt_num(geo_score, digits=2, signed=False)} with category counts {geopolitical.get('category_counts') or geopolitical.get('event_buckets') or {}} and relevance {geopolitical.get('relevance_label') or 'unclear'}."
            if geo_score is not None
            else _coverage_note(geopolitical_entry, "Geopolitical"),
            f"Peer-relative context versus {relative.get('sector') or 'the local comparison set'} is running at {_fmt_num(relative.get('relative_strength_percentile') * 100 if relative.get('relative_strength_percentile') is not None else None, digits=0, signed=False)} percentile strength."
            if relative.get("peer_count") and relative.get("relative_strength_percentile") is not None
            else _coverage_note(relative_entry, "Peer-relative"),
            _fmt_list(macro_notes) if macro_notes else None,
            relative_summary.get("market_relative_note"),
            relative_summary.get("sector_relative_note"),
        ]
    )


def _event_catalyst_risk_text(
    event_domain: Dict[str, Any],
    signal_payload: Dict[str, Any],
) -> str:
    meta = event_domain.get("meta") or {}
    classification = str(event_domain.get("event_risk_classification") or "unknown").replace("_", " ")
    titles = event_domain.get("major_event_titles") or []
    return _join_sentences(
        [
            f"Event posture currently reads as {classification}, with event overhang {_fmt_num(event_domain.get('event_overhang_score'), digits=1, signed=False)} / 100, uncertainty {_fmt_num(event_domain.get('event_uncertainty_score'), digits=1, signed=False)} / 100, and catalyst burst {_fmt_num(event_domain.get('catalyst_burst_score'), digits=1, signed=False)} / 100."
            if event_domain
            else _coverage_note(meta, "Event"),
            f"Days to next estimated major event: {_fmt_num(event_domain.get('days_to_next_event'), digits=0, signed=False)}; days since the last major event: {_fmt_num(event_domain.get('days_since_last_major_event'), digits=0, signed=False)}."
            if event_domain.get("days_to_next_event") is not None or event_domain.get("days_since_last_major_event") is not None
            else None,
            f"Earnings-window flag is {event_domain.get('earnings_window_flag')}, post-event instability flag is {event_domain.get('post_event_instability_flag')}."
            if event_domain.get("earnings_window_flag") is not None or event_domain.get("post_event_instability_flag") is not None
            else None,
            f"Recent catalyst headlines are {_fmt_list(titles[:4])}."
            if titles
            else None,
            f"Event suppression flags currently active are {_fmt_list((signal_payload.get('suppression_flags') or []))}."
            if "event_overhang" in (signal_payload.get("suppression_flags") or [])
            else None,
            meta.get("data_quality_note"),
        ]
    )


def _liquidity_execution_fragility_text(
    liquidity_domain: Dict[str, Any],
    signal_payload: Dict[str, Any],
) -> str:
    meta = liquidity_domain.get("meta") or {}
    return _join_sentences(
        [
            f"Implementation posture reads as {str(liquidity_domain.get('tradability_state') or 'unknown').replace('_', ' ')}, with liquidity quality {_fmt_num(liquidity_domain.get('liquidity_quality_score'), digits=1, signed=False)} / 100, execution cleanliness {_fmt_num(liquidity_domain.get('execution_cleanliness_score'), digits=1, signed=False)} / 100, and implementation fragility {_fmt_num(liquidity_domain.get('implementation_fragility_score'), digits=1, signed=False)} / 100."
            if liquidity_domain
            else _coverage_note(meta, "Liquidity"),
            f"Gap instability is {_fmt_num(liquidity_domain.get('gap_instability_score'), digits=1, signed=False)} / 100, range instability is {_fmt_num(liquidity_domain.get('range_instability_score'), digits=1, signed=False)} / 100, turnover stability is {_fmt_num(liquidity_domain.get('turnover_stability_score'), digits=1, signed=False)} / 100, and friction proxy is {_fmt_num(liquidity_domain.get('friction_proxy_score'), digits=1, signed=False)} / 100."
            if liquidity_domain
            else None,
            f"Tradability caution is {_fmt_num(liquidity_domain.get('tradability_caution_score'), digits=1, signed=False)} / 100 and overnight gap risk is {_fmt_num(liquidity_domain.get('overnight_gap_risk_score'), digits=1, signed=False)} / 100."
            if liquidity_domain
            else None,
            "The canonical signal is explicitly suppressing confidence because the setup is implementation-fragile."
            if "implementation_fragility" in (signal_payload.get("suppression_flags") or [])
            else None,
            meta.get("data_quality_note"),
        ]
    )


def _market_breadth_internal_state_text(
    breadth_domain: Dict[str, Any],
    signal_payload: Dict[str, Any],
) -> str:
    meta = breadth_domain.get("meta") or {}
    breadth_state = str(breadth_domain.get("breadth_state") or "unknown").replace("_", " ")
    return _join_sentences(
        [
            f"Internal market state currently reads as {breadth_state}, with breadth confirmation {_fmt_num(breadth_domain.get('breadth_confirmation_score'), digits=1, signed=False)} / 100, participation breadth {_fmt_num(breadth_domain.get('participation_breadth_score'), digits=1, signed=False)} / 100, and breadth thrust {_fmt_num(breadth_domain.get('breadth_thrust_proxy'), digits=1, signed=False)} / 100."
            if breadth_domain
            else _coverage_note(meta, "Breadth"),
            f"Cross-sectional dispersion is {_fmt_num(breadth_domain.get('cross_sectional_dispersion_proxy'), digits=1, signed=False)} / 100, sector dispersion is {_fmt_num(breadth_domain.get('sector_dispersion_proxy'), digits=1, signed=False)} / 100, and internal divergence is {_fmt_num(breadth_domain.get('internal_market_divergence_score'), digits=1, signed=False)} / 100."
            if breadth_domain
            else None,
            f"Leadership concentration is {_fmt_num(breadth_domain.get('leadership_concentration_score'), digits=1, signed=False)} / 100, leader strength is {_fmt_num(breadth_domain.get('leader_strength_score'), digits=1, signed=False)} / 100, laggard pressure is {_fmt_num(breadth_domain.get('laggard_pressure_score'), digits=1, signed=False)} / 100, and leadership instability is {_fmt_num(breadth_domain.get('leadership_instability_score'), digits=1, signed=False)} / 100."
            if breadth_domain
            else None,
            "Breadth is not confirming the move cleanly, which is one of the reasons the raw score is being dampened."
            if "weak_breadth" in (signal_payload.get("suppression_flags") or [])
            else None,
            meta.get("data_quality_note"),
        ]
    )


def _cross_asset_confirmation_text(
    cross_asset_domain: Dict[str, Any],
    signal_payload: Dict[str, Any],
) -> str:
    meta = cross_asset_domain.get("meta") or {}
    return _join_sentences(
        [
            f"Cross-asset context is anchored to benchmark {cross_asset_domain.get('benchmark_proxy') or 'n/a'} and sector proxy {cross_asset_domain.get('sector_proxy') or 'n/a'}, with benchmark confirmation {_fmt_num(cross_asset_domain.get('benchmark_confirmation_score'), digits=1, signed=False)} / 100 and sector confirmation {_fmt_num(cross_asset_domain.get('sector_confirmation_score'), digits=1, signed=False)} / 100."
            if cross_asset_domain
            else _coverage_note(meta, "Cross-asset depth"),
            f"Macro-asset alignment is {_fmt_num(cross_asset_domain.get('macro_asset_alignment_score'), digits=1, signed=False)} / 100, beta context is {_fmt_num(cross_asset_domain.get('beta_context_score'), digits=1, signed=False)} / 100, and cross-asset conflict is {_fmt_num(cross_asset_domain.get('cross_asset_conflict_score'), digits=1, signed=False)} / 100."
            if cross_asset_domain
            else None,
            f"Cross-asset divergence is {_fmt_num(cross_asset_domain.get('cross_asset_divergence_score'), digits=1, signed=False)} / 100, with idiosyncratic strength {_fmt_num(cross_asset_domain.get('idiosyncratic_strength_score'), digits=1, signed=False)} / 100 versus idiosyncratic weakness {_fmt_num(cross_asset_domain.get('idiosyncratic_weakness_score'), digits=1, signed=False)} / 100."
            if cross_asset_domain
            else None,
            "Cross-asset contradiction is explicitly suppressing the canonical score."
            if "cross_asset_conflict" in (signal_payload.get("suppression_flags") or [])
            else None,
            meta.get("data_quality_note"),
        ]
    )


def _stress_spillover_text(
    stress_domain: Dict[str, Any],
    signal_payload: Dict[str, Any],
) -> str:
    meta = stress_domain.get("meta") or {}
    return _join_sentences(
        [
            f"Stress posture currently shows market stress {_fmt_num(stress_domain.get('market_stress_score'), digits=1, signed=False)} / 100, spillover risk {_fmt_num(stress_domain.get('spillover_risk_score'), digits=1, signed=False)} / 100, and contagion risk {_fmt_num(stress_domain.get('contagion_risk_proxy'), digits=1, signed=False)} / 100."
            if stress_domain
            else _coverage_note(meta, "Stress"),
            f"Correlation-breakdown proxy is {_fmt_num(stress_domain.get('correlation_breakdown_proxy'), digits=1, signed=False)} / 100, volatility shock is {_fmt_num(stress_domain.get('volatility_shock_score'), digits=1, signed=False)} / 100, and stress transition is {_fmt_num(stress_domain.get('stress_transition_score'), digits=1, signed=False)} / 100."
            if stress_domain
            else None,
            f"Defensive regime flag is {stress_domain.get('defensive_regime_flag')}, unstable environment flag is {stress_domain.get('unstable_environment_flag')}."
            if stress_domain.get("defensive_regime_flag") is not None or stress_domain.get("unstable_environment_flag") is not None
            else None,
            f"Active confidence-suppression notes are {_fmt_list(signal_payload.get('adjusted_confidence_notes') or [])}."
            if signal_payload.get("adjusted_confidence_notes")
            else None,
            meta.get("data_quality_note"),
        ]
    )


def _risk_quality_analysis_text(
    signal: Dict[str, Any],
    quality: Dict[str, Any],
    warnings: Sequence[str],
    why_signal: Dict[str, Any],
    strategy: Dict[str, Any],
    fragility: Dict[str, Any],
    domain_agreement: Dict[str, Any],
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
            f"Fragility markers show instability {_fmt_num(fragility.get('instability_score'), digits=1, signed=False)} / 100, clean setup {_fmt_num(fragility.get('clean_setup_score'), digits=1, signed=False)} / 100, and conflict-driven confidence penalty {_fmt_num(domain_agreement.get('confidence_penalty_from_conflict'), digits=1, signed=False)} / 100."
            if fragility or domain_agreement
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
    scenario_matrix = strategy.get("scenario_matrix") or {}
    execution = strategy.get("execution_posture") or {}
    confirmation_triggers = strategy.get("confirmation_triggers") or []
    return _join_sentences(
        [
            strategy.get("strategy_summary")
            or strategy.get("base_case")
            or f"The base case is to carry a {strategy_signal} stance on the {horizon} horizon.",
            f"Internal posture is {strategy.get('strategy_posture') or strategy_signal.lower()}, with actionability {_fmt_num(strategy.get('actionability_score'), digits=1, signed=False)} / 100, confidence {_fmt_num(strategy.get('confidence_score'), digits=1, signed=False)} / 100, and time-horizon fit {strategy.get('time_horizon_fit') or horizon}."
            if strategy
            else None,
            f"Base case: {(scenario_matrix.get('base') or {}).get('summary') or strategy.get('base_case')}."
            if scenario_matrix.get("base") or strategy.get("base_case")
            else None,
            f"Bull case: {(scenario_matrix.get('bull') or {}).get('summary') or strategy.get('upside_case')}."
            if scenario_matrix.get("bull") or strategy.get("upside_case")
            else None,
            f"Bear case: {(scenario_matrix.get('bear') or {}).get('summary') or strategy.get('downside_case')}."
            if scenario_matrix.get("bear") or strategy.get("downside_case")
            else None,
            f"Stress case: {(scenario_matrix.get('stress') or {}).get('summary') or strategy.get('stress_case')}."
            if scenario_matrix.get("stress") or strategy.get("stress_case")
            else None,
            f"Participant fit is {_fmt_list(participant_fit)}."
            if participant_fit
            else "Participant fit remains broad because the setup is not resolving into one narrow style bucket.",
            f"Execution posture prefers {(execution.get('preferred_posture') or 'staged_watch').replace('_', ' ')}, with urgency {(execution.get('urgency_level') or 'low').replace('_', ' ')} and patience {(execution.get('patience_level') or 'high').replace('_', ' ')}."
            if execution
            else None,
            f"Upgrade / confirmation triggers are {_fmt_list(confirmation_triggers)}."
            if confirmation_triggers
            else "No single confirmation trigger dominates; the engine is looking for a cleaner evidence stack rather than one isolated event.",
        ]
    )


def _evaluation_research_text(evaluation: Dict[str, Any]) -> str:
    if not evaluation:
        return (
            "Phase 6 evaluation is not attached to this report yet, so the platform is not adding a historical performance or calibration overlay to the current read."
        )
    return _join_sentences(
        [
            evaluation.get("evaluation_summary"),
            evaluation.get("confidence_reliability_summary"),
            evaluation.get("regime_usefulness_summary"),
        ]
    )


def _deployment_readiness_text(readiness: Dict[str, Any]) -> str:
    if not readiness:
        return (
            "Capital-readiness controls are not attached to this report yet, so the platform is not expressing a deployment-permission or staged live-use posture."
        )
    mode = (readiness.get("deployment_mode") or {}).get("active_mode") or "research_only"
    model = readiness.get("model_readiness") or {}
    permission = readiness.get("deployment_permission") or {}
    return _join_sentences(
        [
            f"Deployment mode is {mode} and current permission is {permission.get('deployment_permission') or 'analysis_only'}.",
            f"Model readiness is {model.get('model_readiness_status') or 'unknown'} at {_fmt_num(model.get('live_readiness_score'), digits=1, signed=False)} / 100.",
            f"Trust tier is {permission.get('trust_tier') or 'blocked'}, human review required is {permission.get('human_review_required')}, and minimum required review is {permission.get('minimum_required_review') or 'analyst_review'}.",
            f"Blockers are {_fmt_list(permission.get('deployment_blockers') or model.get('live_readiness_blockers') or [])}.",
        ]
    )


def _deployment_permission_text(readiness: Dict[str, Any]) -> str:
    if not readiness:
        return "Deployment permission analysis will attach once the capital-readiness layer is available."
    permission = readiness.get("deployment_permission") or {}
    admission = readiness.get("signal_admission_control") or {}
    drift = readiness.get("drift_monitor") or {}
    return _join_sentences(
        [
            permission.get("deployment_rationale"),
            f"Admission control says strategy={admission.get('admitted_for_strategy')}, paper={admission.get('admitted_for_paper')}, live={admission.get('admitted_for_live')}.",
            f"Primary admission decision is {admission.get('primary_decision') or 'unknown'}.",
            f"Pause recommended is {drift.get('pause_recommended')}, degrade-to-paper recommended is {drift.get('degrade_to_paper_recommended')}, and increased review required is {drift.get('increased_review_required')}.",
        ]
    )


def _risk_budget_text(readiness: Dict[str, Any]) -> str:
    if not readiness:
        return "Risk-budget and exposure-caution framing will attach once deployment controls are available."
    risk_budget = readiness.get("risk_budgeting") or {}
    return _join_sentences(
        [
            f"Risk budget tier is {risk_budget.get('risk_budget_tier') or 'none'} with {risk_budget.get('exposure_caution_level') or 'extreme'} exposure caution.",
            f"Fragility-adjusted size band is {risk_budget.get('fragility_adjusted_size_band') or 'none'}, confidence-adjusted size band is {risk_budget.get('confidence_adjusted_size_band') or 'none'}, and the maximum risk mode allowed is {risk_budget.get('maximum_risk_mode_allowed') or 'research_only'}.",
            risk_budget.get("concentration_warning"),
            risk_budget.get("diversification_warning"),
            risk_budget.get("correlation_context_warning"),
        ]
    )


def _rollout_stage_text(readiness: Dict[str, Any]) -> str:
    if not readiness:
        return "Rollout-stage guidance will attach once deployment controls are available."
    rollout = readiness.get("rollout_workflow") or {}
    return _join_sentences(
        [
            f"Current rollout stage is {rollout.get('rollout_stage') or 'historical_validation'}, readiness checkpoint is {rollout.get('readiness_checkpoint') or 'watch'}, and the next eligible stage is {rollout.get('next_eligible_stage') or 'none'}.",
            f"Promotion criteria are {_fmt_list(rollout.get('promotion_criteria') or [])}.",
            f"Demotion criteria are {_fmt_list(rollout.get('demotion_criteria') or [])}.",
            f"Stage transition notes are {_fmt_list(rollout.get('stage_transition_notes') or [])}.",
            f"Rollback reason is {rollout.get('rollback_reason')}."
            if rollout.get("rollback_reason")
            else None,
        ]
    )


def _portfolio_context_text(portfolio: Dict[str, Any]) -> str:
    if not portfolio:
        return (
            "Portfolio-construction overlays are not attached to this report yet, so the platform is not expressing rank, portfolio fit, or watchlist priority."
        )
    current = portfolio.get("current_candidate") or {}
    return _join_sentences(
        [
            f"Portfolio rank is {_fmt_num(current.get('portfolio_rank'), digits=0, signed=False)} of {_fmt_num(len(portfolio.get('cohort_ranking') or []), digits=0, signed=False)}, with ranked opportunity score {_fmt_num(current.get('ranked_opportunity_score'), digits=1, signed=False)} / 100 and portfolio candidate score {_fmt_num(current.get('portfolio_candidate_score'), digits=1, signed=False)} / 100.",
            f"Candidate classification is {current.get('candidate_classification') or 'watchlist_candidate'}, deployability rank is {_fmt_num(current.get('deployability_rank'), digits=1, signed=False)} / 100, and watchlist priority is {_fmt_num(current.get('watchlist_priority_score'), digits=1, signed=False)} / 100.",
            f"Size band is {current.get('size_band') or 'watchlist only'}, weight band is {current.get('weight_band') or '0.00x live weight'}, and caution level is {current.get('caution_level') or 'high'}.",
        ]
    )


def _portfolio_fit_text(portfolio: Dict[str, Any]) -> str:
    if not portfolio:
        return "Portfolio-fit analysis will attach once the portfolio-construction layer is available."
    current = portfolio.get("current_candidate") or {}
    return _join_sentences(
        [
            f"Overlap score is {_fmt_num(current.get('overlap_score'), digits=1, signed=False)} / 100, redundancy score is {_fmt_num(current.get('redundancy_score'), digits=1, signed=False)} / 100, and diversification contribution is {_fmt_num(current.get('diversification_contribution_score'), digits=1, signed=False)} / 100.",
            f"The most redundant tracked peer is {current.get('most_redundant_symbol') or 'none'}, and portfolio fit quality is {_fmt_num(current.get('portfolio_fit_quality'), digits=1, signed=False)} / 100.",
            f"Exposure warnings are {_fmt_list((current.get('active_warnings') or []))}."
            if current.get("active_warnings")
            else f"Diversification status is {current.get('diversification_status') or 'balanced'}.",
        ]
    )


def _execution_quality_text(portfolio: Dict[str, Any]) -> str:
    if not portfolio:
        return "Execution-quality analysis will attach once the portfolio-construction layer is available."
    current = portfolio.get("current_candidate") or {}
    return _join_sentences(
        [
            f"Execution quality is {_fmt_num(current.get('execution_quality_score'), digits=1, signed=False)} / 100 with friction penalty {_fmt_num(current.get('friction_penalty'), digits=1, signed=False)} and turnover penalty {_fmt_num(current.get('turnover_penalty'), digits=1, signed=False)}.",
            f"Urgency is {current.get('urgency_level') or 'measured'}, patience is {current.get('patience_level') or 'high'}, wait-for-better-entry is {current.get('wait_for_better_entry_flag')}, and confirmation preferred is {current.get('confirmation_preferred_flag')}.",
        ]
    )


def _portfolio_risk_model_text(risk_model: Dict[str, Any]) -> str:
    if not risk_model:
        return "Portfolio risk-model context will attach once the realized covariance and exposure layer runs."
    return str(
        risk_model.get("portfolio_risk_model_summary")
        or "Portfolio risk-model summary is unavailable."
    )


def _hidden_overlap_redundancy_text(risk_model: Dict[str, Any]) -> str:
    if not risk_model:
        return "Hidden overlap and redundancy analysis will attach once the portfolio risk-model layer runs."
    return str(
        risk_model.get("hidden_overlap_redundancy_analysis")
        or "Hidden overlap analysis is unavailable."
    )


def _factor_exposure_summary_text(risk_model: Dict[str, Any]) -> str:
    if not risk_model:
        return "Factor exposure summary will attach once the portfolio risk-model layer runs."
    return str(
        risk_model.get("factor_exposure_summary")
        or "Factor exposure summary is unavailable."
    )


def _concentration_cluster_risk_text(risk_model: Dict[str, Any]) -> str:
    if not risk_model:
        return "Concentration and cluster-risk analysis will attach once the portfolio risk-model layer runs."
    return str(
        risk_model.get("concentration_cluster_risk_analysis")
        or "Concentration and cluster-risk analysis is unavailable."
    )


def _replacement_diversification_text(risk_model: Dict[str, Any]) -> str:
    if not risk_model:
        return "Replacement and diversification notes will attach once the portfolio risk-model layer runs."
    return str(
        risk_model.get("replacement_diversification_analysis")
        or "Replacement and diversification analysis is unavailable."
    )


def _portfolio_stress_fragility_text(risk_model: Dict[str, Any]) -> str:
    if not risk_model:
        return "Portfolio stress and fragility overlays will attach once the portfolio risk-model layer runs."
    return str(
        risk_model.get("portfolio_stress_fragility_summary")
        or "Portfolio stress and fragility analysis is unavailable."
    )


def _risks_invalidators_text(
    why_signal: Dict[str, Any],
    strategy: Dict[str, Any],
) -> str:
    missing = why_signal.get("missing_data_warnings") or []
    freshness = why_signal.get("freshness_warnings") or []
    invalidators = strategy.get("invalidation_conditions") or []
    invalidation_map = strategy.get("invalidators") or {}
    vetoes = strategy.get("fragility_vetoes") or []
    uncertainty_notes = strategy.get("uncertainty_notes") or []
    return _join_sentences(
        [
            f"Key weaknesses are {_fmt_driver_list(why_signal.get('top_negative_drivers') or [])}.",
            f"Active fragility / veto dampeners are {_fmt_list(item.get('name', '').replace('_', ' ') for item in vetoes)}."
            if vetoes
            else "No hard fragility veto is active, but the setup is still being checked for instability and evidence weakness.",
            f"Missing-data warnings are {_fmt_list(missing)} and freshness warnings are {_fmt_list(freshness)}."
            if missing or freshness
            else "No major missing-data or freshness escalation is currently dominating the risk framing.",
            f"Formal invalidators are {_fmt_list(invalidators)}."
            if invalidators
            else "Formal invalidators remain scenario-dependent rather than singular.",
            f"Regime invalidators are {_fmt_list((invalidation_map.get('regime_invalidators') or []))}, narrative invalidators are {_fmt_list((invalidation_map.get('narrative_invalidators') or []))}, macro invalidators are {_fmt_list((invalidation_map.get('macro_invalidators') or []))}, and quality invalidators are {_fmt_list((invalidation_map.get('quality_invalidators') or []))}."
            if invalidation_map
            else None,
            f"Deterioration triggers are {_fmt_list((strategy.get('deterioration_triggers') or []))}."
            if strategy.get("deterioration_triggers")
            else None,
            f"Uncertainty notes are {_fmt_list(uncertainty_notes)}."
            if uncertainty_notes
            else None,
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
    domain_confidence = quality.get("domain_confidence") or {}
    confidence_text = ", ".join(
        f"{label}={_fmt_num(value, digits=1, signed=False)}"
        for label, value in domain_confidence.items()
        if value is not None
    )
    return _join_sentences(
        [
            f"Evidence provenance spans {_fmt_list(evidence.get('sources') or [])} plus normalized domains for market, technical, fundamentals, sentiment, macro/cross-asset, geopolitical, relative context, and quality.",
            f"Freshness status is {freshness_summary['overall_status']} with domain states {freshness_summary['domains']}.",
            f"Domain source map is {quality.get('source_map') or {}}, and fallback logic was used in {_fmt_list(fallback_domains)}."
            if fallback_domains
            else f"Domain source map is {quality.get('source_map') or {}}.",
            f"Domain confidence currently reads {confidence_text}."
            if confidence_text
            else None,
            f"Provider notes are {_fmt_list(quality.get('provider_notes') or [])}, external data fabric status is {(data_bundle.get('external_data_fabric') or {}).get('status') or 'disabled'}, reason codes are {_fmt_list(reason_codes)}, and strategy components are {strategy_component_scores}.",
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
    strategy = report.get("strategy") or {}
    canonical_lineage = report.get("canonical_lineage") or {}
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
                strategy.get("final_signal"),
                (report.get("signal") or {}).get("final_action"),
                (report.get("signal") or {}).get("action"),
            ),
            "conviction_tier": _first_available(
                strategy.get("conviction_tier"),
                report.get("conviction_tier"),
            ),
            "strategy_posture": _first_available(
                strategy.get("strategy_posture"),
                report.get("strategy_posture"),
            ),
            "actionability_score": _first_available(
                strategy.get("actionability_score"),
                report.get("actionability_score"),
            ),
            "freshness_status": _first_available(
                (report.get("freshness_summary") or {}).get("overall_status"),
                _freshness_status(report.get("data_bundle")),
            ),
            "report_version": report.get("report_version"),
            "strategy_version": _first_available(
                strategy.get("strategy_version"),
                report.get("strategy_version"),
            ),
            "snapshot_id": canonical_lineage.get("snapshot_id"),
            "snapshot_version": canonical_lineage.get("snapshot_version"),
            "feature_version": canonical_lineage.get("feature_version"),
            "signal_version": canonical_lineage.get("signal_version"),
            "deployment_mode": report.get("deployment_mode"),
            "deployment_permission": report.get("deployment_permission"),
            "trust_tier": report.get("trust_tier"),
            "live_readiness_status": report.get("model_readiness_status"),
            "live_readiness_score": report.get("live_readiness_score"),
            "rollout_stage": report.get("rollout_stage"),
            "candidate_classification": report.get("candidate_classification"),
            "ranked_opportunity_score": report.get("ranked_opportunity_score"),
            "portfolio_fit_quality": report.get("portfolio_fit_quality"),
            "size_band": report.get("size_band"),
            "portfolio_risk_model_version": report.get("portfolio_risk_model_version"),
            "hidden_overlap_score": report.get("hidden_overlap_score"),
            "portfolio_stress_score": report.get("portfolio_stress_score"),
            "replacement_candidate": report.get("replacement_candidate"),
            "setup_archetype": ((report.get("setup_archetype") or {}).get("archetype_name")),
            "research_version": report.get("research_version"),
            "learning_priority": report.get("learning_priority"),
            "validation_version": report.get("validation_version"),
        }
    )


def attach_evaluation_context(
    report: Dict[str, Any],
    evaluation: Dict[str, Any],
    *,
    prediction_record_id: Optional[str] = None,
    evaluation_artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    updated = sanitize_payload({**report})
    research_text = _evaluation_research_text(evaluation)
    updated["report_version"] = "2.1"
    updated["prediction_record_id"] = prediction_record_id
    updated["evaluation_artifact_id"] = evaluation_artifact_id
    updated["evaluation"] = evaluation
    updated["evaluation_summary"] = evaluation.get("evaluation_summary")
    updated["confidence_reliability_summary"] = evaluation.get(
        "confidence_reliability_summary"
    )
    updated["regime_usefulness_summary"] = evaluation.get("regime_usefulness_summary")
    updated["evaluation_research_analysis"] = research_text
    updated["overall_analysis"] = _join_sentences(
        [
            updated.get("overall_analysis"),
            f"Historical evaluation context: {evaluation.get('evaluation_summary')}"
            if evaluation.get("evaluation_summary")
            else None,
        ]
    )
    updated["strategy_view"] = _join_sentences(
        [
            updated.get("strategy_view"),
            f"Research scorecard context: {evaluation.get('confidence_reliability_summary')}"
            if evaluation.get("confidence_reliability_summary")
            else None,
        ]
    )
    updated["risk_quality_analysis"] = _join_sentences(
        [
            updated.get("risk_quality_analysis"),
            f"Regime and failure-mode context: {evaluation.get('regime_usefulness_summary')}"
            if evaluation.get("regime_usefulness_summary")
            else None,
        ]
    )
    updated["evidence_provenance"] = _join_sentences(
        [
            updated.get("evidence_provenance"),
            "Evaluation provenance is point-in-time: prediction records are stored at analysis time and only linked to forward bars once the horizon matures."
            if evaluation
            else None,
        ]
    )
    evidence_map = dict(updated.get("evidence_map") or {})
    evidence_map["evaluation_research_analysis"] = [
        "evaluation.prediction_linkage_summary",
        "evaluation.signal_scorecard",
        "evaluation.strategy_scorecard",
        "evaluation.calibration_summary",
        "evaluation.regime_breakdown",
        "evaluation.factor_attribution_summary",
    ]
    updated["evidence_map"] = evidence_map
    return sanitize_payload(updated)


def attach_deployment_context(
    report: Dict[str, Any],
    readiness: Dict[str, Any],
    *,
    readiness_artifact_id: Optional[str] = None,
    deployment_audit_artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    updated = sanitize_payload({**report})
    deployment_summary = _deployment_readiness_text(readiness)
    permission_text = _deployment_permission_text(readiness)
    risk_budget_text = _risk_budget_text(readiness)
    rollout_text = _rollout_stage_text(readiness)

    mode = readiness.get("deployment_mode") or {}
    model = readiness.get("model_readiness") or {}
    permission = readiness.get("deployment_permission") or {}
    risk_budget = readiness.get("risk_budgeting") or {}
    rollout = readiness.get("rollout_workflow") or {}
    drift = readiness.get("drift_monitor") or {}
    audit_snapshot = readiness.get("audit_snapshot") or {}

    updated["report_version"] = "2.2"
    updated["deployment_readiness_artifact_id"] = readiness_artifact_id
    updated["deployment_audit_artifact_id"] = deployment_audit_artifact_id
    updated["deployment_readiness"] = readiness
    updated["deployment_mode"] = mode.get("active_mode")
    updated["rollout_stage"] = rollout.get("rollout_stage")
    updated["deployment_permission"] = permission.get("deployment_permission")
    updated["deployment_blockers"] = permission.get("deployment_blockers") or []
    updated["deployment_rationale"] = permission.get("deployment_rationale")
    updated["trust_tier"] = permission.get("trust_tier")
    updated["minimum_required_review"] = permission.get("minimum_required_review")
    updated["human_review_required"] = permission.get("human_review_required")
    updated["model_readiness_status"] = model.get("model_readiness_status")
    updated["model_readiness_notes"] = model.get("model_readiness_notes") or []
    updated["live_readiness_score"] = model.get("live_readiness_score")
    updated["live_readiness_blockers"] = model.get("live_readiness_blockers") or []
    updated["recent_degradation_flags"] = model.get("recent_degradation_flags") or []
    updated["evidence_quality_summary"] = model.get("evidence_quality_summary")
    updated["risk_budget_tier"] = risk_budget.get("risk_budget_tier")
    updated["exposure_caution_level"] = risk_budget.get("exposure_caution_level")
    updated["fragility_adjusted_size_band"] = risk_budget.get("fragility_adjusted_size_band")
    updated["confidence_adjusted_size_band"] = risk_budget.get("confidence_adjusted_size_band")
    updated["maximum_risk_mode_allowed"] = risk_budget.get("maximum_risk_mode_allowed")
    updated["pause_recommended"] = drift.get("pause_recommended")
    updated["degrade_to_paper_recommended"] = drift.get("degrade_to_paper_recommended")
    updated["increased_review_required"] = drift.get("increased_review_required")
    updated["degraded_reliability_regimes"] = drift.get("degraded_reliability_regimes") or []
    updated["drift_alerts"] = drift.get("drift_alerts") or []
    updated["deployment_risk_alerts"] = drift.get("deployment_risk_alerts") or []
    updated["readiness_checkpoint"] = rollout.get("readiness_checkpoint")
    updated["promotion_criteria"] = rollout.get("promotion_criteria") or []
    updated["demotion_criteria"] = rollout.get("demotion_criteria") or []
    updated["rollback_reason"] = rollout.get("rollback_reason")
    updated["stage_transition_notes"] = rollout.get("stage_transition_notes") or []
    updated["live_use_audit_snapshot"] = audit_snapshot
    updated["deployment_readiness_summary"] = deployment_summary
    updated["deployment_permission_analysis"] = permission_text
    updated["risk_budget_exposure_analysis"] = risk_budget_text
    updated["rollout_stage_summary"] = rollout_text
    updated["overall_analysis"] = _join_sentences(
        [
            updated.get("overall_analysis"),
            f"Deployment-readiness overlay: {deployment_summary}",
        ]
    )
    updated["strategy_view"] = _join_sentences(
        [
            updated.get("strategy_view"),
            f"Deployment permission overlay: {permission_text}",
            f"Risk-budget overlay: {risk_budget_text}",
        ]
    )
    updated["risk_quality_analysis"] = _join_sentences(
        [
            updated.get("risk_quality_analysis"),
            f"Rollout and drift discipline: {rollout_text}",
            f"Deployment alerts: {_fmt_list(drift.get('deployment_risk_alerts') or drift.get('drift_alerts') or [])}.",
        ]
    )
    updated["evidence_provenance"] = _join_sentences(
        [
            updated.get("evidence_provenance"),
            "Deployment-readiness provenance is point-in-time and audit-backed: mode, blockers, trust tier, rollout stage, and review requirements are stored with each readiness decision.",
        ]
    )
    evidence_map = dict(updated.get("evidence_map") or {})
    evidence_map["deployment_readiness_summary"] = [
        "deployment_readiness.deployment_mode",
        "deployment_readiness.model_readiness",
        "deployment_readiness.signal_admission_control",
    ]
    evidence_map["deployment_permission_analysis"] = [
        "deployment_readiness.deployment_permission",
        "deployment_readiness.drift_monitor",
        "deployment_readiness.audit_snapshot",
    ]
    evidence_map["risk_budget_exposure_analysis"] = [
        "deployment_readiness.risk_budgeting",
        "deployment_readiness.model_readiness.component_scores",
    ]
    evidence_map["rollout_stage_summary"] = [
        "deployment_readiness.rollout_workflow",
        "deployment_readiness.prior_audit_summary",
    ]
    updated["evidence_map"] = evidence_map
    return sanitize_payload(updated)


def attach_portfolio_context(
    report: Dict[str, Any],
    portfolio: Dict[str, Any],
    *,
    portfolio_construction_artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    updated = sanitize_payload({**report})
    current = portfolio.get("current_candidate") or {}
    workflow = portfolio.get("workflow") or {}
    portfolio_context_text = _portfolio_context_text(portfolio)
    portfolio_fit_text = _portfolio_fit_text(portfolio)
    execution_quality_text = _execution_quality_text(portfolio)

    updated["report_version"] = "2.3"
    updated["portfolio_construction_artifact_id"] = portfolio_construction_artifact_id
    updated["portfolio_construction"] = portfolio
    updated["ranked_opportunity_score"] = current.get("ranked_opportunity_score")
    updated["portfolio_candidate_score"] = current.get("portfolio_candidate_score")
    updated["watchlist_priority_score"] = current.get("watchlist_priority_score")
    updated["deployability_rank"] = current.get("deployability_rank")
    updated["quality_vs_fragility_ratio"] = current.get("quality_vs_fragility_ratio")
    updated["confidence_adjusted_rank"] = current.get("confidence_adjusted_rank")
    updated["conviction_adjusted_rank"] = current.get("conviction_adjusted_rank")
    updated["candidate_classification"] = current.get("candidate_classification")
    updated["candidate_blockers"] = current.get("candidate_blockers") or []
    updated["portfolio_rank"] = current.get("portfolio_rank")
    updated["overlap_score"] = current.get("overlap_score")
    updated["redundancy_score"] = current.get("redundancy_score")
    updated["diversification_contribution_score"] = current.get("diversification_contribution_score")
    updated["most_redundant_symbol"] = current.get("most_redundant_symbol")
    updated["cluster_membership"] = current.get("cluster_membership") or []
    updated["exposure_family"] = current.get("exposure_family")
    updated["portfolio_fit_quality"] = current.get("portfolio_fit_quality")
    updated["concentration_warning"] = current.get("concentration_warning")
    updated["cluster_concentration_warning"] = current.get("cluster_concentration_warning")
    updated["sector_crowding_warning"] = current.get("sector_crowding_warning")
    updated["fragility_cluster_warning"] = current.get("fragility_cluster_warning")
    updated["macro_exposure_warning"] = current.get("macro_exposure_warning")
    updated["theme_exposure_warning"] = current.get("theme_exposure_warning")
    updated["diversification_status"] = current.get("diversification_status")
    updated["size_band"] = current.get("size_band")
    updated["weight_band"] = current.get("weight_band")
    updated["risk_budget_band"] = current.get("risk_budget_band")
    updated["fragility_adjustment"] = current.get("fragility_adjustment")
    updated["confidence_adjustment"] = current.get("confidence_adjustment")
    updated["concentration_adjustment"] = current.get("concentration_adjustment")
    updated["overlap_adjustment"] = current.get("overlap_adjustment")
    updated["deployment_mode_adjustment"] = current.get("deployment_mode_adjustment")
    updated["max_priority_allowed"] = current.get("max_priority_allowed")
    updated["caution_level"] = current.get("caution_level")
    updated["execution_quality_score"] = current.get("execution_quality_score")
    updated["friction_penalty"] = current.get("friction_penalty")
    updated["turnover_penalty"] = current.get("turnover_penalty")
    updated["wait_for_better_entry_flag"] = current.get("wait_for_better_entry_flag")
    updated["confirmation_preferred_flag"] = current.get("confirmation_preferred_flag")
    updated["portfolio_context_summary"] = portfolio_context_text
    updated["portfolio_fit_analysis"] = portfolio_fit_text
    updated["execution_quality_analysis"] = execution_quality_text
    updated["portfolio_workflow_summary"] = portfolio.get("portfolio_workflow_summary")
    updated["top_peer_overlaps"] = portfolio.get("top_peer_overlaps") or []
    updated["cohort_ranking"] = portfolio.get("cohort_ranking") or []
    updated["priority_shift_flag"] = workflow.get("priority_shift_flag")
    updated["rebalance_attention_flag"] = workflow.get("rebalance_attention_flag")
    updated["candidate_upgrade_reason"] = workflow.get("candidate_upgrade_reason")
    updated["candidate_downgrade_reason"] = workflow.get("candidate_downgrade_reason")
    updated["replacement_candidate_notes"] = workflow.get("replacement_candidate_notes")
    updated["rotation_pressure_score"] = workflow.get("rotation_pressure_score")
    updated["overall_analysis"] = _join_sentences(
        [
            updated.get("overall_analysis"),
            f"Portfolio context: {portfolio_context_text}",
        ]
    )
    updated["strategy_view"] = _join_sentences(
        [
            updated.get("strategy_view"),
            f"Portfolio fit: {portfolio_fit_text}",
            f"Execution quality: {execution_quality_text}",
        ]
    )
    updated["risk_quality_analysis"] = _join_sentences(
        [
            updated.get("risk_quality_analysis"),
            f"Workflow and rotation: {portfolio.get('portfolio_workflow_summary')}",
        ]
    )
    updated["evidence_provenance"] = _join_sentences(
        [
            updated.get("evidence_provenance"),
            "Portfolio-construction provenance is cohort-based: rank, overlap, diversification, and size-band outputs are computed from the stored report cohort available at analysis time.",
        ]
    )
    evidence_map = dict(updated.get("evidence_map") or {})
    evidence_map["portfolio_context_summary"] = [
        "portfolio_construction.current_candidate",
        "portfolio_construction.cohort_ranking",
    ]
    evidence_map["portfolio_fit_analysis"] = [
        "portfolio_construction.current_candidate.peer_overlaps",
        "portfolio_construction.current_candidate.active_warnings",
    ]
    evidence_map["execution_quality_analysis"] = [
        "portfolio_construction.current_candidate.execution_quality_score",
        "portfolio_construction.current_candidate.friction_penalty",
        "portfolio_construction.current_candidate.turnover_penalty",
    ]
    updated["evidence_map"] = evidence_map
    return sanitize_payload(updated)


def attach_portfolio_risk_context(
    report: Dict[str, Any],
    portfolio_risk_model: Dict[str, Any],
    *,
    portfolio_risk_model_artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    updated = sanitize_payload({**report})
    current = portfolio_risk_model.get("current_candidate") or {}
    overlay = portfolio_risk_model.get("portfolio_overlay") or {}
    risk_summary = _portfolio_risk_model_text(portfolio_risk_model)
    overlap_text = _hidden_overlap_redundancy_text(portfolio_risk_model)
    exposure_text = _factor_exposure_summary_text(portfolio_risk_model)
    concentration_text = _concentration_cluster_risk_text(portfolio_risk_model)
    replacement_text = _replacement_diversification_text(portfolio_risk_model)
    stress_text = _portfolio_stress_fragility_text(portfolio_risk_model)

    updated["report_version"] = "2.6"
    updated["portfolio_risk_model_artifact_id"] = portfolio_risk_model_artifact_id
    updated["portfolio_risk_model"] = portfolio_risk_model
    updated["portfolio_risk_model_version"] = portfolio_risk_model.get(
        "portfolio_risk_model_version"
    )
    updated["portfolio_fit_rank"] = current.get("portfolio_fit_rank")
    updated["marginal_portfolio_utility"] = current.get("marginal_portfolio_utility")
    updated["portfolio_contribution_score"] = current.get("portfolio_contribution_score")
    updated["marginal_fragility_penalty"] = current.get("marginal_fragility_penalty")
    updated["marginal_diversification_bonus"] = current.get(
        "marginal_diversification_bonus"
    )
    updated["replacement_value_score"] = current.get("replacement_value_score")
    updated["replacement_candidate"] = current.get("replacement_candidate")
    updated["substitution_score"] = current.get("substitution_score")
    updated["better_alternative_flag"] = current.get("better_alternative_flag")
    updated["diversification_upgrade_flag"] = current.get(
        "diversification_upgrade_flag"
    )
    updated["overlap_reduction_flag"] = current.get("overlap_reduction_flag")
    updated["portfolio_quality_upgrade_reason"] = current.get(
        "portfolio_quality_upgrade_reason"
    )
    updated["hidden_overlap_score"] = current.get("hidden_overlap_score")
    updated["complementarity_score"] = current.get("complementarity_score")
    updated["overlap_rationale"] = current.get("overlap_rationale")
    updated["overlap_drivers"] = current.get("overlap_drivers") or []
    updated["overlap_confidence"] = current.get("overlap_confidence")
    updated["factor_exposure_vector"] = current.get("factor_exposure_vector") or {}
    updated["factor_loading_summary"] = current.get("factor_loading_summary") or []
    updated["style_affinity"] = current.get("style_affinity")
    updated["macro_exposure_profile"] = current.get("macro_exposure_profile") or {}
    updated["fragility_exposure_profile"] = current.get(
        "fragility_exposure_profile"
    ) or {}
    updated["exposure_confidence"] = current.get("exposure_confidence")
    updated["exposure_cluster"] = current.get("exposure_cluster")
    updated["concentration_profile"] = overlay.get("concentration_profile") or {}
    updated["hidden_concentration_score"] = overlay.get("hidden_concentration_score")
    updated["cluster_risk_score"] = overlay.get("cluster_risk_score")
    updated["fragility_cluster_risk"] = overlay.get("fragility_cluster_risk")
    updated["event_cluster_risk"] = overlay.get("event_cluster_risk")
    updated["macro_cluster_risk"] = overlay.get("macro_cluster_risk")
    updated["narrative_cluster_risk"] = overlay.get("narrative_cluster_risk")
    updated["diversification_health_score"] = overlay.get(
        "diversification_health_score"
    )
    updated["portfolio_stress_score"] = overlay.get("portfolio_stress_score")
    updated["portfolio_fragility_score"] = overlay.get("portfolio_fragility_score")
    updated["correlation_breakdown_risk"] = overlay.get(
        "correlation_breakdown_risk"
    )
    updated["stress_concentration_warning"] = overlay.get(
        "stress_concentration_warning"
    )
    updated["clustered_gap_risk_warning"] = overlay.get(
        "clustered_gap_risk_warning"
    )
    updated["unstable_portfolio_state_flag"] = overlay.get(
        "unstable_portfolio_state_flag"
    )
    updated["portfolio_risk_warnings"] = overlay.get("portfolio_risk_warnings") or []
    updated["candidate_classification"] = current.get("candidate_classification") or updated.get(
        "candidate_classification"
    )
    updated["candidate_blockers"] = current.get("candidate_blockers") or updated.get(
        "candidate_blockers"
    ) or []
    updated["portfolio_fit_quality"] = current.get("portfolio_fit_quality") or updated.get(
        "portfolio_fit_quality"
    )
    updated["portfolio_rank"] = current.get("portfolio_fit_rank") or updated.get(
        "portfolio_rank"
    )
    updated["overlap_score"] = current.get("overlap_score") or updated.get("overlap_score")
    updated["redundancy_score"] = current.get("redundancy_score") or updated.get(
        "redundancy_score"
    )
    updated["diversification_contribution_score"] = current.get(
        "diversification_contribution_score"
    ) or updated.get("diversification_contribution_score")
    updated["most_redundant_symbol"] = current.get("most_redundant_symbol") or updated.get(
        "most_redundant_symbol"
    )
    updated["cohort_ranking"] = portfolio_risk_model.get("cohort_portfolio_risk_ranking") or updated.get(
        "cohort_ranking"
    ) or []
    updated["top_peer_overlaps"] = current.get("pairwise_relationships") or updated.get(
        "top_peer_overlaps"
    ) or []
    updated["cohort_risk_ranking"] = (
        portfolio_risk_model.get("cohort_portfolio_risk_ranking") or []
    )
    updated["portfolio_risk_model_summary"] = risk_summary
    updated["hidden_overlap_redundancy_analysis"] = overlap_text
    updated["factor_exposure_summary"] = exposure_text
    updated["concentration_cluster_risk_analysis"] = concentration_text
    updated["replacement_diversification_analysis"] = replacement_text
    updated["portfolio_stress_fragility_summary"] = stress_text
    updated["portfolio_context_summary"] = _join_sentences(
        [
            updated.get("portfolio_context_summary"),
            risk_summary,
        ]
    )
    updated["portfolio_fit_analysis"] = _join_sentences(
        [
            updated.get("portfolio_fit_analysis"),
            overlap_text,
            concentration_text,
        ]
    )
    updated["execution_quality_analysis"] = _join_sentences(
        [
            updated.get("execution_quality_analysis"),
            replacement_text,
            stress_text,
        ]
    )
    updated["overall_analysis"] = _join_sentences(
        [
            updated.get("overall_analysis"),
            f"Portfolio risk model: {risk_summary}",
        ]
    )
    updated["strategy_view"] = _join_sentences(
        [
            updated.get("strategy_view"),
            f"Hidden overlap: {overlap_text}",
            f"Factor exposures: {exposure_text}",
        ]
    )
    updated["risk_quality_analysis"] = _join_sentences(
        [
            updated.get("risk_quality_analysis"),
            f"Concentration and cluster risk: {concentration_text}",
            f"Portfolio stress and fragility: {stress_text}",
        ]
    )
    updated["evidence_provenance"] = _join_sentences(
        [
            updated.get("evidence_provenance"),
            "Phase 11 portfolio risk-model provenance uses point-in-time realized return histories where available, falls back to low-confidence synthetic proxies when history is thin, and combines covariance, factor exposure, overlap, substitution, and portfolio-stress overlays into the current portfolio-fit view.",
        ]
    )
    evidence_map = dict(updated.get("evidence_map") or {})
    evidence_map["portfolio_risk_model_summary"] = [
        "portfolio_risk_model.current_candidate",
        "portfolio_risk_model.portfolio_overlay",
    ]
    evidence_map["hidden_overlap_redundancy_analysis"] = [
        "portfolio_risk_model.top_pairwise_relationships",
        "portfolio_risk_model.current_candidate.overlap_*",
    ]
    evidence_map["factor_exposure_summary"] = [
        "portfolio_risk_model.current_candidate.factor_exposure_vector",
        "portfolio_risk_model.current_candidate.macro_exposure_profile",
    ]
    evidence_map["concentration_cluster_risk_analysis"] = [
        "portfolio_risk_model.portfolio_overlay.concentration_profile",
        "portfolio_risk_model.portfolio_overlay.portfolio_risk_warnings",
    ]
    evidence_map["replacement_diversification_analysis"] = [
        "portfolio_risk_model.current_candidate.replacement_candidate",
        "portfolio_risk_model.cohort_portfolio_risk_ranking",
    ]
    evidence_map["portfolio_stress_fragility_summary"] = [
        "portfolio_risk_model.portfolio_overlay.portfolio_stress_score",
        "portfolio_risk_model.portfolio_overlay.correlation_breakdown_risk",
    ]
    updated["evidence_map"] = evidence_map
    return sanitize_payload(updated)


def _learning_summary_text(learning: Dict[str, Any]) -> str:
    active = learning.get("active_setup_archetype") or {}
    queue = learning.get("improvement_queue") or []
    drift_alerts = learning.get("drift_alerts") or []
    top_queue = queue[0] if queue else {}
    top_drift = drift_alerts[0] if drift_alerts else {}
    return (
        f"Phase 10 continuous learning classifies the active setup as {active.get('archetype_name') or 'unknown'}, "
        f"with deployment caution {active.get('deployment_caution_level') or 'unknown'}. "
        f"Top improvement priority is {top_queue.get('title') or 'maintain observation mode'}, "
        f"and the leading drift issue is {top_drift.get('affected_component') or 'none active'}."
    )


def _regime_learning_text(learning: Dict[str, Any]) -> str:
    return str(
        learning.get("regime_learning_summary")
        or "Regime-conditioned learning is not yet populated."
    )


def _adaptation_queue_text(learning: Dict[str, Any]) -> str:
    return str(
        learning.get("adaptation_queue_summary")
        or "Adaptation and reweighting candidates are not yet populated."
    )


def _experiment_registry_text(learning: Dict[str, Any]) -> str:
    return str(
        learning.get("experiment_registry_summary")
        or "Experiment-registry context is not yet populated."
    )


def _archetype_motif_text(learning: Dict[str, Any]) -> str:
    return str(
        learning.get("archetype_motif_summary")
        or "Archetype and motif context is not yet populated."
    )


def attach_learning_context(
    report: Dict[str, Any],
    learning: Dict[str, Any],
    *,
    learning_artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    updated = sanitize_payload({**report})
    active = learning.get("active_setup_archetype") or {}
    motifs = learning.get("motif_discovery") or {}
    experiment_registry = learning.get("experiment_registry") or {}
    top_queue = (learning.get("improvement_queue") or [{}])[0]
    top_drift = (learning.get("drift_alerts") or [{}])[0]
    learning_summary = _learning_summary_text(learning)
    regime_summary = _regime_learning_text(learning)
    adaptation_summary = _adaptation_queue_text(learning)
    experiment_summary = _experiment_registry_text(learning)
    archetype_summary = _archetype_motif_text(learning)

    updated["report_version"] = "2.4"
    updated["learning_artifact_id"] = learning_artifact_id
    updated["continuous_learning"] = learning
    updated["research_version"] = learning.get("continuous_learning_version")
    updated["setup_archetype"] = active
    updated["active_motifs"] = motifs.get("active_motifs") or []
    updated["regime_conditioned_learnings"] = learning.get("regime_conditioned_learnings") or []
    updated["reweighting_candidates"] = learning.get("reweighting_candidates") or []
    updated["research_hypotheses"] = learning.get("research_hypotheses") or []
    updated["interaction_candidates"] = learning.get("feature_interaction_candidates") or []
    updated["learning_drift_alerts"] = learning.get("drift_alerts") or []
    updated["experiment_registry"] = experiment_registry
    updated["signal_family_library"] = learning.get("signal_family_library") or {}
    updated["motif_library"] = motifs.get("motif_library") or []
    updated["improvement_queue"] = learning.get("improvement_queue") or []
    updated["learning_priority"] = top_queue.get("priority") or top_drift.get("severity") or "observe"
    updated["learning_summary"] = learning_summary
    updated["regime_learning_summary"] = regime_summary
    updated["adaptation_queue_summary"] = adaptation_summary
    updated["experiment_registry_summary"] = experiment_summary
    updated["archetype_motif_summary"] = archetype_summary
    updated["overall_analysis"] = _join_sentences(
        [
            updated.get("overall_analysis"),
            f"Continuous-learning overlay: {learning_summary}",
        ]
    )
    updated["evaluation_research_analysis"] = _join_sentences(
        [
            updated.get("evaluation_research_analysis"),
            f"Regime learning: {regime_summary}",
            f"Adaptation queue: {adaptation_summary}",
            f"Experiment registry: {experiment_summary}",
            f"Archetypes and motifs: {archetype_summary}",
        ]
    )
    updated["strategy_view"] = _join_sentences(
        [
            updated.get("strategy_view"),
            f"Setup archetype: {active.get('archetype_name') or 'unknown'} with deployment caution {active.get('deployment_caution_level') or 'unknown'}.",
        ]
    )
    updated["risk_quality_analysis"] = _join_sentences(
        [
            updated.get("risk_quality_analysis"),
            f"Learning drift monitor: {top_drift.get('affected_component') or 'no acute drift alert'}; {top_drift.get('severity') or 'stable'} severity.",
        ]
    )
    updated["evidence_provenance"] = _join_sentences(
        [
            updated.get("evidence_provenance"),
            "Continuous-learning provenance is derived from stored evaluation, deployment-readiness, and portfolio-construction artifacts. Adaptation ideas remain proposals until reviewed and approved.",
        ]
    )
    evidence_map = dict(updated.get("evidence_map") or {})
    evidence_map["learning_summary"] = [
        "continuous_learning.active_setup_archetype",
        "continuous_learning.improvement_queue",
        "continuous_learning.drift_alerts",
    ]
    evidence_map["regime_learning_summary"] = [
        "continuous_learning.regime_conditioned_learnings",
    ]
    evidence_map["adaptation_queue_summary"] = [
        "continuous_learning.reweighting_candidates",
        "continuous_learning.research_hypotheses",
    ]
    evidence_map["experiment_registry_summary"] = [
        "continuous_learning.experiment_registry",
    ]
    evidence_map["archetype_motif_summary"] = [
        "continuous_learning.signal_family_library",
        "continuous_learning.motif_discovery",
    ]
    updated["evidence_map"] = evidence_map
    return sanitize_payload(updated)


def _canonical_validation_summary_text(validation: Dict[str, Any]) -> str:
    return str(
        validation.get("validation_summary")
        or "Canonical research-truth validation is not yet populated."
    )


def _walkforward_validation_text(validation: Dict[str, Any]) -> str:
    return str(
        validation.get("walkforward_validation_summary")
        or "Walk-forward validation is not yet populated."
    )


def _net_of_friction_validation_text(validation: Dict[str, Any]) -> str:
    return str(
        validation.get("net_of_friction_summary")
        or "Net-of-friction validation is not yet populated."
    )


def _suppression_readiness_validation_text(validation: Dict[str, Any]) -> str:
    return str(
        validation.get("suppression_readiness_validation_summary")
        or "Suppression and readiness validation is not yet populated."
    )


def _drawdown_invalidation_validation_text(validation: Dict[str, Any]) -> str:
    return str(
        validation.get("drawdown_invalidation_summary")
        or "MAE / MFE / invalidation validation is not yet populated."
    )


def attach_canonical_validation_context(
    report: Dict[str, Any],
    validation: Dict[str, Any],
    *,
    canonical_validation_artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    updated = sanitize_payload({**report})
    validation_summary = _canonical_validation_summary_text(validation)
    walkforward_summary = _walkforward_validation_text(validation)
    net_summary = _net_of_friction_validation_text(validation)
    suppression_summary = _suppression_readiness_validation_text(validation)
    drawdown_summary = _drawdown_invalidation_validation_text(validation)

    updated["report_version"] = "2.5"
    updated["canonical_validation_artifact_id"] = canonical_validation_artifact_id
    updated["canonical_validation"] = validation
    updated["validation_version"] = validation.get("validation_version")
    updated["canonical_validation_summary"] = validation_summary
    updated["walkforward_validation_summary"] = walkforward_summary
    updated["net_of_friction_validation_summary"] = net_summary
    updated["suppression_readiness_validation_summary"] = suppression_summary
    updated["drawdown_invalidation_validation_summary"] = drawdown_summary
    updated["overall_analysis"] = _join_sentences(
        [
            updated.get("overall_analysis"),
            f"Canonical research truth: {validation_summary}",
        ]
    )
    updated["evaluation_research_analysis"] = _join_sentences(
        [
            updated.get("evaluation_research_analysis"),
            f"Walk-forward: {walkforward_summary}",
            f"Net of friction: {net_summary}",
            f"Suppression and readiness: {suppression_summary}",
            f"Drawdown and invalidation: {drawdown_summary}",
        ]
    )
    updated["strategy_view"] = _join_sentences(
        [
            updated.get("strategy_view"),
            f"Canonical validation says: {suppression_summary}",
        ]
    )
    updated["risk_quality_analysis"] = _join_sentences(
        [
            updated.get("risk_quality_analysis"),
            f"Excursion and invalidation truth: {drawdown_summary}",
        ]
    )
    updated["evidence_provenance"] = _join_sentences(
        [
            updated.get("evidence_provenance"),
            "Canonical research-truth provenance is point-in-time: the backtest and walk-forward layer validates the same unified feature and signal engine, and assistant validation artifacts replay stored prediction records against later realized bars.",
        ]
    )
    evidence_map = dict(updated.get("evidence_map") or {})
    evidence_map["canonical_validation_summary"] = [
        "canonical_validation.prediction_linkage_summary",
        "canonical_validation.signal_scorecard",
        "canonical_validation.net_return_summary",
    ]
    evidence_map["walkforward_validation_summary"] = [
        "canonical_validation.walkforward_summary",
        "canonical_validation.walkforward_runs",
    ]
    evidence_map["net_of_friction_validation_summary"] = [
        "canonical_validation.gross_return_summary",
        "canonical_validation.net_return_summary",
        "canonical_validation.friction_cost_summary",
        "canonical_validation.liquidity_bucket_cost_summary",
    ]
    evidence_map["suppression_readiness_validation_summary"] = [
        "canonical_validation.readiness_scorecard",
        "canonical_validation.suppression_effect_summary",
        "canonical_validation.ranking_scorecard",
    ]
    evidence_map["drawdown_invalidation_validation_summary"] = [
        "canonical_validation.mae_mfe_summary",
        "canonical_validation.failure_modes",
    ]
    updated["evidence_map"] = evidence_map
    return sanitize_payload(updated)


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
    canonical_lineage = sanitize_payload(
        (job_context.get("canonical_lineage") or {})
    )

    strategy_signal = (strategy.get("final_signal") or action).upper()
    strategy_confidence = _first_available(strategy.get("confidence"), confidence)
    confidence_score = strategy.get("confidence_score")
    conviction_tier = strategy.get("conviction_tier") or "unknown"
    fragility_tier = strategy.get("fragility_tier") or "unknown"
    strategy_posture = strategy.get("strategy_posture") or strategy_signal.lower()
    actionability_score = strategy.get("actionability_score")
    participant_fit = strategy.get("participant_fit") or []
    execution_posture = strategy.get("execution_posture") or {}
    scenario_matrix = strategy.get("scenario_matrix") or {}
    invalidation_map = strategy.get("invalidators") or {}
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
    canonical_feature_vector = ((data_bundle.get("canonical_alpha_core") or {}).get("feature_vector") or {})
    canonical_signal_payload = ((data_bundle.get("canonical_alpha_core") or {}).get("signal_payload") or {})
    event_catalyst_risk = data_bundle.get("event_catalyst_risk") or {}
    liquidity_execution_fragility = data_bundle.get("liquidity_execution_fragility") or {}
    market_breadth_internals = data_bundle.get("market_breadth_internals") or {}
    cross_asset_confirmation = data_bundle.get("cross_asset_confirmation") or {}
    stress_spillover_conditions = data_bundle.get("stress_spillover_conditions") or {}
    composites = feature_factor_bundle.get("composite_intelligence") or {}
    proprietary_scores = feature_factor_bundle.get("proprietary_scores") or {}
    factor_groups = feature_factor_bundle.get("factor_groups") or {}
    regime_intelligence = feature_factor_bundle.get("regime_intelligence") or feature_factor_bundle.get("regime_engine") or {}
    fragility_intelligence = feature_factor_bundle.get("fragility_intelligence") or feature_factor_bundle.get("volatility_risk_microstructure") or {}
    sentiment_intelligence = feature_factor_bundle.get("sentiment_narrative_intelligence") or feature_factor_bundle.get("sentiment_intelligence") or {}
    macro_intelligence = feature_factor_bundle.get("macro_alignment") or feature_factor_bundle.get("macro_sensitivity") or {}
    domain_agreement = feature_factor_bundle.get("domain_agreement") or {}
    conviction_components = feature_factor_bundle.get("conviction_components") or {}
    opportunity_quality_components = feature_factor_bundle.get("opportunity_quality_components") or {}
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
        "confidence_score": confidence_score,
        "conviction_tier": conviction_tier,
        "fragility_tier": fragility_tier,
        "strategy_posture": strategy_posture,
        "actionability_score": actionability_score,
    }

    signal_summary = _join_sentences(
        [
            f"As of {as_of_text}, the assistant pipeline lands on a {strategy_signal} posture for {symbol}, framed on the {horizon} horizon under {risk_mode} risk mode.",
            f"The underlying signal engine prints {action} with score {_fmt_num(score)} and confidence {_fmt_num(confidence)}, while the strategy layer converts that into {strategy_signal} / {strategy_posture} with probability-like confidence {_fmt_num(strategy_confidence)} ({_fmt_num(confidence_score, digits=1, signed=False)} / 100), {conviction_tier} conviction, {fragility_tier} fragility, and actionability {_fmt_num(actionability_score, digits=1, signed=False)} / 100.",
            f"Structural quality is {_fmt_num(composites.get('Market Structure Integrity Score'), digits=1, signed=False)} / 100, regime stability is {_fmt_num(composites.get('Regime Stability Score'), digits=1, signed=False)} / 100, signal fragility is {_fmt_num(composites.get('Signal Fragility Index'), digits=1, signed=False)} / 100, and opportunity quality is {_fmt_num(composites.get('Opportunity Quality Score'), digits=1, signed=False)} / 100.",
            f"Depth overlays currently show event risk {str(event_catalyst_risk.get('event_risk_classification') or 'unknown').replace('_', ' ')}, implementation fragility {_fmt_num(liquidity_execution_fragility.get('implementation_fragility_score'), digits=1, signed=False)} / 100, breadth state {str(market_breadth_internals.get('breadth_state') or 'unknown').replace('_', ' ')}, cross-asset conflict {_fmt_num(cross_asset_confirmation.get('cross_asset_conflict_score'), digits=1, signed=False)} / 100, and market stress {_fmt_num(stress_spillover_conditions.get('market_stress_score'), digits=1, signed=False)} / 100."
            if event_catalyst_risk or liquidity_execution_fragility or market_breadth_internals or cross_asset_confirmation or stress_spillover_conditions
            else None,
            f"Cross-domain agreement is {_fmt_num(domain_agreement.get('domain_agreement_score'), digits=1, signed=False)} / 100 versus conflict {_fmt_num(domain_agreement.get('domain_conflict_score'), digits=1, signed=False)} / 100; the strongest confirming domains are {_fmt_list(item.get('domain') for item in (domain_agreement.get('strongest_confirming_domains') or []))}, while conflicts are concentrated in {_fmt_list(item.get('domain') for item in (domain_agreement.get('strongest_conflicting_domains') or []))}.",
            f"The dominant regime reads {regime}, freshness is {freshness_summary['overall_status']}, participant fit is {_fmt_list(participant_fit)}, and the main positive drivers are {_fmt_driver_list(why_signal['top_positive_drivers'])}.",
            f"The main risks are {_fmt_driver_list(why_signal['top_negative_drivers'])}, with scenario framing set to {job_context.get('scenario') or 'base'} and execution posture {(execution_posture.get('preferred_posture') or 'staged_watch').replace('_', ' ')}.",
            f"Canonical suppression flags are {_fmt_list(canonical_signal_payload.get('suppression_flags') or [])}."
            if canonical_signal_payload.get("suppression_flags")
            else None,
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
        fragility_intelligence,
        domain_agreement,
        quality,
        data_bundle,
    )

    sentiment_analysis = _sentiment_analysis_text(
        sentiment,
        sentiment_intelligence,
        composites,
        data_bundle,
    )

    macro_geopolitical_analysis = _macro_geopolitical_analysis_text(
        macro,
        macro_intelligence,
        geopolitical,
        relative,
        data_bundle,
    )
    event_catalyst_risk_analysis = _event_catalyst_risk_text(
        event_catalyst_risk,
        canonical_signal_payload,
    )
    liquidity_execution_fragility_analysis = _liquidity_execution_fragility_text(
        liquidity_execution_fragility,
        canonical_signal_payload,
    )
    market_breadth_internal_state_analysis = _market_breadth_internal_state_text(
        market_breadth_internals,
        canonical_signal_payload,
    )
    cross_asset_confirmation_analysis = _cross_asset_confirmation_text(
        cross_asset_confirmation,
        canonical_signal_payload,
    )
    stress_spillover_analysis = _stress_spillover_text(
        stress_spillover_conditions,
        canonical_signal_payload,
    )

    risk_quality_analysis = _risk_quality_analysis_text(
        signal,
        quality,
        warnings,
        why_signal,
        strategy,
        fragility_intelligence,
        domain_agreement,
        data_bundle,
    )
    risk_quality_analysis = _join_sentences(
        [
            risk_quality_analysis,
            f"False-positive suppression flags are {_fmt_list(canonical_signal_payload.get('suppression_flags') or [])}, and adjusted confidence notes are {_fmt_list(canonical_signal_payload.get('adjusted_confidence_notes') or [])}."
            if canonical_signal_payload.get("suppression_flags") or canonical_signal_payload.get("adjusted_confidence_notes")
            else None,
        ]
    )

    overall_analysis = _join_sentences(
        [
            f"The unified system view on {symbol} is {strategy_signal} / {strategy_posture}. That posture is not coming from one score alone; it is the result of trend, mean-reversion, sentiment, macro-alignment, quality/fundamental, relative-strength, evidence-quality, and fragility-veto components being fused inside the strategy layer.",
            f"The strongest evidence for the thesis is {_fmt_driver_list(why_signal['top_positive_drivers'])}. The strongest evidence against it is {_fmt_driver_list(why_signal['top_negative_drivers'])}.",
            f"Structural integrity {_fmt_num(composites.get('Market Structure Integrity Score'), digits=1)} / 100, fundamental durability {_fmt_num(composites.get('Fundamental Durability Score'), digits=1)} / 100, macro alignment {_fmt_num(composites.get('Macro Alignment Score'), digits=1)} / 100, and cross-domain conviction {_fmt_num(composites.get('Cross-Domain Conviction Score'), digits=1)} / 100 are being weighed against fragility {_fmt_num(composites.get('Signal Fragility Index'), digits=1)} / 100 and crowding {_fmt_num(composites.get('Narrative Crowding Index'), digits=1)} / 100.",
            f"Depth realism overlays show event overhang {_fmt_num(canonical_feature_vector.get('event_overhang_score'), digits=1, signed=False)} / 100, implementation fragility {_fmt_num(canonical_feature_vector.get('implementation_fragility_score'), digits=1, signed=False)} / 100, breadth confirmation {_fmt_num(canonical_feature_vector.get('breadth_confirmation_score'), digits=1, signed=False)} / 100, cross-asset conflict {_fmt_num(canonical_feature_vector.get('cross_asset_conflict_score'), digits=1, signed=False)} / 100, and market stress {_fmt_num(canonical_feature_vector.get('market_stress_score'), digits=1, signed=False)} / 100."
            if canonical_feature_vector
            else None,
            f"The final posture stays at {strategy_signal} because raw signal action {action}, regime {regime}, opportunity-quality score {_fmt_num(composites.get('Opportunity Quality Score'), digits=1)} / 100, and execution framing {(execution_posture.get('preferred_posture') or 'staged_watch').replace('_', ' ')} outweigh the current detractors, but the system is explicitly least certain where {strategy.get('where_least_certain') or 'cross-domain disagreement is highest'}.",
            f"Suppression logic remains active through {_fmt_list(canonical_signal_payload.get('suppression_flags') or [])}, which is why the platform is treating superficially attractive setups more defensively when event windows, liquidity fragility, weak breadth, cross-asset conflict, or stress spillover are elevated."
            if canonical_signal_payload.get("suppression_flags")
            else None,
            f"Scenario discipline matters: base case is {(scenario_matrix.get('base') or {}).get('summary') or strategy.get('base_case')}, bull transition requires {_fmt_list((strategy.get('confirmation_triggers') or []))}, bear deterioration comes through {_fmt_list((strategy.get('deterioration_triggers') or []))}, and top invalidators are {_fmt_list((invalidation_map.get('top_invalidators') or []))}.",
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
    evaluation_research_analysis = (
        "Phase 6 evaluation will attach a point-in-time research scorecard once prediction records and matured outcomes are available for this cohort."
    )
    evidence_map = {
        "signal_summary": [
            "signal.action",
            "strategy.final_signal",
            "strategy.component_scores",
            "feature_factor_bundle.composite_intelligence",
            "feature_factor_bundle.proprietary_scores",
            "feature_factor_bundle.domain_agreement",
        ],
        "technical_analysis": [
            "data_bundle.market_price_volume",
            "data_bundle.technical_market_structure",
            "key_features.ret_1d/ret_5d/ret_21d",
            "feature_factor_bundle.factor_groups.market_structure",
        ],
        "fundamental_analysis": [
            "data_bundle.fundamental_filing",
            "feature_factor_bundle.fundamental_intelligence",
        ],
        "statistical_analysis": [
            "feature_factor_bundle.factor_groups.market_structure",
            "feature_factor_bundle.fragility_intelligence",
            "feature_factor_bundle.regime_intelligence",
            "feature_factor_bundle.domain_agreement",
        ],
        "sentiment_analysis": [
            "data_bundle.sentiment_narrative_flow",
            "feature_factor_bundle.sentiment_narrative_intelligence",
        ],
        "macro_geopolitical_analysis": [
            "data_bundle.macro_cross_asset",
            "data_bundle.geopolitical_policy",
            "data_bundle.relative_context",
            "feature_factor_bundle.macro_alignment",
        ],
        "event_catalyst_risk_analysis": [
            "data_bundle.event_catalyst_risk",
            "canonical_alpha_core.feature_vector.event_*",
            "canonical_alpha_core.signal_payload.event_penalties",
        ],
        "liquidity_execution_fragility_analysis": [
            "data_bundle.liquidity_execution_fragility",
            "canonical_alpha_core.feature_vector.implementation_fragility_*",
            "canonical_alpha_core.signal_payload.liquidity_penalties",
        ],
        "market_breadth_internal_state_analysis": [
            "data_bundle.market_breadth_internals",
            "canonical_alpha_core.feature_vector.breadth_*",
        ],
        "cross_asset_confirmation_analysis": [
            "data_bundle.cross_asset_confirmation",
            "canonical_alpha_core.feature_vector.cross_asset_*",
        ],
        "stress_spillover_analysis": [
            "data_bundle.stress_spillover_conditions",
            "canonical_alpha_core.feature_vector.market_stress_*",
            "canonical_alpha_core.signal_payload.stress_penalties",
        ],
        "strategy_view": [
            "strategy.strategy_summary",
            "strategy.strategy_posture",
            "strategy.actionability_score",
            "strategy.scenario_matrix",
            "strategy.execution_posture",
            "strategy.base_case",
            "strategy.upside_case",
            "strategy.downside_case",
            "strategy.invalidation_conditions",
        ],
        "risks_weaknesses_invalidators": [
            "quality.warnings",
            "quality.anomaly_flags",
            "strategy.confidence_degraders",
            "strategy.fragility_vetoes",
            "strategy.invalidators",
        ],
    }

    report = {
        "report_kind": ANALYSIS_REPORT_KIND,
        "report_version": "2.1",
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
        "canonical_lineage": canonical_lineage,
        "snapshot_id": canonical_lineage.get("snapshot_id"),
        "snapshot_version": canonical_lineage.get("snapshot_version"),
        "feature_version": canonical_lineage.get("feature_version"),
        "signal_version": canonical_lineage.get("signal_version"),
        "freshness_summary": freshness_summary,
        "signal": signal_view,
        "key_features": key_features,
        "quality": quality,
        "evidence": evidence,
        "data_bundle": data_bundle,
        "domain_availability": domain_availability,
        "feature_factor_bundle": feature_factor_bundle,
        "event_catalyst_risk": event_catalyst_risk,
        "liquidity_execution_fragility": liquidity_execution_fragility,
        "market_breadth_internals": market_breadth_internals,
        "cross_asset_confirmation": cross_asset_confirmation,
        "stress_spillover_conditions": stress_spillover_conditions,
        "proprietary_scores": proprietary_scores,
        "factor_groups": factor_groups,
        "regime_intelligence": regime_intelligence,
        "fragility_intelligence": fragility_intelligence,
        "domain_agreement": domain_agreement,
        "conviction_components": conviction_components,
        "opportunity_quality_components": opportunity_quality_components,
        "strategy": strategy,
        "strategy_summary": strategy.get("strategy_summary"),
        "strategy_posture": strategy_posture,
        "confidence_score": confidence_score,
        "actionability_score": actionability_score,
        "participant_fit": participant_fit,
        "scenario_matrix": scenario_matrix,
        "invalidators": invalidation_map,
        "confirmation_triggers": strategy.get("confirmation_triggers") or [],
        "deterioration_triggers": strategy.get("deterioration_triggers") or [],
        "fragility_vetoes": strategy.get("fragility_vetoes") or [],
        "execution_posture": execution_posture,
        "uncertainty_notes": strategy.get("uncertainty_notes") or [],
        "strategy_version": strategy.get("strategy_version"),
        "why_this_signal": why_signal,
        "top_positive_drivers": why_signal["top_positive_drivers"],
        "top_negative_drivers": why_signal["top_negative_drivers"],
        "confidence_modifiers": why_signal["confidence_modifiers"],
        "missing_data_warnings": why_signal["missing_data_warnings"],
        "freshness_warnings": why_signal["freshness_warnings"],
        "evidence_map": evidence_map,
        "evaluation": {},
        "evaluation_summary": None,
        "confidence_reliability_summary": None,
        "regime_usefulness_summary": None,
        "evaluation_research_analysis": evaluation_research_analysis,
        "deployment_readiness": {},
        "deployment_mode": None,
        "rollout_stage": None,
        "deployment_permission": None,
        "deployment_blockers": [],
        "deployment_rationale": None,
        "trust_tier": None,
        "minimum_required_review": None,
        "human_review_required": None,
        "model_readiness_status": None,
        "model_readiness_notes": [],
        "live_readiness_score": None,
        "live_readiness_blockers": [],
        "recent_degradation_flags": [],
        "evidence_quality_summary": None,
        "risk_budget_tier": None,
        "exposure_caution_level": None,
        "fragility_adjusted_size_band": None,
        "confidence_adjusted_size_band": None,
        "maximum_risk_mode_allowed": None,
        "pause_recommended": None,
        "degrade_to_paper_recommended": None,
        "increased_review_required": None,
        "degraded_reliability_regimes": [],
        "drift_alerts": [],
        "deployment_risk_alerts": [],
        "readiness_checkpoint": None,
        "promotion_criteria": [],
        "demotion_criteria": [],
        "rollback_reason": None,
        "stage_transition_notes": [],
        "live_use_audit_snapshot": {},
        "deployment_readiness_summary": (
            "Phase 8 deployment-readiness controls will attach a staged capital-readiness summary, trust tier, and live-use blockers once the readiness layer runs."
        ),
        "deployment_permission_analysis": (
            "Deployment-permission analysis will attach once capital-readiness controls are available."
        ),
        "risk_budget_exposure_analysis": (
            "Risk-budget and exposure-caution framing will attach once capital-readiness controls are available."
        ),
        "rollout_stage_summary": (
            "Rollout-stage guidance will attach once staged deployment controls are available."
        ),
        "portfolio_construction": {},
        "portfolio_construction_artifact_id": None,
        "ranked_opportunity_score": None,
        "portfolio_candidate_score": None,
        "watchlist_priority_score": None,
        "deployability_rank": None,
        "quality_vs_fragility_ratio": None,
        "confidence_adjusted_rank": None,
        "conviction_adjusted_rank": None,
        "candidate_classification": None,
        "candidate_blockers": [],
        "portfolio_rank": None,
        "overlap_score": None,
        "redundancy_score": None,
        "diversification_contribution_score": None,
        "most_redundant_symbol": None,
        "cluster_membership": [],
        "exposure_family": None,
        "portfolio_fit_quality": None,
        "concentration_warning": None,
        "cluster_concentration_warning": None,
        "sector_crowding_warning": None,
        "fragility_cluster_warning": None,
        "macro_exposure_warning": None,
        "theme_exposure_warning": None,
        "diversification_status": None,
        "size_band": None,
        "weight_band": None,
        "risk_budget_band": None,
        "fragility_adjustment": None,
        "confidence_adjustment": None,
        "concentration_adjustment": None,
        "overlap_adjustment": None,
        "deployment_mode_adjustment": None,
        "max_priority_allowed": None,
        "caution_level": None,
        "execution_quality_score": None,
        "friction_penalty": None,
        "turnover_penalty": None,
        "wait_for_better_entry_flag": None,
        "confirmation_preferred_flag": None,
        "portfolio_context_summary": (
            "Phase 9 portfolio-construction overlays will attach rank, portfolio fit, overlap, and size-band context once the portfolio layer runs."
        ),
        "portfolio_fit_analysis": (
            "Portfolio-fit and diversification analysis will attach once the portfolio-construction layer is available."
        ),
        "execution_quality_analysis": (
            "Execution-quality and friction analysis will attach once the portfolio-construction layer is available."
        ),
        "portfolio_workflow_summary": (
            "Watchlist, candidate-priority, and rotation workflow guidance will attach once the portfolio-construction layer is available."
        ),
        "portfolio_risk_model": {},
        "portfolio_risk_model_artifact_id": None,
        "portfolio_risk_model_version": None,
        "portfolio_fit_rank": None,
        "marginal_portfolio_utility": None,
        "portfolio_contribution_score": None,
        "marginal_fragility_penalty": None,
        "marginal_diversification_bonus": None,
        "replacement_value_score": None,
        "replacement_candidate": None,
        "substitution_score": None,
        "better_alternative_flag": None,
        "diversification_upgrade_flag": None,
        "overlap_reduction_flag": None,
        "portfolio_quality_upgrade_reason": None,
        "hidden_overlap_score": None,
        "complementarity_score": None,
        "overlap_rationale": None,
        "overlap_drivers": [],
        "overlap_confidence": None,
        "factor_exposure_vector": {},
        "factor_loading_summary": [],
        "style_affinity": None,
        "macro_exposure_profile": {},
        "fragility_exposure_profile": {},
        "exposure_confidence": None,
        "exposure_cluster": None,
        "concentration_profile": {},
        "hidden_concentration_score": None,
        "cluster_risk_score": None,
        "fragility_cluster_risk": None,
        "event_cluster_risk": None,
        "macro_cluster_risk": None,
        "narrative_cluster_risk": None,
        "diversification_health_score": None,
        "portfolio_stress_score": None,
        "portfolio_fragility_score": None,
        "correlation_breakdown_risk": None,
        "stress_concentration_warning": None,
        "clustered_gap_risk_warning": None,
        "unstable_portfolio_state_flag": None,
        "portfolio_risk_warnings": [],
        "cohort_risk_ranking": [],
        "portfolio_risk_model_summary": (
            "Phase 11 portfolio risk-model overlays will attach realized covariance, factor exposure, hidden overlap, concentration, and marginal utility context once the portfolio risk-model layer runs."
        ),
        "hidden_overlap_redundancy_analysis": (
            "Hidden overlap and redundancy analysis will attach once the portfolio risk-model layer is available."
        ),
        "factor_exposure_summary": (
            "Factor exposure summary will attach once the portfolio risk-model layer is available."
        ),
        "concentration_cluster_risk_analysis": (
            "Concentration and cluster-risk analysis will attach once the portfolio risk-model layer is available."
        ),
        "replacement_diversification_analysis": (
            "Replacement and diversification analysis will attach once the portfolio risk-model layer is available."
        ),
        "portfolio_stress_fragility_summary": (
            "Portfolio stress and fragility overlays will attach once the portfolio risk-model layer is available."
        ),
        "top_peer_overlaps": [],
        "cohort_ranking": [],
        "priority_shift_flag": None,
        "rebalance_attention_flag": None,
        "candidate_upgrade_reason": None,
        "candidate_downgrade_reason": None,
        "replacement_candidate_notes": None,
        "rotation_pressure_score": None,
        "continuous_learning": {},
        "learning_artifact_id": None,
        "research_version": None,
        "setup_archetype": {},
        "active_motifs": [],
        "regime_conditioned_learnings": [],
        "reweighting_candidates": [],
        "research_hypotheses": [],
        "interaction_candidates": [],
        "learning_drift_alerts": [],
        "experiment_registry": {},
        "signal_family_library": {},
        "motif_library": [],
        "improvement_queue": [],
        "learning_priority": None,
        "learning_summary": (
            "Phase 10 continuous-learning overlays will attach regime learnings, adaptation candidates, drift alerts, experiments, and archetype research once the learning layer runs."
        ),
        "regime_learning_summary": (
            "Regime-conditioned learning context will attach once the continuous-learning layer is available."
        ),
        "adaptation_queue_summary": (
            "Reweighting candidates and research hypotheses will attach once the continuous-learning layer is available."
        ),
        "experiment_registry_summary": (
            "Experiment-registry and approval-workflow context will attach once the continuous-learning layer is available."
        ),
        "archetype_motif_summary": (
            "Signal-family archetypes and motif discovery context will attach once the continuous-learning layer is available."
        ),
        "signal_summary": signal_summary,
        "technical_analysis": technical_analysis,
        "fundamental_analysis": fundamental_analysis,
        "statistical_analysis": statistical_analysis,
        "sentiment_analysis": sentiment_analysis,
        "macro_geopolitical_analysis": macro_geopolitical_analysis,
        "event_catalyst_risk_analysis": event_catalyst_risk_analysis,
        "liquidity_execution_fragility_analysis": liquidity_execution_fragility_analysis,
        "market_breadth_internal_state_analysis": market_breadth_internal_state_analysis,
        "cross_asset_confirmation_analysis": cross_asset_confirmation_analysis,
        "stress_spillover_analysis": stress_spillover_analysis,
        "risk_quality_analysis": risk_quality_analysis,
        "overall_analysis": overall_analysis,
        "strategy_view": strategy_view,
        "risks_weaknesses_invalidators": risks_weaknesses_invalidators,
        "evidence_provenance": evidence_provenance,
        "suppression_flags": canonical_signal_payload.get("suppression_flags") or [],
        "adjusted_confidence_notes": canonical_signal_payload.get("adjusted_confidence_notes") or [],
    }
    return sanitize_payload(report)
