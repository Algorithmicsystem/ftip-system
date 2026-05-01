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
            "deployment_mode": report.get("deployment_mode"),
            "deployment_permission": report.get("deployment_permission"),
            "trust_tier": report.get("trust_tier"),
            "live_readiness_status": report.get("model_readiness_status"),
            "live_readiness_score": report.get("live_readiness_score"),
            "rollout_stage": report.get("rollout_stage"),
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

    signal_summary = " ".join(
        [
            f"As of {as_of_text}, the assistant pipeline lands on a {strategy_signal} posture for {symbol}, framed on the {horizon} horizon under {risk_mode} risk mode.",
            f"The underlying signal engine prints {action} with score {_fmt_num(score)} and confidence {_fmt_num(confidence)}, while the strategy layer converts that into {strategy_signal} / {strategy_posture} with probability-like confidence {_fmt_num(strategy_confidence)} ({_fmt_num(confidence_score, digits=1, signed=False)} / 100), {conviction_tier} conviction, {fragility_tier} fragility, and actionability {_fmt_num(actionability_score, digits=1, signed=False)} / 100.",
            f"Structural quality is {_fmt_num(composites.get('Market Structure Integrity Score'), digits=1, signed=False)} / 100, regime stability is {_fmt_num(composites.get('Regime Stability Score'), digits=1, signed=False)} / 100, signal fragility is {_fmt_num(composites.get('Signal Fragility Index'), digits=1, signed=False)} / 100, and opportunity quality is {_fmt_num(composites.get('Opportunity Quality Score'), digits=1, signed=False)} / 100.",
            f"Cross-domain agreement is {_fmt_num(domain_agreement.get('domain_agreement_score'), digits=1, signed=False)} / 100 versus conflict {_fmt_num(domain_agreement.get('domain_conflict_score'), digits=1, signed=False)} / 100; the strongest confirming domains are {_fmt_list(item.get('domain') for item in (domain_agreement.get('strongest_confirming_domains') or []))}, while conflicts are concentrated in {_fmt_list(item.get('domain') for item in (domain_agreement.get('strongest_conflicting_domains') or []))}.",
            f"The dominant regime reads {regime}, freshness is {freshness_summary['overall_status']}, participant fit is {_fmt_list(participant_fit)}, and the main positive drivers are {_fmt_driver_list(why_signal['top_positive_drivers'])}.",
            f"The main risks are {_fmt_driver_list(why_signal['top_negative_drivers'])}, with scenario framing set to {job_context.get('scenario') or 'base'} and execution posture {(execution_posture.get('preferred_posture') or 'staged_watch').replace('_', ' ')}.",
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

    overall_analysis = " ".join(
        [
            f"The unified system view on {symbol} is {strategy_signal} / {strategy_posture}. That posture is not coming from one score alone; it is the result of trend, mean-reversion, sentiment, macro-alignment, quality/fundamental, relative-strength, evidence-quality, and fragility-veto components being fused inside the strategy layer.",
            f"The strongest evidence for the thesis is {_fmt_driver_list(why_signal['top_positive_drivers'])}. The strongest evidence against it is {_fmt_driver_list(why_signal['top_negative_drivers'])}.",
            f"Structural integrity {_fmt_num(composites.get('Market Structure Integrity Score'), digits=1)} / 100, fundamental durability {_fmt_num(composites.get('Fundamental Durability Score'), digits=1)} / 100, macro alignment {_fmt_num(composites.get('Macro Alignment Score'), digits=1)} / 100, and cross-domain conviction {_fmt_num(composites.get('Cross-Domain Conviction Score'), digits=1)} / 100 are being weighed against fragility {_fmt_num(composites.get('Signal Fragility Index'), digits=1)} / 100 and crowding {_fmt_num(composites.get('Narrative Crowding Index'), digits=1)} / 100.",
            f"The final posture stays at {strategy_signal} because raw signal action {action}, regime {regime}, opportunity-quality score {_fmt_num(composites.get('Opportunity Quality Score'), digits=1)} / 100, and execution framing {(execution_posture.get('preferred_posture') or 'staged_watch').replace('_', ' ')} outweigh the current detractors, but the system is explicitly least certain where {strategy.get('where_least_certain') or 'cross-domain disagreement is highest'}.",
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
        "freshness_summary": freshness_summary,
        "signal": signal_view,
        "key_features": key_features,
        "quality": quality,
        "evidence": evidence,
        "data_bundle": data_bundle,
        "domain_availability": domain_availability,
        "feature_factor_bundle": feature_factor_bundle,
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
