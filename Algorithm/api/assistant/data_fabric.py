from __future__ import annotations

import datetime as dt
import math
import re
import statistics
from collections import Counter
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from api import config
from api.data_providers.alphavantage import fetch_company_overview
from api.data_providers.bars import fetch_reference_bars
from api.data_providers.errors import ProviderError
from api.data_providers.finnhub import (
    fetch_basic_financials,
    fetch_company_news,
    fetch_company_profile,
)
from api.data_providers.fred import fetch_series as fetch_fred_series
from api.data_providers.gdelt import search_articles as search_gdelt_articles
from api.data_providers.gnews import search_news as search_gnews
from api.data_providers.newsapi import search_news as search_newsapi
from api.data_providers.sec_edgar import fetch_company_filing_profile
from api.data_providers.world_bank import fetch_indicator as fetch_world_bank_indicator

_SECTOR_PROXY_MAP = {
    "technology": "XLK",
    "financial services": "XLF",
    "financial": "XLF",
    "healthcare": "XLV",
    "energy": "XLE",
    "industrials": "XLI",
    "consumer defensive": "XLP",
    "consumer staples": "XLP",
    "utilities": "XLU",
    "communication services": "XLC",
    "consumer cyclical": "XLY",
    "materials": "XLB",
    "real estate": "XLRE",
}
_TITLE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
_TITLE_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "after",
    "amid",
    "over",
    "under",
    "stock",
    "stocks",
    "shares",
    "company",
    "corp",
    "quarter",
    "market",
    "markets",
    "earnings",
    "analyst",
    "price",
    "today",
    "says",
}
_POSITIVE_HEADLINE_WORDS = {
    "beats",
    "beat",
    "growth",
    "record",
    "surge",
    "strong",
    "upgrade",
    "expands",
    "wins",
    "positive",
}
_NEGATIVE_HEADLINE_WORDS = {
    "misses",
    "miss",
    "cuts",
    "cut",
    "warning",
    "probe",
    "lawsuit",
    "downgrade",
    "weak",
    "decline",
    "risk",
}
_EVENT_BUCKETS = {
    "policy_regulation": {"regulation", "regulatory", "antitrust", "license", "export", "ban", "sec", "doj", "ftc", "fda"},
    "trade_supply_chain": {"tariff", "sanction", "trade", "supply", "chip", "shipping", "export"},
    "conflict_security": {"war", "conflict", "missile", "cyber", "attack", "china", "taiwan", "russia", "ukraine", "middle east"},
    "elections_policy": {"election", "congress", "senate", "president", "administration", "white house"},
    "macro_rates_inflation": {"fed", "rates", "yield", "inflation", "cpi", "pce", "payrolls"},
}
_FRED_SERIES = {
    "rates": "DGS10",
    "policy_rate": "FEDFUNDS",
    "inflation": "CPIAUCSL",
    "labor": "UNRATE",
    "growth": "GDPC1",
    "credit": "BAMLC0A0CM",
}
_WORLD_BANK_INDICATORS = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "inflation": "FP.CPI.TOTL.ZG",
    "unemployment": "SL.UEM.TOTL.ZS",
}


def enrich_data_bundle(
    *,
    job_context: Dict[str, Any],
    symbol_meta: Dict[str, Any],
    data_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    if not config.data_fabric_enabled():
        return {
            "enabled": False,
            "status": "disabled",
            "provider_notes": ["External data fabric is disabled."],
        }

    symbol = str(job_context.get("symbol") or "").upper()
    as_of_date = dt.date.fromisoformat(str(job_context.get("as_of_date")))

    market_overlay = _build_market_overlay(symbol, as_of_date, data_bundle.get("market_price_volume") or {})
    fundamentals_overlay = _build_fundamental_overlay(symbol, symbol_meta)
    news_overlay = _build_news_overlay(symbol, symbol_meta, as_of_date)
    macro_overlay = _build_macro_overlay(symbol_meta)
    cross_asset_overlay = _build_cross_asset_overlay(symbol_meta, as_of_date)
    geopolitical_overlay = _build_geopolitical_overlay(news_overlay)
    quality_overlay = _build_quality_overlay(
        base_quality=data_bundle.get("quality_provenance") or {},
        overlays={
            "market_price_volume": market_overlay,
            "fundamental_filing": fundamentals_overlay,
            "sentiment_narrative_flow": news_overlay,
            "macro_cross_asset": macro_overlay,
            "geopolitical_policy": geopolitical_overlay,
            "relative_context": cross_asset_overlay,
        },
    )

    return {
        "enabled": True,
        "status": "ok",
        "domains": {
            "market_price_volume": market_overlay,
            "fundamental_filing": fundamentals_overlay,
            "sentiment_narrative_flow": news_overlay,
            "macro_cross_asset": macro_overlay,
            "geopolitical_policy": geopolitical_overlay,
            "relative_context": cross_asset_overlay,
            "quality_provenance": quality_overlay,
        },
    }


def merge_into_data_bundle(
    *,
    data_bundle: Dict[str, Any],
    overlay: Dict[str, Any],
) -> Dict[str, Any]:
    merged = deepcopy(data_bundle)
    for domain, payload in (overlay.get("domains") or {}).items():
        merged[domain] = _deep_merge(merged.get(domain) or {}, payload)
    merged["external_data_fabric"] = {
        "enabled": overlay.get("enabled"),
        "status": overlay.get("status"),
    }
    return merged


def _build_market_overlay(
    symbol: str,
    as_of_date: dt.date,
    base_market: Dict[str, Any],
) -> Dict[str, Any]:
    providers = []
    notes = []
    verification_gap_pct = None
    external_status = "limited"
    try:
        rows = fetch_reference_bars(symbol, as_of_date - dt.timedelta(days=90), as_of_date)
        providers = sorted({str(row.get("source")) for row in rows if row.get("source")})
        latest_external_close = rows[-1].get("close") if rows else None
        base_close = base_market.get("latest_close")
        if latest_external_close not in (None, 0) and base_close not in (None, 0):
            verification_gap_pct = float(base_close) / float(latest_external_close) - 1.0
        external_status = "fresh" if rows else "limited"
    except Exception as exc:
        notes.append(f"external market verification unavailable: {exc}")
        rows = []
    return {
        "external_verification": {
            "verification_gap_pct": verification_gap_pct,
            "latest_external_close": rows[-1].get("close") if rows else None,
            "latest_external_date": rows[-1].get("as_of_date") if rows else None,
            "source": rows[-1].get("source") if rows else None,
        },
        "meta": {
            "external_sources": providers,
            "external_status": external_status,
            "external_notes": notes,
        },
    }


def _build_fundamental_overlay(symbol: str, symbol_meta: Dict[str, Any]) -> Dict[str, Any]:
    provider_payloads: Dict[str, Any] = {}
    provider_status: Dict[str, Any] = {}
    notes: List[str] = []

    for provider_name, fetcher in (
        ("sec_edgar", lambda: fetch_company_filing_profile(symbol)),
        ("finnhub_profile", lambda: fetch_company_profile(symbol)),
        ("finnhub_basic_financials", lambda: fetch_basic_financials(symbol)),
        ("alphavantage_overview", lambda: fetch_company_overview(symbol)),
    ):
        payload, status = _safe_call(fetcher)
        provider_status[provider_name] = status
        if payload is not None:
            provider_payloads[provider_name] = payload
        elif status.get("note"):
            notes.append(f"{provider_name}: {status['note']}")

    sec_profile = provider_payloads.get("sec_edgar") or {}
    finnhub_metrics = provider_payloads.get("finnhub_basic_financials") or {}
    alpha_overview = provider_payloads.get("alphavantage_overview") or {}
    finnhub_profile = provider_payloads.get("finnhub_profile") or {}

    growth_quality = _bounded_score(
        _mean(
            [
                alpha_overview.get("quarterly_revenue_growth_yoy"),
                finnhub_metrics.get("revenue_growth_ttm_yoy"),
            ]
        ),
        low=-0.25,
        high=0.35,
    )
    profitability_quality = _bounded_score(
        _mean(
            [
                alpha_overview.get("profit_margin"),
                alpha_overview.get("operating_margin_ttm"),
                finnhub_metrics.get("net_margin"),
                finnhub_metrics.get("operating_margin_ttm"),
            ]
        ),
        low=-0.05,
        high=0.3,
    )
    balance_sheet_resilience = _bounded_score(
        _mean(
            [
                finnhub_metrics.get("current_ratio_quarterly"),
                finnhub_metrics.get("quick_ratio_quarterly"),
                _inverse_metric(finnhub_metrics.get("total_debt_to_equity_quarterly"), cap=3.0),
            ]
        ),
        low=0.0,
        high=2.0,
    )

    sources = sorted(provider_payloads)
    filing_recency_days = sec_profile.get("filing_recency_days")
    status = (
        "fresh"
        if filing_recency_days is not None and filing_recency_days <= 120
        else "stale_but_usable"
        if filing_recency_days is not None and filing_recency_days <= 180
        else "limited"
    )
    return {
        "filing_recency_days": filing_recency_days,
        "coverage_flags": sec_profile.get("coverage_flags") or {},
        "company_profile": {
            "name": symbol_meta.get("name") or finnhub_profile.get("name"),
            "sector": symbol_meta.get("sector") or alpha_overview.get("sector"),
            "industry": alpha_overview.get("industry") or finnhub_profile.get("finnhub_industry"),
            "exchange": symbol_meta.get("exchange") or finnhub_profile.get("exchange"),
            "country": symbol_meta.get("country") or finnhub_profile.get("country"),
            "currency": symbol_meta.get("currency") or finnhub_profile.get("currency"),
        },
        "durability_proxies": {
            "growth_quality": growth_quality,
            "profitability_quality": profitability_quality,
            "balance_sheet_resilience": balance_sheet_resilience,
            "reporting_stability": 1.0 if filing_recency_days is not None and filing_recency_days <= 120 else 0.5 if filing_recency_days is not None else 0.25,
        },
        "provider_snapshot": provider_payloads,
        "meta": {
            "sources": sources,
            "status": status,
            "coverage_score": _bounded_ratio(len(sources), 4),
            "provider_status": provider_status,
            "notes": notes,
        },
    }


def _build_news_overlay(
    symbol: str,
    symbol_meta: Dict[str, Any],
    as_of_date: dt.date,
) -> Dict[str, Any]:
    to_ts = dt.datetime.combine(as_of_date, dt.time.max).replace(tzinfo=dt.timezone.utc)
    from_ts = to_ts - dt.timedelta(days=max(config.data_fabric_news_days(), 1))
    company_name = str(symbol_meta.get("name") or symbol).strip()
    query = f'"{symbol}" OR "{company_name}"'

    provider_payloads: Dict[str, List[Dict[str, Any]]] = {}
    provider_status: Dict[str, Dict[str, Any]] = {}
    for provider_name, fetcher in (
        ("gnews", lambda: search_gnews(query, from_ts=from_ts, to_ts=to_ts, max_items=config.data_fabric_news_limit())),
        ("newsapi", lambda: search_newsapi(query, from_ts=from_ts, to_ts=to_ts, max_items=config.data_fabric_news_limit())),
        ("finnhub_news", lambda: fetch_company_news(symbol, from_ts.date(), to_ts.date())),
        ("gdelt", lambda: search_gdelt_articles(query, from_ts=from_ts, to_ts=to_ts, max_records=config.data_fabric_news_limit())),
    ):
        payload, status = _safe_call(fetcher)
        provider_status[provider_name] = status
        provider_payloads[provider_name] = payload or []

    articles = _dedupe_articles(provider_payloads)
    titles = [str(item.get("title") or "") for item in articles]
    source_breakdown = {
        provider_name: len(payload)
        for provider_name, payload in provider_payloads.items()
        if payload
    }
    sentiment_bias = _headline_sentiment_bias(titles)
    novelty_ratio = _bounded_ratio(len({_normalize_title(title) for title in titles if title}), len(titles))
    top_narratives = _extract_topics(titles, symbol)
    contradiction_score = _contradiction_score(titles)
    latest_published_at = articles[0].get("published_at") if articles else None

    return {
        "headline_count": len(articles),
        "attention_crowding": _attention_multiple(source_breakdown, len(articles)),
        "novelty_ratio": novelty_ratio,
        "disagreement_score": contradiction_score,
        "aggregated_sentiment_bias": sentiment_bias,
        "source_breakdown": source_breakdown,
        "top_narratives": top_narratives,
        "aggregated_headlines": articles[:10],
        "provider_snapshot": provider_payloads,
        "meta": {
            "sources": sorted(source_breakdown),
            "latest_news_published_at": latest_published_at,
            "coverage_score": _bounded_ratio(len(articles), 12),
            "status": "fresh" if articles else "limited",
            "provider_status": provider_status,
        },
    }


def _build_macro_overlay(symbol_meta: Dict[str, Any]) -> Dict[str, Any]:
    provider_status: Dict[str, Dict[str, Any]] = {}
    fred_payloads: Dict[str, Any] = {}
    for label, series_id in _FRED_SERIES.items():
        payload, status = _safe_call(lambda series_id=series_id: fetch_fred_series(series_id, limit=12))
        provider_status[f"fred:{label}"] = status
        if payload is not None:
            fred_payloads[label] = payload

    country = _world_bank_country(symbol_meta.get("country"))
    world_bank_payloads: Dict[str, Any] = {}
    for label, indicator in _WORLD_BANK_INDICATORS.items():
        payload, status = _safe_call(
            lambda indicator=indicator: fetch_world_bank_indicator(country=country, indicator=indicator, per_page=8)
        )
        provider_status[f"world_bank:{label}"] = status
        if payload is not None:
            world_bank_payloads[label] = payload

    fred_snapshot = {
        label: _series_snapshot(payload.get("observations") or [])
        for label, payload in fred_payloads.items()
    }
    world_bank_snapshot = {
        label: _series_snapshot(payload.get("observations") or [])
        for label, payload in world_bank_payloads.items()
    }
    macro_regime = _infer_macro_regime(fred_snapshot, world_bank_snapshot)

    return {
        "fred_series": fred_snapshot,
        "world_bank_series": world_bank_snapshot,
        "macro_regime_context": macro_regime,
        "meta": {
            "sources": sorted({"fred" if fred_payloads else None, "world_bank" if world_bank_payloads else None} - {None}),
            "coverage_score": _bounded_ratio(len(fred_payloads) + len(world_bank_payloads), len(_FRED_SERIES) + len(_WORLD_BANK_INDICATORS)),
            "status": "fresh" if fred_payloads else "limited",
            "provider_status": provider_status,
        },
    }


def _build_cross_asset_overlay(
    symbol_meta: Dict[str, Any],
    as_of_date: dt.date,
) -> Dict[str, Any]:
    sector = str(symbol_meta.get("sector") or "").strip().lower()
    sector_proxy = _SECTOR_PROXY_MAP.get(sector)
    proxy_universe = [item for item in [sector_proxy, "SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "UUP"] if item]
    proxy_snapshot: Dict[str, Any] = {}
    provider_status: Dict[str, Dict[str, Any]] = {}
    for proxy in proxy_universe:
        payload, status = _safe_call(
            lambda proxy=proxy: fetch_reference_bars(proxy, as_of_date - dt.timedelta(days=90), as_of_date)
        )
        provider_status[proxy] = status
        if payload:
            proxy_snapshot[proxy] = _bars_snapshot(payload)
    benchmark_symbol = sector_proxy or "SPY"
    benchmark = proxy_snapshot.get(benchmark_symbol) or proxy_snapshot.get("SPY")
    return {
        "benchmark_context": {
            "benchmark_symbol": benchmark_symbol if benchmark else None,
            "benchmark_ret_21d": benchmark.get("ret_21d") if benchmark else None,
            "benchmark_vol_21d": benchmark.get("vol_21d") if benchmark else None,
        },
        "sector_proxy": sector_proxy,
        "proxy_snapshot": proxy_snapshot,
        "meta": {
            "sources": sorted(
                {
                    str(snapshot.get("source"))
                    for snapshot in proxy_snapshot.values()
                    if snapshot.get("source")
                }
            ),
            "coverage_score": _bounded_ratio(len(proxy_snapshot), len(proxy_universe)),
            "status": "fresh" if proxy_snapshot else "limited",
            "provider_status": provider_status,
        },
    }


def _build_geopolitical_overlay(news_overlay: Dict[str, Any]) -> Dict[str, Any]:
    articles = list(news_overlay.get("aggregated_headlines") or [])
    counts = {bucket: 0 for bucket in _EVENT_BUCKETS}
    relevant: List[Dict[str, Any]] = []
    for article in articles:
        title = str(article.get("title") or "").lower()
        matched = [bucket for bucket, keywords in _EVENT_BUCKETS.items() if any(keyword in title for keyword in keywords)]
        if not matched:
            continue
        for bucket in matched:
            counts[bucket] += 1
        relevant.append(
            {
                "title": article.get("title"),
                "published_at": article.get("published_at"),
                "matches": matched,
                "source": article.get("source"),
            }
        )

    total_articles = len(articles) or 1
    weighted_hits = (
        counts["policy_regulation"]
        + 1.2 * counts["trade_supply_chain"]
        + 1.4 * counts["conflict_security"]
        + 0.9 * counts["elections_policy"]
        + 0.8 * counts["macro_rates_inflation"]
    )
    event_intensity = min(weighted_hits / total_articles, 1.0)
    confidence = 0.75 if len(relevant) >= 4 else 0.45 if relevant else 0.15
    return {
        "event_buckets": counts,
        "event_intensity_score": event_intensity,
        "confidence": confidence,
        "recent_event_summary": relevant[:8],
        "meta": {
            "sources": sorted(
                {
                    str(item.get("source"))
                    for item in relevant
                    if item.get("source")
                }
            ),
            "coverage_score": _bounded_ratio(len(articles), 10),
            "status": "fresh" if articles else "limited",
            "note": "Event tagging is heuristic and should be read as directional context rather than a precision event model.",
        },
    }


def _build_quality_overlay(
    *,
    base_quality: Dict[str, Any],
    overlays: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    sources: Dict[str, List[str]] = {}
    freshness_summary: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = list(base_quality.get("warnings") or [])
    provider_notes: List[str] = []
    for domain, payload in overlays.items():
        meta = payload.get("meta") or {}
        sources[domain] = list(meta.get("sources") or [])
        freshness_summary[domain] = {
            "status": meta.get("status") or meta.get("external_status") or "unknown",
            "sources": list(meta.get("sources") or meta.get("external_sources") or []),
        }
        for note in meta.get("notes") or meta.get("external_notes") or []:
            provider_notes.append(str(note))
        if freshness_summary[domain]["status"] in {"limited", "stale", "stale_but_usable"}:
            warnings.append(f"{domain} coverage is {freshness_summary[domain]['status']}.")
    return {
        "source_map": sources,
        "freshness_summary": freshness_summary,
        "provider_notes": provider_notes,
        "warnings": list(dict.fromkeys(warnings)),
        "meta": {
            "status": "fresh" if not provider_notes else "mixed",
            "coverage_score": _bounded_ratio(
                sum(1 for item in freshness_summary.values() if item.get("status") == "fresh"),
                len(freshness_summary),
            ),
        },
    }


def _safe_call(fetcher: Callable[[], Any]) -> Tuple[Any, Dict[str, Any]]:
    try:
        payload = fetcher()
        size = len(payload) if isinstance(payload, (list, dict)) else None
        return payload, {"status": "ok", "size": size}
    except ProviderError as exc:
        return None, {"status": "error", "note": exc.reason_detail}
    except Exception as exc:
        return None, {"status": "error", "note": str(exc)}


def _series_snapshot(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not observations:
        return {}
    latest = observations[0]
    previous = observations[1] if len(observations) > 1 else {}
    latest_value = _safe_float(latest.get("value"))
    previous_value = _safe_float(previous.get("value"))
    delta = latest_value - previous_value if latest_value is not None and previous_value is not None else None
    return {
        "latest": latest_value,
        "previous": previous_value,
        "delta": delta,
        "latest_date": latest.get("date"),
    }


def _infer_macro_regime(
    fred_snapshot: Dict[str, Dict[str, Any]],
    world_bank_snapshot: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    rates = fred_snapshot.get("rates") or {}
    inflation = fred_snapshot.get("inflation") or {}
    labor = fred_snapshot.get("labor") or {}
    growth = world_bank_snapshot.get("gdp_growth") or fred_snapshot.get("growth") or {}
    regime = "neutral"
    if (rates.get("latest") or 0) > 4.0 and (inflation.get("delta") or 0) > 0:
        regime = "tightening_inflationary"
    elif (labor.get("latest") or 0) > 5.0 and (growth.get("latest") or 0) < 2.0:
        regime = "growth_softening"
    elif (growth.get("latest") or 0) > 2.5 and (labor.get("latest") or 0) < 4.5:
        regime = "growth_supportive"
    return {
        "regime": regime,
        "rates": rates,
        "inflation": inflation,
        "labor": labor,
        "growth": growth,
    }


def _bars_snapshot(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    closes = [row.get("close") for row in rows if row.get("close") is not None]
    if not closes:
        return {}
    latest = closes[-1]
    return {
        "latest_close": latest,
        "ret_21d": _pct_change(latest, closes[-22]) if len(closes) >= 22 else None,
        "vol_21d": _realized_vol(closes, 21),
        "source": rows[-1].get("source"),
    }


def _dedupe_articles(provider_payloads: Dict[str, Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for provider_name, items in provider_payloads.items():
        for item in items:
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            published_at = item.get("published_at")
            if not title or not url or not isinstance(published_at, dt.datetime):
                continue
            key = url or _normalize_title(title)
            current = deduped.get(key)
            if current is None:
                deduped[key] = {
                    "title": title,
                    "url": url,
                    "published_at": published_at.isoformat(),
                    "source": provider_name,
                    "content_snippet": item.get("content_snippet"),
                }
            else:
                current["source"] = "|".join(
                    sorted(set(str(current.get("source") or "").split("|") + [provider_name]))
                ).strip("|")
                if not current.get("content_snippet") and item.get("content_snippet"):
                    current["content_snippet"] = item.get("content_snippet")
    output = list(deduped.values())
    output.sort(key=lambda row: row.get("published_at"), reverse=True)
    return output


def _headline_sentiment_bias(titles: Sequence[str]) -> Optional[float]:
    if not titles:
        return None
    scores = []
    for title in titles:
        tokens = {
            token.lower()
            for token in _TITLE_TOKEN_RE.findall(title or "")
            if token.lower() not in _STOPWORDS
        }
        pos = len(tokens & _POSITIVE_HEADLINE_WORDS)
        neg = len(tokens & _NEGATIVE_HEADLINE_WORDS)
        scores.append((pos - neg) / max(len(tokens), 1))
    return float(statistics.fmean(scores)) if scores else None


def _contradiction_score(titles: Sequence[str]) -> Optional[float]:
    if not titles:
        return None
    positive_hits = 0
    negative_hits = 0
    for title in titles:
        lowered = title.lower()
        if any(word in lowered for word in _POSITIVE_HEADLINE_WORDS):
            positive_hits += 1
        if any(word in lowered for word in _NEGATIVE_HEADLINE_WORDS):
            negative_hits += 1
    total = positive_hits + negative_hits
    if total == 0:
        return 0.0
    return (2 * min(positive_hits, negative_hits)) / total


def _extract_topics(titles: Sequence[str], symbol: str) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    ignored = {symbol.lower(), symbol.lower().replace(".", ""), *(_STOPWORDS)}
    for title in titles:
        for token in _TITLE_TOKEN_RE.findall(title or ""):
            lowered = token.lower()
            if lowered in ignored:
                continue
            counter[lowered] += 1
    return [{"topic": topic, "count": count} for topic, count in counter.most_common(6)]


def _attention_multiple(source_breakdown: Dict[str, int], headline_count: int) -> Optional[float]:
    if headline_count <= 0:
        return None
    providers = max(len(source_breakdown), 1)
    return headline_count / providers


def _normalize_title(title: str) -> str:
    return _TITLE_NORMALIZE_RE.sub(" ", (title or "").strip().lower()).strip()


def _world_bank_country(country: Any) -> str:
    value = str(country or "").upper()
    if value == "CA":
        return "CA"
    return "US"


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(statistics.fmean(clean))


def _inverse_metric(value: Optional[float], *, cap: float) -> Optional[float]:
    if value is None:
        return None
    return max(cap - min(float(value), cap), 0.0)


def _bounded_score(value: Optional[float], *, low: float, high: float) -> Optional[float]:
    if value is None or math.isclose(high, low):
        return None
    clipped = max(low, min(high, float(value)))
    return 100.0 * ((clipped - low) / (high - low))


def _bounded_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator_value <= 0:
        return None
    return max(0.0, min(numerator_value / denominator_value, 1.0))


def _pct_change(current: Any, prior: Any) -> Optional[float]:
    current_value = _safe_float(current)
    prior_value = _safe_float(prior)
    if current_value is None or prior_value in (None, 0.0):
        return None
    return current_value / prior_value - 1.0


def _realized_vol(close_values: Sequence[Any], window: int) -> Optional[float]:
    returns: List[float] = []
    previous: Optional[float] = None
    for value in close_values:
        current = _safe_float(value)
        if current is None:
            previous = None
            continue
        if previous not in (None, 0.0):
            returns.append(current / previous - 1.0)
        previous = current
    if len(returns) < window:
        return None
    sigma = statistics.pstdev(returns[-window:])
    return float(sigma * math.sqrt(252.0))


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged
