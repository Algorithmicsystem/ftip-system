from __future__ import annotations

import datetime as dt
import math
import re
import statistics
from collections import Counter
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from api import config
from api.assistant.coverage import availability_payload
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
_NARRATIVE_BUCKETS = {
    "earnings_guidance": {"earnings", "guidance", "revenue", "margin", "forecast", "quarter", "results"},
    "policy_regulation": {"regulation", "regulatory", "antitrust", "export", "ban", "probe", "license", "policy"},
    "product_ai_cycle": {"ai", "gpu", "chip", "datacenter", "model", "launch", "semiconductor", "factory"},
    "demand_growth": {"demand", "growth", "orders", "backlog", "adoption", "expansion", "wins"},
    "supply_chain": {"supply", "shipping", "factory", "inventory", "capacity", "component"},
    "capital_markets": {"upgrade", "downgrade", "target", "valuation", "buyback", "offering"},
    "legal_risk": {"lawsuit", "investigation", "sec", "doj", "ftc", "litigation", "fraud"},
    "macro_market": {"rates", "inflation", "yield", "fed", "macro", "recession", "payrolls"},
}
_CROSS_ASSET_PROXY_DEFS = {
    "SPY": {"bucket": "broad_market", "label": "S&P 500"},
    "QQQ": {"bucket": "broad_market", "label": "NASDAQ 100"},
    "IWM": {"bucket": "broad_market", "label": "Russell 2000"},
    "TLT": {"bucket": "rates", "label": "Long-duration Treasuries"},
    "GLD": {"bucket": "commodities", "label": "Gold"},
    "USO": {"bucket": "commodities", "label": "Oil"},
    "UUP": {"bucket": "fx", "label": "US Dollar"},
}
_SECTOR_MACRO_NOTES = {
    "technology": "Technology setups are more sensitive to rates and liquidity conditions than slower-growth sectors.",
    "financial services": "Financial setups are especially sensitive to rates, credit spreads, and curve behavior.",
    "financial": "Financial setups are especially sensitive to rates, credit spreads, and curve behavior.",
    "energy": "Energy setups are influenced by growth and commodity context more than pure duration moves.",
    "consumer defensive": "Defensive consumer names usually care more about inflation and labor resilience than cyclical beta.",
    "consumer staples": "Defensive consumer names usually care more about inflation and labor resilience than cyclical beta.",
    "industrials": "Industrial setups are more exposed to growth and supply-chain conditions than pure multiple expansion.",
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
    fundamentals_overlay = _build_fundamental_overlay(symbol, symbol_meta, as_of_date)
    news_overlay = _build_news_overlay(symbol, symbol_meta, as_of_date)
    cross_asset_overlay = _build_cross_asset_overlay(
        symbol_meta,
        as_of_date,
        base_market=data_bundle.get("market_price_volume") or {},
    )
    macro_overlay = _build_macro_overlay(symbol_meta, cross_asset_overlay)
    geopolitical_overlay = _build_geopolitical_overlay(news_overlay, symbol_meta)
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
    fallback_source: List[str] = []
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        rows = fetch_reference_bars(symbol, as_of_date - dt.timedelta(days=90), as_of_date)
        providers = sorted({str(row.get("source")) for row in rows if row.get("source")})
        latest_external_close = rows[-1].get("close") if rows else None
        base_close = base_market.get("latest_close")
        if latest_external_close not in (None, 0) and base_close not in (None, 0):
            verification_gap_pct = float(base_close) / float(latest_external_close) - 1.0
        external_status = "fresh" if rows else "limited"
        fallback_source = providers
    except Exception as exc:
        notes.append(f"external market verification unavailable: {exc}")
        rows = []
        latest_external_close = None
    verification_status = "aligned"
    if verification_gap_pct is not None and abs(verification_gap_pct) >= 0.02:
        verification_status = "divergent"
    elif verification_gap_pct is None:
        verification_status = "unverified"
    return {
        "external_verification": {
            "verification_gap_pct": verification_gap_pct,
            "verification_status": verification_status,
            "latest_external_close": latest_external_close,
            "latest_external_date": rows[-1].get("as_of_date") if rows else None,
            "source": rows[-1].get("source") if rows else None,
        },
        "freshness": {
            "fetched_at": fetched_at,
            "data_as_of": rows[-1].get("as_of_date") if rows else None,
            "freshness_status": external_status,
        },
        "provenance": {
            "sources_used": providers,
            "fallback_used": bool(fallback_source),
            "fallback_source": fallback_source,
            "notes": notes,
            "confidence": 0.9 if verification_status == "aligned" else 0.6 if rows else 0.25,
        },
        "meta": {
            "external_sources": providers,
            "external_status": external_status,
            "external_notes": notes,
            **availability_payload(
                has_data=bool(rows),
                coverage_score=1.0 if rows else 0.0,
                freshness_status=external_status,
                missing_reason="unavailable" if not rows else None,
                fallback_used=bool(fallback_source),
                fallback_source=fallback_source,
                data_quality_note=(
                    "External market verification is aligned with the primary instrument tape."
                    if verification_status == "aligned"
                    else "External market verification is available but shows a noticeable source gap."
                    if verification_status == "divergent"
                    else "External market verification is currently unavailable."
                ),
            ),
            "fetched_at": fetched_at,
            "data_as_of": rows[-1].get("as_of_date") if rows else None,
            "confidence": 0.9 if verification_status == "aligned" else 0.6 if rows else 0.25,
        },
    }


def _build_fundamental_overlay(
    symbol: str,
    symbol_meta: Dict[str, Any],
    as_of_date: dt.date,
) -> Dict[str, Any]:
    provider_payloads: Dict[str, Any] = {}
    provider_status: Dict[str, Any] = {}
    notes: List[str] = []
    finnhub_profile, finnhub_profile_status = _safe_call(lambda: fetch_company_profile(symbol))
    provider_status["finnhub_profile"] = finnhub_profile_status
    if finnhub_profile is not None:
        provider_payloads["finnhub_profile"] = finnhub_profile
    elif finnhub_profile_status.get("note"):
        notes.append(f"finnhub_profile: {finnhub_profile_status['note']}")

    alpha_overview, alpha_status = _safe_call(lambda: fetch_company_overview(symbol))
    provider_status["alphavantage_overview"] = alpha_status
    if alpha_overview is not None:
        provider_payloads["alphavantage_overview"] = alpha_overview
    elif alpha_status.get("note"):
        notes.append(f"alphavantage_overview: {alpha_status['note']}")

    sec_profile, sec_status = _safe_call(
        lambda: fetch_company_filing_profile(
            symbol,
            company_name=(
                symbol_meta.get("name")
                or (finnhub_profile or {}).get("name")
                or (alpha_overview or {}).get("name")
            ),
            as_of_date=as_of_date,
        )
    )
    provider_status["sec_edgar"] = sec_status
    if sec_profile is not None:
        provider_payloads["sec_edgar"] = sec_profile
    elif sec_status.get("note"):
        notes.append(f"sec_edgar: {sec_status['note']}")

    finnhub_metrics, finnhub_metrics_status = _safe_call(lambda: fetch_basic_financials(symbol))
    provider_status["finnhub_basic_financials"] = finnhub_metrics_status
    if finnhub_metrics is not None:
        provider_payloads["finnhub_basic_financials"] = finnhub_metrics
    elif finnhub_metrics_status.get("note"):
        notes.append(f"finnhub_basic_financials: {finnhub_metrics_status['note']}")

    sec_profile = sec_profile or {}
    finnhub_metrics = finnhub_metrics or {}
    alpha_overview = alpha_overview or {}
    finnhub_profile = finnhub_profile or {}

    normalized_metrics = dict((sec_profile.get("normalized_metrics") or {}))
    normalized_metrics["revenue_growth_yoy"] = _first_non_null(
        normalized_metrics.get("revenue_growth_yoy"),
        alpha_overview.get("quarterly_revenue_growth_yoy"),
        finnhub_metrics.get("revenue_growth_ttm_yoy"),
    )
    normalized_metrics["operating_margin"] = _first_non_null(
        normalized_metrics.get("operating_margin"),
        alpha_overview.get("operating_margin_ttm"),
        finnhub_metrics.get("operating_margin_ttm"),
    )
    normalized_metrics["net_margin"] = _first_non_null(
        normalized_metrics.get("net_margin"),
        alpha_overview.get("profit_margin"),
        finnhub_metrics.get("net_margin"),
    )
    normalized_metrics["current_ratio"] = _first_non_null(
        normalized_metrics.get("current_ratio"),
        finnhub_metrics.get("current_ratio_quarterly"),
    )
    normalized_metrics["cash_ratio"] = _first_non_null(
        normalized_metrics.get("cash_ratio"),
        finnhub_metrics.get("quick_ratio_quarterly"),
    )
    normalized_metrics["debt_to_equity"] = _first_non_null(
        normalized_metrics.get("debt_to_equity"),
        finnhub_metrics.get("total_debt_to_equity_quarterly"),
    )
    normalized_metrics["return_on_assets"] = _first_non_null(
        normalized_metrics.get("return_on_assets"),
        alpha_overview.get("return_on_assets_ttm"),
        finnhub_metrics.get("roa_ttm"),
    )
    normalized_metrics["return_on_equity"] = _first_non_null(
        normalized_metrics.get("return_on_equity"),
        alpha_overview.get("return_on_equity_ttm"),
        finnhub_metrics.get("roe_ttm"),
    )

    coverage_flags = dict((sec_profile.get("coverage_flags") or {}))
    coverage = {
        "revenue_growth_coverage": normalized_metrics.get("revenue_growth_yoy") is not None,
        "profitability_coverage": any(
            normalized_metrics.get(field) is not None
            for field in ("operating_margin", "net_margin", "return_on_assets", "return_on_equity")
        ),
        "margin_coverage": any(
            normalized_metrics.get(field) is not None
            for field in ("gross_margin", "operating_margin", "net_margin")
        ),
        "balance_sheet_resilience_coverage": any(
            normalized_metrics.get(field) is not None
            for field in ("current_ratio", "cash_ratio", "liabilities_to_assets")
        ),
        "cash_flow_coverage": any(
            normalized_metrics.get(field) is not None
            for field in ("free_cash_flow", "free_cash_flow_margin", "positive_fcf_ratio")
        ),
        "leverage_liquidity_coverage": any(
            normalized_metrics.get(field) is not None
            for field in ("current_ratio", "cash_ratio", "debt_to_equity")
        ),
        "filing_recency_coverage": sec_profile.get("filing_recency_days") is not None,
        "reporting_quality_coverage": any(
            (sec_profile.get("quality_proxies") or {}).get(field) is not None
            for field in ("reporting_completeness_score", "reporting_quality_proxy")
        ),
    }
    coverage_flags.update(coverage)
    coverage_score = _bounded_ratio(
        sum(1 for ok in coverage.values() if ok),
        len(coverage),
    ) or sec_profile.get("coverage_score") or 0.0
    missingness_flags = list(sec_profile.get("missingness_flags") or [])
    missingness_flags.extend(
        f"{field}_missing"
        for field, ok in coverage.items()
        if not ok and f"{field}_missing" not in missingness_flags
    )

    sec_quality = sec_profile.get("quality_proxies") or {}
    quality_proxies = {
        "filing_recency_score": _first_non_null(
            sec_quality.get("filing_recency_score"),
            _bounded_score(
                1.0 - min((sec_profile.get("filing_recency_days") or 365), 365) / 365.0,
                low=0.0,
                high=1.0,
            ),
        ),
        "reporting_completeness_score": _first_non_null(
            sec_quality.get("reporting_completeness_score"),
            coverage_score * 100.0,
        ),
        "reporting_quality_proxy": _first_non_null(
            sec_quality.get("reporting_quality_proxy"),
            _mean(
                [
                    sec_quality.get("reporting_quality_proxy"),
                    _first_non_null(sec_profile.get("margin_stability"), normalized_metrics.get("operating_margin_stability")),
                ]
            ),
        ),
        "business_quality_durability": _first_non_null(
            sec_quality.get("business_quality_durability"),
            _mean(
                [
                    _bounded_score(normalized_metrics.get("revenue_growth_yoy"), low=-0.2, high=0.3),
                    _bounded_score(_mean([normalized_metrics.get("operating_margin"), normalized_metrics.get("net_margin")]), low=0.0, high=0.3),
                    _bounded_score(normalized_metrics.get("positive_fcf_ratio"), low=0.0, high=1.0),
                    _bounded_score(
                        _mean([normalized_metrics.get("current_ratio"), _inverse_metric(normalized_metrics.get("debt_to_equity"), cap=3.0)]),
                        low=0.0,
                        high=2.0,
                    ),
                ]
            ),
        ),
        "balance_sheet_resilience": _first_non_null(
            sec_quality.get("balance_sheet_resilience"),
            _bounded_score(
                _mean([normalized_metrics.get("current_ratio"), _inverse_metric(normalized_metrics.get("debt_to_equity"), cap=3.0)]),
                low=0.0,
                high=2.0,
            ),
        ),
        "cash_flow_durability": _first_non_null(
            sec_quality.get("cash_flow_durability"),
            _bounded_score(
                _mean([normalized_metrics.get("positive_fcf_ratio"), normalized_metrics.get("free_cash_flow_margin")]),
                low=0.0,
                high=0.3,
            ),
        ),
    }

    strengths = list(sec_profile.get("strength_summary") or [])
    weaknesses = list(sec_profile.get("weakness_summary") or [])
    coverage_caveats = list(sec_profile.get("coverage_caveats") or [])
    strengths.extend(_secondary_strengths(normalized_metrics))
    weaknesses.extend(_secondary_weaknesses(normalized_metrics))
    if provider_status["sec_edgar"].get("status") != "ok":
        coverage_caveats.append("SEC filing backbone is unavailable, so the fundamental layer is relying more heavily on secondary enrichment.")
    if provider_status["alphavantage_overview"].get("status") != "ok" and provider_status["finnhub_basic_financials"].get("status") != "ok":
        coverage_caveats.append("Secondary enrichment from Finnhub and Alpha Vantage is unavailable or thin.")
    filing_recency_days = sec_profile.get("filing_recency_days")
    status = (
        "fresh"
        if filing_recency_days is not None and filing_recency_days <= 120
        else "stale_but_usable"
        if filing_recency_days is not None and filing_recency_days <= 210
        else "limited"
    )

    latest_quarter = (
        ((sec_profile.get("statement_snapshot") or {}).get("latest_quarter"))
        or sec_profile.get("latest_quarter")
        or {}
    )
    if latest_quarter and latest_quarter.get("op_margin") is None:
        latest_quarter = {
            **latest_quarter,
            "op_margin": latest_quarter.get("operating_margin"),
        }
    latest_annual = ((sec_profile.get("statement_snapshot") or {}).get("latest_annual")) or {}
    latest_balance_sheet = (
        ((sec_profile.get("statement_snapshot") or {}).get("latest_balance_sheet")) or {}
    )
    sources = sorted(provider_payloads)
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "mapping": sec_profile.get("mapping") or {},
        "filing_backbone": sec_profile.get("filing_backbone") or {},
        "statement_snapshot": {
            "latest_quarter": latest_quarter,
            "prior_year_quarter": ((sec_profile.get("statement_snapshot") or {}).get("prior_year_quarter")) or {},
            "latest_annual": latest_annual,
            "latest_balance_sheet": latest_balance_sheet,
            "quarterly_series": ((sec_profile.get("statement_snapshot") or {}).get("quarterly_series")) or [],
            "annual_series": ((sec_profile.get("statement_snapshot") or {}).get("annual_series")) or [],
        },
        "coverage_flags": coverage_flags,
        "coverage_score": coverage_score,
        "missingness_flags": missingness_flags,
        "company_profile": {
            "name": symbol_meta.get("name") or finnhub_profile.get("name") or sec_profile.get("name"),
            "sector": symbol_meta.get("sector") or alpha_overview.get("sector"),
            "industry": alpha_overview.get("industry") or finnhub_profile.get("finnhub_industry"),
            "exchange": symbol_meta.get("exchange") or finnhub_profile.get("exchange"),
            "country": symbol_meta.get("country") or finnhub_profile.get("country"),
            "currency": symbol_meta.get("currency") or finnhub_profile.get("currency"),
            "market_cap": _first_non_null(
                finnhub_metrics.get("market_cap"),
                alpha_overview.get("market_cap"),
                finnhub_profile.get("market_capitalization"),
            ),
        },
        "normalized_metrics": normalized_metrics,
        "quality_proxies": quality_proxies,
        "strength_summary": _dedupe_text(strengths)[:5],
        "weakness_summary": _dedupe_text(weaknesses)[:5],
        "coverage_caveats": _dedupe_text(coverage_caveats + notes)[:8],
        "provider_snapshot": provider_payloads,
        "filing_recency_days": filing_recency_days,
        "latest_quarter": latest_quarter,
        "revenue_growth_yoy": normalized_metrics.get("revenue_growth_yoy"),
        "margin_stability": _first_non_null(
            sec_profile.get("margin_stability"),
            normalized_metrics.get("operating_margin_stability"),
            normalized_metrics.get("gross_margin_stability"),
        ),
        "positive_fcf_ratio": _first_non_null(
            sec_profile.get("positive_fcf_ratio"),
            normalized_metrics.get("positive_fcf_ratio"),
        ),
        "durability_proxies": {
            "growth_quality": _bounded_score(
                normalized_metrics.get("revenue_growth_yoy"),
                low=-0.25,
                high=0.35,
            ),
            "profitability_quality": _bounded_score(
                _mean([normalized_metrics.get("operating_margin"), normalized_metrics.get("net_margin")]),
                low=-0.05,
                high=0.3,
            ),
            "balance_sheet_resilience": quality_proxies.get("balance_sheet_resilience"),
            "reporting_stability": _first_non_null(
                quality_proxies.get("reporting_quality_proxy"),
                100.0 * coverage_score,
            ),
        },
        "freshness": {
            "fetched_at": fetched_at,
            "data_as_of": _first_non_null(
                latest_quarter.get("report_date"),
                latest_annual.get("report_date"),
                ((sec_profile.get("filing_backbone") or {}).get("latest_filing_date")),
            ),
            "freshness_status": status,
        },
        "provenance": {
            "sources_used": sources,
            "provider_status": provider_status,
            "fallback_used": bool(sec_profile) is False and bool(provider_payloads),
            "fallback_source": [source for source in sources if source != "sec_edgar"],
            "confidence": _provider_confidence(
                provider_status,
                sources_used=sources,
                coverage_score=coverage_score,
            ),
            "notes": _dedupe_text(coverage_caveats + notes)[:8],
        },
        "meta": {
            "sources": sources,
            "sources_used": sources,
            "primary_source": "sec_edgar" if sec_profile else "secondary_enrichment",
            "status": status,
            "coverage_score": coverage_score,
            "missingness": max(0.0, 1.0 - coverage_score),
            "provider_status": provider_status,
            "notes": notes,
            "fetched_at": fetched_at,
            "updated_at": fetched_at,
            "latest_report_date": _first_non_null(
                latest_quarter.get("report_date"),
                latest_annual.get("report_date"),
                ((sec_profile.get("filing_backbone") or {}).get("latest_filing_date")),
            ),
            "fallback_notes": _dedupe_text(coverage_caveats + notes)[:8],
            "confidence": _provider_confidence(
                provider_status,
                sources_used=sources,
                coverage_score=coverage_score,
            ),
            **availability_payload(
                has_data=bool(sec_profile or provider_payloads),
                coverage_score=coverage_score,
                freshness_status=status,
                missing_reason="unavailable" if not sec_profile and not provider_payloads else None,
                fallback_used=bool(sec_profile) is False and bool(provider_payloads),
                fallback_source=[source for source in sources if source != "sec_edgar"],
                data_quality_note=(
                    "Fundamental coverage is SEC-anchored and supplemented with secondary enrichment."
                    if sec_profile
                    else "Fundamental coverage is relying on secondary enrichment because the SEC backbone is unavailable."
                ),
            ),
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
    source_mix = _source_mix(source_breakdown, provider_status, len(articles))
    sentiment_bias = _headline_sentiment_bias(titles)
    novelty_ratio = _bounded_ratio(len({_normalize_title(title) for title in titles if title}), len(titles))
    contradiction_score = _contradiction_score(titles)
    top_narratives = _extract_topics(titles, symbol)
    topic_clusters = _topic_clusters(articles, symbol)
    topic_bucket_counts = _topic_bucket_counts(titles)
    gdelt_articles = list(provider_payloads.get("gdelt") or [])
    gdelt_tone = _mean(_safe_float(item.get("tone")) for item in gdelt_articles)
    event_overlay = {
        "gdelt_article_count": len(gdelt_articles),
        "gdelt_tone_average": gdelt_tone,
        "event_theme_counts": _topic_bucket_counts([str(item.get("title") or "") for item in gdelt_articles]),
        "coverage_note": (
            "GDELT is contributing broader policy and event coverage on top of finance-news APIs."
            if gdelt_articles
            else "GDELT did not add incremental event coverage in the active window."
        ),
    }
    latest_published_at = articles[0].get("published_at") if articles else None
    attention_score = _bounded_score(_attention_multiple(source_breakdown, len(articles)), low=1.0, high=8.0)
    persistence_score = _narrative_persistence_score(topic_clusters, len(articles))
    sentiment_confidence = _narrative_confidence(
        headline_count=len(articles),
        source_count=len(source_breakdown),
        contradiction_score=contradiction_score,
        topic_clusters=topic_clusters,
    )
    coverage_score = _bounded_ratio(
        len(articles) + len(source_breakdown),
        18,
    )
    status = "fresh" if articles else "limited"
    coverage_note = (
        "Narrative coverage is broad enough to support a directional sentiment overlay."
        if len(articles) >= 8 and len(source_breakdown) >= 3
        else "Narrative coverage is present but still somewhat sparse, so tone should be read as supportive context rather than a standalone signal."
        if articles
        else "Narrative coverage is currently unavailable because no recent articles survived multi-source normalization."
    )
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "headline_count": len(articles),
        "source_count": len(source_breakdown),
        "source_mix": source_mix,
        "sentiment_level_proxy": sentiment_bias,
        "sentiment_summary": {
            "bias": sentiment_bias,
            "tone_label": _sentiment_tone_label(sentiment_bias),
            "gdelt_tone_average": gdelt_tone,
            "supportive_ratio": _supportive_ratio(titles),
            "caution_ratio": _caution_ratio(titles),
        },
        "sentiment_confidence": sentiment_confidence,
        "attention_crowding": _attention_multiple(source_breakdown, len(articles)),
        "attention_score": attention_score,
        "novelty_ratio": novelty_ratio,
        "novelty_score": _bounded_score(novelty_ratio, low=0.2, high=1.0),
        "repetition_score": _bounded_score(1.0 - (novelty_ratio or 0.0), low=0.0, high=0.8),
        "persistence_score": persistence_score,
        "disagreement_score": contradiction_score,
        "contradiction_score": contradiction_score,
        "aggregated_sentiment_bias": sentiment_bias,
        "source_breakdown": source_breakdown,
        "top_narratives": top_narratives,
        "dominant_topics": top_narratives,
        "topic_clusters": topic_clusters,
        "topic_buckets": topic_bucket_counts,
        "event_overlay": event_overlay,
        "aggregated_headlines": articles[:10],
        "recent_headlines": articles[:10],
        "provider_snapshot": provider_payloads,
        "freshness": {
            "fetched_at": fetched_at,
            "data_as_of": latest_published_at,
            "freshness_status": status,
            "latest_news_published_at": latest_published_at,
        },
        "provenance": {
            "sources_used": sorted(source_breakdown),
            "source_mix": source_mix,
            "provider_status": provider_status,
            "fallback_used": any(
                status_info.get("status") != "ok" for status_info in provider_status.values()
            )
            and bool(articles),
            "fallback_source": sorted(source_breakdown),
            "confidence": sentiment_confidence,
            "notes": [
                coverage_note,
                event_overlay["coverage_note"],
            ],
        },
        "meta": {
            "sources": sorted(source_breakdown),
            "sources_used": sorted(source_breakdown),
            "latest_news_published_at": latest_published_at,
            "coverage_score": coverage_score,
            "status": status,
            "provider_status": provider_status,
            "confidence": sentiment_confidence,
            "fetched_at": fetched_at,
            "data_as_of": latest_published_at,
            **availability_payload(
                has_data=bool(articles),
                coverage_score=coverage_score,
                freshness_status=status,
                missing_reason="unavailable" if not articles else None,
                fallback_used=bool(articles)
                and any(status_info.get("status") != "ok" for status_info in provider_status.values()),
                fallback_source=sorted(source_breakdown),
                data_quality_note=coverage_note,
            ),
        },
    }


def _build_macro_overlay(
    symbol_meta: Dict[str, Any],
    cross_asset_overlay: Dict[str, Any],
) -> Dict[str, Any]:
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
    rates_context = _macro_dimension_context("rates", fred_snapshot.get("rates") or {}, higher_is_tighter=True)
    policy_rate_context = _macro_dimension_context(
        "policy_rate",
        fred_snapshot.get("policy_rate") or {},
        higher_is_tighter=True,
    )
    inflation_context = _macro_dimension_context(
        "inflation",
        _first_non_null(fred_snapshot.get("inflation"), world_bank_snapshot.get("inflation")) or {},
        higher_is_tighter=True,
    )
    growth_context = _macro_dimension_context(
        "growth",
        _first_non_null(world_bank_snapshot.get("gdp_growth"), fred_snapshot.get("growth")) or {},
        higher_is_supportive=True,
    )
    labor_context = _macro_dimension_context(
        "labor",
        _first_non_null(fred_snapshot.get("labor"), world_bank_snapshot.get("unemployment")) or {},
        higher_is_tighter=False,
    )
    credit_context = _macro_dimension_context(
        "credit",
        fred_snapshot.get("credit") or {},
        higher_is_tighter=True,
    )
    liquidity_context = _liquidity_context(policy_rate_context, credit_context, cross_asset_overlay)
    sector = str(symbol_meta.get("sector") or "").strip().lower()
    alignment_notes = _macro_alignment_notes(
        sector=sector,
        regime=macro_regime,
        cross_asset_overlay=cross_asset_overlay,
    )
    country_relevance = "supplemental" if _world_bank_country(symbol_meta.get("country")) == "US" else "primary"
    coverage_score = _bounded_ratio(
        len(fred_payloads) + len(world_bank_payloads),
        len(_FRED_SERIES) + len(_WORLD_BANK_INDICATORS),
    )
    sources_used = sorted(
        {"fred" if fred_payloads else None, "world_bank" if world_bank_payloads else None} - {None}
    )
    confidence = _provider_confidence(provider_status, sources_used=sources_used, coverage_score=coverage_score)
    status = "fresh" if fred_payloads else "limited"
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "fred_series": fred_snapshot,
        "world_bank_series": world_bank_snapshot,
        "macro_regime_context": macro_regime,
        "macro_regime_summary": {
            "regime": macro_regime.get("regime"),
            "summary": _macro_regime_summary(
                macro_regime,
                rates_context=rates_context,
                inflation_context=inflation_context,
                growth_context=growth_context,
                labor_context=labor_context,
                liquidity_context=liquidity_context,
            ),
            "confidence": confidence,
        },
        "rates_context": rates_context,
        "policy_rate_context": policy_rate_context,
        "inflation_context": inflation_context,
        "growth_context": growth_context,
        "labor_context": labor_context,
        "liquidity_context": liquidity_context,
        "credit_context": credit_context,
        "country_context": {
            "country": _world_bank_country(symbol_meta.get("country")),
            "relevance": country_relevance,
            "world_bank_indicators": world_bank_snapshot,
        },
        "macro_alignment_notes": alignment_notes,
        "freshness": {
            "fetched_at": fetched_at,
            "data_as_of": _latest_snapshot_date([fred_snapshot, world_bank_snapshot]),
            "freshness_status": status,
        },
        "provenance": {
            "sources_used": sources_used,
            "provider_status": provider_status,
            "fallback_used": False,
            "fallback_source": [],
            "confidence": confidence,
            "notes": alignment_notes,
        },
        "meta": {
            "sources": sources_used,
            "sources_used": sources_used,
            "coverage_score": coverage_score,
            "status": status,
            "provider_status": provider_status,
            "confidence": confidence,
            "fetched_at": fetched_at,
            "data_as_of": _latest_snapshot_date([fred_snapshot, world_bank_snapshot]),
            **availability_payload(
                has_data=bool(fred_payloads or world_bank_payloads),
                coverage_score=coverage_score,
                freshness_status=status,
                missing_reason="unavailable" if not fred_payloads and not world_bank_payloads else None,
                data_quality_note=(
                    "Macro coverage is broad enough to frame the active setup across rates, inflation, growth, labor, and liquidity."
                    if coverage_score is not None and coverage_score >= 0.6
                    else "Macro coverage is present but still partial, so regime framing should be treated as context rather than a dominant signal."
                ),
            ),
        },
    }


def _build_cross_asset_overlay(
    symbol_meta: Dict[str, Any],
    as_of_date: dt.date,
    *,
    base_market: Dict[str, Any],
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
            proxy_snapshot[proxy] = {
                **_bars_snapshot(payload),
                "symbol": proxy,
            }
    benchmark_symbol = sector_proxy or "SPY"
    benchmark = proxy_snapshot.get(benchmark_symbol) or proxy_snapshot.get("SPY")
    stock_ret_21d = base_market.get("ret_21d")
    stock_ret_63d = base_market.get("ret_63d")
    benchmarks = {
        proxy: snapshot
        for proxy, snapshot in proxy_snapshot.items()
        if _CROSS_ASSET_PROXY_DEFS.get(proxy, {}).get("bucket") == "broad_market"
    }
    broad_market_context = {
        "major_benchmarks": benchmarks,
        "average_ret_21d": _mean(snapshot.get("ret_21d") for snapshot in benchmarks.values()),
        "average_vol_21d": _mean(snapshot.get("vol_21d") for snapshot in benchmarks.values()),
        "risk_tone": _risk_tone_label(_mean(snapshot.get("ret_21d") for snapshot in benchmarks.values())),
    }
    sector_context = {
        "sector_proxy": sector_proxy,
        "sector_proxy_snapshot": proxy_snapshot.get(sector_proxy) if sector_proxy else None,
        "relative_vs_sector_ret_21d": _relative_gap(stock_ret_21d, (proxy_snapshot.get(sector_proxy) or {}).get("ret_21d")),
        "relative_vs_sector_ret_63d": _relative_gap(stock_ret_63d, (proxy_snapshot.get(sector_proxy) or {}).get("ret_63d")),
        "coverage_note": (
            f"Sector context is anchored to {sector_proxy}."
            if sector_proxy and proxy_snapshot.get(sector_proxy)
            else "Sector-specific proxy coverage is unavailable, so broader-market benchmarks are being used."
        ),
    }
    rates_fx_commodities = {
        "rates_proxy": proxy_snapshot.get("TLT"),
        "fx_proxy": proxy_snapshot.get("UUP"),
        "commodity_proxies": {
            "gold": proxy_snapshot.get("GLD"),
            "oil": proxy_snapshot.get("USO"),
        },
    }
    relative_move_summary = {
        "vs_benchmark_ret_21d": _relative_gap(stock_ret_21d, benchmark.get("ret_21d") if benchmark else None),
        "vs_benchmark_ret_63d": _relative_gap(stock_ret_63d, benchmark.get("ret_63d") if benchmark else None),
        "vs_sector_ret_21d": sector_context.get("relative_vs_sector_ret_21d"),
        "market_relative_note": _relative_note(
            stock_ret_21d,
            benchmark.get("ret_21d") if benchmark else None,
            benchmark.get("symbol") if benchmark else None,
        ),
        "sector_relative_note": _relative_note(
            stock_ret_21d,
            (proxy_snapshot.get(sector_proxy) or {}).get("ret_21d") if sector_proxy else None,
            sector_proxy,
        ),
    }
    sources_used = sorted(
        {
            str(snapshot.get("source"))
            for snapshot in proxy_snapshot.values()
            if snapshot.get("source")
        }
    )
    coverage_score = _bounded_ratio(len(proxy_snapshot), len(proxy_universe))
    confidence = _provider_confidence(provider_status, sources_used=sources_used, coverage_score=coverage_score)
    fallback_used = bool(benchmark and sector_proxy and benchmark.get("symbol") != sector_proxy)
    fallback_source = [benchmark.get("symbol")] if fallback_used and benchmark else []
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "benchmark_proxy": benchmark.get("symbol") if benchmark else None,
        "benchmark_ret_21d": benchmark.get("ret_21d") if benchmark else None,
        "benchmark_vol_21d": benchmark.get("vol_21d") if benchmark else None,
        "benchmark_context": {
            "benchmark_symbol": benchmark.get("symbol") if benchmark else None,
            "benchmark_ret_21d": benchmark.get("ret_21d") if benchmark else None,
            "benchmark_vol_21d": benchmark.get("vol_21d") if benchmark else None,
        },
        "sector_proxy": sector_proxy,
        "sector_context": sector_context,
        "broad_market_context": broad_market_context,
        "rates_fx_commodities_context": rates_fx_commodities,
        "relative_move_summary": relative_move_summary,
        "proxy_snapshot": proxy_snapshot,
        "freshness": {
            "fetched_at": fetched_at,
            "data_as_of": _latest_snapshot_date([proxy_snapshot]),
            "freshness_status": "fresh" if proxy_snapshot else "limited",
        },
        "provenance": {
            "sources_used": sources_used,
            "provider_status": provider_status,
            "fallback_used": fallback_used,
            "fallback_source": fallback_source,
            "confidence": confidence,
            "notes": _dedupe_text(
                [
                    sector_context.get("coverage_note"),
                    relative_move_summary.get("market_relative_note"),
                    relative_move_summary.get("sector_relative_note"),
                ]
            ),
        },
        "meta": {
            "sources": sources_used,
            "sources_used": sources_used,
            "coverage_score": coverage_score,
            "status": "fresh" if proxy_snapshot else "limited",
            "provider_status": provider_status,
            "confidence": confidence,
            "fetched_at": fetched_at,
            "data_as_of": _latest_snapshot_date([proxy_snapshot]),
            **availability_payload(
                has_data=bool(proxy_snapshot),
                coverage_score=coverage_score,
                freshness_status="fresh" if proxy_snapshot else "limited",
                missing_reason="unavailable" if not proxy_snapshot else None,
                fallback_used=fallback_used,
                fallback_source=fallback_source,
                data_quality_note=(
                    sector_context.get("coverage_note")
                    if sector_context.get("coverage_note")
                    else "Cross-asset proxy coverage is currently unavailable."
                ),
            ),
        },
    }


def _build_geopolitical_overlay(
    news_overlay: Dict[str, Any],
    symbol_meta: Dict[str, Any],
) -> Dict[str, Any]:
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
    top_bucket = max(counts, key=counts.get) if any(counts.values()) else None
    relevance_label = (
        "material"
        if event_intensity >= 0.45
        else "background"
        if event_intensity >= 0.18
        else "weak"
    )
    relevance_note = (
        f"Event relevance is {relevance_label}, led by {top_bucket.replace('_', ' ')} themes."
        if top_bucket and counts[top_bucket] > 0
        else f"Geopolitical relevance appears weak for the current {symbol_meta.get('sector') or 'equity'} setup."
    )
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "event_buckets": counts,
        "event_intensity_score": event_intensity,
        "event_theme_summary": top_bucket,
        "relevance_label": relevance_label,
        "confidence": confidence,
        "recent_event_summary": relevant[:8],
        "coverage_confidence": confidence,
        "event_overlay": news_overlay.get("event_overlay") or {},
        "freshness": {
            "fetched_at": fetched_at,
            "data_as_of": articles[0].get("published_at") if articles else None,
            "freshness_status": "fresh" if articles else "limited",
        },
        "provenance": {
            "sources_used": sorted(
                {
                    str(item.get("source"))
                    for item in relevant
                    if item.get("source")
                }
            ),
            "provider_status": ((news_overlay.get("meta") or {}).get("provider_status") or {}),
            "fallback_used": False,
            "fallback_source": [],
            "confidence": confidence,
            "notes": [
                relevance_note,
                "Event tagging is heuristic and should be read as directional context rather than a precision event model.",
            ],
        },
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
            "confidence": confidence,
            "fetched_at": fetched_at,
            "data_as_of": articles[0].get("published_at") if articles else None,
            **availability_payload(
                has_data=bool(articles),
                coverage_score=_bounded_ratio(len(articles), 10),
                freshness_status="fresh" if articles else "limited",
                missing_reason="unavailable" if not articles else None,
                relevant=(relevance_label != "weak") if articles else True,
                data_quality_note=relevance_note,
            ),
        },
    }


def _build_quality_overlay(
    *,
    base_quality: Dict[str, Any],
    overlays: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    sources: Dict[str, List[str]] = {}
    freshness_summary: Dict[str, Dict[str, Any]] = {}
    provider_status_map: Dict[str, Dict[str, Any]] = {}
    domain_confidence: Dict[str, Optional[float]] = {}
    domain_notes: Dict[str, List[str]] = {}
    warnings: List[str] = list(base_quality.get("warnings") or [])
    provider_notes: List[str] = []
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    for domain, payload in overlays.items():
        meta = payload.get("meta") or {}
        sources[domain] = list(meta.get("sources") or [])
        freshness_summary[domain] = {
            "status": meta.get("status") or meta.get("external_status") or "unknown",
            "sources": list(meta.get("sources") or meta.get("external_sources") or []),
            "data_as_of": meta.get("data_as_of") or (payload.get("freshness") or {}).get("data_as_of"),
        }
        provider_status_map[domain] = dict(meta.get("provider_status") or {})
        domain_confidence[domain] = _safe_float(meta.get("confidence"))
        domain_notes[domain] = _dedupe_text(
            [
                *(meta.get("notes") or meta.get("external_notes") or []),
                meta.get("data_quality_note"),
                *((payload.get("provenance") or {}).get("notes") or []),
                (payload.get("freshness") or {}).get("freshness_status"),
            ]
        )
        for note in meta.get("notes") or meta.get("external_notes") or []:
            provider_notes.append(str(note))
        if freshness_summary[domain]["status"] in {"limited", "stale", "stale_but_usable"}:
            warnings.append(f"{domain} coverage is {freshness_summary[domain]['status']}.")
        if meta.get("fallback_used"):
            warnings.append(
                f"{domain} is relying on fallback sources {', '.join(meta.get('fallback_source') or [])}."
            )
    coverage_score = _bounded_ratio(
        sum(1 for item in freshness_summary.values() if item.get("status") == "fresh"),
        len(freshness_summary),
    )
    confidence = _mean(domain_confidence.values())
    return {
        "source_map": sources,
        "freshness_summary": freshness_summary,
        "provider_status_map": provider_status_map,
        "domain_confidence": domain_confidence,
        "domain_notes": domain_notes,
        "provider_notes": provider_notes,
        "warnings": list(dict.fromkeys(warnings)),
        "freshness": {
            "fetched_at": fetched_at,
            "data_as_of": _latest_snapshot_date([freshness_summary]),
            "freshness_status": "fresh" if coverage_score == 1.0 else "mixed",
        },
        "provenance": {
            "sources_used": sorted({source for values in sources.values() for source in values}),
            "provider_status": provider_status_map,
            "fallback_used": any(
                bool((overlays.get(domain, {}).get("meta") or {}).get("fallback_used"))
                for domain in overlays
            ),
            "fallback_source": sorted(
                {
                    source
                    for domain in overlays
                    for source in ((overlays.get(domain, {}).get("meta") or {}).get("fallback_source") or [])
                }
            ),
            "confidence": confidence,
            "notes": provider_notes,
        },
        "meta": {
            "status": "fresh" if not provider_notes else "mixed",
            "coverage_score": coverage_score,
            "confidence": confidence,
            "fetched_at": fetched_at,
            "data_as_of": _latest_snapshot_date([freshness_summary]),
            **availability_payload(
                has_data=bool(freshness_summary),
                coverage_score=coverage_score,
                freshness_status="fresh" if not provider_notes else "stale_but_usable",
                missing_reason="unavailable" if not freshness_summary else None,
                fallback_used=any(
                    bool((overlays.get(domain, {}).get("meta") or {}).get("fallback_used"))
                    for domain in overlays
                ),
                fallback_source=sorted(
                    {
                        source
                        for domain in overlays
                        for source in ((overlays.get(domain, {}).get("meta") or {}).get("fallback_source") or [])
                    }
                ),
                data_quality_note=(
                    "Domain provenance, freshness, and fallback state are populated across the enriched data fabric."
                    if freshness_summary
                    else "No enriched domain provenance was captured."
                ),
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
        "ret_63d": _pct_change(latest, closes[-64]) if len(closes) >= 64 else None,
        "vol_21d": _realized_vol(closes, 21),
        "latest_date": rows[-1].get("as_of_date"),
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


def _source_mix(
    source_breakdown: Dict[str, int],
    provider_status: Dict[str, Dict[str, Any]],
    headline_count: int,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for source, count in sorted(source_breakdown.items(), key=lambda item: (-item[1], item[0])):
        items.append(
            {
                "source": source,
                "count": count,
                "share": _bounded_ratio(count, headline_count),
                "status": (provider_status.get(source) or {}).get("status"),
            }
        )
    return items


def _topic_bucket_counts(titles: Sequence[str]) -> Dict[str, int]:
    counts = {bucket: 0 for bucket in _NARRATIVE_BUCKETS}
    for title in titles:
        lowered = title.lower()
        for bucket, keywords in _NARRATIVE_BUCKETS.items():
            if any(keyword in lowered for keyword in keywords):
                counts[bucket] += 1
    return counts


def _topic_clusters(
    articles: Sequence[Dict[str, Any]],
    symbol: str,
) -> List[Dict[str, Any]]:
    titles = [str(item.get("title") or "") for item in articles]
    bucket_counts = _topic_bucket_counts(titles)
    normalized_titles = [_normalize_title(title) for title in titles if title]
    clusters: List[Dict[str, Any]] = []
    for bucket, count in sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0])):
        if count <= 0:
            continue
        examples = []
        for article in articles:
            title = str(article.get("title") or "")
            lowered = title.lower()
            if any(keyword in lowered for keyword in _NARRATIVE_BUCKETS[bucket]):
                normalized = _normalize_title(title)
                if normalized not in examples:
                    examples.append(normalized)
            if len(examples) >= 3:
                break
        clusters.append(
            {
                "topic": bucket,
                "count": count,
                "share": _bounded_ratio(count, len(normalized_titles)),
                "examples": examples,
            }
        )
    if clusters:
        return clusters
    return _extract_topics(titles, symbol)


def _narrative_persistence_score(
    topic_clusters: Sequence[Dict[str, Any]],
    headline_count: int,
) -> Optional[float]:
    if not topic_clusters or headline_count <= 0:
        return None
    dominant = max((cluster.get("count") or 0) for cluster in topic_clusters)
    return _bounded_score(_bounded_ratio(dominant, headline_count), low=0.1, high=0.7)


def _narrative_confidence(
    *,
    headline_count: int,
    source_count: int,
    contradiction_score: Optional[float],
    topic_clusters: Sequence[Dict[str, Any]],
) -> Optional[float]:
    confidence = _mean(
        [
            _bounded_ratio(headline_count, 12),
            _bounded_ratio(source_count, 4),
            1.0 - min(contradiction_score or 0.0, 1.0),
            _bounded_ratio(len(topic_clusters), 4),
        ]
    )
    return _bounded_score(confidence, low=0.0, high=1.0)


def _sentiment_tone_label(value: Optional[float]) -> str:
    if value is None:
        return "unclear"
    if value >= 0.15:
        return "constructive"
    if value <= -0.15:
        return "negative"
    return "mixed"


def _supportive_ratio(titles: Sequence[str]) -> Optional[float]:
    if not titles:
        return None
    hits = sum(
        1 for title in titles if any(word in title.lower() for word in _POSITIVE_HEADLINE_WORDS)
    )
    return _bounded_ratio(hits, len(titles))


def _caution_ratio(titles: Sequence[str]) -> Optional[float]:
    if not titles:
        return None
    hits = sum(
        1 for title in titles if any(word in title.lower() for word in _NEGATIVE_HEADLINE_WORDS)
    )
    return _bounded_ratio(hits, len(titles))


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


def _macro_dimension_context(
    label: str,
    snapshot: Dict[str, Any],
    *,
    higher_is_tighter: bool = False,
    higher_is_supportive: bool = False,
) -> Dict[str, Any]:
    latest = _safe_float(snapshot.get("latest"))
    delta = _safe_float(snapshot.get("delta"))
    direction = "stable"
    if delta is not None:
        if delta > 0:
            direction = "rising"
        elif delta < 0:
            direction = "falling"
    interpretation = "neutral"
    if higher_is_tighter and latest is not None:
        interpretation = "tightening" if latest > 0 else "neutral"
    elif higher_is_supportive and latest is not None:
        interpretation = "supportive" if latest > 0 else "softening"
    elif label == "labor" and latest is not None:
        interpretation = "softer" if latest > 4.5 else "firm"
    confidence = _bounded_score(_bounded_ratio(2 if latest is not None else 0, 2), low=0.0, high=1.0)
    return {
        "latest": latest,
        "delta": delta,
        "latest_date": snapshot.get("latest_date"),
        "direction": direction,
        "interpretation": interpretation,
        "confidence": confidence,
    }


def _liquidity_context(
    policy_rate_context: Dict[str, Any],
    credit_context: Dict[str, Any],
    cross_asset_overlay: Dict[str, Any],
) -> Dict[str, Any]:
    rates_proxy = ((cross_asset_overlay.get("rates_fx_commodities_context") or {}).get("rates_proxy")) or {}
    conditions = _mean(
        [
            _bounded_ratio(policy_rate_context.get("latest"), 6.0),
            _bounded_ratio(credit_context.get("latest"), 6.0),
            _bounded_ratio(abs(rates_proxy.get("ret_21d") or 0.0), 0.15),
        ]
    )
    regime = "neutral"
    if conditions is not None and conditions >= 0.65:
        regime = "restrictive"
    elif conditions is not None and conditions <= 0.35:
        regime = "supportive"
    return {
        "conditions_score": _bounded_score(conditions, low=0.0, high=1.0),
        "regime": regime,
        "rates_proxy": rates_proxy,
    }


def _macro_alignment_notes(
    *,
    sector: str,
    regime: Dict[str, Any],
    cross_asset_overlay: Dict[str, Any],
) -> List[str]:
    notes = []
    sector_note = _SECTOR_MACRO_NOTES.get(sector)
    if sector_note:
        notes.append(sector_note)
    benchmark = cross_asset_overlay.get("benchmark_proxy")
    if benchmark:
        notes.append(f"Cross-asset macro framing is anchored to {benchmark}.")
    regime_label = regime.get("regime")
    if regime_label == "tightening_inflationary":
        notes.append("The macro backdrop is leaning tighter, which can pressure valuation-sensitive setups.")
    elif regime_label == "growth_supportive":
        notes.append("The macro backdrop remains broadly growth-supportive rather than overtly defensive.")
    elif regime_label == "growth_softening":
        notes.append("Growth-sensitive setups should be read against a softening macro backdrop.")
    return _dedupe_text(notes)


def _macro_regime_summary(
    regime: Dict[str, Any],
    *,
    rates_context: Dict[str, Any],
    inflation_context: Dict[str, Any],
    growth_context: Dict[str, Any],
    labor_context: Dict[str, Any],
    liquidity_context: Dict[str, Any],
) -> str:
    parts = [
        f"rates are {rates_context.get('direction')}",
        f"inflation is {inflation_context.get('direction')}",
        f"growth is {growth_context.get('interpretation') or growth_context.get('direction')}",
        f"labor is {labor_context.get('interpretation') or labor_context.get('direction')}",
        f"liquidity conditions are {liquidity_context.get('regime')}",
    ]
    return f"{(regime.get('regime') or 'neutral').replace('_', ' ')} regime: " + ", ".join(parts) + "."


def _latest_snapshot_date(collections: Sequence[Dict[str, Any]]) -> Optional[str]:
    values: List[str] = []
    for collection in collections:
        for value in collection.values():
            if isinstance(value, dict):
                candidate = value.get("latest_date") or value.get("data_as_of")
                if candidate:
                    values.append(str(candidate))
    return max(values) if values else None


def _provider_confidence(
    provider_status: Dict[str, Dict[str, Any]],
    *,
    sources_used: Sequence[str],
    coverage_score: Optional[float],
) -> Optional[float]:
    success_count = sum(1 for status in provider_status.values() if status.get("status") == "ok")
    provider_ratio = _bounded_ratio(success_count, max(len(provider_status), 1))
    source_ratio = _bounded_ratio(len(sources_used), max(len(provider_status), 1))
    raw = _mean([provider_ratio, source_ratio, coverage_score])
    return _bounded_score(raw, low=0.0, high=1.0)


def _relative_gap(base_value: Optional[float], comparator: Optional[float]) -> Optional[float]:
    if base_value is None or comparator is None:
        return None
    return base_value - comparator


def _relative_note(base_value: Optional[float], comparator: Optional[float], label: Optional[str]) -> Optional[str]:
    if base_value is None or comparator is None or not label:
        return None
    gap = base_value - comparator
    if gap >= 0.03:
        return f"The stock is outperforming {label} on a 21-day basis."
    if gap <= -0.03:
        return f"The stock is lagging {label} on a 21-day basis."
    return f"The stock is moving broadly in line with {label}."


def _risk_tone_label(value: Optional[float]) -> str:
    if value is None:
        return "mixed"
    if value >= 0.04:
        return "risk_on"
    if value <= -0.04:
        return "risk_off"
    return "mixed"


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _secondary_strengths(metrics: Dict[str, Any]) -> List[str]:
    strengths: List[str] = []
    revenue_growth = metrics.get("revenue_growth_yoy")
    if revenue_growth is not None and revenue_growth >= 0.12:
        strengths.append(
            f"Secondary enrichment confirms revenue growth around {revenue_growth * 100:.1f}%."
        )
    roe = metrics.get("return_on_equity")
    if roe is not None and roe >= 0.12:
        strengths.append(
            f"Return on equity is supportive at {roe * 100:.1f}%."
        )
    cash_flow_margin = metrics.get("free_cash_flow_margin")
    if cash_flow_margin is not None and cash_flow_margin >= 0.1:
        strengths.append(
            f"Free-cash-flow margin is positive at {cash_flow_margin * 100:.1f}%."
        )
    return strengths


def _secondary_weaknesses(metrics: Dict[str, Any]) -> List[str]:
    weaknesses: List[str] = []
    debt_to_equity = metrics.get("debt_to_equity")
    if debt_to_equity is not None and debt_to_equity >= 1.5:
        weaknesses.append(
            f"Secondary enrichment indicates leverage is elevated at {debt_to_equity:.2f} debt-to-equity."
        )
    current_ratio = metrics.get("current_ratio")
    if current_ratio is not None and current_ratio < 1.0:
        weaknesses.append(
            f"Current ratio is below 1.0 at {current_ratio:.2f}."
        )
    revenue_growth = metrics.get("revenue_growth_yoy")
    if revenue_growth is not None and revenue_growth < 0:
        weaknesses.append(
            f"Revenue growth is negative at {revenue_growth * 100:.1f}% year over year."
        )
    return weaknesses


def _dedupe_text(items: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


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
