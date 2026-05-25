from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Dict, List, Optional

from api import config

from .alphavantage import fetch_earnings_intelligence
from .bars import fetch_daily_bars_with_meta
from .errors import ProviderError, ProviderUnavailable, SymbolNoData
from .finnhub import fetch_company_news
from .gnews import search_news as search_gnews
from .newsapi import search_news as search_newsapi
from .quality import provider_confidence_grade
from .sec_edgar import fetch_company_filing_profile


_CONNECTOR_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "premium_market_data": {
        "title": "Premium Market Data",
        "domain": "market_data",
        "capabilities": ["daily_bars", "reference_bars", "snapshot_bars"],
        "required_envs": ["MASSIVE_API_KEY|POLYGON_API_KEY"],
    },
    "premium_news_intel": {
        "title": "Premium News Intelligence",
        "domain": "news_intelligence",
        "capabilities": ["headline_search", "company_news", "event_headlines"],
        "required_envs": ["FINNHUB_API_KEY|GNEWS_API_KEY|NEWS_API_KEY"],
    },
    "filings_intel": {
        "title": "Filings Intelligence",
        "domain": "filings",
        "capabilities": ["filing_backbone", "fundamental_backbone", "filing_events"],
        "required_envs": ["SEC_USER_AGENT"],
    },
    "estimates_or_earnings_intel": {
        "title": "Estimates / Earnings Intelligence",
        "domain": "earnings_estimates",
        "capabilities": ["earnings_history", "surprise_history", "estimate_support"],
        "required_envs": ["ALPHAVANTAGE_API_KEY"],
    },
}


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _configured_connector_vendors(connector_type: str) -> List[str]:
    if connector_type == "premium_market_data":
        return ["massive_polygon"] if config.massive_api_key() else []
    if connector_type == "premium_news_intel":
        configured: List[str] = []
        if config.finnhub_api_key():
            configured.append("finnhub")
        if config.gnews_api_key():
            configured.append("gnews")
        if config.news_api_key():
            configured.append("newsapi")
        return configured
    if connector_type == "filings_intel":
        return ["sec_edgar"] if config.sec_user_agent() else []
    if connector_type == "estimates_or_earnings_intel":
        return ["alphavantage"] if config.alphavantage_api_key() else []
    return []


def _credential_present(connector_type: str) -> bool:
    if connector_type == "filings_intel":
        return bool(config.sec_user_agent())
    return bool(_configured_connector_vendors(connector_type))


def _readiness_status(connector_type: str) -> str:
    if connector_type == "filings_intel":
        return "public_ready" if config.sec_user_agent() else "misconfigured"
    return "configured" if _credential_present(connector_type) else "missing_credentials"


def _quality_summary(
    *,
    connector_type: str,
    readiness_status: str,
    live_probe_status: str,
    configured_vendors: List[str],
) -> str:
    title = _CONNECTOR_CAPABILITIES[connector_type]["title"]
    vendor_text = ", ".join(configured_vendors) if configured_vendors else "none configured"
    return (
        f"{title} is {readiness_status.replace('_', ' ')} with vendors {vendor_text} "
        f"and live probe state {live_probe_status.replace('_', ' ')}."
    )


def _probe_market_data(sample_symbol: str) -> Dict[str, Any]:
    end_date = dt.datetime.now(dt.timezone.utc).date()
    start_date = end_date - dt.timedelta(days=15)
    rows, metadata = fetch_daily_bars_with_meta(
        sample_symbol,
        start_date,
        end_date,
        preferred_provider="massive_polygon",
        disabled_providers=["alphavantage", "stooq", "yfinance"],
    )
    return {
        "provider_name": metadata.get("provider_name"),
        "row_count": len(rows),
        "latest_as_of_date": rows[-1]["as_of_date"].isoformat() if rows else None,
        "freshness_status": metadata.get("freshness_status"),
        "confidence_grade": metadata.get("confidence_grade"),
        "response_quality": metadata.get("response_quality"),
        "best_source_used": metadata.get("best_source_used"),
        "fallback_chain_used": metadata.get("fallback_chain_used") or [],
        "source_warning_flags": metadata.get("source_warning_flags") or [],
        "source_strength_summary": metadata.get("source_strength_summary"),
    }


def _probe_news_intel(sample_symbol: str) -> Dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    from_ts = now - dt.timedelta(days=7)
    if config.finnhub_api_key():
        items = fetch_company_news(sample_symbol, from_ts.date(), now.date())
        provider_name = "finnhub"
    elif config.gnews_api_key():
        items = search_gnews(
            f'"{sample_symbol}" stock',
            from_ts=from_ts,
            to_ts=now,
            max_items=8,
        )
        provider_name = "gnews"
    elif config.news_api_key():
        items = search_newsapi(
            f'"{sample_symbol}" stock',
            from_ts=from_ts,
            to_ts=now,
            max_items=8,
        )
        provider_name = "newsapi"
    else:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            "no premium news credentials configured",
            provider_name="premium_news_intel",
            source_type="news",
        )
    latest = items[0] if items else {}
    published_at = latest.get("published_at")
    return {
        "provider_name": provider_name,
        "item_count": len(items),
        "latest_published_at": published_at.isoformat() if hasattr(published_at, "isoformat") else None,
        "freshness_status": "fresh" if items else "historical",
        "confidence_grade": provider_confidence_grade(0.72 if items else 0.48),
        "response_quality": "complete" if items else "partial",
        "best_source_used": provider_name,
        "fallback_chain_used": [provider_name],
        "source_warning_flags": [] if items else ["partial_result"],
        "source_strength_summary": (
            f"{provider_name} returned {len(items)} premium-news item(s) for {sample_symbol}."
        ),
    }


def _probe_filings_intel(sample_symbol: str) -> Dict[str, Any]:
    payload = fetch_company_filing_profile(sample_symbol)
    filing_backbone = payload.get("filing_backbone") or {}
    recent_filings = filing_backbone.get("recent_filings") or []
    return {
        "provider_name": "sec_edgar",
        "latest_filing_date": filing_backbone.get("latest_filing_date"),
        "latest_form": filing_backbone.get("latest_form"),
        "recent_filing_count": len(recent_filings),
        "freshness_status": filing_backbone.get("status") or "unknown",
        "confidence_grade": provider_confidence_grade(0.82 if recent_filings else 0.55),
        "response_quality": "complete" if recent_filings else "partial",
        "best_source_used": "sec_edgar",
        "fallback_chain_used": ["sec_edgar"],
        "source_warning_flags": [] if recent_filings else ["partial_result"],
        "source_strength_summary": (
            f"SEC filings intelligence returned {len(recent_filings)} recent filing record(s) for {sample_symbol}."
        ),
    }


def _probe_estimates_intel(sample_symbol: str) -> Dict[str, Any]:
    payload = fetch_earnings_intelligence(sample_symbol)
    quarters = payload.get("recent_quarters") or []
    latest = payload.get("latest_quarter") or {}
    return {
        "provider_name": "alphavantage",
        "quarter_count": len(quarters),
        "latest_reported_date": (
            latest.get("reported_date").isoformat()
            if hasattr(latest.get("reported_date"), "isoformat")
            else latest.get("reported_date")
        ),
        "estimate_revision_support": payload.get("estimate_revision_support"),
        "freshness_status": payload.get("freshness_status") or "unknown",
        "confidence_grade": provider_confidence_grade(0.74 if quarters else 0.5),
        "response_quality": "complete" if quarters else "partial",
        "best_source_used": "alphavantage",
        "fallback_chain_used": ["alphavantage"],
        "source_warning_flags": [] if quarters else ["partial_result"],
        "source_strength_summary": (
            f"Estimates intelligence parsed {len(quarters)} earnings quarter(s) for {sample_symbol}."
        ),
    }


def _classify_failure(exc: Exception) -> str:
    if isinstance(exc, SymbolNoData):
        return "no_data"
    if isinstance(exc, ProviderUnavailable):
        return "provider_unavailable"
    if isinstance(exc, ProviderError):
        return "provider_error"
    return "unexpected_error"


def probe_premium_connector(
    connector_type: str,
    *,
    sample_symbol: str = "NVDA",
    execute_live: bool = False,
) -> Dict[str, Any]:
    connector_type = str(connector_type or "").strip()
    definition = _CONNECTOR_CAPABILITIES.get(connector_type)
    if definition is None:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"unknown premium connector {connector_type}",
            provider_name=connector_type,
        )
    configured_vendors = _configured_connector_vendors(connector_type)
    readiness_status = _readiness_status(connector_type)
    live_probe_status = "not_attempted"
    last_execution_result: Dict[str, Any] = {}
    error_summary = None
    failure_reason_classification = None
    if execute_live and readiness_status not in {"missing_credentials", "misconfigured"}:
        live_probe_status = "running"
        probe_map: Dict[str, Callable[[str], Dict[str, Any]]] = {
            "premium_market_data": _probe_market_data,
            "premium_news_intel": _probe_news_intel,
            "filings_intel": _probe_filings_intel,
            "estimates_or_earnings_intel": _probe_estimates_intel,
        }
        try:
            last_execution_result = probe_map[connector_type](sample_symbol)
            live_probe_status = "probe_succeeded"
        except Exception as exc:  # pragma: no cover - defensive for live paths
            live_probe_status = "probe_failed"
            error_summary = str(exc)
            failure_reason_classification = _classify_failure(exc)
    elif execute_live:
        live_probe_status = "blocked_missing_credentials"
        failure_reason_classification = "missing_credentials"
        error_summary = "connector is not configured for live execution"

    return {
        "connector_type": connector_type,
        "title": definition["title"],
        "domain": definition["domain"],
        "capabilities": list(definition["capabilities"]),
        "required_envs": list(definition["required_envs"]),
        "configured_vendors": configured_vendors,
        "credential_present": _credential_present(connector_type),
        "config_present": bool(configured_vendors) or connector_type == "filings_intel",
        "readiness_status": readiness_status,
        "execute_live": bool(execute_live),
        "sample_symbol": sample_symbol,
        "live_probe_status": live_probe_status,
        "last_execution_result": last_execution_result,
        "error_summary": error_summary,
        "failure_reason_classification": failure_reason_classification,
        "data_quality_summary": _quality_summary(
            connector_type=connector_type,
            readiness_status=readiness_status,
            live_probe_status=live_probe_status,
            configured_vendors=configured_vendors,
        ),
        "checked_at": _now_utc(),
    }


def list_premium_connector_summaries(
    *,
    sample_symbol: str = "NVDA",
    execute_live: bool = False,
) -> List[Dict[str, Any]]:
    return [
        probe_premium_connector(
            connector_type,
            sample_symbol=sample_symbol,
            execute_live=execute_live,
        )
        for connector_type in _CONNECTOR_CAPABILITIES
    ]


def build_premium_connector_overview(
    *,
    sample_symbol: str = "NVDA",
    execute_live: bool = False,
) -> Dict[str, Any]:
    connectors = list_premium_connector_summaries(
        sample_symbol=sample_symbol,
        execute_live=execute_live,
    )
    configured_count = sum(
        1 for item in connectors if item.get("readiness_status") in {"configured", "public_ready"}
    )
    probe_ready_count = sum(
        1 for item in connectors if item.get("live_probe_status") == "probe_succeeded"
    )
    warnings = [
        item["connector_type"]
        for item in connectors
        if item.get("readiness_status") == "missing_credentials"
        or item.get("live_probe_status") in {"probe_failed", "blocked_missing_credentials"}
    ]
    status = (
        "ready"
        if configured_count == len(connectors) and not warnings
        else "partial"
        if configured_count > 0
        else "limited"
    )
    return {
        "status": status,
        "connector_count": len(connectors),
        "configured_count": configured_count,
        "live_probe_success_count": probe_ready_count,
        "warnings": warnings,
        "connectors": connectors,
        "summary": (
            f"Premium connector readiness is {status} with {configured_count}/{len(connectors)} "
            f"connector slot(s) configured and {probe_ready_count} live probe success(es)."
        ),
    }
