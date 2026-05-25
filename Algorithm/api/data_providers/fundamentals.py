from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Tuple

import importlib.util

from api.source_governance import active_source_profile, source_allowed

from .alphavantage import fetch_quarterly_fundamentals as fetch_alphavantage_quarterly
from .errors import ProviderUnavailable, SymbolNoData
from .quality import (
    ordered_provider_candidates,
    provider_attempt,
    provider_result_metadata,
)
from .symbols import canonical_symbol

_yf_spec = importlib.util.find_spec("yfinance")
if _yf_spec:
    import yfinance as yf
else:  # pragma: no cover - optional dependency
    yf = None


def fetch_fundamentals_quarterly(symbol: str) -> List[Dict[str, object]]:
    rows, _metadata = fetch_fundamentals_quarterly_with_meta(symbol)
    return rows


def fetch_fundamentals_quarterly_with_meta(
    symbol: str,
) -> Tuple[List[Dict[str, object]], Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    provider_plan = ordered_provider_candidates(
        [
            ("alphavantage", lambda: fetch_alphavantage_quarterly(symbol)),
            ("yfinance", lambda: _fetch_yfinance_quarterly(symbol)),
        ],
        capability="quarterly_fundamentals",
    )
    blocked = 0
    for index, (provider_name, fetcher) in enumerate(provider_plan):
        fallback_used = index > 0
        if not source_allowed(provider_name):
            blocked += 1
            attempts.append(
                provider_attempt(
                    provider_name,
                    status="blocked",
                    source_type="fundamentals",
                    reason_code="BLOCKED_BY_SOURCE_PROFILE",
                    reason_detail="blocked by source profile",
                    response_quality="degraded",
                    fallback_used=fallback_used,
                )
            )
            continue
        try:
            rows = fetcher()
            attempts.append(
                provider_attempt(
                    provider_name,
                    status="success",
                    source_type="fundamentals",
                    response_quality="complete" if rows else "partial",
                    fallback_used=fallback_used,
                )
            )
            latest_date = max(
                (
                    row.get("report_date") or row.get("fiscal_period_end")
                    for row in rows
                    if row.get("report_date") or row.get("fiscal_period_end")
                ),
                default=None,
            )
            return rows, provider_result_metadata(
                provider_name,
                source_type="fundamentals",
                end_date=latest_date,
                fallback_used=fallback_used,
                response_quality="complete" if rows else "partial",
                attempts=attempts,
                capability="quarterly_fundamentals",
            )
        except ProviderUnavailable as exc:
            attempts.append(
                provider_attempt(
                    provider_name,
                    status="failed",
                    source_type="fundamentals",
                    reason_code=exc.reason_code,
                    reason_detail=exc.reason_detail,
                    response_quality="degraded",
                    fallback_used=fallback_used,
                )
            )
        except SymbolNoData:
            raise

    if blocked == len(provider_plan):
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"no fundamentals providers are allowed under source profile {active_source_profile()}",
        )
    raise ProviderUnavailable(
        "PROVIDER_UNAVAILABLE",
        "quarterly fundamentals unavailable",
        provider_name="fundamentals_multi_source",
        source_type="fundamentals",
        metadata={
            "provider_context": provider_result_metadata(
                "fundamentals_multi_source",
                source_type="fundamentals",
                response_quality="degraded",
                attempts=attempts,
                capability="quarterly_fundamentals",
                partial_result=False,
            )
        },
    )


def _fetch_yfinance_quarterly(symbol: str) -> List[Dict[str, object]]:
    if yf is None:
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "yfinance not installed")
    symbol = canonical_symbol(symbol)
    ticker = yf.Ticker(symbol)
    financials = ticker.quarterly_financials
    if financials is None or financials.empty:
        raise SymbolNoData(
            "NO_DATA",
            "no quarterly financials",
            provider_name="yfinance",
            source_type="fundamentals",
        )

    results: List[Dict[str, object]] = []
    for col in financials.columns:
        try:
            fiscal_end = (
                col.date()
                if hasattr(col, "date")
                else dt.date.fromisoformat(str(col)[:10])
            )
        except Exception:
            continue
        revenue = financials.get("Total Revenue", {}).get(col)
        gross_profit = financials.get("Gross Profit", {}).get(col)
        op_income = financials.get("Operating Income", {}).get(col)
        results.append(
            {
                "symbol": symbol,
                "fiscal_period_end": fiscal_end,
                "report_date": fiscal_end,
                "revenue": float(revenue) if revenue is not None else None,
                "eps": None,
                "gross_margin": (
                    float(gross_profit) / float(revenue)
                    if revenue and gross_profit
                    else None
                ),
                "op_margin": (
                    float(op_income) / float(revenue) if revenue and op_income else None
                ),
                "fcf": None,
                "source": "yfinance",
            }
        )

    if not results:
        raise SymbolNoData(
            "NO_DATA",
            "no quarterly rows parsed",
            provider_name="yfinance",
            source_type="fundamentals",
        )
    return results
