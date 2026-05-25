from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import requests

from api import config

from .errors import ProviderUnavailable, SymbolNoData
from .symbols import canonical_symbol

BASE_URL = "https://www.alphavantage.co/query"


def _request(params: Dict[str, Any]) -> Dict[str, Any]:
    api_key = config.alphavantage_api_key()
    if not api_key:
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "ALPHAVANTAGE_API_KEY not set")
    response = requests.get(
        BASE_URL,
        params={**params, "apikey": api_key},
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"Alpha Vantage HTTP {response.status_code}",
        )
    payload = response.json()
    if payload.get("Note") or payload.get("Information"):
        message = payload.get("Note") or payload.get("Information") or "Alpha Vantage unavailable"
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", str(message))
    if payload.get("Error Message"):
        raise SymbolNoData("NO_DATA", str(payload["Error Message"]))
    return payload


def fetch_daily_adjusted_bars(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> List[Dict[str, object]]:
    payload = _request(
        {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": canonical_symbol(symbol),
            "outputsize": "full",
        }
    )
    series = payload.get("Time Series (Daily)") or {}
    if not isinstance(series, dict) or not series:
        raise SymbolNoData("NO_DATA", "Alpha Vantage returned no daily bars")

    rows: List[Dict[str, object]] = []
    for as_of_text, raw in series.items():
        try:
            as_of_date = dt.date.fromisoformat(as_of_text)
        except Exception:
            continue
        if as_of_date < start_date or as_of_date > end_date:
            continue
        rows.append(
            {
                "symbol": canonical_symbol(symbol),
                "as_of_date": as_of_date,
                "open": _float_or_none(raw.get("1. open")),
                "high": _float_or_none(raw.get("2. high")),
                "low": _float_or_none(raw.get("3. low")),
                "close": _float_or_none(raw.get("5. adjusted close") or raw.get("4. close")),
                "volume": _int_or_none(raw.get("6. volume")),
                "source": "alphavantage",
            }
        )
    rows.sort(key=lambda item: item["as_of_date"])
    if not rows:
        raise SymbolNoData("NO_DATA", "Alpha Vantage returned no bars in range")
    return rows


def fetch_company_overview(symbol: str) -> Dict[str, Any]:
    payload = _request({"function": "OVERVIEW", "symbol": canonical_symbol(symbol)})
    if not isinstance(payload, dict) or not payload:
        raise SymbolNoData("NO_DATA", "Alpha Vantage overview is empty")
    if not payload.get("Symbol"):
        raise SymbolNoData("NO_DATA", "Alpha Vantage overview missing symbol")
    return {
        "symbol": canonical_symbol(symbol),
        "name": payload.get("Name"),
        "sector": payload.get("Sector"),
        "industry": payload.get("Industry"),
        "market_cap": _float_or_none(payload.get("MarketCapitalization")),
        "pe_ratio": _float_or_none(payload.get("PERatio")),
        "peg_ratio": _float_or_none(payload.get("PEGRatio")),
        "beta": _float_or_none(payload.get("Beta")),
        "profit_margin": _float_or_none(payload.get("ProfitMargin")),
        "operating_margin_ttm": _float_or_none(payload.get("OperatingMarginTTM")),
        "return_on_assets_ttm": _float_or_none(payload.get("ReturnOnAssetsTTM")),
        "return_on_equity_ttm": _float_or_none(payload.get("ReturnOnEquityTTM")),
        "revenue_ttm": _float_or_none(payload.get("RevenueTTM")),
        "gross_profit_ttm": _float_or_none(payload.get("GrossProfitTTM")),
        "diluted_eps_ttm": _float_or_none(payload.get("DilutedEPSTTM")),
        "analyst_target_price": _float_or_none(payload.get("AnalystTargetPrice")),
        "dividend_yield": _float_or_none(payload.get("DividendYield")),
        "quarterly_earnings_growth_yoy": _float_or_none(
            payload.get("QuarterlyEarningsGrowthYOY")
        ),
        "quarterly_revenue_growth_yoy": _float_or_none(
            payload.get("QuarterlyRevenueGrowthYOY")
        ),
        "latest_quarter": payload.get("LatestQuarter"),
        "fiscal_year_end": payload.get("FiscalYearEnd"),
        "source": "alphavantage",
    }


def fetch_earnings_intelligence(symbol: str) -> Dict[str, Any]:
    payload = _request({"function": "EARNINGS", "symbol": canonical_symbol(symbol)})
    quarterly = payload.get("quarterlyEarnings") or []
    if not isinstance(quarterly, list) or not quarterly:
        raise SymbolNoData("NO_DATA", "Alpha Vantage returned no earnings history")

    quarters: List[Dict[str, Any]] = []
    for item in quarterly[:12]:
        fiscal_date_text = str(item.get("fiscalDateEnding") or "").strip()
        reported_date_text = str(item.get("reportedDate") or "").strip()
        try:
            fiscal_period_end = dt.date.fromisoformat(fiscal_date_text)
        except Exception:
            continue
        reported_date = None
        if reported_date_text:
            try:
                reported_date = dt.date.fromisoformat(reported_date_text)
            except Exception:
                reported_date = None
        surprise_pct = _surprise_pct_or_none(item.get("surprisePercentage"))
        surprise = _float_or_none(item.get("surprise"))
        reported_eps = _float_or_none(item.get("reportedEPS"))
        estimated_eps = _float_or_none(item.get("estimatedEPS"))
        quarters.append(
            {
                "symbol": canonical_symbol(symbol),
                "fiscal_period_end": fiscal_period_end,
                "reported_date": reported_date,
                "reported_eps": reported_eps,
                "estimated_eps": estimated_eps,
                "surprise": surprise,
                "surprise_pct": surprise_pct,
                "surprise_direction": (
                    "beat"
                    if surprise is not None and surprise > 0
                    else "miss"
                    if surprise is not None and surprise < 0
                    else "inline"
                ),
                "source": "alphavantage",
            }
        )
    if not quarters:
        raise SymbolNoData("NO_DATA", "Alpha Vantage earnings history could not be parsed")

    latest = quarters[0]
    recent = quarters[:4]
    beat_count = sum(1 for item in recent if item.get("surprise_direction") == "beat")
    miss_count = sum(1 for item in recent if item.get("surprise_direction") == "miss")
    average_surprise_pct = _mean(
        [item.get("surprise_pct") for item in recent if item.get("surprise_pct") is not None]
    )
    estimate_revision_support = _estimate_revision_support(
        average_surprise_pct=average_surprise_pct,
        beat_count=beat_count,
        miss_count=miss_count,
        sample_size=len(recent),
    )
    freshness_status = _earnings_freshness_status(latest.get("reported_date"))
    return {
        "symbol": canonical_symbol(symbol),
        "recent_quarters": quarters[:8],
        "latest_quarter": latest,
        "beat_rate_4q": beat_count / max(len(recent), 1),
        "miss_rate_4q": miss_count / max(len(recent), 1),
        "average_surprise_pct_4q": average_surprise_pct,
        "estimate_revision_support": estimate_revision_support,
        "freshness_status": freshness_status,
        "source": "alphavantage",
        "meta": {
            "sources": ["alphavantage"],
            "primary_source": "alphavantage",
            "coverage_score": 1.0 if quarters else 0.0,
            "confidence": 78.0 if quarters else 0.0,
            "latest_reported_date": (
                latest.get("reported_date").isoformat()
                if isinstance(latest.get("reported_date"), dt.date)
                else None
            ),
            "status": freshness_status,
        },
    }


def fetch_quarterly_fundamentals(symbol: str) -> List[Dict[str, object]]:
    income = _request({"function": "INCOME_STATEMENT", "symbol": canonical_symbol(symbol)})
    cashflow = _request({"function": "CASH_FLOW", "symbol": canonical_symbol(symbol)})
    earnings = _request({"function": "EARNINGS", "symbol": canonical_symbol(symbol)})

    income_rows = {
        str(item.get("fiscalDateEnding")): item
        for item in income.get("quarterlyReports") or []
        if item.get("fiscalDateEnding")
    }
    cash_rows = {
        str(item.get("fiscalDateEnding")): item
        for item in cashflow.get("quarterlyReports") or []
        if item.get("fiscalDateEnding")
    }
    earnings_rows = {
        str(item.get("fiscalDateEnding")): item
        for item in earnings.get("quarterlyEarnings") or []
        if item.get("fiscalDateEnding")
    }
    fiscal_dates = sorted(set(income_rows) | set(cash_rows) | set(earnings_rows), reverse=True)
    if not fiscal_dates:
        raise SymbolNoData("NO_DATA", "Alpha Vantage returned no quarterly fundamentals")

    rows: List[Dict[str, object]] = []
    for fiscal_date in fiscal_dates[:8]:
        income_row = income_rows.get(fiscal_date) or {}
        cash_row = cash_rows.get(fiscal_date) or {}
        earnings_row = earnings_rows.get(fiscal_date) or {}
        revenue = _float_or_none(income_row.get("totalRevenue"))
        gross_profit = _float_or_none(income_row.get("grossProfit"))
        operating_income = _float_or_none(income_row.get("operatingIncome"))
        operating_cashflow = _float_or_none(cash_row.get("operatingCashflow"))
        capital_expenditures = abs(_float_or_none(cash_row.get("capitalExpenditures")) or 0.0)
        free_cash_flow = _float_or_none(cash_row.get("freeCashFlow"))
        if free_cash_flow is None and operating_cashflow is not None:
            free_cash_flow = operating_cashflow - capital_expenditures
        try:
            period_end = dt.date.fromisoformat(fiscal_date)
        except Exception:
            continue
        rows.append(
            {
                "symbol": canonical_symbol(symbol),
                "fiscal_period_end": period_end,
                "report_date": period_end,
                "revenue": revenue,
                "eps": _float_or_none(
                    earnings_row.get("reportedEPS") or earnings_row.get("reportedEPS")
                ),
                "gross_margin": (
                    float(gross_profit) / float(revenue)
                    if revenue not in (None, 0.0) and gross_profit is not None
                    else None
                ),
                "op_margin": (
                    float(operating_income) / float(revenue)
                    if revenue not in (None, 0.0) and operating_income is not None
                    else None
                ),
                "fcf": free_cash_flow,
                "source": "alphavantage",
            }
        )
    if not rows:
        raise SymbolNoData("NO_DATA", "Alpha Vantage quarterly fundamentals could not be parsed")
    return rows


def _mean(values: List[Optional[float]]) -> Optional[float]:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return float(sum(defined) / len(defined))


def _surprise_pct_or_none(value: Any) -> Optional[float]:
    raw = _float_or_none(value)
    if raw is None:
        return None
    if abs(raw) <= 1.0:
        return raw * 100.0
    return raw


def _estimate_revision_support(
    *,
    average_surprise_pct: Optional[float],
    beat_count: int,
    miss_count: int,
    sample_size: int,
) -> Optional[float]:
    if sample_size <= 0:
        return None
    surprise_component = 0.0 if average_surprise_pct is None else max(
        min(average_surprise_pct, 20.0),
        -20.0,
    )
    beat_component = ((beat_count - miss_count) / sample_size) * 18.0
    return round(max(min(50.0 + surprise_component * 1.2 + beat_component, 100.0), 0.0), 2)


def _earnings_freshness_status(reported_date: Any) -> str:
    if not isinstance(reported_date, dt.date):
        return "unknown"
    age_days = max((dt.datetime.now(dt.timezone.utc).date() - reported_date).days, 0)
    if age_days <= 45:
        return "fresh"
    if age_days <= 120:
        return "recent"
    return "historical"


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, "", "None", "null", "NoneType"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> Optional[int]:
    number = _float_or_none(value)
    return int(number) if number is not None else None
