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
