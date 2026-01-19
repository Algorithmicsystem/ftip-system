from __future__ import annotations

import datetime as dt
from typing import Dict, List

import importlib.util

from .bars import ProviderUnavailable, SymbolNoData
from .symbols import canonical_symbol

_yf_spec = importlib.util.find_spec("yfinance")
if _yf_spec:
    import yfinance as yf
else:  # pragma: no cover - optional dependency
    yf = None


def fetch_fundamentals_quarterly(symbol: str) -> List[Dict[str, object]]:
    if yf is None:
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "yfinance not installed")
    symbol = canonical_symbol(symbol)
    ticker = yf.Ticker(symbol)
    financials = ticker.quarterly_financials
    if financials is None or financials.empty:
        raise SymbolNoData("NO_DATA", "no quarterly financials")

    results: List[Dict[str, object]] = []
    for col in financials.columns:
        try:
            fiscal_end = col.date() if hasattr(col, "date") else dt.date.fromisoformat(str(col)[:10])
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
                "gross_margin": float(gross_profit) / float(revenue)
                if revenue and gross_profit
                else None,
                "op_margin": float(op_income) / float(revenue)
                if revenue and op_income
                else None,
                "fcf": None,
                "source": "yfinance",
            }
        )

    if not results:
        raise SymbolNoData("NO_DATA", "no quarterly rows parsed")
    return results
