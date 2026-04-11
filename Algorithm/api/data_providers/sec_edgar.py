from __future__ import annotations

import datetime as dt
import math
import re
import statistics
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from api import config

from .errors import ProviderUnavailable, SymbolNoData
from .symbols import canonical_symbol

BASE_URL = "https://data.sec.gov"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_PERIODIC_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "40-F"}
_ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "40-F"}
_QUARTERLY_FORMS = {"10-Q", "10-Q/A"}
_NAME_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_METRIC_TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cash_from_operations": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capital_expenditures": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
    ],
    "assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "total_debt": [
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "DebtInstrumentCarryingAmount",
    ],
}
_MONEY_UNITS = ("USD",)


def fetch_company_filing_profile(
    symbol: str,
    *,
    company_name: Optional[str] = None,
    as_of_date: Optional[dt.date] = None,
) -> Dict[str, Any]:
    mapping = resolve_company_mapping(symbol, company_name=company_name)
    cik = mapping["cik"]
    submissions = fetch_submissions(cik)
    companyfacts = fetch_companyfacts(cik)
    filing_history = _build_filing_history(submissions, as_of_date=as_of_date)
    fact_bundle = _build_companyfacts_bundle(companyfacts, filing_history)
    coverage_flags, coverage_score, missingness_flags = _coverage_profile(
        fact_bundle, filing_history
    )
    quality_proxies = _quality_proxies(
        fact_bundle=fact_bundle,
        coverage_score=coverage_score,
        filing_recency_days=filing_history.get("filing_recency_days"),
    )
    strengths, weaknesses, coverage_caveats = _strengths_and_weaknesses(
        fact_bundle=fact_bundle,
        coverage_flags=coverage_flags,
        filing_history=filing_history,
        mapping=mapping,
    )
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    latest_report_date = (
        (((fact_bundle.get("latest_quarter") or {}).get("report_date")))
        or (((fact_bundle.get("latest_annual") or {}).get("report_date")))
        or (((filing_history.get("latest_10q") or {}).get("filing_date")))
        or (((filing_history.get("latest_10k") or {}).get("filing_date")))
    )
    latest_quarter = fact_bundle.get("latest_quarter") or {}
    margin_series = [
        point.get("operating_margin")
        for point in fact_bundle.get("quarterly_series") or []
        if point.get("operating_margin") is not None
    ]
    margin_stability = _margin_stability(margin_series)
    positive_fcf_ratio = _positive_ratio(
        point.get("free_cash_flow")
        for point in fact_bundle.get("quarterly_series") or []
    )

    return {
        "symbol": canonical_symbol(symbol),
        "cik": cik,
        "name": submissions.get("name") or mapping.get("matched_company_name"),
        "sic": submissions.get("sic"),
        "sic_description": submissions.get("sicDescription"),
        "fiscal_year_end": submissions.get("fiscalYearEnd"),
        "mapping": mapping,
        "filing_backbone": filing_history,
        "statement_snapshot": {
            "latest_quarter": fact_bundle.get("latest_quarter"),
            "prior_year_quarter": fact_bundle.get("prior_year_quarter"),
            "latest_annual": fact_bundle.get("latest_annual"),
            "latest_balance_sheet": fact_bundle.get("latest_balance_sheet"),
            "quarterly_series": fact_bundle.get("quarterly_series"),
            "annual_series": fact_bundle.get("annual_series"),
        },
        "normalized_metrics": fact_bundle.get("normalized_metrics"),
        "quality_proxies": quality_proxies,
        "coverage_flags": coverage_flags,
        "coverage_score": coverage_score,
        "missingness_flags": missingness_flags,
        "strength_summary": strengths,
        "weakness_summary": weaknesses,
        "coverage_caveats": coverage_caveats,
        "latest_filing_date": filing_history.get("latest_filing_date"),
        "latest_form": filing_history.get("latest_form"),
        "filing_recency_days": filing_history.get("filing_recency_days"),
        "latest_quarter": latest_quarter,
        "revenue_growth_yoy": (fact_bundle.get("normalized_metrics") or {}).get(
            "revenue_growth_yoy"
        ),
        "margin_stability": margin_stability,
        "positive_fcf_ratio": positive_fcf_ratio,
        "source": "sec_edgar",
        "meta": {
            "sources": ["sec_edgar"],
            "primary_source": "sec_edgar",
            "fetched_at": fetched_at,
            "latest_report_date": latest_report_date,
            "status": filing_history.get("status"),
            "coverage_score": coverage_score,
            "missingness": 1.0 - coverage_score,
            "mapping_status": mapping.get("match_type"),
        },
    }


@lru_cache(maxsize=1)
def _ticker_records() -> List[Dict[str, str]]:
    response = requests.get(
        _TICKER_MAP_URL,
        headers=_headers(),
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"SEC ticker map HTTP {response.status_code}",
        )
    payload = response.json()
    records: List[Dict[str, str]] = []
    if isinstance(payload, dict):
        for row in payload.values():
            ticker = str(row.get("ticker") or "").strip().upper()
            title = str(row.get("title") or "").strip()
            cik = str(row.get("cik_str") or "").strip()
            if ticker and cik:
                records.append(
                    {
                        "ticker": ticker,
                        "ticker_normalized": _normalize_sec_ticker(ticker),
                        "title": title,
                        "title_normalized": _normalize_company_name(title),
                        "cik": cik.zfill(10),
                    }
                )
    if not records:
        raise SymbolNoData("NO_DATA", "SEC ticker map was empty")
    return records


def resolve_company_mapping(
    symbol: str,
    *,
    company_name: Optional[str] = None,
) -> Dict[str, Any]:
    requested_symbol = canonical_symbol(symbol)
    base_ticker = (
        requested_symbol.split(".")[0]
        if requested_symbol.endswith((".TO", ".V"))
        else requested_symbol
    )
    ticker_normalized = _normalize_sec_ticker(base_ticker)
    requested_name_normalized = _normalize_company_name(company_name)
    notes: List[str] = []
    records = _ticker_records()

    exact_record = next((row for row in records if row["ticker"] == base_ticker), None)
    if exact_record:
        return _mapping_payload(
            requested_symbol=requested_symbol,
            requested_company_name=company_name,
            match_type="exact_ticker",
            record=exact_record,
            notes=notes,
        )

    normalized_record = next(
        (row for row in records if row["ticker_normalized"] == ticker_normalized), None
    )
    if normalized_record:
        notes.append(
            f"Resolved {base_ticker} through normalized SEC ticker {normalized_record['ticker']}."
        )
        return _mapping_payload(
            requested_symbol=requested_symbol,
            requested_company_name=company_name,
            match_type="normalized_ticker",
            record=normalized_record,
            notes=notes,
        )

    if requested_name_normalized:
        exact_name = next(
            (row for row in records if row["title_normalized"] == requested_name_normalized),
            None,
        )
        if exact_name:
            notes.append(
                f"Ticker {base_ticker} was not present in SEC map; matched company title {exact_name['title']} instead."
            )
            return _mapping_payload(
                requested_symbol=requested_symbol,
                requested_company_name=company_name,
                match_type="company_name_exact",
                record=exact_name,
                notes=notes,
            )

        fuzzy_name = next(
            (
                row
                for row in records
                if requested_name_normalized and requested_name_normalized in row["title_normalized"]
            ),
            None,
        )
        if fuzzy_name:
            notes.append(
                f"Ticker {base_ticker} was not present in SEC map; fuzzy-matched company title {fuzzy_name['title']}."
            )
            return _mapping_payload(
                requested_symbol=requested_symbol,
                requested_company_name=company_name,
                match_type="company_name_fuzzy",
                record=fuzzy_name,
                notes=notes,
            )

    raise SymbolNoData(
        "NO_DATA",
        f"SEC ticker/company mapping failed for {requested_symbol}"
        + (f" ({company_name})" if company_name else ""),
    )


def resolve_cik(symbol: str, company_name: Optional[str] = None) -> str:
    return resolve_company_mapping(symbol, company_name=company_name)["cik"]


def fetch_submissions(cik: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_URL}/submissions/CIK{str(cik).zfill(10)}.json",
        headers=_headers(),
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"SEC submissions HTTP {response.status_code}",
        )
    payload = response.json()
    if not isinstance(payload, dict) or not payload:
        raise SymbolNoData("NO_DATA", "SEC submissions payload was empty")
    return payload


def fetch_companyfacts(cik: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_URL}/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json",
        headers=_headers(),
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            f"SEC companyfacts HTTP {response.status_code}",
        )
    payload = response.json()
    if not isinstance(payload, dict) or not payload:
        raise SymbolNoData("NO_DATA", "SEC companyfacts payload was empty")
    return payload


def _build_filing_history(
    submissions: Dict[str, Any],
    *,
    as_of_date: Optional[dt.date],
) -> Dict[str, Any]:
    filings = ((submissions.get("filings") or {}).get("recent") or {})
    rows: List[Dict[str, Any]] = []
    for accession, filed_at, form, document in zip(
        filings.get("accessionNumber") or [],
        filings.get("filingDate") or [],
        filings.get("form") or [],
        filings.get("primaryDocument") or [],
    ):
        rows.append(
            {
                "accession_number": accession,
                "filing_date": filed_at,
                "form": form,
                "primary_document": document,
            }
        )

    latest_periodic = next(
        (row for row in rows if str(row.get("form") or "") in _PERIODIC_FORMS),
        rows[0] if rows else None,
    )
    latest_10k = next(
        (row for row in rows if str(row.get("form") or "") in _ANNUAL_FORMS),
        None,
    )
    latest_10q = next(
        (row for row in rows if str(row.get("form") or "") in _QUARTERLY_FORMS),
        None,
    )

    latest_filing_date = latest_periodic.get("filing_date") if latest_periodic else None
    filing_recency_days = None
    if latest_filing_date:
        reference_date = as_of_date or dt.datetime.now(dt.timezone.utc).date()
        try:
            filing_recency_days = (
                reference_date - dt.date.fromisoformat(str(latest_filing_date))
            ).days
        except Exception:
            filing_recency_days = None

    status = "limited"
    if filing_recency_days is not None and filing_recency_days <= 120:
        status = "fresh"
    elif filing_recency_days is not None and filing_recency_days <= 210:
        status = "stale_but_usable"

    return {
        "latest_form": latest_periodic.get("form") if latest_periodic else None,
        "latest_filing_date": latest_filing_date,
        "filing_recency_days": filing_recency_days,
        "latest_10k": latest_10k,
        "latest_10q": latest_10q,
        "recent_filings": rows[:12],
        "status": status,
    }


def _build_companyfacts_bundle(
    companyfacts: Dict[str, Any],
    filing_history: Dict[str, Any],
) -> Dict[str, Any]:
    facts = (companyfacts.get("facts") or {}).get("us-gaap") or {}
    quarterly_metrics = {
        metric: _fact_series(facts, tags, kind="duration", quarterly=True)
        for metric, tags in _METRIC_TAGS.items()
        if metric
        in {
            "revenue",
            "gross_profit",
            "operating_income",
            "net_income",
            "cash_from_operations",
            "capital_expenditures",
        }
    }
    annual_metrics = {
        metric: _fact_series(facts, tags, kind="duration", annual=True)
        for metric, tags in _METRIC_TAGS.items()
        if metric
        in {
            "revenue",
            "gross_profit",
            "operating_income",
            "net_income",
            "cash_from_operations",
            "capital_expenditures",
        }
    }
    instant_metrics = {
        metric: _fact_series(facts, tags, kind="instant")
        for metric, tags in _METRIC_TAGS.items()
        if metric
        in {
            "assets",
            "current_assets",
            "liabilities",
            "current_liabilities",
            "equity",
            "cash",
            "total_debt",
        }
    }

    quarterly_series = _compose_quarterly_series(quarterly_metrics)
    annual_series = _compose_annual_series(annual_metrics)
    latest_quarter = quarterly_series[0] if quarterly_series else {}
    prior_year_quarter = _find_year_ago_quarter(quarterly_series, latest_quarter)
    latest_annual = annual_series[0] if annual_series else {}
    latest_balance_sheet = {
        metric: series[0]["value"] if series else None
        for metric, series in instant_metrics.items()
    }
    if any(value is not None for value in latest_balance_sheet.values()):
        latest_balance_sheet["report_date"] = _first_available(
            *(series[0].get("report_date") for series in instant_metrics.values() if series)
        )
        latest_balance_sheet["fiscal_period_end"] = _first_available(
            *(series[0].get("period_end") for series in instant_metrics.values() if series)
        )

    normalized_metrics = _normalized_metrics(
        latest_quarter=latest_quarter,
        prior_year_quarter=prior_year_quarter,
        latest_annual=latest_annual,
        latest_balance_sheet=latest_balance_sheet,
        quarterly_series=quarterly_series,
    )

    return {
        "latest_quarter": latest_quarter,
        "prior_year_quarter": prior_year_quarter,
        "latest_annual": latest_annual,
        "latest_balance_sheet": latest_balance_sheet,
        "quarterly_series": quarterly_series[:6],
        "annual_series": annual_series[:3],
        "normalized_metrics": normalized_metrics,
    }


def _fact_series(
    facts: Dict[str, Any],
    tags: Sequence[str],
    *,
    kind: str,
    quarterly: bool = False,
    annual: bool = False,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for tag in tags:
        concept = facts.get(tag) or {}
        units = concept.get("units") or {}
        for unit in _MONEY_UNITS:
            for point in units.get(unit) or []:
                normalized = _normalize_fact_point(point, tag=tag, kind=kind)
                if normalized is None:
                    continue
                if quarterly and not _is_quarterly_point(normalized):
                    continue
                if annual and not _is_annual_point(normalized):
                    continue
                points.append(normalized)
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for point in sorted(points, key=_fact_sort_key, reverse=True):
        key = (
            str(point.get("period_end") or ""),
            str(point.get("form") or ""),
        )
        deduped.setdefault(key, point)
    return list(deduped.values())


def _normalize_fact_point(
    raw: Dict[str, Any],
    *,
    tag: str,
    kind: str,
) -> Optional[Dict[str, Any]]:
    value = _safe_float(raw.get("val"))
    if value is None:
        return None
    period_end = _parse_date(raw.get("end"))
    report_date = _parse_date(raw.get("filed"))
    if period_end is None:
        return None
    form = str(raw.get("form") or "")
    if form and form not in _PERIODIC_FORMS:
        return None
    days = _duration_days(raw.get("start"), raw.get("end")) if kind == "duration" else None
    return {
        "tag": tag,
        "value": value,
        "period_end": period_end.isoformat(),
        "report_date": report_date.isoformat() if report_date else None,
        "form": form,
        "days": days,
        "fp": raw.get("fp"),
        "fy": raw.get("fy"),
    }


def _compose_quarterly_series(metrics: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    period_rows: Dict[str, Dict[str, Any]] = {}
    for metric_name, series in metrics.items():
        for point in series[:8]:
            row = period_rows.setdefault(
                point["period_end"],
                {
                    "fiscal_period_end": point["period_end"],
                    "report_date": point.get("report_date"),
                    "form": point.get("form"),
                },
            )
            row[metric_name] = point.get("value")
            row["report_date"] = _first_available(row.get("report_date"), point.get("report_date"))
            row["form"] = _first_available(row.get("form"), point.get("form"))
    composed = list(period_rows.values())
    composed.sort(key=lambda item: item.get("fiscal_period_end") or "", reverse=True)
    for row in composed:
        _attach_margin_fields(row)
    return composed


def _compose_annual_series(metrics: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    period_rows: Dict[str, Dict[str, Any]] = {}
    for metric_name, series in metrics.items():
        for point in series[:6]:
            row = period_rows.setdefault(
                point["period_end"],
                {
                    "fiscal_period_end": point["period_end"],
                    "report_date": point.get("report_date"),
                    "form": point.get("form"),
                },
            )
            row[metric_name] = point.get("value")
            row["report_date"] = _first_available(row.get("report_date"), point.get("report_date"))
            row["form"] = _first_available(row.get("form"), point.get("form"))
    composed = list(period_rows.values())
    composed.sort(key=lambda item: item.get("fiscal_period_end") or "", reverse=True)
    for row in composed:
        _attach_margin_fields(row)
    return composed


def _find_year_ago_quarter(
    quarterly_series: Sequence[Dict[str, Any]],
    latest_quarter: Dict[str, Any],
) -> Dict[str, Any]:
    latest_period_end = _parse_date(latest_quarter.get("fiscal_period_end"))
    if latest_period_end is None:
        return quarterly_series[3] if len(quarterly_series) >= 4 else {}
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for row in quarterly_series[1:]:
        period_end = _parse_date(row.get("fiscal_period_end"))
        if period_end is None:
            continue
        delta = abs((latest_period_end - period_end).days)
        if 300 <= delta <= 430:
            candidates.append((abs(delta - 365), row))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    return quarterly_series[3] if len(quarterly_series) >= 4 else {}


def _normalized_metrics(
    *,
    latest_quarter: Dict[str, Any],
    prior_year_quarter: Dict[str, Any],
    latest_annual: Dict[str, Any],
    latest_balance_sheet: Dict[str, Any],
    quarterly_series: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    revenue_growth_yoy = _pct_change(
        latest_quarter.get("revenue"),
        prior_year_quarter.get("revenue"),
    )
    cash_flow_margin = _ratio(
        latest_quarter.get("cash_from_operations"),
        latest_quarter.get("revenue"),
    )
    capex = latest_quarter.get("capital_expenditures")
    cash_from_operations = latest_quarter.get("cash_from_operations")
    free_cash_flow = None
    if cash_from_operations is not None:
        free_cash_flow = cash_from_operations - abs(capex or 0.0)
    free_cash_flow_margin = _ratio(free_cash_flow, latest_quarter.get("revenue"))
    quarterly_fcf_values = []
    for row in quarterly_series:
        cfo = row.get("cash_from_operations")
        row_capex = row.get("capital_expenditures")
        quarterly_fcf_values.append(
            None if cfo is None else cfo - abs(row_capex or 0.0)
        )
    positive_fcf_ratio = _positive_ratio(quarterly_fcf_values)
    operating_margin_series = [
        row.get("operating_margin")
        for row in quarterly_series
        if row.get("operating_margin") is not None
    ]
    gross_margin_series = [
        row.get("gross_margin")
        for row in quarterly_series
        if row.get("gross_margin") is not None
    ]
    net_margin_series = [
        row.get("net_margin")
        for row in quarterly_series
        if row.get("net_margin") is not None
    ]
    current_ratio = _ratio(
        latest_balance_sheet.get("current_assets"),
        latest_balance_sheet.get("current_liabilities"),
    )
    cash_ratio = _ratio(
        latest_balance_sheet.get("cash"),
        latest_balance_sheet.get("current_liabilities"),
    )
    debt_to_equity = _ratio(
        latest_balance_sheet.get("total_debt"),
        latest_balance_sheet.get("equity"),
    )
    liabilities_to_assets = _ratio(
        latest_balance_sheet.get("liabilities"),
        latest_balance_sheet.get("assets"),
    )
    return_on_assets = _ratio(
        latest_annual.get("net_income"),
        latest_balance_sheet.get("assets"),
    )
    return_on_equity = _ratio(
        latest_annual.get("net_income"),
        latest_balance_sheet.get("equity"),
    )
    return {
        "revenue_growth_yoy": revenue_growth_yoy,
        "gross_margin": latest_quarter.get("gross_margin"),
        "operating_margin": latest_quarter.get("operating_margin"),
        "net_margin": latest_quarter.get("net_margin"),
        "cash_flow_margin": cash_flow_margin,
        "free_cash_flow": free_cash_flow,
        "free_cash_flow_margin": free_cash_flow_margin,
        "current_ratio": current_ratio,
        "cash_ratio": cash_ratio,
        "debt_to_equity": debt_to_equity,
        "liabilities_to_assets": liabilities_to_assets,
        "return_on_assets": return_on_assets,
        "return_on_equity": return_on_equity,
        "positive_fcf_ratio": positive_fcf_ratio,
        "gross_margin_stability": _margin_stability(gross_margin_series),
        "operating_margin_stability": _margin_stability(operating_margin_series),
        "net_margin_stability": _margin_stability(net_margin_series),
    }


def _coverage_profile(
    fact_bundle: Dict[str, Any],
    filing_history: Dict[str, Any],
) -> Tuple[Dict[str, bool], float, List[str]]:
    metrics = fact_bundle.get("normalized_metrics") or {}
    latest_quarter = fact_bundle.get("latest_quarter") or {}
    latest_balance_sheet = fact_bundle.get("latest_balance_sheet") or {}
    coverage_flags = {
        "filing_backbone": filing_history.get("latest_10k") is not None
        or filing_history.get("latest_10q") is not None,
        "revenue_growth": metrics.get("revenue_growth_yoy") is not None,
        "profitability": any(
            latest_quarter.get(field) is not None
            for field in ("operating_income", "net_income")
        ),
        "margins": any(
            metrics.get(field) is not None
            for field in ("gross_margin", "operating_margin", "net_margin")
        ),
        "balance_sheet_resilience": any(
            latest_balance_sheet.get(field) is not None
            for field in ("assets", "equity", "cash")
        ),
        "cash_flow": any(
            metrics.get(field) is not None
            for field in ("cash_flow_margin", "free_cash_flow", "positive_fcf_ratio")
        ),
        "leverage_liquidity": any(
            metrics.get(field) is not None
            for field in ("current_ratio", "cash_ratio", "debt_to_equity")
        ),
    }
    coverage_score = sum(1 for ok in coverage_flags.values() if ok) / max(
        len(coverage_flags), 1
    )
    missingness_flags = [
        f"{dimension}_missing"
        for dimension, ok in coverage_flags.items()
        if not ok
    ]
    return coverage_flags, coverage_score, missingness_flags


def _quality_proxies(
    *,
    fact_bundle: Dict[str, Any],
    coverage_score: float,
    filing_recency_days: Optional[int],
) -> Dict[str, Any]:
    metrics = fact_bundle.get("normalized_metrics") or {}
    filing_recency_score = _bounded_score(
        1.0 - min((filing_recency_days or 365), 365) / 365.0,
        low=0.0,
        high=1.0,
    )
    profitability_strength = _bounded_score(
        _mean(
            [
                metrics.get("gross_margin"),
                metrics.get("operating_margin"),
                metrics.get("net_margin"),
            ]
        ),
        low=0.0,
        high=0.4,
    )
    resilience_strength = _bounded_score(
        _mean(
            [
                metrics.get("current_ratio"),
                _inverse_metric(metrics.get("debt_to_equity"), cap=3.0),
            ]
        ),
        low=0.0,
        high=2.0,
    )
    cash_flow_durability = _bounded_score(
        _mean(
            [
                metrics.get("positive_fcf_ratio"),
                metrics.get("free_cash_flow_margin"),
            ]
        ),
        low=0.0,
        high=0.3,
    )
    reporting_quality_proxy = _mean(
        [
            filing_recency_score,
            coverage_score * 100.0,
            _mean(
                [
                    metrics.get("gross_margin_stability"),
                    metrics.get("operating_margin_stability"),
                    metrics.get("net_margin_stability"),
                ]
            ),
        ]
    )
    business_quality_durability = _mean(
        [
            _bounded_score(metrics.get("revenue_growth_yoy"), low=-0.2, high=0.3),
            profitability_strength,
            resilience_strength,
            cash_flow_durability,
            filing_recency_score,
        ]
    )
    return {
        "filing_recency_score": filing_recency_score,
        "reporting_completeness_score": coverage_score * 100.0,
        "reporting_quality_proxy": reporting_quality_proxy,
        "business_quality_durability": business_quality_durability,
        "cash_flow_durability": cash_flow_durability,
        "profitability_strength": profitability_strength,
        "balance_sheet_resilience": resilience_strength,
    }


def _strengths_and_weaknesses(
    *,
    fact_bundle: Dict[str, Any],
    coverage_flags: Dict[str, bool],
    filing_history: Dict[str, Any],
    mapping: Dict[str, Any],
) -> Tuple[List[str], List[str], List[str]]:
    metrics = fact_bundle.get("normalized_metrics") or {}
    strengths: List[str] = []
    weaknesses: List[str] = []
    caveats: List[str] = []

    revenue_growth = metrics.get("revenue_growth_yoy")
    if revenue_growth is not None and revenue_growth >= 0.1:
        strengths.append(
            f"Quarterly revenue growth is running at {revenue_growth * 100:.1f}% year over year."
        )
    elif revenue_growth is not None and revenue_growth < 0:
        weaknesses.append(
            f"Quarterly revenue is contracting {abs(revenue_growth) * 100:.1f}% versus the year-ago quarter."
        )

    operating_margin = metrics.get("operating_margin")
    if operating_margin is not None and operating_margin >= 0.2:
        strengths.append(
            f"Operating margin is healthy at {operating_margin * 100:.1f}%."
        )
    elif operating_margin is not None and operating_margin < 0.08:
        weaknesses.append(
            f"Operating margin is thin at {operating_margin * 100:.1f}%."
        )

    current_ratio = metrics.get("current_ratio")
    if current_ratio is not None and current_ratio >= 1.5:
        strengths.append(
            f"Current ratio of {current_ratio:.2f} indicates good near-term liquidity."
        )
    elif current_ratio is not None and current_ratio < 1.0:
        weaknesses.append(
            f"Current ratio of {current_ratio:.2f} suggests tighter liquidity."
        )

    debt_to_equity = metrics.get("debt_to_equity")
    if debt_to_equity is not None and debt_to_equity > 1.5:
        weaknesses.append(
            f"Debt-to-equity is elevated at {debt_to_equity:.2f}."
        )

    positive_fcf_ratio = metrics.get("positive_fcf_ratio")
    if positive_fcf_ratio is not None and positive_fcf_ratio >= 0.75:
        strengths.append(
            "Recent quarterly free-cash-flow coverage is consistently positive."
        )
    elif positive_fcf_ratio is not None and positive_fcf_ratio < 0.5:
        weaknesses.append(
            "Free-cash-flow durability is uneven across recent quarters."
        )

    filing_recency_days = filing_history.get("filing_recency_days")
    if filing_recency_days is not None and filing_recency_days > 150:
        caveats.append(
            f"Latest periodic filing is {filing_recency_days} days old, so the filing layer is no longer fresh."
        )
    if mapping.get("match_type") not in {"exact_ticker", "normalized_ticker"}:
        caveats.append(
            f"SEC mapping required {mapping.get('match_type')} fallback rather than a direct ticker match."
        )
    for dimension, ok in coverage_flags.items():
        if not ok:
            caveats.append(f"{dimension.replace('_', ' ')} coverage is missing or partial.")
    return strengths[:4], weaknesses[:4], caveats[:6]


def _mapping_payload(
    *,
    requested_symbol: str,
    requested_company_name: Optional[str],
    match_type: str,
    record: Dict[str, Any],
    notes: List[str],
) -> Dict[str, Any]:
    return {
        "requested_symbol": requested_symbol,
        "requested_company_name": requested_company_name,
        "match_type": match_type,
        "matched_ticker": record.get("ticker"),
        "matched_company_name": record.get("title"),
        "cik": record.get("cik"),
        "notes": notes,
    }


def _normalize_sec_ticker(value: str) -> str:
    cleaned = str(value or "").strip().upper()
    return cleaned.replace(".", "").replace("-", "").replace("/", "")


def _normalize_company_name(value: Optional[str]) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = _NAME_TOKEN_RE.sub(" ", text)
    text = re.sub(r"\b(inc|corp|corporation|holdings|group|plc|ltd|limited|company|co)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_quarterly_point(point: Dict[str, Any]) -> bool:
    days = point.get("days")
    form = str(point.get("form") or "")
    fp = str(point.get("fp") or "")
    if days is not None and 70 <= days <= 120:
        return True
    if form in _QUARTERLY_FORMS:
        return True
    if form in _ANNUAL_FORMS and fp.startswith("Q") and fp != "FY":
        return True
    return False


def _is_annual_point(point: Dict[str, Any]) -> bool:
    days = point.get("days")
    form = str(point.get("form") or "")
    fp = str(point.get("fp") or "")
    if days is not None and days >= 280:
        return True
    if form in _ANNUAL_FORMS and fp in {"FY", ""}:
        return True
    return False


def _fact_sort_key(point: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(point.get("period_end") or ""),
        str(point.get("report_date") or ""),
    )


def _duration_days(start: Any, end: Any) -> Optional[int]:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    if start_date is None or end_date is None:
        return None
    return max((end_date - start_date).days, 0)


def _parse_date(value: Any) -> Optional[dt.date]:
    if value in (None, ""):
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _attach_margin_fields(row: Dict[str, Any]) -> None:
    revenue = row.get("revenue")
    row["gross_margin"] = _ratio(row.get("gross_profit"), revenue)
    row["operating_margin"] = _ratio(row.get("operating_income"), revenue)
    row["net_margin"] = _ratio(row.get("net_income"), revenue)
    cfo = row.get("cash_from_operations")
    capex = row.get("capital_expenditures")
    row["free_cash_flow"] = None if cfo is None else cfo - abs(capex or 0.0)


def _ratio(numerator: Any, denominator: Any) -> Optional[float]:
    numerator_value = _safe_float(numerator)
    denominator_value = _safe_float(denominator)
    if numerator_value is None or denominator_value in (None, 0.0):
        return None
    return numerator_value / denominator_value


def _pct_change(current: Any, prior: Any) -> Optional[float]:
    current_value = _safe_float(current)
    prior_value = _safe_float(prior)
    if current_value is None or prior_value in (None, 0.0):
        return None
    return current_value / prior_value - 1.0


def _margin_stability(values: Iterable[Any]) -> Optional[float]:
    clean = [float(value) for value in values if _safe_float(value) is not None]
    if len(clean) < 2:
        return None
    sigma = statistics.pstdev(clean)
    return max(0.0, 100.0 * (1.0 - min(sigma, 0.2) / 0.2))


def _positive_ratio(values: Iterable[Any]) -> Optional[float]:
    clean = [_safe_float(value) for value in values]
    present = [value for value in clean if value is not None]
    if not present:
        return None
    return sum(1 for value in present if value > 0) / len(present)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(statistics.fmean(clean))


def _bounded_score(value: Optional[float], *, low: float, high: float) -> Optional[float]:
    if value is None or math.isclose(high, low):
        return None
    clipped = max(low, min(high, float(value)))
    return 100.0 * ((clipped - low) / (high - low))


def _inverse_metric(value: Optional[float], *, cap: float) -> Optional[float]:
    if value is None:
        return None
    return max(cap - min(float(value), cap), 0.0)


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


def _first_available(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": config.sec_user_agent(),
        "Accept": "application/json",
        "Host": "data.sec.gov",
    }
