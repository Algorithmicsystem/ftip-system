import datetime as dt

from api.assistant import reports
from api.data_providers import sec_edgar


def test_resolve_company_mapping_normalizes_share_class_ticker(monkeypatch):
    monkeypatch.setattr(
        sec_edgar,
        "_ticker_records",
        lambda: [
            {
                "ticker": "BRK-B",
                "ticker_normalized": "BRKB",
                "title": "Berkshire Hathaway Inc",
                "title_normalized": "berkshire hathaway",
                "cik": "0001067983",
            }
        ],
    )

    mapping = sec_edgar.resolve_company_mapping("BRK.B")

    assert mapping["cik"] == "0001067983"
    assert mapping["match_type"] == "normalized_ticker"
    assert mapping["matched_ticker"] == "BRK-B"


def test_resolve_company_mapping_falls_back_to_company_name(monkeypatch):
    monkeypatch.setattr(
        sec_edgar,
        "_ticker_records",
        lambda: [
            {
                "ticker": "META",
                "ticker_normalized": "META",
                "title": "Meta Platforms Inc",
                "title_normalized": "meta platforms",
                "cik": "0001326801",
            }
        ],
    )

    mapping = sec_edgar.resolve_company_mapping(
        "FB",
        company_name="Meta Platforms, Inc.",
    )

    assert mapping["cik"] == "0001326801"
    assert mapping["match_type"] == "company_name_exact"
    assert "matched company title" in mapping["notes"][0]


def test_fetch_company_filing_profile_extracts_normalized_sec_metrics(monkeypatch):
    monkeypatch.setattr(
        sec_edgar,
        "_ticker_records",
        lambda: [
            {
                "ticker": "NVDA",
                "ticker_normalized": "NVDA",
                "title": "NVIDIA CORP",
                "title_normalized": "nvidia",
                "cik": "0001045810",
            }
        ],
    )
    monkeypatch.setattr(
        sec_edgar,
        "fetch_submissions",
        lambda cik: {
            "name": "NVIDIA CORP",
            "sic": "3674",
            "sicDescription": "Semiconductors",
            "fiscalYearEnd": "0131",
            "filings": {
                "recent": {
                    "accessionNumber": ["0001", "0002", "0003"],
                    "filingDate": ["2026-03-01", "2025-12-15", "2025-03-01"],
                    "form": ["10-Q", "10-Q", "10-K"],
                    "primaryDocument": ["q1.htm", "q2.htm", "annual.htm"],
                }
            },
        },
    )
    monkeypatch.setattr(
        sec_edgar,
        "fetch_companyfacts",
        lambda cik: {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-11-01",
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fp": "Q4",
                                    "fy": 2026,
                                    "val": 1200,
                                },
                                {
                                    "start": "2024-11-01",
                                    "end": "2025-01-31",
                                    "filed": "2025-03-01",
                                    "form": "10-K",
                                    "fp": "Q4",
                                    "fy": 2025,
                                    "val": 1000,
                                },
                                {
                                    "start": "2025-02-01",
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-K",
                                    "fp": "FY",
                                    "fy": 2026,
                                    "val": 4200,
                                },
                            ]
                        }
                    },
                    "GrossProfit": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-11-01",
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fp": "Q4",
                                    "fy": 2026,
                                    "val": 720,
                                }
                            ]
                        }
                    },
                    "OperatingIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-11-01",
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fp": "Q4",
                                    "fy": 2026,
                                    "val": 360,
                                }
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-11-01",
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fp": "Q4",
                                    "fy": 2026,
                                    "val": 300,
                                },
                                {
                                    "start": "2025-02-01",
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-K",
                                    "fp": "FY",
                                    "fy": 2026,
                                    "val": 980,
                                },
                            ]
                        }
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-11-01",
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fp": "Q4",
                                    "fy": 2026,
                                    "val": 330,
                                }
                            ]
                        }
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-11-01",
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fp": "Q4",
                                    "fy": 2026,
                                    "val": -80,
                                }
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fy": 2026,
                                    "val": 9000,
                                }
                            ]
                        }
                    },
                    "AssetsCurrent": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fy": 2026,
                                    "val": 4200,
                                }
                            ]
                        }
                    },
                    "Liabilities": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fy": 2026,
                                    "val": 3600,
                                }
                            ]
                        }
                    },
                    "LiabilitiesCurrent": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fy": 2026,
                                    "val": 1400,
                                }
                            ]
                        }
                    },
                    "StockholdersEquity": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fy": 2026,
                                    "val": 5400,
                                }
                            ]
                        }
                    },
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fy": 2026,
                                    "val": 2100,
                                }
                            ]
                        }
                    },
                    "LongTermDebtNoncurrent": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2026-01-31",
                                    "filed": "2026-03-01",
                                    "form": "10-Q",
                                    "fy": 2026,
                                    "val": 900,
                                }
                            ]
                        }
                    },
                }
            }
        },
    )

    profile = sec_edgar.fetch_company_filing_profile(
        "NVDA",
        company_name="NVIDIA Corp",
        as_of_date=dt.date(2026, 4, 11),
    )

    assert profile["mapping"]["match_type"] == "exact_ticker"
    assert profile["filing_backbone"]["latest_10q"]["filing_date"] == "2026-03-01"
    assert round(profile["normalized_metrics"]["revenue_growth_yoy"], 4) == 0.2
    assert profile["normalized_metrics"]["operating_margin"] == 0.3
    assert profile["normalized_metrics"]["current_ratio"] == 3.0
    assert round(profile["normalized_metrics"]["debt_to_equity"], 4) == round(900 / 5400, 4)
    assert profile["coverage_flags"]["cash_flow"] is True
    assert profile["coverage_score"] > 0.7
    assert profile["meta"]["latest_report_date"] == "2026-03-01"
    assert profile["strength_summary"]


def test_fundamental_report_section_is_populated_when_coverage_exists():
    report = reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2026-04-11",
        horizon="swing",
        risk_mode="balanced",
        signal={"action": "BUY", "score": 0.8, "confidence": 0.7},
        key_features={},
        quality={"fundamentals_ok": True, "warnings": []},
        evidence={"sources": ["signals_daily", "sec_edgar"]},
        data_bundle={
            "fundamental_filing": {
                "filing_backbone": {
                    "latest_form": "10-Q",
                    "latest_filing_date": "2026-03-01",
                    "latest_10k": {"filing_date": "2025-03-01"},
                    "latest_10q": {"filing_date": "2026-03-01"},
                },
                "statement_snapshot": {
                    "latest_quarter": {"revenue": 1200, "report_date": "2026-03-01"},
                    "latest_annual": {"report_date": "2026-03-01", "net_income": 980},
                    "latest_balance_sheet": {"assets": 9000, "equity": 5400},
                },
                "normalized_metrics": {
                    "revenue_growth_yoy": 0.2,
                    "operating_margin": 0.3,
                    "net_margin": 0.25,
                    "free_cash_flow_margin": 0.21,
                    "current_ratio": 3.0,
                    "cash_ratio": 1.5,
                    "debt_to_equity": 0.17,
                    "liabilities_to_assets": 0.4,
                },
                "quality_proxies": {
                    "reporting_quality_proxy": 84.0,
                    "business_quality_durability": 81.0,
                },
                "coverage_score": 0.9,
                "strength_summary": ["Quarterly revenue growth is running at 20.0% year over year."],
                "weakness_summary": ["Debt-to-equity is manageable rather than elevated."],
                "coverage_caveats": ["Cash-flow detail is based on quarterly companyfacts only."],
                "meta": {"sources": ["sec_edgar", "finnhub_basic_financials"], "status": "fresh"},
            },
            "quality_provenance": {"freshness_summary": {}},
        },
        feature_factor_bundle={"composite_intelligence": {"Fundamental Durability Score": 78.0}},
        strategy={"final_signal": "BUY", "confidence": 0.72, "conviction_tier": "moderate"},
    )

    assert "Latest periodic filing is 10-Q dated 2026-03-01" in report["fundamental_analysis"]
    assert "Quarterly revenue is 1200" in report["fundamental_analysis"]
    assert "Fundamental strengths are" in report["fundamental_analysis"]


def test_fundamental_report_section_stays_explicit_when_thin():
    report = reports.build_analysis_report(
        symbol="AAPL",
        as_of_date="2026-04-11",
        horizon="swing",
        risk_mode="balanced",
        signal={"action": "HOLD", "score": 0.1, "confidence": 0.5},
        key_features={},
        quality={"fundamentals_ok": False, "warnings": []},
        evidence={"sources": ["signals_daily"]},
        data_bundle={
            "fundamental_filing": {
                "missingness_flags": ["revenue_growth_coverage_missing", "cash_flow_coverage_missing"],
                "meta": {"sources": ["sec_edgar"], "status": "limited"},
            },
            "quality_provenance": {"freshness_summary": {}},
        },
        feature_factor_bundle={"composite_intelligence": {}},
        strategy={"final_signal": "HOLD", "confidence": 0.5},
    )

    assert "Fundamental coverage is explicitly missing" in report["fundamental_analysis"] or "Fundamental data is still thin" in report["fundamental_analysis"]
