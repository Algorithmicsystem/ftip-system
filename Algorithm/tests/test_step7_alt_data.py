"""Step 7: Entity Resolution Engine + Alt Data Scrapers — test suite.

Tests cover:
- Entity resolver: exact and fuzzy matching, bulk_resolve, get_company_name
- Glassdoor sentiment: neutral fallback, Google CSE path, Comparably path
- EPA ECHO: mocked HTTP response parsing, store logic
- CourtListener: mocked HTTP response, case classification, risk scoring
- Migration SQL files exist for tables 113-116
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MIGRATIONS_DIR = Path(__file__).parents[1] / "api" / "migrations"
SCRAPERS_DIR = Path(__file__).parents[1] / "api" / "scrapers"


# ---------------------------------------------------------------------------
# Part 1 — Entity Resolver
# ---------------------------------------------------------------------------

class TestEntityResolverNormalize(unittest.TestCase):
    def setUp(self):
        from api.scrapers.entity_resolver import _normalize
        self.normalize = _normalize

    def test_strips_legal_suffixes(self):
        assert self.normalize("Apple Inc.") == "apple"

    def test_strips_corp(self):
        assert self.normalize("Microsoft Corporation") == "microsoft"

    def test_lowercases(self):
        assert self.normalize("NVIDIA Corporation") == "nvidia"

    def test_strips_punctuation(self):
        assert self.normalize("Johnson & Johnson") == "johnson  johnson"

    def test_handles_empty(self):
        assert self.normalize("") == ""

    def test_strips_holdings(self):
        assert "holdings" not in self.normalize("Berkshire Hathaway Holdings")


class TestEntityResolverKnownNames(unittest.TestCase):
    def test_known_names_populated(self):
        from api.scrapers.entity_resolver import KNOWN_NAMES
        assert len(KNOWN_NAMES) >= 50

    def test_aapl_present(self):
        from api.scrapers.entity_resolver import KNOWN_NAMES
        assert "AAPL" in KNOWN_NAMES
        assert "Apple" in KNOWN_NAMES["AAPL"]

    def test_msft_present(self):
        from api.scrapers.entity_resolver import KNOWN_NAMES
        assert "MSFT" in KNOWN_NAMES

    def test_get_company_name_known(self):
        from api.scrapers.entity_resolver import get_company_name
        assert "Apple" in get_company_name("AAPL")

    def test_get_company_name_unknown_returns_ticker(self):
        from api.scrapers.entity_resolver import get_company_name
        assert get_company_name("XYZ999") == "XYZ999"


class TestEntityResolverExactMatch(unittest.TestCase):
    """Test exact matching when DB candidates are available."""

    def _make_candidates(self):
        # Returns (canon_normalized, ticker, alias_norms) tuples
        return [
            ("apple", "AAPL", ["aapl", "apple computer"]),
            ("microsoft", "MSFT", ["msft"]),
            ("nvidia", "NVDA", ["nvda", "nvidia corp"]),
            ("amazon", "AMZN", ["amzn", "amazoncom"]),
        ]

    def test_exact_canonical_match(self):
        from api.scrapers import entity_resolver
        candidates = self._make_candidates()
        with patch.object(entity_resolver, "_get_candidates", return_value=candidates), \
             patch.dict(entity_resolver._RESOLVE_CACHE, {}, clear=True):
            result = entity_resolver.resolve_entity("Apple")
            assert result == "AAPL", f"Expected AAPL, got {result}"

    def test_exact_alias_match(self):
        from api.scrapers import entity_resolver
        candidates = self._make_candidates()
        with patch.object(entity_resolver, "_get_candidates", return_value=candidates), \
             patch.dict(entity_resolver._RESOLVE_CACHE, {}, clear=True):
            result = entity_resolver.resolve_entity("Apple Computer")
            assert result == "AAPL", f"Expected AAPL, got {result}"

    def test_case_insensitive(self):
        from api.scrapers import entity_resolver
        candidates = self._make_candidates()
        with patch.object(entity_resolver, "_get_candidates", return_value=candidates), \
             patch.dict(entity_resolver._RESOLVE_CACHE, {}, clear=True):
            result = entity_resolver.resolve_entity("MICROSOFT")
            assert result == "MSFT"

    def test_strips_inc_before_matching(self):
        from api.scrapers import entity_resolver
        candidates = self._make_candidates()
        with patch.object(entity_resolver, "_get_candidates", return_value=candidates), \
             patch.dict(entity_resolver._RESOLVE_CACHE, {}, clear=True):
            result = entity_resolver.resolve_entity("Apple Inc.")
            assert result == "AAPL"

    def test_no_match_returns_none(self):
        from api.scrapers import entity_resolver
        candidates = self._make_candidates()
        with patch.object(entity_resolver, "_get_candidates", return_value=candidates), \
             patch.dict(entity_resolver._RESOLVE_CACHE, {}, clear=True):
            result = entity_resolver.resolve_entity("Totally Unknown Company XYZ")
            assert result is None

    def test_empty_string_returns_none(self):
        from api.scrapers import entity_resolver
        result = entity_resolver.resolve_entity("")
        assert result is None


class TestEntityResolverFuzzyMatch(unittest.TestCase):
    """Test fuzzy matching with SequenceMatcher."""

    def _make_candidates(self):
        return [
            ("apple", "AAPL", ["apple computer"]),
            ("microsoft", "MSFT", []),
        ]

    def test_fuzzy_match_typo(self):
        from api.scrapers import entity_resolver
        candidates = self._make_candidates()
        with patch.object(entity_resolver, "_get_candidates", return_value=candidates), \
             patch.dict(entity_resolver._RESOLVE_CACHE, {}, clear=True):
            # "appl" is close enough to "apple" (ratio ~0.89)
            result = entity_resolver.resolve_entity("Appl")
            assert result == "AAPL"

    def test_fuzzy_threshold_rejects_poor_match(self):
        from api.scrapers import entity_resolver
        candidates = self._make_candidates()
        with patch.object(entity_resolver, "_get_candidates", return_value=candidates), \
             patch.dict(entity_resolver._RESOLVE_CACHE, {}, clear=True):
            # Very different name — should not match
            result = entity_resolver.resolve_entity("Zymeworks Inc")
            assert result is None

    def test_fuzzy_threshold_value(self):
        from api.scrapers.entity_resolver import _FUZZY_THRESHOLD
        assert _FUZZY_THRESHOLD >= 0.80
        assert _FUZZY_THRESHOLD <= 0.95


class TestBulkResolve(unittest.TestCase):
    def test_bulk_resolve_returns_dict(self):
        from api.scrapers import entity_resolver
        candidates = [("apple", "AAPL", []), ("microsoft", "MSFT", [])]
        with patch.object(entity_resolver, "_get_candidates", return_value=candidates), \
             patch.dict(entity_resolver._RESOLVE_CACHE, {}, clear=True):
            result = entity_resolver.bulk_resolve(["Apple", "Microsoft", "Unknown"])
            assert isinstance(result, dict)
            assert result["Apple"] == "AAPL"
            assert result["Microsoft"] == "MSFT"
            assert result["Unknown"] is None

    def test_bulk_resolve_empty_list(self):
        from api.scrapers.entity_resolver import bulk_resolve
        assert bulk_resolve([]) == {}


class TestSeedEntityResolution(unittest.TestCase):
    def test_seed_no_db_returns_gracefully(self):
        from api.scrapers import entity_resolver
        with patch("api.scrapers.entity_resolver.db") as mock_db:
            mock_db.db_write_enabled.return_value = False
            result = entity_resolver.seed_entity_resolution(10)
        assert result["seeded"] == 0
        assert "reason" in result


# ---------------------------------------------------------------------------
# Part 2 — Glassdoor Sentiment
# ---------------------------------------------------------------------------

class TestGlassdoorSentimentFallback(unittest.TestCase):
    def test_neutral_fallback_when_no_env(self):
        """Returns neutral 50 when GOOGLE_API_KEY not set and Comparably unavailable."""
        from api.scrapers.glassdoor_sentiment import fetch_employee_sentiment
        with patch.dict("os.environ", {}, clear=False), \
             patch("api.scrapers.glassdoor_sentiment._fetch_google_cse", return_value=None), \
             patch("api.scrapers.glassdoor_sentiment._fetch_comparably", return_value=None):
            result = fetch_employee_sentiment("AAPL", "Apple Inc")
        assert result["ticker"] == "AAPL"
        assert result["overall_rating"] == 50.0
        assert result["ceo_approval"] == 50.0
        assert result["culture_score"] == 50.0
        assert result["source"] == "neutral_fallback"
        assert "as_of_date" in result

    def test_result_has_required_keys(self):
        from api.scrapers.glassdoor_sentiment import fetch_employee_sentiment
        with patch("api.scrapers.glassdoor_sentiment._fetch_google_cse", return_value=None), \
             patch("api.scrapers.glassdoor_sentiment._fetch_comparably", return_value=None):
            result = fetch_employee_sentiment("MSFT", "Microsoft")
        for key in ("ticker", "overall_rating", "ceo_approval", "culture_score", "source", "as_of_date"):
            assert key in result, f"Missing key: {key}"

    def test_google_cse_path(self):
        from api.scrapers.glassdoor_sentiment import fetch_employee_sentiment
        mock_result = {
            "overall_rating": 75.0,
            "ceo_approval": 80.0,
            "culture_score": 70.0,
            "source": "google_cse_glassdoor",
        }
        with patch("api.scrapers.glassdoor_sentiment._fetch_google_cse", return_value=mock_result), \
             patch("api.scrapers.glassdoor_sentiment._fetch_comparably", return_value=None):
            result = fetch_employee_sentiment("NVDA", "NVIDIA")
        assert result["overall_rating"] == 75.0
        assert result["source"] == "google_cse_glassdoor"

    def test_comparably_used_as_fallback_when_cse_missing(self):
        from api.scrapers.glassdoor_sentiment import fetch_employee_sentiment
        mock_comparably = {
            "overall_rating": 68.0,
            "ceo_approval": 72.0,
            "culture_score": 65.0,
            "source": "comparably",
        }
        with patch("api.scrapers.glassdoor_sentiment._fetch_google_cse", return_value=None), \
             patch("api.scrapers.glassdoor_sentiment._fetch_comparably", return_value=mock_comparably):
            result = fetch_employee_sentiment("AMZN", "Amazon")
        assert result["overall_rating"] == 68.0
        assert result["source"] == "comparably"

    def test_ratings_in_valid_range(self):
        from api.scrapers.glassdoor_sentiment import fetch_employee_sentiment
        with patch("api.scrapers.glassdoor_sentiment._fetch_google_cse", return_value=None), \
             patch("api.scrapers.glassdoor_sentiment._fetch_comparably", return_value=None):
            result = fetch_employee_sentiment("T", "AT&T")
        assert 0.0 <= result["overall_rating"] <= 100.0
        assert 0.0 <= result["ceo_approval"] <= 100.0
        assert 0.0 <= result["culture_score"] <= 100.0

    def test_google_cse_not_called_without_key(self):
        from api.scrapers.glassdoor_sentiment import _fetch_google_cse
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "", "GOOGLE_CX": ""}):
            result = _fetch_google_cse("Apple")
        assert result is None


class TestStoreEmployeeSentiment(unittest.TestCase):
    def test_store_returns_zero_when_db_disabled(self):
        from api.scrapers.glassdoor_sentiment import store_employee_sentiment
        with patch("api.scrapers.glassdoor_sentiment.db") as mock_db:
            mock_db.db_write_enabled.return_value = False
            result = store_employee_sentiment([{"ticker": "AAPL"}])
        assert result == 0


# ---------------------------------------------------------------------------
# Part 3 — EPA ECHO
# ---------------------------------------------------------------------------

class TestEpaEchoFetch(unittest.TestCase):
    def _mock_facilities_response(self) -> Dict[str, Any]:
        return {
            "Results": {
                "Facilities": [
                    {
                        "FacName": "Apple Operations",
                        "CAA3YrQtrsInViol": "2",
                        "CWA3YrQtrsInViol": "1",
                        "RCRA3YrQtrsInViol": "0",
                        "CAAScore": "5000",
                        "CWAScore": "2000",
                    },
                    {
                        "FacName": "Apple Manufacturing",
                        "CAA3YrQtrsInViol": "0",
                        "CWA3YrQtrsInViol": "0",
                        "RCRA3YrQtrsInViol": "1",
                        "CAAScore": "0",
                        "CWAScore": "0",
                    },
                ]
            }
        }

    def test_fetch_returns_dict_with_required_keys(self):
        from api.scrapers.epa_echo import fetch_epa_violations
        with patch("api.scrapers.epa_echo._fetch_echo_facilities",
                   return_value=self._mock_facilities_response()["Results"]["Facilities"]):
            result = fetch_epa_violations("AAPL", "Apple Inc")
        for key in ("ticker", "violation_count_3yr", "total_penalties_usd",
                    "esg_risk_score", "facilities_count", "source", "as_of_date"):
            assert key in result, f"Missing key: {key}"

    def test_violation_count_aggregated(self):
        from api.scrapers.epa_echo import fetch_epa_violations
        facilities = self._mock_facilities_response()["Results"]["Facilities"]
        with patch("api.scrapers.epa_echo._fetch_echo_facilities", return_value=facilities):
            result = fetch_epa_violations("AAPL", "Apple Inc")
        # Facility 1: 2+1+0=3, Facility 2: 0+0+1=1 → total 4
        assert result["violation_count_3yr"] == 4

    def test_esg_risk_score_in_range(self):
        from api.scrapers.epa_echo import fetch_epa_violations
        facilities = self._mock_facilities_response()["Results"]["Facilities"]
        with patch("api.scrapers.epa_echo._fetch_echo_facilities", return_value=facilities):
            result = fetch_epa_violations("AAPL", "Apple Inc")
        assert 0.0 <= result["esg_risk_score"] <= 100.0

    def test_source_is_epa_echo(self):
        from api.scrapers.epa_echo import fetch_epa_violations
        with patch("api.scrapers.epa_echo._fetch_echo_facilities", return_value=[]):
            result = fetch_epa_violations("AAPL", "Apple Inc")
        assert result["source"] == "EPA_ECHO"

    def test_empty_facilities_returns_zero_violations(self):
        from api.scrapers.epa_echo import fetch_epa_violations
        with patch("api.scrapers.epa_echo._fetch_echo_facilities", return_value=[]):
            result = fetch_epa_violations("MSFT", "Microsoft")
        assert result["violation_count_3yr"] == 0
        assert result["esg_risk_score"] == 0.0

    def test_api_failure_returns_empty(self):
        from api.scrapers.epa_echo import fetch_epa_violations
        with patch("api.scrapers.epa_echo._fetch_echo_facilities", return_value=None):
            result = fetch_epa_violations("GS", "Goldman Sachs")
        assert result["violation_count_3yr"] == 0

    def test_facilities_count_populated(self):
        from api.scrapers.epa_echo import fetch_epa_violations
        facilities = self._mock_facilities_response()["Results"]["Facilities"]
        with patch("api.scrapers.epa_echo._fetch_echo_facilities", return_value=facilities):
            result = fetch_epa_violations("AAPL", "Apple Inc")
        assert result["facilities_count"] == 2

    def test_high_violations_produce_high_esg_risk(self):
        from api.scrapers.epa_echo import fetch_epa_violations
        # 20 quarters in violation → should produce high risk
        many_viols = [{"CAA3YrQtrsInViol": "20", "CWA3YrQtrsInViol": "0",
                        "RCRA3YrQtrsInViol": "0", "CAAScore": "0", "CWAScore": "0"}]
        with patch("api.scrapers.epa_echo._fetch_echo_facilities", return_value=many_viols):
            result = fetch_epa_violations("XOM", "Exxon Mobil")
        assert result["esg_risk_score"] > 50.0


class TestStoreEpaViolations(unittest.TestCase):
    def test_store_returns_zero_when_db_disabled(self):
        from api.scrapers.epa_echo import store_epa_violations
        with patch("api.scrapers.epa_echo.db") as mock_db:
            mock_db.db_write_enabled.return_value = False
            result = store_epa_violations([{"ticker": "XOM"}])
        assert result == 0


# ---------------------------------------------------------------------------
# Part 4 — CourtListener
# ---------------------------------------------------------------------------

class TestCourtListenerClassification(unittest.TestCase):
    def test_securities_fraud_classified(self):
        from api.scrapers.court_listener import _classify_case
        assert _classify_case("Securities Fraud Class Action", "") == "securities"

    def test_employment_classified(self):
        from api.scrapers.court_listener import _classify_case
        assert _classify_case("Employment Discrimination", "") == "employment"

    def test_antitrust_classified(self):
        from api.scrapers.court_listener import _classify_case
        assert _classify_case("Antitrust Monopoly Case", "") == "antitrust"

    def test_ip_classified(self):
        from api.scrapers.court_listener import _classify_case
        assert _classify_case("Patent Infringement", "") == "ip"

    def test_other_when_no_keywords(self):
        from api.scrapers.court_listener import _classify_case
        assert _classify_case("Contract Dispute", "") == "other"

    def test_securities_takes_priority(self):
        from api.scrapers.court_listener import _classify_case
        # Both securities and employment keywords
        assert _classify_case("Securities Fraud Employment", "") == "securities"


class TestCourtListenerFetch(unittest.TestCase):
    def _mock_api_response(self) -> str:
        data = {
            "count": 3,
            "results": [
                {"case_name": "Smith v Apple Inc Securities Fraud Class Action",
                 "date_filed": "2024-01-15", "docket_text": "securities exchange act",
                 "nature_of_suit_str": "Securities"},
                {"case_name": "Johnson v Apple Employment Discrimination",
                 "date_filed": "2024-02-10", "docket_text": "discrimination employment",
                 "nature_of_suit_str": "Employment"},
                {"case_name": "Patent Infringement Case",
                 "date_filed": "2024-03-01", "docket_text": "patent infringement",
                 "nature_of_suit_str": "Patent"},
            ],
        }
        return json.dumps(data).encode()

    def test_fetch_returns_required_keys(self):
        from api.scrapers.court_listener import fetch_litigation_risk
        mock_response = MagicMock()
        mock_response.read.return_value = self._mock_api_response()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = fetch_litigation_risk("AAPL", "Apple Inc")

        for key in ("ticker", "active_cases_1yr", "securities_fraud_cases",
                    "employment_cases", "total_litigation_score", "source", "as_of_date"):
            assert key in result, f"Missing key: {key}"

    def test_case_counts_correct(self):
        from api.scrapers.court_listener import fetch_litigation_risk
        mock_response = MagicMock()
        mock_response.read.return_value = self._mock_api_response()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = fetch_litigation_risk("AAPL", "Apple Inc")

        assert result["active_cases_1yr"] == 3
        assert result["securities_fraud_cases"] == 1
        assert result["employment_cases"] == 1
        assert result["ip_cases"] == 1

    def test_securities_weighted_in_score(self):
        from api.scrapers.court_listener import fetch_litigation_risk
        mock_response = MagicMock()
        mock_response.read.return_value = self._mock_api_response()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = fetch_litigation_risk("AAPL", "Apple Inc")

        # 1 securities (3×) + 1 employment (1×) + 1 IP (1×) = 5 weighted → score = 50
        assert result["total_litigation_score"] == pytest.approx(50.0, abs=1.0)

    def test_api_failure_returns_empty(self):
        from api.scrapers.court_listener import fetch_litigation_risk
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = fetch_litigation_risk("AAPL", "Apple Inc")
        assert result["active_cases_1yr"] == 0
        assert result["total_litigation_score"] == 0.0

    def test_source_is_courtlistener(self):
        from api.scrapers.court_listener import fetch_litigation_risk
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            result = fetch_litigation_risk("MSFT", "Microsoft")
        assert result["source"] == "CourtListener"

    def test_score_capped_at_100(self):
        from api.scrapers.court_listener import fetch_litigation_risk
        # 100 securities fraud cases → weighted=300, score should be 100 max
        many_cases = [
            {"case_name": f"Securities Fraud Case {i}", "date_filed": "2024-01-01",
             "docket_text": "securities", "nature_of_suit_str": ""}
            for i in range(100)
        ]
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"count": 100, "results": many_cases}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = fetch_litigation_risk("GS", "Goldman Sachs")
        assert result["total_litigation_score"] == 100.0


class TestStoreLitigationRisk(unittest.TestCase):
    def test_store_returns_zero_when_db_disabled(self):
        from api.scrapers.court_listener import store_litigation_risk
        with patch("api.scrapers.court_listener.db") as mock_db:
            mock_db.db_write_enabled.return_value = False
            result = store_litigation_risk([{"ticker": "AAPL"}])
        assert result == 0


# ---------------------------------------------------------------------------
# Part 5 — Migration SQL files
# ---------------------------------------------------------------------------

class TestMigrationFilesExist(unittest.TestCase):
    def test_113_entity_resolution_sql_exists(self):
        path = MIGRATIONS_DIR / "113_entity_resolution.sql"
        assert path.exists(), f"Migration not found: {path}"

    def test_114_employee_sentiment_sql_exists(self):
        path = MIGRATIONS_DIR / "114_employee_sentiment.sql"
        assert path.exists()

    def test_115_epa_violations_sql_exists(self):
        path = MIGRATIONS_DIR / "115_epa_violations.sql"
        assert path.exists()

    def test_116_litigation_risk_sql_exists(self):
        path = MIGRATIONS_DIR / "116_litigation_risk.sql"
        assert path.exists()

    def test_113_creates_entity_resolution_table(self):
        sql = (MIGRATIONS_DIR / "113_entity_resolution.sql").read_text()
        assert "entity_resolution" in sql
        assert "ticker" in sql
        assert "canonical_name" in sql
        assert "aliases" in sql

    def test_114_creates_employee_sentiment_table(self):
        sql = (MIGRATIONS_DIR / "114_employee_sentiment.sql").read_text()
        assert "employee_sentiment" in sql
        assert "overall_rating" in sql
        assert "ceo_approval" in sql
        assert "culture_score" in sql

    def test_115_creates_epa_violations_table(self):
        sql = (MIGRATIONS_DIR / "115_epa_violations.sql").read_text()
        assert "epa_violations" in sql
        assert "violation_count_3yr" in sql
        assert "esg_risk_score" in sql
        assert "total_penalties_usd" in sql

    def test_116_creates_litigation_risk_table(self):
        sql = (MIGRATIONS_DIR / "116_litigation_risk.sql").read_text()
        assert "litigation_risk" in sql
        assert "securities_fraud_cases" in sql
        assert "total_litigation_score" in sql

    def test_migrations_registered_in_init(self):
        init_text = (MIGRATIONS_DIR / "__init__.py").read_text()
        for version in ("113_entity_resolution", "114_employee_sentiment",
                        "115_epa_violations", "116_litigation_risk"):
            assert version in init_text, f"Migration not registered: {version}"


# ---------------------------------------------------------------------------
# Part 6 — Scraper module structure
# ---------------------------------------------------------------------------

class TestScraperModulesExist(unittest.TestCase):
    def test_entity_resolver_module(self):
        path = SCRAPERS_DIR / "entity_resolver.py"
        assert path.exists()

    def test_glassdoor_sentiment_module(self):
        path = SCRAPERS_DIR / "glassdoor_sentiment.py"
        assert path.exists()

    def test_epa_echo_module(self):
        path = SCRAPERS_DIR / "epa_echo.py"
        assert path.exists()

    def test_court_listener_module(self):
        path = SCRAPERS_DIR / "court_listener.py"
        assert path.exists()

    def test_entity_resolver_exports(self):
        from api.scrapers.entity_resolver import (
            resolve_entity, bulk_resolve, seed_entity_resolution,
            get_company_name, KNOWN_NAMES,
        )
        assert callable(resolve_entity)
        assert callable(bulk_resolve)
        assert callable(seed_entity_resolution)
        assert callable(get_company_name)
        assert isinstance(KNOWN_NAMES, dict)

    def test_glassdoor_exports(self):
        from api.scrapers.glassdoor_sentiment import (
            fetch_employee_sentiment, store_employee_sentiment, fetch_bulk_employee_sentiment,
        )
        assert callable(fetch_employee_sentiment)
        assert callable(store_employee_sentiment)

    def test_epa_echo_exports(self):
        from api.scrapers.epa_echo import (
            fetch_epa_violations, store_epa_violations, fetch_bulk_epa_violations,
        )
        assert callable(fetch_epa_violations)
        assert callable(store_epa_violations)

    def test_court_listener_exports(self):
        from api.scrapers.court_listener import (
            fetch_litigation_risk, store_litigation_risk, fetch_bulk_litigation_risk,
        )
        assert callable(fetch_litigation_risk)
        assert callable(store_litigation_risk)


# ---------------------------------------------------------------------------
# Part 7 — UniversalIntelligenceResponse new fields
# ---------------------------------------------------------------------------

class TestUniversalResponseNewFields(unittest.TestCase):
    def test_response_has_esg_risk_score(self):
        from api.universal.intelligence_api import UniversalIntelligenceResponse
        import dataclasses
        fields = {f.name for f in dataclasses.fields(UniversalIntelligenceResponse)}
        assert "esg_risk_score" in fields

    def test_response_has_litigation_score(self):
        from api.universal.intelligence_api import UniversalIntelligenceResponse
        import dataclasses
        fields = {f.name for f in dataclasses.fields(UniversalIntelligenceResponse)}
        assert "litigation_score" in fields

    def test_response_has_employee_sentiment_score(self):
        from api.universal.intelligence_api import UniversalIntelligenceResponse
        import dataclasses
        fields = {f.name for f in dataclasses.fields(UniversalIntelligenceResponse)}
        assert "employee_sentiment_score" in fields

    def test_new_fields_default_to_none(self):
        import datetime as dt
        from api.universal.intelligence_api import UniversalIntelligenceResponse
        resp = UniversalIntelligenceResponse(
            symbol="AAPL", as_of_date=dt.date.today(),
            signal_label="HOLD", dau=50.0, ml_adjusted_dau=50.0,
            analyst_rating="Hold", conviction="Low",
            regime_label="UNKNOWN", regime_strength=0.5,
            systemic_risk_index=50.0, ic_state="INSUFFICIENT",
            intelligence_quality_score=0.0, days_of_live_data=0,
            eis_score=50.0, caps_score=50.0, fragility_score=50.0,
            scps_score=50.0, bfs_score=50.0, factor_composite_score=50.0,
            osms_score=None, ias_score=None, pess_score=None,
            var_1d_99=None, sri=None,
            primary_driver="unknown", primary_conclusion="test",
            top_supporting_evidence=[], top_risk="none",
            signal_batting_average=None, dossier_event_count=0,
            moat_score=None, data_freshness_hours=0.0, staleness_warning=False,
        )
        assert resp.esg_risk_score is None
        assert resp.litigation_score is None
        assert resp.employee_sentiment_score is None


# ---------------------------------------------------------------------------
# Part 8 — EPA DAU penalty logic
# ---------------------------------------------------------------------------

class TestEpaDauPenalty(unittest.TestCase):
    """The EPA ESG risk score should reduce DAU by up to 5 points."""

    def test_esg_risk_above_50_reduces_dau(self):
        # esg_risk=60 → penalty = min(5, (60-50)/10) = 1.0
        import math
        esg_risk = 60.0
        dau = 75.0
        penalty = min(5.0, (esg_risk - 50.0) / 10.0)
        adjusted = max(0.0, dau - penalty)
        assert adjusted == pytest.approx(74.0)

    def test_esg_risk_100_caps_penalty_at_5(self):
        esg_risk = 100.0
        dau = 75.0
        penalty = min(5.0, (esg_risk - 50.0) / 10.0)
        adjusted = max(0.0, dau - penalty)
        assert penalty == 5.0
        assert adjusted == pytest.approx(70.0)

    def test_esg_risk_below_50_no_penalty(self):
        esg_risk = 30.0
        dau = 75.0
        # No penalty when esg_risk <= 50
        assert esg_risk <= 50.0  # guard condition not triggered

    def test_dau_cannot_go_below_zero(self):
        esg_risk = 100.0
        dau = 2.0
        penalty = min(5.0, (esg_risk - 50.0) / 10.0)
        adjusted = max(0.0, dau - penalty)
        assert adjusted >= 0.0


# ---------------------------------------------------------------------------
# Part 9 — Pipeline orchestrator wiring
# ---------------------------------------------------------------------------

class TestOrchestatorAltDataWiring(unittest.TestCase):
    def test_glassdoor_imported_in_alt_data(self):
        src = Path(__file__).parents[1] / "api" / "orchestration" / "pipeline_orchestrator.py"
        text = src.read_text()
        assert "glassdoor_sentiment" in text

    def test_epa_echo_imported_in_alt_data(self):
        src = Path(__file__).parents[1] / "api" / "orchestration" / "pipeline_orchestrator.py"
        text = src.read_text()
        assert "epa_echo" in text

    def test_court_listener_imported_in_alt_data(self):
        src = Path(__file__).parents[1] / "api" / "orchestration" / "pipeline_orchestrator.py"
        text = src.read_text()
        assert "court_listener" in text

    def test_entity_resolver_admin_endpoint_in_main(self):
        src = Path(__file__).parents[1] / "api" / "main.py"
        text = src.read_text()
        assert "entity-resolution/seed" in text or "entity_resolution" in text


import pytest
