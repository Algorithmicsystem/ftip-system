"""Entity Resolution Engine.

Maps arbitrary company name strings to exchange ticker symbols using
exact and fuzzy matching against a PostgreSQL lookup table seeded from
the universe registry.

Usage:
    from api.scrapers.entity_resolver import resolve_entity, bulk_resolve
    ticker = resolve_entity("Apple Computer Inc")   # → "AAPL"
    mapping = bulk_resolve(["Alphabet", "NVIDIA Corp"])  # → {name: ticker}
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from api import db

logger = logging.getLogger(__name__)

# Minimum SequenceMatcher ratio to accept a fuzzy match
_FUZZY_THRESHOLD = 0.85

# Common legal suffixes to strip before comparing
_STRIP_SUFFIXES = re.compile(
    r"\b(inc\.?|corp(?:oration)?\.?|co\.?|ltd\.?|llc\.?|plc\.?|group|holdings?|"
    r"international|industries|technologies|systems|services|solutions|"
    r"communications?|enterprises?|partners?|company|companies)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^\w\s]")

# In-memory caches
_RESOLVE_CACHE: Dict[str, Optional[str]] = {}
_CANDIDATES_CACHE: Optional[List[Tuple[str, str, List[str]]]] = None

# Canonical company names for the top universe symbols
# Used both by the entity resolver and by scrapers that need ticker→name lookup
KNOWN_NAMES: Dict[str, str] = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc",
    "GOOGL": "Alphabet Inc",
    "GOOG": "Alphabet Inc",
    "META": "Meta Platforms Inc",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla Inc",
    "LLY": "Eli Lilly and Company",
    "JPM": "JPMorgan Chase",
    "V": "Visa Inc",
    "UNH": "UnitedHealth Group",
    "XOM": "Exxon Mobil Corporation",
    "MA": "Mastercard Inc",
    "AVGO": "Broadcom Inc",
    "JNJ": "Johnson and Johnson",
    "PG": "Procter and Gamble",
    "HD": "Home Depot",
    "COST": "Costco Wholesale Corporation",
    "ABBV": "AbbVie Inc",
    "MRK": "Merck and Co",
    "CVX": "Chevron Corporation",
    "BAC": "Bank of America",
    "WMT": "Walmart Inc",
    "KO": "Coca-Cola Company",
    "CRM": "Salesforce Inc",
    "ACN": "Accenture",
    "NFLX": "Netflix Inc",
    "AMD": "Advanced Micro Devices",
    "ADBE": "Adobe Inc",
    "TMO": "Thermo Fisher Scientific",
    "TXN": "Texas Instruments",
    "WFC": "Wells Fargo",
    "CSCO": "Cisco Systems",
    "DIS": "Walt Disney Company",
    "ORCL": "Oracle Corporation",
    "ABT": "Abbott Laboratories",
    "INTU": "Intuit Inc",
    "PFE": "Pfizer Inc",
    "PM": "Philip Morris International",
    "MCD": "McDonalds Corporation",
    "IBM": "International Business Machines",
    "GS": "Goldman Sachs",
    "CAT": "Caterpillar Inc",
    "HON": "Honeywell International",
    "AMGN": "Amgen Inc",
    "BA": "Boeing Company",
    "GE": "General Electric",
    "RTX": "Raytheon Technologies",
    "SPGI": "S&P Global",
    "AXP": "American Express",
    "BKNG": "Booking Holdings",
    "QCOM": "QUALCOMM Incorporated",
    "LOW": "Lowes Companies",
    "PLD": "Prologis",
    "DE": "Deere and Company",
    "T": "AT&T Inc",
    "VZ": "Verizon Communications",
    "NEE": "NextEra Energy",
    "UPS": "United Parcel Service",
    "LIN": "Linde",
    "NKE": "Nike Inc",
    "ISRG": "Intuitive Surgical",
    "SBUX": "Starbucks Corporation",
    "ETN": "Eaton Corporation",
    "SLB": "SLB",
    "BMY": "Bristol Myers Squibb",
    "MDT": "Medtronic",
    "GILD": "Gilead Sciences",
    "ADI": "Analog Devices",
    "PANW": "Palo Alto Networks",
    "LRCX": "Lam Research",
    "VRTX": "Vertex Pharmaceuticals",
    "ZTS": "Zoetis Inc",
    "ADP": "Automatic Data Processing",
    "AMAT": "Applied Materials",
    "MMC": "Marsh and McLennan",
    "CB": "Chubb Limited",
    "AON": "Aon",
    "CI": "Cigna Group",
    "SYK": "Stryker Corporation",
    "REGN": "Regeneron Pharmaceuticals",
    "BSX": "Boston Scientific",
    "BDX": "Becton Dickinson",
    "ICE": "Intercontinental Exchange",
    "CME": "CME Group",
    "MCO": "Moodys Corporation",
    "MSCI": "MSCI Inc",
    "MCK": "McKesson Corporation",
    "CAH": "Cardinal Health",
    "CVS": "CVS Health Corporation",
    "WBA": "Walgreens Boots Alliance",
    "LMT": "Lockheed Martin",
    "NOC": "Northrop Grumman",
    "GD": "General Dynamics",
    "RTX": "Raytheon Technologies",
    "HII": "Huntington Ingalls Industries",
    "PLTR": "Palantir Technologies",
    "BAH": "Booz Allen Hamilton",
    "SAIC": "Science Applications International",
    "LDOS": "Leidos Holdings",
    "LHX": "L3Harris Technologies",
    "F": "Ford Motor Company",
    "GM": "General Motors",
    "C": "Citigroup",
    "MS": "Morgan Stanley",
    "BLK": "BlackRock",
    "SCHW": "Charles Schwab",
    "USB": "US Bancorp",
    "PNC": "PNC Financial Services",
    "TFC": "Truist Financial",
    "COF": "Capital One Financial",
    "AIG": "American International Group",
    "MET": "MetLife",
    "PRU": "Prudential Financial",
    "ALL": "Allstate Corporation",
    "HUM": "Humana Inc",
    "ELV": "Elevance Health",
    "CNC": "Centene Corporation",
    "MOH": "Molina Healthcare",
}


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation and common legal suffixes."""
    n = name.lower()
    n = _PUNCT.sub("", n)
    n = _STRIP_SUFFIXES.sub(" ", n)
    return n.strip()


def _load_candidates() -> List[Tuple[str, str, List[str]]]:
    """Load (canonical_normalized, ticker, [alias_normalized]) from DB."""
    from api import db
    if not db.db_read_enabled():
        return []
    try:
        rows = db.safe_fetchall(
            "SELECT ticker, canonical_name, COALESCE(aliases, '[]'::jsonb) FROM entity_resolution"
        )
        if not rows:
            return []
        result = []
        for ticker, canonical, aliases in rows:
            canon_norm = _normalize(str(canonical))
            alias_list = aliases if isinstance(aliases, list) else []
            alias_norms = [_normalize(str(a)) for a in alias_list]
            result.append((canon_norm, str(ticker), alias_norms))
        return result
    except Exception as exc:
        logger.debug("entity_resolver.load_candidates_failed err=%s", exc)
        return []


def _get_candidates() -> List[Tuple[str, str, List[str]]]:
    global _CANDIDATES_CACHE
    if _CANDIDATES_CACHE is None:
        _CANDIDATES_CACHE = _load_candidates()
    return _CANDIDATES_CACHE


def invalidate_cache() -> None:
    """Clear all in-memory caches (call after seeding or DB updates)."""
    global _CANDIDATES_CACHE
    _CANDIDATES_CACHE = None
    _RESOLVE_CACHE.clear()


def resolve_entity(name: str) -> Optional[str]:
    """Map a company name string to a ticker symbol.

    Resolution order:
    1. In-memory session cache (fast path).
    2. Exact match on normalized canonical_name or any alias in DB.
    3. Fuzzy match via difflib.SequenceMatcher (threshold 0.85).

    Returns the ticker symbol or None if no confident match found.
    """
    if not name or not name.strip():
        return None

    norm = _normalize(name)
    if norm in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[norm]

    candidates = _get_candidates()

    # Exact match (DB)
    for canon_norm, ticker, alias_norms in candidates:
        if norm == canon_norm or norm in alias_norms:
            _RESOLVE_CACHE[norm] = ticker
            return ticker

    # Fuzzy match
    best_ratio = 0.0
    best_ticker: Optional[str] = None
    for canon_norm, ticker, alias_norms in candidates:
        for candidate_name in [canon_norm] + alias_norms:
            ratio = difflib.SequenceMatcher(None, norm, candidate_name).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_ticker = ticker

    result = best_ticker if best_ratio >= _FUZZY_THRESHOLD else None
    _RESOLVE_CACHE[norm] = result
    return result


def bulk_resolve(names: List[str]) -> Dict[str, Optional[str]]:
    """Resolve a list of company names to ticker symbols.

    Returns a dict mapping each input name to its resolved ticker (or None).
    """
    return {name: resolve_entity(name) for name in names}


def get_company_name(ticker: str) -> str:
    """Look up the canonical company name for a ticker symbol.

    Returns the ticker itself when no name is known.
    """
    return KNOWN_NAMES.get(ticker.upper(), ticker)


def seed_entity_resolution(max_symbols: int = 100) -> Dict[str, int]:
    """Seed the entity_resolution table from the universe registry.

    Seeds up to max_symbols tickers not yet present in the DB.
    Returns {seeded, skipped, errors}.
    """
    from api import db
    if not db.db_write_enabled():
        return {"seeded": 0, "skipped": 0, "errors": 0, "reason": "db_writes_disabled"}

    from api.universe_registry import BOOTSTRAP_TIER1

    try:
        existing_rows = db.safe_fetchall("SELECT ticker FROM entity_resolution") or []
        existing = {str(r[0]) for r in existing_rows}
    except Exception:
        existing = set()

    to_seed = [t for t in BOOTSTRAP_TIER1 if t not in existing][:max_symbols]
    seeded = skipped = errors = 0

    for ticker in to_seed:
        try:
            canonical = KNOWN_NAMES.get(ticker, ticker)
            aliases_list: List[str] = [ticker]
            norm_canonical = _normalize(canonical)
            if norm_canonical and norm_canonical not in aliases_list:
                aliases_list.append(norm_canonical)

            db.safe_execute(
                """
                INSERT INTO entity_resolution (ticker, canonical_name, aliases)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (ticker) DO UPDATE SET
                    canonical_name = EXCLUDED.canonical_name,
                    aliases        = EXCLUDED.aliases,
                    updated_at     = now()
                """,
                (ticker, canonical, json.dumps(aliases_list)),
            )
            seeded += 1
        except Exception as exc:
            logger.debug("entity_resolver.seed_failed ticker=%s err=%s", ticker, exc)
            errors += 1

    invalidate_cache()
    logger.info(
        "entity_resolver.seed_complete seeded=%d skipped=%d errors=%d",
        seeded, skipped, errors,
    )
    return {"seeded": seeded, "skipped": skipped, "errors": errors}
