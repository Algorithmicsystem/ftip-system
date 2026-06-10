"""Earnings transcript / management tone scraper.

Uses SEC 8-K filings (earnings press releases) as a proxy for management
confidence and computes a Management Confidence Score (MCS) 0-100.

Keyword scoring approach:
  positive keywords add +1.5 each (capped)
  negative keywords subtract -1.5 each (capped)
  normalized to 0-100 scale
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SEC_HEADERS = {
    "User-Agent": "FTIP-System contact@ftip.ai",
    "Accept-Encoding": "gzip, deflate",
}
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/{primary_doc}"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

POSITIVE_WORDS = {
    "record", "strong", "accelerating", "momentum", "confident", "exceeded",
    "outperform", "growth", "expansion", "beat", "raised", "increase",
    "profitable", "robust", "resilient", "innovative", "breakthrough",
    "improving", "positive", "solid", "opportunity", "demand", "strength",
}

NEGATIVE_WORDS = {
    "challenging", "headwinds", "uncertain", "cautious", "declining",
    "miss", "missed", "shortfall", "weaker", "decrease", "pressure",
    "difficult", "concern", "risk", "deteriorating", "slowdown", "tariff",
    "inflation", "recession", "restructuring", "impairment", "loss",
}


def _get_json(url: str, timeout: int = 15) -> Optional[Any]:
    try:
        import httpx
        resp = httpx.get(url, headers=_SEC_HEADERS, timeout=timeout, follow_redirects=True)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.debug("earnings_transcript.get_failed url=%s error=%s", url, exc)
    return None


def _get_text(url: str, timeout: int = 15) -> Optional[str]:
    try:
        import httpx
        resp = httpx.get(url, headers=_SEC_HEADERS, timeout=timeout, follow_redirects=True)
        if resp.status_code == 200:
            return resp.text
    except Exception as exc:
        logger.debug("earnings_transcript.get_text_failed url=%s error=%s", url, exc)
    return None


def _lookup_cik(symbol: str) -> Optional[str]:
    data = _get_json(_TICKERS_URL, timeout=20)
    if not data:
        return None
    sym_upper = symbol.upper()
    for entry in data.values():
        if isinstance(entry, dict) and entry.get("ticker", "").upper() == sym_upper:
            return str(entry["cik_str"]).zfill(10)
    return None


def _score_text(text: str) -> Dict[str, Any]:
    """Count positive and negative keyword hits in *text*."""
    lower = text.lower()
    pos_hits = sum(1 for w in POSITIVE_WORDS if w in lower)
    neg_hits = sum(1 for w in NEGATIVE_WORDS if w in lower)
    total = pos_hits + neg_hits
    if total == 0:
        return {"positive_hits": 0, "negative_hits": 0, "mcs_score": 50.0}
    # Clamp raw delta to ±20, then map to 0-100
    delta = min(max(pos_hits - neg_hits, -20), 20)
    mcs = round((delta + 20) / 40.0 * 100.0, 2)
    return {"positive_hits": pos_hits, "negative_hits": neg_hits, "mcs_score": mcs}


def fetch_transcript_sentiment(symbol: str, days_back: int = 90) -> Dict[str, Any]:
    """Fetch the most recent earnings 8-K filing text and score management tone.

    Returns:
      symbol, mcs_score (0-100), positive_hits, negative_hits,
      filing_date, source_url
    """
    cik = _lookup_cik(symbol)
    if not cik:
        logger.debug("earnings_transcript.cik_not_found symbol=%s", symbol)
        return _default_transcript_result(symbol)

    submissions = _get_json(_SUBMISSIONS_URL.format(cik=cik), timeout=20)
    if not submissions:
        return _default_transcript_result(symbol)

    cutoff = date.today() - timedelta(days=days_back)
    filings = (submissions.get("filings") or {}).get("recent") or {}
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    # Find most recent 8-K (earnings press release proxy)
    for i, form in enumerate(forms):
        if form not in ("8-K", "8-K/A"):
            continue
        try:
            filing_date = date.fromisoformat(dates[i])
        except Exception:
            continue
        if filing_date < cutoff:
            continue

        acc_no_dashes = accessions[i].replace("-", "")
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        source_url = _EDGAR_ARCHIVE.format(
            cik=cik, acc_no_dashes=acc_no_dashes, primary_doc=primary_doc
        )
        text = _get_text(source_url, timeout=20)
        if not text or len(text) < 200:
            continue

        # Strip HTML tags for clean keyword matching
        clean = re.sub(r"<[^>]+>", " ", text)
        score_data = _score_text(clean)
        return {
            "symbol": symbol,
            "mcs_score": score_data["mcs_score"],
            "positive_hits": score_data["positive_hits"],
            "negative_hits": score_data["negative_hits"],
            "filing_date": filing_date.isoformat(),
            "source_url": source_url,
        }

    return _default_transcript_result(symbol)


def _default_transcript_result(symbol: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "mcs_score": 50.0,
        "positive_hits": 0,
        "negative_hits": 0,
        "filing_date": None,
        "source_url": None,
    }
