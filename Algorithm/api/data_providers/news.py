from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Dict, List

import requests

from api import config

from .errors import ProviderError
from .finnhub import fetch_company_news
from .gdelt import search_articles as search_gdelt_articles
from .gnews import search_news as search_gnews
from .newsapi import search_news as search_newsapi
from .symbols import canonical_symbol

logger = logging.getLogger(__name__)
_TITLE_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


class NewsProviderError(ProviderError):
    pass


def _rss_url_for_symbol(symbol: str) -> str:
    query = f"{symbol} stock"
    return f"https://news.google.com/rss/search?q={query}"  # noqa: S310 - controlled input


def _parse_rss(content: str) -> List[Dict[str, object]]:
    root = ET.fromstring(content)
    items = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub_date = item.findtext("pubDate") or ""
        if not title or not link or not pub_date:
            continue
        try:
            published_at = parsedate_to_datetime(pub_date).astimezone(dt.timezone.utc)
        except Exception:
            continue
        items.append({"title": title, "url": link, "published_at": published_at})
    return items


def fetch_news_items(
    symbol: str, from_ts: dt.datetime, to_ts: dt.datetime
) -> List[Dict[str, object]]:
    symbol = canonical_symbol(symbol)
    candidates: List[Dict[str, object]] = []
    errors: List[str] = []

    for provider_name, fetcher in (
        ("google_news_rss", lambda: _fetch_google_rss(symbol, from_ts, to_ts)),
        (
            "gnews",
            lambda: search_gnews(
                f'"{symbol}" stock',
                from_ts=from_ts,
                to_ts=to_ts,
                max_items=config.data_fabric_news_limit(),
            ),
        ),
        (
            "newsapi",
            lambda: search_newsapi(
                f'"{symbol}" stock',
                from_ts=from_ts,
                to_ts=to_ts,
                max_items=config.data_fabric_news_limit(),
            ),
        ),
        (
            "finnhub_news",
            lambda: fetch_company_news(symbol, from_ts.date(), to_ts.date()),
        ),
        (
            "gdelt",
            lambda: search_gdelt_articles(
                f'"{symbol}"',
                from_ts=from_ts,
                to_ts=to_ts,
                max_records=config.data_fabric_news_limit(),
            ),
        ),
    ):
        try:
            candidates.extend(fetcher())
        except ProviderError as exc:
            errors.append(f"{provider_name}:{exc.reason_detail}")
        except Exception as exc:
            errors.append(f"{provider_name}:{exc}")

    results = _dedupe_news(symbol, candidates, from_ts, to_ts)
    if not results and errors:
        raise NewsProviderError("PROVIDER_UNAVAILABLE", "; ".join(errors[:3]))

    logger.info(
        "news.fetch",
        extra={"symbol": symbol, "count": len(results), "errors": errors[:3]},
    )
    return results


def _fetch_google_rss(
    symbol: str, from_ts: dt.datetime, to_ts: dt.datetime
) -> List[Dict[str, object]]:
    url = _rss_url_for_symbol(symbol)
    try:
        resp = requests.get(url, timeout=config.data_fabric_timeout_seconds())
    except requests.RequestException as exc:
        raise NewsProviderError("PROVIDER_UNAVAILABLE", str(exc)) from exc

    if resp.status_code != 200:
        raise NewsProviderError(
            "PROVIDER_UNAVAILABLE", f"news RSS HTTP {resp.status_code}"
        )

    items = _parse_rss(resp.text)
    results: List[Dict[str, object]] = []
    for item in items:
        published_at = item["published_at"]
        if published_at < from_ts or published_at > to_ts:
            continue
        results.append(
            {
                "published_at": published_at,
                "source": "google_news_rss",
                "title": item["title"],
                "url": item["url"],
                "content_snippet": None,
            }
        )
    return results


def _dedupe_news(
    symbol: str,
    items: List[Dict[str, object]],
    from_ts: dt.datetime,
    to_ts: dt.datetime,
) -> List[Dict[str, object]]:
    deduped: Dict[str, Dict[str, object]] = {}
    for item in items:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        published_at = item.get("published_at")
        if not title or not url or not isinstance(published_at, dt.datetime):
            continue
        if published_at < from_ts or published_at > to_ts:
            continue
        normalized_title = _TITLE_NORMALIZE_RE.sub(" ", title.lower()).strip()
        key = url or normalized_title
        current = deduped.get(key)
        if current is None:
            deduped[key] = {
                "symbol": symbol,
                "published_at": published_at,
                "source": str(item.get("source") or "unknown"),
                "title": title,
                "url": url,
                "url_hash": hashlib.sha256(url.encode("utf-8")).hexdigest(),
                "content_snippet": item.get("content_snippet"),
            }
            continue
        existing_source = str(current.get("source") or "")
        new_source = str(item.get("source") or "")
        if new_source and new_source not in existing_source.split("|"):
            current["source"] = "|".join([part for part in [existing_source, new_source] if part])
        if not current.get("content_snippet") and item.get("content_snippet"):
            current["content_snippet"] = item.get("content_snippet")
    results = list(deduped.values())
    results.sort(key=lambda row: row.get("published_at"), reverse=True)
    return results
