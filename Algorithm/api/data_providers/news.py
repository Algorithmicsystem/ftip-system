from __future__ import annotations

import datetime as dt
import hashlib
import logging
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Dict, List

import requests

from .bars import ProviderError
from .symbols import canonical_symbol

logger = logging.getLogger(__name__)


class NewsProviderError(ProviderError):
    pass


def _rss_url_for_symbol(symbol: str) -> str:
    query = f"{symbol} stock"
    return (
        f"https://news.google.com/rss/search?q={query}"  # noqa: S310 - controlled input
    )


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
    url = _rss_url_for_symbol(symbol)
    try:
        resp = requests.get(url, timeout=15)
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
        url_hash = hashlib.sha256(item["url"].encode("utf-8")).hexdigest()
        results.append(
            {
                "symbol": symbol,
                "published_at": published_at,
                "source": "google_news_rss",
                "title": item["title"],
                "url": item["url"],
                "url_hash": url_hash,
                "content_snippet": None,
            }
        )

    logger.info("news.fetch", extra={"symbol": symbol, "count": len(results)})
    return results
