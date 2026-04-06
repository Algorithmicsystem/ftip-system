from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

import requests

from api import config

from .errors import ProviderUnavailable, SymbolNoData

BASE_URL = "https://gnews.io/api/v4/search"


def search_news(
    query: str,
    *,
    from_ts: dt.datetime,
    to_ts: dt.datetime,
    max_items: int = 10,
) -> List[Dict[str, object]]:
    api_key = config.gnews_api_key()
    if not api_key:
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "GNEWS_API_KEY not set")
    response = requests.get(
        BASE_URL,
        params={
            "q": query,
            "from": _iso_datetime(from_ts),
            "to": _iso_datetime(to_ts),
            "lang": "en",
            "expand": "content",
            "max": max(1, min(max_items, 50)),
            "apikey": api_key,
        },
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", f"GNews HTTP {response.status_code}")
    payload = response.json()
    articles = payload.get("articles") or []
    if not isinstance(articles, list):
        raise SymbolNoData("NO_DATA", "GNews returned no articles")
    results: List[Dict[str, object]] = []
    for article in articles:
        published_at = _parse_datetime(article.get("publishedAt"))
        if published_at is None:
            continue
        results.append(
            {
                "published_at": published_at,
                "source": "gnews",
                "source_name": ((article.get("source") or {}).get("name")),
                "title": article.get("title") or "",
                "url": article.get("url") or "",
                "content_snippet": article.get("description") or article.get("content"),
            }
        )
    return [item for item in results if item.get("title") and item.get("url")]


def _parse_datetime(value: Any) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
            dt.timezone.utc
        )
    except Exception:
        return None


def _iso_datetime(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()
