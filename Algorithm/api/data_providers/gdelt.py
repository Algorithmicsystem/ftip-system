from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

import requests

from api import config

from .errors import ProviderUnavailable, SymbolNoData

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def search_articles(
    query: str,
    *,
    from_ts: dt.datetime,
    to_ts: dt.datetime,
    max_records: int = 15,
) -> List[Dict[str, object]]:
    if not config.gdelt_enabled():
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "GDELT is disabled")
    response = requests.get(
        BASE_URL,
        params={
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "sort": "DateDesc",
            "maxrecords": max(1, min(max_records, 50)),
            "startdatetime": _gdelt_datetime(from_ts),
            "enddatetime": _gdelt_datetime(to_ts),
        },
        timeout=config.data_fabric_timeout_seconds(),
    )
    if response.status_code != 200:
        raise ProviderUnavailable("PROVIDER_UNAVAILABLE", f"GDELT HTTP {response.status_code}")
    payload = response.json()
    articles = payload.get("articles") or []
    if not isinstance(articles, list):
        raise SymbolNoData("NO_DATA", "GDELT returned no articles")
    results: List[Dict[str, object]] = []
    for article in articles:
        published_at = _parse_datetime(article.get("seendate") or article.get("date"))
        if published_at is None:
            continue
        results.append(
            {
                "published_at": published_at,
                "source": "gdelt",
                "source_name": article.get("domain"),
                "title": article.get("title") or "",
                "url": article.get("url") or "",
                "content_snippet": article.get("excerpt"),
                "tone": _float_or_none(article.get("tone")),
                "language": article.get("language"),
                "domain": article.get("domain"),
            }
        )
    return [item for item in results if item.get("title") and item.get("url")]


def _gdelt_datetime(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y%m%d%H%M%S")


def _parse_datetime(value: Any) -> dt.datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y%m%d%H%M"):
        try:
            parsed = dt.datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(
            dt.timezone.utc
        )
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
