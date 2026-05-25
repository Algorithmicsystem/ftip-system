from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from api.assistant.phase3.common import clamp

from .quality import provider_capability_profile


def build_event_intelligence_overlay(
    *,
    symbol: str,
    company_name: Optional[str],
    as_of_date: dt.date,
    fundamentals_overlay: Optional[Dict[str, Any]] = None,
    news_overlay: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fundamentals_overlay = fundamentals_overlay or {}
    news_overlay = news_overlay or {}
    filing_backbone = fundamentals_overlay.get("filing_backbone") or {}
    provider_snapshot = fundamentals_overlay.get("provider_snapshot") or {}
    earnings_intel = provider_snapshot.get("alphavantage_earnings_intel") or {}
    recent_filings = list(filing_backbone.get("recent_filings") or [])
    recent_headlines = list(news_overlay.get("aggregated_headlines") or [])

    events: List[Dict[str, Any]] = []
    events.extend(_filing_events(recent_filings, as_of_date))
    events.extend(_earnings_events(earnings_intel, as_of_date))
    events.extend(_headline_events(recent_headlines, as_of_date, symbol, company_name))

    events.sort(
        key=lambda item: (
            0 if item.get("days_from_as_of") is not None and item.get("days_from_as_of") >= 0 else 1,
            abs(int(item.get("days_from_as_of") or 9999)),
            -float(item.get("event_relevance") or 0.0),
        )
    )
    latest_events = [item for item in events if (item.get("days_from_as_of") or 9999) <= 0][:6]
    upcoming_events = [item for item in events if (item.get("days_from_as_of") or -9999) >= 0][:4]
    event_type_counts = dict(Counter(str(item.get("event_type") or "unknown") for item in events))

    event_risk_score = _mean_score(item.get("event_risk") for item in latest_events[:4]) or 0.0
    event_opportunity_score = _mean_score(item.get("event_opportunity") for item in latest_events[:4]) or 0.0
    event_relevance_score = _mean_score(item.get("event_relevance") for item in latest_events[:4] or events[:4]) or 0.0
    event_freshness_score = _mean_score(item.get("event_freshness_score") for item in latest_events[:4] or events[:4]) or 0.0

    filings_change_signal = _filings_change_signal(
        filing_backbone=filing_backbone,
        fundamentals_overlay=fundamentals_overlay,
    )
    estimate_revision_support = _safe_float(earnings_intel.get("estimate_revision_support"))
    source_strength_support, source_strength_penalty = _source_strength_scores(
        fundamentals_overlay=fundamentals_overlay,
        news_overlay=news_overlay,
    )
    premium_evidence_bonus = _premium_evidence_bonus(
        sources=list((fundamentals_overlay.get("meta") or {}).get("sources") or [])
        + list((news_overlay.get("meta") or {}).get("sources") or [])
    )
    evidence_recency_quality = _evidence_recency_quality(
        as_of_date=as_of_date,
        filing_backbone=filing_backbone,
        earnings_intel=earnings_intel,
        recent_headlines=recent_headlines,
    )
    catalyst_quality = clamp(
        (
            (event_relevance_score * 0.26)
            + (event_opportunity_score * 0.22)
            + (filings_change_signal * 0.18)
            + ((estimate_revision_support or 50.0) * 0.12)
            + (evidence_recency_quality * 0.12)
            + (source_strength_support * 0.10)
        ),
        0.0,
        100.0,
    )
    event_overhang_support_or_penalty = clamp(
        50.0
        + (event_opportunity_score - event_risk_score) * 0.28
        + (catalyst_quality - 50.0) * 0.18
        + (evidence_recency_quality - 50.0) * 0.12
        - max(source_strength_penalty - 50.0, 0.0) * 0.1
        + premium_evidence_bonus * 0.6,
        0.0,
        100.0,
    )
    freshness_label = _event_freshness_label(latest_events, upcoming_events)
    relevance_label = _event_relevance_label(event_relevance_score, event_risk_score, event_opportunity_score)
    coverage_score = min(len(events) / 8.0, 1.0)
    sources_used = sorted(
        {
            source
            for source in (
                list((fundamentals_overlay.get("meta") or {}).get("sources") or [])
                + list((news_overlay.get("meta") or {}).get("sources") or [])
            )
            if source
        }
    )
    confidence = _mean_score(
        [
            _safe_float((fundamentals_overlay.get("meta") or {}).get("confidence")),
            _safe_float((news_overlay.get("meta") or {}).get("confidence")),
        ]
    )
    note_parts = [
        f"{len(events)} structured event(s) were classified across filings, earnings, and headlines."
        if events
        else "No structured event evidence was classified from the available filings, earnings, and headline stack.",
        f"Freshness is {freshness_label} and relevance is {relevance_label}.",
        f"Source strength support is {source_strength_support:.1f} / 100 with penalty {source_strength_penalty:.1f} / 100.",
    ]
    if premium_evidence_bonus > 0:
        note_parts.append(
            f"Premium-evidence bonus is {premium_evidence_bonus:.1f} because the source stack includes higher-grade connectors."
        )
    return {
        "events": events[:16],
        "latest_events": latest_events,
        "upcoming_events": upcoming_events,
        "event_type_counts": event_type_counts,
        "event_freshness": freshness_label,
        "event_relevance": relevance_label,
        "event_risk_score": round(event_risk_score, 2),
        "event_opportunity_score": round(event_opportunity_score, 2),
        "event_overhang_support_or_penalty": round(event_overhang_support_or_penalty, 2),
        "filings_change_signal": round(filings_change_signal, 2),
        "catalyst_quality": round(catalyst_quality, 2),
        "estimate_revision_support": round(estimate_revision_support, 2) if estimate_revision_support is not None else None,
        "source_strength_support": round(source_strength_support, 2),
        "source_strength_penalty": round(source_strength_penalty, 2),
        "premium_evidence_bonus": round(premium_evidence_bonus, 2),
        "evidence_recency_quality": round(evidence_recency_quality, 2),
        "major_event_titles": [item.get("summary") for item in latest_events[:5] if item.get("summary")],
        "days_to_next_event": _first_days(upcoming_events, positive_only=True),
        "days_since_last_major_event": _first_days(latest_events, positive_only=False),
        "source_confidence": round(confidence or 0.0, 2),
        "coverage_note": " ".join(note_parts),
        "provenance": {
            "sources_used": sources_used,
            "fallback_used": bool((news_overlay.get("meta") or {}).get("fallback_used"))
            or bool((fundamentals_overlay.get("meta") or {}).get("fallback_used")),
            "fallback_source": sorted(
                {
                    source
                    for source in (
                        list((news_overlay.get("meta") or {}).get("fallback_source") or [])
                        + list((fundamentals_overlay.get("meta") or {}).get("fallback_source") or [])
                    )
                    if source
                }
            ),
            "confidence": confidence,
            "notes": note_parts,
        },
        "meta": {
            "sources": sources_used,
            "coverage_score": round(coverage_score, 2),
            "status": "fresh" if freshness_label == "fresh" else "mixed" if events else "limited",
            "confidence": round(confidence or 0.0, 2),
            "data_as_of": (
                latest_events[0].get("event_date") if latest_events else filing_backbone.get("latest_filing_date")
            ),
            "data_quality_note": " ".join(note_parts),
            "source_strength_summary": (
                f"Event intelligence uses {', '.join(sources_used) or 'no live sources'} with "
                f"support {source_strength_support:.1f} / 100 and premium bonus {premium_evidence_bonus:.1f}."
            ),
        },
    }


def _filing_events(recent_filings: List[Dict[str, Any]], as_of_date: dt.date) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for item in recent_filings[:8]:
        event_date = _parse_date(item.get("filing_date"))
        if event_date is None:
            continue
        form = str(item.get("form") or "filing").upper()
        risk = 66.0 if form.endswith("/A") or form in {"8-K", "6-K"} else 38.0
        opportunity = 58.0 if form in {"10-Q", "10-K", "20-F", "40-F"} else 42.0
        events.append(
            {
                "event_type": "filing",
                "event_date": event_date.isoformat(),
                "days_from_as_of": (event_date - as_of_date).days,
                "event_freshness": _freshness_label((as_of_date - event_date).days),
                "event_freshness_score": _freshness_score((as_of_date - event_date).days),
                "event_relevance": 84.0 if form in {"10-Q", "10-K", "20-F", "40-F"} else 68.0,
                "event_risk": risk,
                "event_opportunity": opportunity,
                "source_confidence": 82.0,
                "summary": f"{form} filing",
                "derived_from": ["sec_edgar", "fundamental_filing"],
            }
        )
    return events


def _earnings_events(earnings_intel: Dict[str, Any], as_of_date: dt.date) -> List[Dict[str, Any]]:
    latest = earnings_intel.get("latest_quarter") or {}
    event_date = latest.get("reported_date")
    if not isinstance(event_date, dt.date):
        return []
    surprise_pct = _safe_float(latest.get("surprise_pct")) or 0.0
    risk = clamp(50.0 - min(surprise_pct, 0.0) * 1.8, 0.0, 100.0)
    opportunity = clamp(50.0 + max(surprise_pct, 0.0) * 1.8, 0.0, 100.0)
    return [
        {
            "event_type": "earnings",
            "event_date": event_date.isoformat(),
            "days_from_as_of": (event_date - as_of_date).days,
            "event_freshness": _freshness_label((as_of_date - event_date).days),
            "event_freshness_score": _freshness_score((as_of_date - event_date).days),
            "event_relevance": 88.0,
            "event_risk": risk,
            "event_opportunity": opportunity,
            "source_confidence": 74.0,
            "summary": (
                f"Earnings {latest.get('surprise_direction') or 'update'} "
                f"with surprise {surprise_pct:.1f}%"
            ),
            "derived_from": ["alphavantage", "earnings_history"],
        }
    ]


def _headline_events(
    headlines: List[Dict[str, Any]],
    as_of_date: dt.date,
    symbol: str,
    company_name: Optional[str],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    name_token = str(company_name or "").lower()
    for item in headlines[:10]:
        published_at = item.get("published_at")
        if hasattr(published_at, "date"):
            event_date = published_at.date()
        else:
            event_date = _parse_date(published_at)
        if event_date is None:
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        lowered = title.lower()
        event_type = _headline_event_type(lowered)
        if symbol.lower() not in lowered and name_token and name_token not in lowered:
            relevance = 55.0
        else:
            relevance = 76.0 if event_type in {"guidance", "earnings", "filing"} else 68.0
        risk = _headline_risk(lowered)
        opportunity = _headline_opportunity(lowered)
        events.append(
            {
                "event_type": event_type,
                "event_date": event_date.isoformat(),
                "days_from_as_of": (event_date - as_of_date).days,
                "event_freshness": _freshness_label((as_of_date - event_date).days),
                "event_freshness_score": _freshness_score((as_of_date - event_date).days),
                "event_relevance": relevance,
                "event_risk": risk,
                "event_opportunity": opportunity,
                "source_confidence": _headline_source_confidence(item.get("source")),
                "summary": title,
                "derived_from": [str(item.get("source") or "headline"), "headline_news"],
            }
        )
    return events


def _headline_event_type(title: str) -> str:
    if any(token in title for token in ("earnings", "eps", "quarter results", "quarterly results")):
        return "earnings"
    if any(token in title for token in ("guidance", "forecast", "outlook")):
        return "guidance"
    if any(token in title for token in ("10-k", "10-q", "8-k", "filing", "sec")):
        return "filing"
    if any(token in title for token in ("spin-off", "merger", "acquisition", "buyback", "dividend", "offering")):
        return "corporate_action"
    if any(token in title for token in ("upgrade", "downgrade", "price target", "analyst")):
        return "analyst_revision"
    if any(token in title for token in ("ceo", "cfo", "management", "executive")):
        return "management_signal"
    if any(token in title for token in ("fed", "rates", "cpi", "tariff", "election", "policy")):
        return "macro_catalyst"
    return "general"


def _headline_risk(title: str) -> float:
    risk_tokens = ("miss", "cuts", "cut", "warning", "probe", "lawsuit", "downgrade", "weak", "fraud")
    return 72.0 if any(token in title for token in risk_tokens) else 42.0


def _headline_opportunity(title: str) -> float:
    positive_tokens = ("beat", "beats", "raise", "raised", "record", "strong", "upgrade", "wins", "growth")
    return 74.0 if any(token in title for token in positive_tokens) else 44.0


def _headline_source_confidence(source: Any) -> float:
    provider_name = str(source or "").split("|")[0]
    slot = provider_capability_profile(provider_name).get("connector_slot")
    if slot in {"premium_news_intel", "filings_estimates"}:
        return 74.0
    if slot in {"scraping_news_backup", "alt_data_events"}:
        return 58.0
    return 64.0


def _filings_change_signal(
    *,
    filing_backbone: Dict[str, Any],
    fundamentals_overlay: Dict[str, Any],
) -> float:
    filing_recency_days = _safe_float(filing_backbone.get("filing_recency_days"))
    growth = _safe_float((fundamentals_overlay.get("normalized_metrics") or {}).get("revenue_growth_yoy"))
    margin_stability = _safe_float(fundamentals_overlay.get("margin_stability"))
    score = 52.0
    if filing_recency_days is not None:
        score += max(min((120.0 - filing_recency_days) / 2.4, 18.0), -12.0)
    if growth is not None:
        score += max(min(growth * 75.0, 14.0), -12.0)
    if margin_stability is not None:
        score += max(min((margin_stability - 0.5) * 35.0, 10.0), -8.0)
    return clamp(score, 0.0, 100.0)


def _source_strength_scores(
    *,
    fundamentals_overlay: Dict[str, Any],
    news_overlay: Dict[str, Any],
) -> tuple[float, float]:
    fundamental_conf = _safe_float((fundamentals_overlay.get("meta") or {}).get("confidence")) or 42.0
    news_conf = _safe_float((news_overlay.get("meta") or {}).get("confidence")) or 38.0
    support = clamp((fundamental_conf * 0.56) + (news_conf * 0.44), 0.0, 100.0)
    penalty = clamp(
        100.0
        - support
        + (8.0 if (fundamentals_overlay.get("meta") or {}).get("fallback_used") else 0.0)
        + (8.0 if (news_overlay.get("meta") or {}).get("fallback_used") else 0.0),
        0.0,
        100.0,
    )
    return support, penalty


def _premium_evidence_bonus(*, sources: List[str]) -> float:
    premium_slots = {
        provider_capability_profile(source).get("connector_slot")
        for source in sources
        if source
    }
    count = len(
        {
            slot
            for slot in premium_slots
            if slot in {"premium_market_data", "premium_news_intel", "filings_estimates"}
        }
    )
    return float(min(count * 4.0, 12.0))


def _evidence_recency_quality(
    *,
    as_of_date: dt.date,
    filing_backbone: Dict[str, Any],
    earnings_intel: Dict[str, Any],
    recent_headlines: List[Dict[str, Any]],
) -> float:
    filing_date = _parse_date(filing_backbone.get("latest_filing_date"))
    earnings_date = _parse_date((earnings_intel.get("latest_quarter") or {}).get("reported_date"))
    headline_date = None
    if recent_headlines:
        headline_date = _parse_date(
            (recent_headlines[0].get("published_at").date() if hasattr(recent_headlines[0].get("published_at"), "date") else recent_headlines[0].get("published_at"))
        )
    ages = [
        max((as_of_date - item).days, 0)
        for item in [filing_date, earnings_date, headline_date]
        if isinstance(item, dt.date)
    ]
    if not ages:
        return 35.0
    average_age = sum(ages) / len(ages)
    return clamp(95.0 - average_age * 0.75, 0.0, 100.0)


def _event_freshness_label(latest_events: List[Dict[str, Any]], upcoming_events: List[Dict[str, Any]]) -> str:
    if upcoming_events and _first_days(upcoming_events, positive_only=True) is not None and _first_days(upcoming_events, positive_only=True) <= 7:
        return "imminent"
    if latest_events and _first_days(latest_events, positive_only=False) is not None and _first_days(latest_events, positive_only=False) <= 3:
        return "fresh"
    if latest_events:
        return "recent"
    return "limited"


def _event_relevance_label(event_relevance_score: float, event_risk_score: float, event_opportunity_score: float) -> str:
    if event_relevance_score >= 78.0 and max(event_risk_score, event_opportunity_score) >= 62.0:
        return "high"
    if event_relevance_score >= 62.0:
        return "medium"
    return "low"


def _freshness_label(age_days: int) -> str:
    if age_days <= 2:
        return "fresh"
    if age_days <= 10:
        return "recent"
    if age_days <= 35:
        return "aging"
    return "historical"


def _freshness_score(age_days: int) -> float:
    return clamp(95.0 - max(age_days, 0) * 2.2, 0.0, 100.0)


def _first_days(events: List[Dict[str, Any]], *, positive_only: bool) -> Optional[int]:
    for item in events:
        value = item.get("days_from_as_of")
        if value is None:
            continue
        try:
            numeric = int(value)
        except Exception:
            continue
        if positive_only and numeric >= 0:
            return numeric
        if not positive_only and numeric <= 0:
            return abs(numeric)
    return None


def _mean_score(values: Iterable[Any]) -> Optional[float]:
    defined = [float(value) for value in values if _safe_float(value) is not None]
    if not defined:
        return None
    return float(sum(defined) / len(defined))


def _parse_date(value: Any) -> Optional[dt.date]:
    if isinstance(value, dt.date):
        return value
    if hasattr(value, "date"):
        try:
            return value.date()
        except Exception:
            return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
