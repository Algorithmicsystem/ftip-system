from __future__ import annotations
from typing import Any, Dict, List, Optional
from api.assistant.phase3.common import clamp

def compute_nss(news_data: dict) -> float:
    """Narrative Surprise Score (0-100). Higher = more novel, surprising narrative event."""
    novelty_score = news_data.get("novelty_score")
    baseline_similarity = news_data.get("baseline_similarity")  # 0-1, higher = more similar to recent news
    sentiment_score = news_data.get("sentiment_score")  # 0-100, 50=neutral
    entity_relevance = news_data.get("entity_relevance")  # 0-1, 1=company as primary subject

    if novelty_score is None and sentiment_score is None:
        return 0.0

    nov = clamp(float(novelty_score or 50.0), 0.0, 100.0)
    similarity = clamp(float(baseline_similarity or 0.5), 0.0, 1.0)
    sent_mag = clamp(abs(float(sentiment_score or 50.0) - 50.0) * 2.0, 0.0, 100.0)  # 0-100
    relevance = clamp(float(entity_relevance or 0.5), 0.0, 1.0)

    # NSS = novelty × (1-similarity) × sentiment_magnitude × entity_relevance, normalized
    raw = (nov / 100.0) * (1.0 - similarity) * (sent_mag / 100.0) * relevance * 100.0
    # Rescale: max raw ≈ 100 * 1 * 100 * 1 / 100 = 100 but typical range is 0-30 so scale up
    nss = clamp(raw * 3.0, 0.0, 100.0)
    return round(nss, 2)


def compute_nms(news_history: list) -> float:
    """Narrative Momentum Score (0-100). Measures persistence of directional sentiment."""
    if not news_history:
        return 50.0

    sentiments = []
    for item in news_history:
        s = item.get("sentiment_score") if isinstance(item, dict) else None
        if s is not None:
            sentiments.append(float(s))

    if not sentiments:
        return 50.0

    def _dir_score(window: list) -> float:
        if not window:
            return 50.0
        avg = sum(window) / len(window)
        return clamp((avg - 50.0) / 50.0 * 50.0 + 50.0, 0.0, 100.0)

    last_5 = sentiments[-5:] if len(sentiments) >= 5 else sentiments
    last_21 = sentiments[-21:] if len(sentiments) >= 21 else sentiments

    d5_score = _dir_score(last_5)

    # 21d trend: compare first half vs second half
    if len(last_21) >= 10:
        half = len(last_21) // 2
        first_avg = sum(last_21[:half]) / half
        second_avg = sum(last_21[half:]) / (len(last_21) - half)
        delta = second_avg - first_avg
        trend_score = clamp((delta / 25.0) * 50.0 + 50.0, 0.0, 100.0)
    else:
        trend_score = _dir_score(last_21)

    # Novelty shift: detect if recent 3 days diverge from prior 5 days
    inflection = detect_narrative_inflection(news_history)
    novelty_shift_score = 75.0 if inflection["inflection_detected"] else 50.0

    nms = (d5_score * 0.50 + trend_score * 0.30 + novelty_shift_score * 0.20)
    return round(clamp(nms, 0.0, 100.0), 2)


def detect_narrative_inflection(news_history: list) -> dict:
    """Detect narrative inflection by comparing 5-day rolling sentiment windows."""
    sentiments = []
    for item in news_history:
        s = item.get("sentiment_score") if isinstance(item, dict) else None
        if s is not None:
            sentiments.append(float(s))

    if len(sentiments) < 10:
        return {
            "inflection_detected": False,
            "direction_change": "none",
            "confidence": 0.0,
            "days_since_change": 0,
        }

    recent_5 = sentiments[-5:]
    prior_5 = sentiments[-10:-5]
    recent_avg = sum(recent_5) / len(recent_5)
    prior_avg = sum(prior_5) / len(prior_5)
    delta = recent_avg - prior_avg

    inflection = abs(delta) > 25.0
    if delta > 25.0:
        direction_change = "negative_to_positive"
    elif delta < -25.0:
        direction_change = "positive_to_negative"
    else:
        direction_change = "none"

    confidence = clamp(abs(delta) / 50.0, 0.0, 1.0) if inflection else 0.0

    return {
        "inflection_detected": inflection,
        "direction_change": direction_change,
        "confidence": round(confidence, 4),
        "days_since_change": 5 if inflection else 0,
    }
