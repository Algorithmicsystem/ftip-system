"""Unified ExplanationPayload — single struct for all signal explanations.

Replaces the four competing explanation shapes in the codebase:
  - NarrationPayload          (LLM-generated: headline/summary/bullets/…)
  - _why_this_signal()        (deterministic: drivers, warnings, codes)
  - _historical_evidence_payload()  (calibration evidence)
  - explanation_summary: str  (bare string in AxiomArtifact)

Usage::

    # Deterministic-only (no LLM)
    payload = build_explanation_payload(signal_external_dict)

    # With LLM narration overlay
    narration = narrate_payload(...)
    payload = build_explanation_payload(signal_external_dict, narration=narration)

    # JSON for API response
    return payload.model_dump(exclude_none=True)
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class DriverItem(BaseModel):
    feature: str
    description: str
    value: Optional[float] = None
    direction: str          # "supports" | "opposes"
    strength_label: str     # "strong" | "moderate" | "weak"


class NextResearchItem(BaseModel):
    """A structured follow-up research question produced by the narrator."""
    question: str
    category: str = "general"  # fundamentals | technicals | macro | risk | catalyst | general
    priority: str = "medium"   # high | medium | low


# ---------------------------------------------------------------------------
# Main struct
# ---------------------------------------------------------------------------

class ExplanationPayload(BaseModel):
    # --- Deterministic fields (no LLM) ---
    top_drivers: List[DriverItem] = []
    reason_codes: List[str] = []
    regime_context: Optional[str] = None
    signal_age_days: Optional[int] = None
    staleness_label: Optional[str] = None   # fresh | aging | stale | expired
    staleness_score: Optional[float] = None # 0 (fresh) – 100 (very stale)
    confidence_modifiers: List[str] = []
    data_warnings: List[str] = []
    source_table: Optional[str] = None

    # --- LLM-overlay fields (populated by narrate_payload) ---
    headline: Optional[str] = None
    summary: Optional[str] = None
    bullets: List[str] = []
    disclaimer: str = ""
    followups: List[NextResearchItem] = []


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

def _coerce_followup(item: Any) -> NextResearchItem:
    """Convert a string, dict, or NextResearchItem into a NextResearchItem."""
    if isinstance(item, NextResearchItem):
        return item
    if isinstance(item, dict):
        return NextResearchItem(**item)
    return NextResearchItem(question=str(item))


# ---------------------------------------------------------------------------
# Staleness helpers
# ---------------------------------------------------------------------------

def _staleness_label(age_days: int) -> str:
    if age_days <= 2:
        return "fresh"
    if age_days <= 5:
        return "aging"
    if age_days <= 10:
        return "stale"
    return "expired"


def _staleness_score(age_days: int) -> float:
    """0 = fresh, 100 = 10+ days old."""
    return round(min(age_days / 10.0 * 100.0, 100.0), 1)


def _compute_staleness(as_of_str: Optional[str]) -> tuple:
    """Return (age_days, staleness_label, staleness_score) or (None, None, None)."""
    if not as_of_str:
        return None, None, None
    try:
        as_of = dt.date.fromisoformat(str(as_of_str)[:10])
    except (ValueError, TypeError):
        return None, None, None
    age = (dt.date.today() - as_of).days
    age = max(age, 0)
    return age, _staleness_label(age), _staleness_score(age)


# ---------------------------------------------------------------------------
# Driver merging
# ---------------------------------------------------------------------------

def _merge_drivers(
    evidence_for: List[Dict[str, Any]],
    evidence_against: List[Dict[str, Any]],
) -> List[DriverItem]:
    """Merge evidence_for / evidence_against into a unified DriverItem list."""
    items: List[DriverItem] = []
    for item in evidence_for:
        items.append(DriverItem(
            feature=item.get("feature", ""),
            description=item.get("description", ""),
            value=item.get("value"),
            direction="supports",
            strength_label=item.get("strength_label", "weak"),
        ))
    for item in evidence_against:
        items.append(DriverItem(
            feature=item.get("feature", ""),
            description=item.get("description", ""),
            value=item.get("value"),
            direction="opposes",
            strength_label=item.get("strength_label", "weak"),
        ))
    # Sort: strong supporters first, then weak supporters, then opposing
    _order = {"strong": 0, "moderate": 1, "weak": 2}
    items.sort(key=lambda d: (0 if d.direction == "supports" else 1,
                               _order.get(d.strength_label, 3)))
    return items[:8]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_explanation_payload(
    signal: Dict[str, Any],
    *,
    narration: Optional[Any] = None,
    data_warnings: Optional[List[str]] = None,
) -> ExplanationPayload:
    """Build a unified ExplanationPayload from a signal external_payload dict.

    Args:
        signal: Dict from SignalResponse.external_payload() — must contain
                evidence_for, evidence_against, reason_codes, as_of, etc.
        narration: Optional NarrationPayload from narrate_payload(). When
                   provided, headline/summary/bullets/disclaimer/followups
                   are overlaid from it.
        data_warnings: Optional additional data-quality warnings to include.
    """
    evidence_for = list(signal.get("evidence_for") or [])
    evidence_against = list(signal.get("evidence_against") or [])
    top_drivers = _merge_drivers(evidence_for, evidence_against)

    reason_codes = list(signal.get("reason_codes") or [])
    confidence_modifiers = list(signal.get("adjusted_confidence_notes") or [])
    regime_context = signal.get("regime")
    source_table = signal.get("source_table")

    as_of = signal.get("as_of")
    age_days, stal_label, stal_score = _compute_staleness(as_of)

    payload = ExplanationPayload(
        top_drivers=top_drivers,
        reason_codes=reason_codes,
        regime_context=regime_context,
        signal_age_days=age_days,
        staleness_label=stal_label,
        staleness_score=stal_score,
        confidence_modifiers=confidence_modifiers,
        data_warnings=list(data_warnings or []),
        source_table=source_table,
    )

    # LLM overlay
    if narration is not None:
        if isinstance(narration, dict):
            payload.headline = narration.get("headline")
            payload.summary = narration.get("summary")
            payload.bullets = list(narration.get("bullets") or [])
            payload.disclaimer = str(narration.get("disclaimer") or "")
            raw_followups = list(narration.get("followups") or [])
        else:
            payload.headline = getattr(narration, "headline", None)
            payload.summary = getattr(narration, "summary", None)
            payload.bullets = list(getattr(narration, "bullets", None) or [])
            payload.disclaimer = str(getattr(narration, "disclaimer", "") or "")
            raw_followups = list(getattr(narration, "followups", None) or [])
        payload.followups = [_coerce_followup(f) for f in raw_followups]

    return payload


__all__ = [
    "DriverItem",
    "NextResearchItem",
    "ExplanationPayload",
    "build_explanation_payload",
    "compute_signal_staleness",
]


def compute_signal_staleness(as_of_str: Optional[str]) -> Dict[str, Any]:
    """Return staleness dict for standalone use (e.g. by external_payload)."""
    age_days, label, score = _compute_staleness(as_of_str)
    return {
        "signal_age_days": age_days,
        "staleness_label": label,
        "staleness_score": score,
    }
