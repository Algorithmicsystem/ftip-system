"""Regression tests for Phase 14 — NextResearchItem structured follow-ups."""

from __future__ import annotations

from unittest.mock import patch


# ---------------------------------------------------------------------------
# 1. NextResearchItem model
# ---------------------------------------------------------------------------

def test_next_research_item_importable():
    from api.assistant.explanation import NextResearchItem
    assert NextResearchItem


def test_next_research_item_in_all():
    from api.assistant import explanation
    assert "NextResearchItem" in explanation.__all__


def test_next_research_item_fields():
    from api.assistant.explanation import NextResearchItem
    item = NextResearchItem(question="What is the revenue trend?",
                            category="fundamentals", priority="high")
    assert item.question == "What is the revenue trend?"
    assert item.category == "fundamentals"
    assert item.priority == "high"


def test_next_research_item_defaults():
    from api.assistant.explanation import NextResearchItem
    item = NextResearchItem(question="Check macro backdrop")
    assert item.category == "general"
    assert item.priority == "medium"


def test_next_research_item_model_dump():
    from api.assistant.explanation import NextResearchItem
    item = NextResearchItem(question="Watch RSI divergence",
                            category="technicals", priority="low")
    d = item.model_dump()
    assert d == {"question": "Watch RSI divergence",
                 "category": "technicals", "priority": "low"}


# ---------------------------------------------------------------------------
# 2. ExplanationPayload.followups is List[NextResearchItem]
# ---------------------------------------------------------------------------

def test_explanation_payload_followups_accepts_items():
    from api.assistant.explanation import ExplanationPayload, NextResearchItem
    item = NextResearchItem(question="Any upcoming earnings?",
                            category="catalyst", priority="high")
    p = ExplanationPayload(followups=[item])
    assert len(p.followups) == 1
    assert isinstance(p.followups[0], NextResearchItem)
    assert p.followups[0].category == "catalyst"


def test_explanation_payload_followups_default_empty():
    from api.assistant.explanation import ExplanationPayload
    p = ExplanationPayload()
    assert p.followups == []


# ---------------------------------------------------------------------------
# 3. NarrationPayload.followups coercion
# ---------------------------------------------------------------------------

def test_narration_payload_accepts_next_research_items():
    from api.assistant.narration import NarrationPayload
    from api.assistant.explanation import NextResearchItem
    item = NextResearchItem(question="Q1", category="macro", priority="medium")
    n = NarrationPayload(headline="h", summary="s", bullets=[], disclaimer="d",
                         followups=[item])
    assert len(n.followups) == 1
    assert isinstance(n.followups[0], NextResearchItem)


def test_narration_payload_coerces_string_followups():
    from api.assistant.narration import NarrationPayload
    from api.assistant.explanation import NextResearchItem
    n = NarrationPayload(headline="h", summary="s", bullets=[], disclaimer="d",
                         followups=["Check earnings growth", "Review sector rotation"])
    assert len(n.followups) == 2
    assert isinstance(n.followups[0], NextResearchItem)
    assert n.followups[0].question == "Check earnings growth"
    assert n.followups[0].category == "general"
    assert n.followups[0].priority == "medium"


def test_narration_payload_coerces_dict_followups():
    from api.assistant.narration import NarrationPayload
    from api.assistant.explanation import NextResearchItem
    n = NarrationPayload(
        headline="h", summary="s", bullets=[], disclaimer="d",
        followups=[{"question": "Assess rate risk", "category": "macro", "priority": "high"}],
    )
    assert isinstance(n.followups[0], NextResearchItem)
    assert n.followups[0].question == "Assess rate risk"
    assert n.followups[0].category == "macro"
    assert n.followups[0].priority == "high"


def test_narration_payload_followups_default_empty():
    from api.assistant.narration import NarrationPayload
    n = NarrationPayload(headline="h", summary="s", bullets=[], disclaimer="d")
    assert n.followups == []


# ---------------------------------------------------------------------------
# 4. build_explanation_payload overlay
# ---------------------------------------------------------------------------

def test_build_explanation_payload_overlays_next_research_items():
    from api.assistant.explanation import build_explanation_payload, NextResearchItem
    from api.assistant.narration import NarrationPayload
    narration = NarrationPayload(
        headline="h", summary="s", bullets=[], disclaimer="d",
        followups=[NextResearchItem(question="Any catalyst risk?",
                                   category="risk", priority="high")],
    )
    payload = build_explanation_payload({}, narration=narration)
    assert len(payload.followups) == 1
    assert isinstance(payload.followups[0], NextResearchItem)
    assert payload.followups[0].question == "Any catalyst risk?"
    assert payload.followups[0].category == "risk"


def test_build_explanation_payload_coerces_string_followups_from_dict_narration():
    from api.assistant.explanation import build_explanation_payload, NextResearchItem
    narration_dict = {
        "headline": "h", "summary": "s", "bullets": [], "disclaimer": "",
        "followups": ["Plain string question"],
    }
    payload = build_explanation_payload({}, narration=narration_dict)
    assert len(payload.followups) == 1
    assert isinstance(payload.followups[0], NextResearchItem)
    assert payload.followups[0].question == "Plain string question"


def test_build_explanation_payload_no_narration_has_empty_followups():
    from api.assistant.explanation import build_explanation_payload
    payload = build_explanation_payload({})
    assert payload.followups == []


# ---------------------------------------------------------------------------
# 5. _sanitize_output preserves structure
# ---------------------------------------------------------------------------

def test_sanitize_output_sanitizes_followup_question():
    from api.assistant.narration import NarrationPayload, _sanitize_output
    from api.assistant.explanation import NextResearchItem
    model = NarrationPayload(
        headline="h", summary="s", bullets=[], disclaimer="d",
        followups=[NextResearchItem(question="Growth at 42 percent", category="fundamentals", priority="medium")],
    )
    sanitized = _sanitize_output(model, allowed_numbers={"42"})
    assert sanitized.followups[0].question == "Growth at 42 percent"


def test_sanitize_output_redacts_unlisted_numbers_in_question():
    from api.assistant.narration import NarrationPayload, _sanitize_output
    from api.assistant.explanation import NextResearchItem
    model = NarrationPayload(
        headline="h", summary="s", bullets=[], disclaimer="d",
        followups=[NextResearchItem(question="Growth at 99 percent", category="fundamentals", priority="high")],
    )
    sanitized = _sanitize_output(model, allowed_numbers=set())
    assert "99" not in sanitized.followups[0].question
    assert sanitized.followups[0].category == "fundamentals"
    assert sanitized.followups[0].priority == "high"
