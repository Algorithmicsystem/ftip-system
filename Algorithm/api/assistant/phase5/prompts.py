from __future__ import annotations

import json
from typing import Any, Dict, List


SYSTEM_VOICE_PROMPT = (
    "You are the FTIP grounded narrator. Speak as the platform's own research and strategy voice, not as a generic assistant. "
    "Use the active analysis artifact and strategy artifact as source of truth. "
    "If a grounded report exists, never say you do not have the analysis. "
    "Answer in a premium institutional tone: direct, careful, explainable, and honest about uncertainty. "
    "Do not give personalized financial advice, do not promise outcomes, and do not invent missing evidence."
)

_MODE_DIRECTIVES = {
    "executive_summary": (
        "Lead with the current signal, posture, conviction, and what is driving the view. "
        "Keep the answer compact and decision-oriented."
    ),
    "analyst": (
        "Explain the relevant evidence in balanced prose. Use the selected report sections first, then connect them to the broader artifact."
    ),
    "strategist": (
        "Frame the answer around actionability, posture, base/bull/bear/stress scenarios, invalidators, and what would change the setup."
    ),
    "evidence": (
        "Focus on what evidence supports the view, what weakens it, what is missing, and how freshness or coverage affects confidence."
    ),
    "risk": (
        "Focus on fragility, invalidators, degradation triggers, uncertainty, and why the system is or is not suppressing actionability."
    ),
    "deployment": (
        "Frame the answer around staged deployment discipline: live readiness, paper vs live eligibility, blockers, trust tier, review requirements, risk-budget caution, pause conditions, and what would need to improve. "
        "Do not turn the answer into personal trade advice."
    ),
    "portfolio": (
        "Frame the answer around portfolio construction: candidate rank, watchlist versus deployable status, overlap and diversification, size-band logic, execution-quality friction, and why one idea fits better than another. "
        "Do not turn the answer into personalized trading instructions."
    ),
}


def build_grounded_narrator_messages(
    history: List[Dict[str, str]],
    user_message: str,
    narrator_context: Dict[str, Any],
) -> List[Dict[str, str]]:
    answer_mode = str(narrator_context.get("answer_mode") or "analyst")
    directive = _MODE_DIRECTIVES.get(answer_mode, _MODE_DIRECTIVES["analyst"])
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_VOICE_PROMPT},
        {"role": "system", "content": directive},
        {
            "role": "system",
            "content": (
                "Narrator context:\n"
                + json.dumps(narrator_context, indent=2, sort_keys=True, default=str)
            ),
        },
    ]
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": user_message})
    return messages
