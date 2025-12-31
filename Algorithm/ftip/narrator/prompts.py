from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

SYSTEM_PERSONA = (
    "You are the FTIP Narrator. Provide concise, neutral explanations using only the supplied context. "
    "Responses are informational only and must not be financial advice. Do not claim guaranteed profits or share secrets such as keys or environment variables."
)
DISCLAIMER = "Informational only; not financial advice."

MAX_PROMPT_CHARS = 12000


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:  # pragma: no cover - defensive
        return "{}"


def build_context_packet(
    *,
    question: str,
    symbols: List[Dict[str, Any]],
    strategy_graph: Optional[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "question": question,
        "symbols": symbols,
        "strategy_graph": strategy_graph,
        "meta": meta,
    }


def build_ask_prompt(question: str, context: Dict[str, Any], *, safe_mode: bool = True) -> List[Dict[str, str]]:
    system_lines = [SYSTEM_PERSONA]
    if safe_mode:
        system_lines.append("Always remind the user this is informational only and avoid promises of returns.")
    system_prompt = " ".join(system_lines)

    user_content = (
        "Answer the user's question using only the provided context. "
        "If data is missing, acknowledge gaps and stay conservative.\n\n"
        f"Context: {_safe_json(context)}\n\nQuestion: {question}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_explain_prompt(context: Dict[str, Any], *, safe_mode: bool = True) -> List[Dict[str, str]]:
    system_lines = [SYSTEM_PERSONA]
    if safe_mode:
        system_lines.append(
            "Explain the signal, drivers, risks, and invalidation criteria without offering financial advice or guarantees."
        )
    system_prompt = " ".join(system_lines)

    user_content = (
        "Using the context packet, summarize the signal, key drivers, risks, and what would invalidate the thesis."
        " Keep answers succinct (4-8 bullets) and reiterate the informational-only stance.\n\n"
        f"Context: {_safe_json(context)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_strategy_graph_prompt(graph: Dict[str, Any], *, safe_mode: bool = True) -> List[Dict[str, str]]:
    system_lines = [
        "You are the FTIP analyst narrator.",
        "Explain the strategy graph transitions, highlight signal persistence, and summarize notable regime shifts.",
        "Ground the explanation strictly in the provided graph statistics.",
    ]
    if safe_mode:
        system_lines.append("Keep it concise (6-10 sentences) and remind that this is informational only.")
    system_prompt = " ".join(system_lines)

    serialized = _safe_json(graph)
    if len(serialized) > MAX_PROMPT_CHARS:
        serialized = serialized[: MAX_PROMPT_CHARS - 3] + "..."

    user_content = (
        "Summarize the strategy graph using only this JSON. "
        "Discuss node frequency, most common transitions, and recent signals with confidence context.\n\n"
        f"Graph: {serialized}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def summarize_context_used(context: Dict[str, Any]) -> Dict[str, Any]:
    symbols = [item.get("symbol") for item in context.get("symbols", []) if item.get("symbol")]
    strategy_graph = context.get("strategy_graph") or {}
    return {
        "symbols": symbols,
        "strategy_graph_available": bool(strategy_graph),
        "meta": context.get("meta") or {},
    }


__all__ = [
    "build_ask_prompt",
    "build_context_packet",
    "build_explain_prompt",
    "build_strategy_graph_prompt",
    "summarize_context_used",
    "DISCLAIMER",
    "MAX_PROMPT_CHARS",
    "SYSTEM_PERSONA",
]
