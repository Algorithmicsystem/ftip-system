from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Set, Tuple

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from ftip.narrator import client as narrator_client


class NarrationPayload(BaseModel):
    headline: str
    summary: str
    bullets: List[str]
    disclaimer: str
    followups: List[str]


NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _numbers_in_payload(payload: Dict[str, Any]) -> Set[str]:
    serialized = json.dumps(payload, default=str)
    return set(NUMBER_PATTERN.findall(serialized))


def _sanitize_numbers(text: str, allowed: Set[str]) -> str:
    def _replace(match: re.Match) -> str:
        value = match.group(0)
        if value in allowed:
            return value
        return "not available"

    return NUMBER_PATTERN.sub(_replace, text)


def _sanitize_output(model: NarrationPayload, allowed_numbers: Set[str]) -> NarrationPayload:
    return NarrationPayload(
        headline=_sanitize_numbers(model.headline, allowed_numbers),
        summary=_sanitize_numbers(model.summary, allowed_numbers),
        bullets=[_sanitize_numbers(item, allowed_numbers) for item in model.bullets],
        disclaimer=_sanitize_numbers(model.disclaimer, allowed_numbers),
        followups=[_sanitize_numbers(item, allowed_numbers) for item in model.followups],
    )


def _build_prompt(payload: Dict[str, Any], user_message: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are the FTIP narrator. Use only the provided payload to answer. "
        "Only use numeric values present in the payload; if a number is missing say 'not available'. "
        "Return JSON with keys: headline, summary, bullets, disclaimer, followups."
    )
    user_prompt = (
        "Payload:\n"
        f"{json.dumps(payload, default=str)}\n\n"
        f"User message: {user_message}\n\n"
        "Respond with JSON only."
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def _parse_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def narrate_payload(payload: Dict[str, Any], user_message: str, *, trace_id: str | None = None) -> NarrationPayload:
    messages = _build_prompt(payload, user_message)
    reply, _model, _usage = narrator_client.complete_chat(
        messages,
        max_tokens=350,
        temperature=0.2,
        trace_id=trace_id,
    )
    try:
        parsed = _parse_json(reply)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=502, detail={"message": "Narrator response invalid JSON"}) from exc

    try:
        model = NarrationPayload(**parsed)
    except ValidationError as exc:
        raise HTTPException(status_code=502, detail={"message": "Narrator response schema invalid"}) from exc

    allowed_numbers = _numbers_in_payload(payload)
    return _sanitize_output(model, allowed_numbers)
