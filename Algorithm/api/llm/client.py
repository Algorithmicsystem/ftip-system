from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException
from openai import APIError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from api import config

logger = logging.getLogger(__name__)


def _missing_key_exc(trace_id: Optional[str]) -> HTTPException:
    detail = {
        "message": "OpenAI API key not configured. Set OPENAI_API_KEY (legacy OpenAI_ftip-system is supported as a fallback).",
        "trace_id": trace_id,
    }
    return HTTPException(status_code=503, detail=detail)


class LLMClient:
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id
        api_key = config.openai_api_key()
        if not api_key:
            raise _missing_key_exc(trace_id)
        self._client = OpenAI(
            api_key=api_key,
            timeout=config.llm_timeout_seconds(),
            max_retries=config.llm_max_retries(),
        )

    def complete_chat(
        self,
        messages: List[ChatCompletionMessageParam],
        *,
        max_tokens: int = None,
        model: Optional[str] = None,
        temperature: float = None,
    ) -> Tuple[str, str, Dict[str, Optional[int]]]:
        try:
            resp = self._client.chat.completions.create(
                model=model or config.llm_model(),
                messages=messages,
                max_tokens=(
                    max_tokens if max_tokens is not None else config.llm_max_tokens()
                ),
                temperature=(
                    temperature if temperature is not None else config.llm_temperature()
                ),
                response_format={"type": "text"},
            )
            reply = resp.choices[0].message.content or ""
            usage = resp.usage
            usage_map = {
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
            }
            logger.info(
                "narrator.llm.completed",
                extra={
                    "trace_id": self.trace_id,
                    "model": resp.model,
                    "prompt_tokens": usage_map.get("prompt_tokens"),
                    "completion_tokens": usage_map.get("completion_tokens"),
                },
            )
            return reply, resp.model or model or config.llm_model(), usage_map
        except HTTPException:
            raise
        except (APIStatusError, APIError, APITimeoutError) as exc:
            logger.warning(
                "narrator.llm.upstream",
                extra={"trace_id": self.trace_id, "error": str(exc)},
            )
            raise HTTPException(status_code=502, detail="LLM upstream error")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "narrator.llm.unexpected",
                extra={"trace_id": self.trace_id, "error": str(exc)},
            )
            raise HTTPException(status_code=502, detail="LLM upstream error")


def complete_chat(
    messages: List[ChatCompletionMessageParam],
    *,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    trace_id: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Optional[int]]]:
    client = LLMClient(trace_id)
    return client.complete_chat(
        messages, max_tokens=max_tokens, model=model, temperature=temperature
    )


__all__ = ["LLMClient", "complete_chat", "_missing_key_exc"]
