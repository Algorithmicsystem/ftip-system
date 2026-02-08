from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException
from openai import APIError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from api import config

logger = logging.getLogger("ftip.narrator.client")


class NarratorClient:
    def __init__(self, trace_id: Optional[str] = None) -> None:
        self.trace_id = trace_id
        api_key = config.openai_api_key()
        if not api_key:
            raise self._missing_key_exc(trace_id)
        self._client = OpenAI(
            api_key=api_key,
            timeout=config.llm_timeout_seconds(),
            max_retries=0,
        )
        self._retries = max(0, config.llm_max_retries())

    @staticmethod
    def _missing_key_exc(trace_id: Optional[str]) -> HTTPException:
        detail = {
            "message": "OpenAI API key not configured. Set OPENAI_API_KEY (legacy OpenAI_ftip-system supported).",
            "trace_id": trace_id,
        }
        return HTTPException(status_code=503, detail=detail)

    def _request_with_retries(self, **kwargs):
        last_exc: Optional[Exception] = None
        for attempt in range(self._retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except (APIStatusError, APIError, APITimeoutError) as exc:
                last_exc = exc
                backoff = min(2**attempt * 0.5, 5.0)
                logger.warning(
                    "narrator.llm.retry",
                    extra={
                        "trace_id": self.trace_id,
                        "attempt": attempt + 1,
                        "backoff_seconds": backoff,
                    },
                )
                time.sleep(backoff)
        if last_exc:
            logger.warning(
                "narrator.llm.failed",
                extra={"trace_id": self.trace_id, "error": str(last_exc)},
            )
        raise HTTPException(
            status_code=502,
            detail={"message": "LLM upstream error", "trace_id": self.trace_id},
        )

    def complete_chat(
        self,
        messages: List[ChatCompletionMessageParam],
        *,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, str, Dict[str, Optional[int]]]:
        try:
            resp = self._request_with_retries(
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
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "narrator.llm.unexpected",
                extra={"trace_id": self.trace_id, "error": str(exc)},
            )
            raise HTTPException(
                status_code=502,
                detail={"message": "LLM upstream error", "trace_id": self.trace_id},
            )


def complete_chat(
    messages: List[ChatCompletionMessageParam],
    *,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    trace_id: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Optional[int]]]:
    client = NarratorClient(trace_id)
    return client.complete_chat(
        messages,
        max_tokens=max_tokens,
        model=model,
        temperature=temperature,
    )


__all__ = ["NarratorClient", "complete_chat"]
