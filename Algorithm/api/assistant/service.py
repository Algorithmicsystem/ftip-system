from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException
from openai import APIError, APIStatusError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from api import config
from api.assistant import prompts
from api.assistant.storage import AssistantStorage, storage

logger = logging.getLogger(__name__)


def _build_client() -> OpenAI:
    api_key = config.openai_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="LLM API key not configured")
    return OpenAI(api_key=api_key, timeout=config.llm_timeout_seconds(), max_retries=config.llm_max_retries())


def _safe_completion(messages: List[ChatCompletionMessageParam]) -> Tuple[str, str, Dict[str, Optional[int]]]:
    try:
        client = _build_client()
        resp = client.chat.completions.create(
            model=config.llm_model(),
            messages=messages,
            max_tokens=400,
        )
        reply = resp.choices[0].message.content or ""
        usage = resp.usage
        usage_map = {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        }
        return reply, resp.model or config.llm_model(), usage_map
    except HTTPException:
        raise
    except APIStatusError as exc:
        logger.warning("LLM upstream error: %s", exc)
        raise HTTPException(status_code=502, detail="LLM upstream error")
    except APIError as exc:
        logger.warning("LLM API error: %s", exc)
        raise HTTPException(status_code=502, detail="LLM upstream error")
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM unexpected error: %s", exc)
        raise HTTPException(status_code=502, detail="LLM upstream error")


def _ensure_llm_enabled() -> None:
    if not config.llm_enabled():
        raise HTTPException(status_code=503, detail="Assistant is disabled (set FTIP_LLM_ENABLED=1 to enable)")


def _prepare_history(session_id: Optional[str], store: AssistantStorage) -> Tuple[str, List[Dict[str, str]]]:
    if session_id is None:
        new_session = store.create_session()
        return new_session, []

    session = store.get_session(session_id)
    if session is None:
        new_session = store.create_session()
        return new_session, []

    messages = store.get_messages(session_id=session_id)
    history: List[Dict[str, str]] = []
    for msg in messages:
        history.append({"role": msg["role"], "content": msg["content"]})
    return session_id, history


def chat_with_assistant(request: Dict[str, Any], store: AssistantStorage = storage) -> Dict[str, Any]:
    _ensure_llm_enabled()

    session_id = request.get("session_id")
    message = request.get("message") or ""
    context = request.get("context")

    sid, history = _prepare_history(session_id, store)
    if message:
        store.add_message(sid, "user", message)

    messages: List[ChatCompletionMessageParam] = prompts.build_chat_messages(history, message, context)
    reply, model_used, usage = _safe_completion(messages)

    store.add_message(
        sid,
        "assistant",
        reply,
        model=model_used,
        tokens_in=usage.get("prompt_tokens"),
        tokens_out=usage.get("completion_tokens"),
    )

    return {
        "session_id": sid,
        "reply": reply,
        "citations": [prompts.system_capabilities()],
    }


def explain_signal(
    payload: Dict[str, Any],
    *,
    signal_fetcher: Optional[Callable[[str, str, int], Any]] = None,
    store: AssistantStorage = storage,
) -> Dict[str, Any]:
    _ensure_llm_enabled()

    if signal_fetcher is None:
        from api import main as api_main  # late import to avoid circular

        signal_fetcher = api_main.compute_signal_for_symbol

    symbol = payload.get("symbol")
    as_of = payload.get("as_of")
    lookback = int(payload.get("lookback") or 252)

    result = signal_fetcher(symbol, as_of, lookback)
    summary = prompts.summarize_signal(result.model_dump()) if hasattr(result, "model_dump") else prompts.summarize_signal(result)

    history: List[Dict[str, str]] = []
    messages = prompts.build_chat_messages(history, f"Explain this signal: {summary}", None)
    reply, model_used, usage = _safe_completion(messages)

    session_id = payload.get("session_id") or store.create_session(metadata={"symbol": symbol, "as_of": as_of})
    store.add_message(session_id, "system", summary)
    store.add_message(session_id, "assistant", reply, model=model_used, tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"))
    store.save_artifact(session_id, "signal_explanation", {"symbol": symbol, "as_of": as_of, "lookback": lookback, "summary": summary})

    return {"session_id": session_id, "reply": reply, "citations": ["signal_summary"]}


def explain_backtest(
    payload: Dict[str, Any],
    *,
    backtest_runner: Optional[Callable[..., Any]] = None,
    store: AssistantStorage = storage,
) -> Dict[str, Any]:
    _ensure_llm_enabled()

    backtest_request_model = None
    if backtest_runner is None:
        from api import main as api_main

        backtest_runner = api_main.portfolio_backtest
        backtest_request_model = api_main.PortfolioBacktestRequest

    req_obj = backtest_request_model(**payload) if backtest_request_model else payload
    result = backtest_runner(req_obj)  # type: ignore[arg-type]
    summary = prompts.summarize_backtest(result.model_dump()) if hasattr(result, "model_dump") else prompts.summarize_backtest(result)

    messages = prompts.build_chat_messages([], f"Explain this backtest: {summary}", None)
    reply, model_used, usage = _safe_completion(messages)

    session_id = payload.get("session_id") or store.create_session(metadata={"kind": "backtest"})
    store.add_message(session_id, "system", summary)
    store.add_message(session_id, "assistant", reply, model=model_used, tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"))
    store.save_artifact(session_id, "backtest_explanation", {"summary": summary, "request": payload})

    return {"session_id": session_id, "reply": reply, "citations": ["backtest_summary"]}


def title_session(session_id: str, hint: Optional[str] = None, store: AssistantStorage = storage) -> Dict[str, Any]:
    _ensure_llm_enabled()

    history = store.get_messages(session_id=session_id)
    summary_text = hint or "Summarize this conversation with a short title."
    messages = prompts.build_chat_messages(history, summary_text, None)
    reply, model_used, usage = _safe_completion(messages)

    store.add_message(session_id, "assistant", reply, model=model_used, tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"))
    store.upsert_session_metadata(session_id, {"title": reply})
    return {"session_id": session_id, "title": reply}
