from __future__ import annotations

import logging
import time
import uuid
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Request

from api import config
from api.assistant import service
from api.assistant.models import (
    ChatRequest,
    ChatResponse,
    ExplainBacktestRequest,
    ExplainSignalRequest,
    SessionResponse,
    TitleSessionRequest,
    TitleSessionResponse,
)
from api.assistant.storage import storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["assistant"])


class MemoryRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.hits: Dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        bucket = self.hits.get(key, [])
        bucket = [ts for ts in bucket if ts > now - self.window_seconds]
        if len(bucket) >= self.max_requests:
            self.hits[key] = bucket
            return False
        bucket.append(now)
        self.hits[key] = bucket
        return True


_limiter = MemoryRateLimiter(config.assistant_rate_limit())


def rate_limit(request: Request) -> None:
    if config.assistant_rate_limit() <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    if not _limiter.allow(ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")


def request_id_dependency(request: Request) -> str:
    req_id = str(uuid.uuid4())
    request.state.request_id = req_id
    return req_id


def ensure_enabled() -> None:
    if not config.llm_enabled():
        raise HTTPException(status_code=503, detail="Assistant is disabled (set FTIP_LLM_ENABLED=1 to enable)")


@router.get("/health")
def assistant_health() -> Dict[str, bool]:
    return {
        "status": "ok",
        "llm_enabled": config.llm_enabled(),
        "db_enabled": config.db_enabled(),
    }


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    request_id: str = Depends(request_id_dependency),
    __: None = Depends(rate_limit),
):
    ensure_enabled()
    result = service.chat_with_assistant(payload.model_dump())
    logger.info("assistant.chat", extra={"request_id": request_id})
    return ChatResponse(**result)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def session_details(session_id: str, __: None = Depends(rate_limit)):
    ensure_enabled()
    session = storage.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = storage.get_messages(session_id=session_id)
    return SessionResponse(session_id=session_id, title=session.get("title"), metadata=session.get("metadata"), messages=messages)


@router.post("/explain/signal")
async def explain_signal_endpoint(
    payload: ExplainSignalRequest,
    request_id: str = Depends(request_id_dependency),
    __: None = Depends(rate_limit),
):
    ensure_enabled()
    response = service.explain_signal(payload.model_dump())
    logger.info("assistant.explain_signal", extra={"request_id": request_id})
    return response


@router.post("/explain/backtest")
async def explain_backtest_endpoint(
    payload: ExplainBacktestRequest,
    request_id: str = Depends(request_id_dependency),
    __: None = Depends(rate_limit),
):
    ensure_enabled()
    response = service.explain_backtest(payload.model_dump())
    logger.info("assistant.explain_backtest", extra={"request_id": request_id})
    return response


@router.post("/title_session", response_model=TitleSessionResponse)
async def title_session_endpoint(
    payload: TitleSessionRequest,
    request_id: str = Depends(request_id_dependency),
    __: None = Depends(rate_limit),
):
    ensure_enabled()
    result = service.title_session(payload.session_id, hint=payload.hint)
    logger.info("assistant.title_session", extra={"request_id": request_id})
    return TitleSessionResponse(**result)
