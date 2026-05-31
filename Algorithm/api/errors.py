"""Canonical error response helpers for the FTIP API.

Single source of truth for error envelope shape.  All routes should use these
helpers rather than hand-rolling ``{"error": ...}`` dicts, so that clients
always see a consistent structure.

Canonical envelope::

    {
        "error": {
            "type": "validation_error",
            "message": "symbol is required",
            "trace_id": "abc123"
        },
        "trace_id": "abc123"
    }

For routes that do not carry a trace_id (internal job routes, etc.) use
``simple_error()`` which omits the trace field.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse


def _new_trace() -> str:
    return uuid.uuid4().hex


def err_response(
    err_type: str,
    message: str,
    status_code: int = 400,
    *,
    trace_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """Return a JSONResponse with the canonical FTIP error envelope."""
    tid = trace_id or _new_trace()
    payload: Dict[str, Any] = {
        "error": {"type": err_type, "message": message, "trace_id": tid},
        "trace_id": tid,
    }
    if extra:
        payload.update(extra)
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers={"X-Trace-Id": tid},
    )


def simple_error(
    err_type: str,
    message: str,
    status_code: int = 400,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """Lightweight error response without trace overhead (internal/job routes)."""
    payload: Dict[str, Any] = {"error": err_type, "detail": message}
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


__all__ = ["err_response", "simple_error"]
