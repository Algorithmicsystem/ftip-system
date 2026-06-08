"""Phase 19.4: Usage logging middleware."""
from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api import db

logger = logging.getLogger(__name__)

_SKIP_PATHS = {"/health", "/db/health", "/version"}


class UsageLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000.0

        try:
            tenant_id: Optional[str] = None
            if hasattr(request.state, "tenant") and request.state.tenant:
                tenant_id = request.state.tenant.get("tenant_id")
            if tenant_id is None:
                for header in ("x-api-key", "x-ftip-api-key"):
                    val = request.headers.get(header)
                    if val:
                        tenant_id = f"anon:{val[:8]}"
                        break

            path = request.url.path
            method = request.method
            status_code = response.status_code

            symbol: Optional[str] = None
            path_parts = path.strip("/").split("/")
            for part in path_parts:
                if part.isupper() and 1 <= len(part) <= 5:
                    symbol = part
                    break

            if db.db_write_enabled():
                db.safe_execute(
                    """
                    INSERT INTO api_usage_log
                        (tenant_id, endpoint, method, status_code, response_time_ms, symbol, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (tenant_id, path, method, status_code, round(elapsed_ms, 2),
                     symbol, dt.datetime.utcnow()),
                )
        except Exception as exc:
            logger.debug("usage_middleware.log_failed error=%s", exc)

        try:
            from api.cloud.performance import perf_tracker
            is_error = response.status_code >= 400
            perf_tracker.record(path, elapsed_ms, is_error=is_error)
        except Exception:
            pass

        return response
