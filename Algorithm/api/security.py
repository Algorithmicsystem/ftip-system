import json
import os
import time
import uuid
from collections import deque
from typing import Dict, List, Optional, Tuple

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from api.ops import metrics_tracker


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _parse_api_keys() -> List[str]:
    csv_keys = _env("FTIP_API_KEYS")
    json_keys = _env("FTIP_API_KEYS_JSON")

    keys: List[str] = []
    if csv_keys:
        keys.extend([k.strip() for k in csv_keys.split(",") if k.strip()])
    if json_keys:
        try:
            parsed = json.loads(json_keys)
            if isinstance(parsed, list):
                keys.extend([str(k).strip() for k in parsed if str(k).strip()])
        except Exception:
            pass
    return sorted(set(keys))


_ALLOWED_ORIGINS: Optional[List[str]] = None
_API_KEYS: Optional[List[str]] = None


def get_allowed_origins() -> List[str]:
    global _ALLOWED_ORIGINS
    if _ALLOWED_ORIGINS is None:
        raw = _env("FTIP_ALLOWED_ORIGINS", "") or ""
        _ALLOWED_ORIGINS = [o.strip() for o in raw.split(",") if o.strip()]
    return _ALLOWED_ORIGINS


def get_api_keys() -> List[str]:
    global _API_KEYS
    if _API_KEYS is None:
        _API_KEYS = _parse_api_keys()
    return _API_KEYS


def allow_public_docs() -> bool:
    return (_env("FTIP_PUBLIC_DOCS", "0") or "0") == "1"


class RateLimiter:
    def __init__(self, rpm: int = 60) -> None:
        self.rpm = max(1, int(rpm))
        self.window_seconds = 60
        self._hits: Dict[str, deque] = {}

    def check(self, key: str) -> Tuple[bool, Optional[int]]:
        now = time.monotonic()
        window_start = now - self.window_seconds
        bucket = self._hits.get(key) or deque()
        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= self.rpm:
            retry_after = int(max(1, self.window_seconds - (now - bucket[0])))
            return False, retry_after

        bucket.append(now)
        self._hits[key] = bucket
        return True, None


def json_error_response(err_type: str, message: str, trace_id: str, status_code: int) -> JSONResponse:
    payload = {"error": {"type": err_type, "message": message, "trace_id": trace_id}, "trace_id": trace_id}
    return JSONResponse(status_code=status_code, content=payload, headers={"X-Trace-Id": trace_id})


def require_api_key_if_needed(request: Request, trace_id: str) -> Optional[JSONResponse]:
    path = request.url.path
    method = request.method.upper()
    public_docs = allow_public_docs()
    if path in {"/health", "/version", "/db/health", "/prosperity/health"}:
        return None
    if public_docs and method == "GET" and path in {"/docs", "/openapi.json"}:
        return None
    if not path.startswith("/prosperity"):
        return None

    keys = get_api_keys()
    provided = request.headers.get("X-FTIP-API-Key")
    if not provided:
        return json_error_response("auth_error", "missing API key", trace_id, 401)
    if provided not in keys:
        return json_error_response("auth_error", "invalid API key", trace_id, 401)

    request.state.api_key = provided
    return None


def enforce_rate_limit(request: Request, limiter: RateLimiter, trace_id: str) -> Optional[JSONResponse]:
    path = request.url.path
    if path in {"/health", "/version", "/db/health", "/prosperity/health"}:
        return None

    key = getattr(request.state, "api_key", None)
    if not key:
        client = request.client.host if request.client else "unknown"
        key = f"ip:{client}"
    allowed, retry_after = limiter.check(str(key))
    if allowed:
        return None

    metrics_tracker.record_rate_limit_hit()
    headers = {"Retry-After": str(retry_after or 60), "X-Trace-Id": trace_id}
    payload = {"error": {"type": "rate_limit", "message": "rate limit exceeded", "trace_id": trace_id}, "trace_id": trace_id}
    return JSONResponse(status_code=429, content=payload, headers=headers)


def add_cors_middleware(app) -> None:
    origins = get_allowed_origins()
    if not origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def trace_id_from_request(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", None)
    if not trace_id:
        trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
        request.state.trace_id = trace_id
    return trace_id
