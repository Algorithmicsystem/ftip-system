import os
import time
import uuid
from collections import deque
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from api.ops import metrics_tracker


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


_ALLOWED_ORIGINS: Optional[List[str]] = None
_API_KEYS: Optional[List[str]] = None
_AUTH_STATUS_LOGGED = False


def get_allowed_origins() -> List[str]:
    global _ALLOWED_ORIGINS
    if _ALLOWED_ORIGINS is None:
        raw = _env("FTIP_ALLOWED_ORIGINS", "") or ""
        _ALLOWED_ORIGINS = [o.strip() for o in raw.split(",") if o.strip()]
    return _ALLOWED_ORIGINS


def get_allowed_api_keys() -> List[str]:
    """Source of truth for API keys (merged + trimmed)."""

    global _API_KEYS
    if _API_KEYS is not None:
        return _API_KEYS

    keys: List[str] = []
    seen = set()
    for env_var in ("FTIP_API_KEY", "FTIP_API_KEYS", "FTIP_API_KEY_PRIMARY"):
        raw = _env(env_var, "") or ""
        if not raw:
            continue
        parts = raw.split(",") if env_var == "FTIP_API_KEYS" else [raw]
        for part in parts:
            key = part.strip()
            if key and key not in seen:
                keys.append(key)
                seen.add(key)

    _API_KEYS = keys
    return _API_KEYS


def auth_enabled() -> bool:
    return len(get_allowed_api_keys()) > 0


def allow_public_docs() -> bool:
    return (_env("FTIP_PUBLIC_DOCS", "0") or "0") == "1"


def query_key_enabled() -> bool:
    return (_env("FTIP_ALLOW_QUERY_KEY", "0") or "0") == "1"


def auth_status_public() -> bool:
    return (_env("FTIP_AUTH_STATUS_PUBLIC", "0") or "0") == "1"


def accepted_auth_modes() -> List[str]:
    modes = ["x-ftip-api-key", "bearer"]
    if query_key_enabled():
        modes.append("query")
    return modes


def auth_status_payload() -> Dict[str, object]:
    keys = get_allowed_api_keys()
    return {
        "auth_enabled": auth_enabled(),
        "keys_configured": len(keys),
        "accepted_modes": accepted_auth_modes(),
        "query_key_enabled": query_key_enabled(),
    }


def log_auth_config(logger) -> None:
    global _AUTH_STATUS_LOGGED
    if _AUTH_STATUS_LOGGED:
        return
    payload = auth_status_payload()
    logger.info(
        "auth.config",
        extra={
            "auth_enabled": payload["auth_enabled"],
            "keys_configured": payload["keys_configured"],
            "accepted_modes": payload["accepted_modes"],
        },
    )
    _AUTH_STATUS_LOGGED = True


def reset_auth_cache() -> None:
    global _API_KEYS, _AUTH_STATUS_LOGGED
    _API_KEYS = None
    _AUTH_STATUS_LOGGED = False


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


def unauthorized_response(trace_id: str) -> JSONResponse:
    payload = {"error": {"type": "http_error", "message": "unauthorized", "trace_id": trace_id}, "trace_id": trace_id}
    return JSONResponse(status_code=401, content=payload, headers={"X-Trace-Id": trace_id})


def get_provided_api_key(request: Request) -> Optional[str]:
    headers = request.headers or {}

    header_key = headers.get("x-ftip-api-key")
    if header_key:
        return header_key.strip()

    auth_header = headers.get("authorization")
    if auth_header:
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()

    if query_key_enabled():
        query_key = request.query_params.get("api_key")
        if query_key:
            return query_key.strip()

    return None


def validate_api_key(request: Request) -> Optional[str]:
    keys = get_allowed_api_keys()
    if not keys:
        return None

    provided = (get_provided_api_key(request) or "").strip()
    if provided and provided in keys:
        request.state.api_key = provided
        return provided

    raise HTTPException(status_code=401, detail="unauthorized")


def require_prosperity_api_key(request: Request) -> Optional[str]:
    if not auth_enabled():
        return None
    return validate_api_key(request)


def require_api_key_if_needed(request: Request, trace_id: str) -> Optional[JSONResponse]:
    path = request.url.path
    method = request.method.upper()
    public_docs = allow_public_docs()

    if path in {"/health", "/version", "/db/health"}:
        return None

    if path == "/auth/status" and (auth_status_public() or not auth_enabled()):
        return None

    if public_docs and method == "GET" and path in {"/docs", "/openapi.json"}:
        return None

    if not path.startswith("/prosperity") and path != "/auth/status":
        return None

    if not auth_enabled():
        return None

    try:
        validate_api_key(request)
    except HTTPException:
        return unauthorized_response(trace_id)
    return None


def enforce_rate_limit(request: Request, limiter: RateLimiter, trace_id: str) -> Optional[JSONResponse]:
    path = request.url.path
    if path in {"/health", "/version", "/db/health"}:
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
