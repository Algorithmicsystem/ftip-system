"""Phase 19.7: Billing tiers and quota enforcement."""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse

# ---------------------------------------------------------------------------
# Billing tier definitions
# ---------------------------------------------------------------------------

BILLING_TIERS: Dict[str, Dict[str, Any]] = {
    "free": {
        "price_usd": 0,
        "monthly_calls": 1_000,
        "rpm": 10,
        "description": "Free tier — 1K calls/month, 10 RPM",
        "features": ["signal_query", "universe_scores"],
    },
    "starter": {
        "price_usd": 199,
        "monthly_calls": 10_000,
        "rpm": 60,
        "description": "Starter — $199/mo, 10K calls/month, 60 RPM",
        "features": ["signal_query", "universe_scores", "briefing", "explain"],
    },
    "professional": {
        "price_usd": 999,
        "monthly_calls": 100_000,
        "rpm": 300,
        "description": "Professional — $999/mo, 100K calls/month, 300 RPM",
        "features": ["signal_query", "universe_scores", "briefing", "explain",
                     "competitive", "macro", "webhooks"],
    },
    "institutional": {
        "price_usd": 4_999,
        "monthly_calls": None,  # unlimited
        "rpm": 1_000,
        "description": "Institutional — $4,999/mo, unlimited calls, 1K RPM",
        "features": ["*"],
    },
}


# ---------------------------------------------------------------------------
# Quota enforcer
# ---------------------------------------------------------------------------

class QuotaEnforcer:
    """Sliding-window rate limiter + monthly call quota."""

    def __init__(self) -> None:
        self._windows: Dict[str, deque] = {}
        self._monthly: Dict[str, int] = {}
        self._month_key: Dict[str, str] = {}

    def _current_month(self) -> str:
        t = time.gmtime()
        return f"{t.tm_year}-{t.tm_mon:02d}"

    def _get_tier_config(self, tier: str) -> Dict[str, Any]:
        return BILLING_TIERS.get(tier, BILLING_TIERS["free"])

    def check_rate_limit(self, key: str, tier: str) -> Tuple[bool, Optional[int]]:
        """Check sliding-window RPM. Returns (allowed, retry_after_seconds)."""
        cfg = self._get_tier_config(tier)
        rpm = int(cfg["rpm"])
        if rpm <= 0:
            return True, None

        now = time.monotonic()
        window_start = now - 60.0
        bucket = self._windows.setdefault(key, deque())

        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= rpm:
            retry_after = max(1, int(60.0 - (now - bucket[0])))
            return False, retry_after

        bucket.append(now)
        return True, None

    def check_monthly_quota(self, key: str, tier: str) -> Tuple[bool, int]:
        """Check monthly call quota. Returns (allowed, calls_remaining)."""
        cfg = self._get_tier_config(tier)
        limit = cfg.get("monthly_calls")
        if limit is None:
            return True, -1  # unlimited

        month = self._current_month()
        prev_month = self._month_key.get(key)
        if prev_month != month:
            self._monthly[key] = 0
            self._month_key[key] = month

        used = self._monthly.get(key, 0)
        if used >= limit:
            return False, 0

        self._monthly[key] = used + 1
        return True, max(0, limit - used - 1)

    def enforce(
        self,
        request: Request,
        tier: str,
        trace_id: str,
    ) -> Optional[JSONResponse]:
        """Run both checks; return 429 response if any limit exceeded."""
        key = _extract_key(request)

        rate_ok, retry_after = self.check_rate_limit(key, tier)
        if not rate_ok:
            return _rate_limit_response(trace_id, retry_after or 60, "rate_limit")

        quota_ok, remaining = self.check_monthly_quota(key, tier)
        if not quota_ok:
            return _rate_limit_response(trace_id, 0, "monthly_quota_exceeded",
                                        extra={"calls_remaining": 0})

        return None

    def get_usage_summary(self, key: str, tier: str) -> Dict[str, Any]:
        cfg = self._get_tier_config(tier)
        month = self._current_month()
        prev = self._month_key.get(key)
        calls_used = self._monthly.get(key, 0) if prev == month else 0
        limit = cfg.get("monthly_calls")
        return {
            "tier": tier,
            "calls_used_this_month": calls_used,
            "monthly_limit": limit,
            "calls_remaining": max(0, limit - calls_used) if limit is not None else None,
            "rpm_limit": cfg["rpm"],
            "billing_month": month,
        }


def _extract_key(request: Request) -> str:
    for h in ("x-ftip-api-key", "x-api-key"):
        v = request.headers.get(h)
        if v:
            return v[:32]
    client = getattr(request.client, "host", "unknown") if request.client else "unknown"
    return f"ip:{client}"


def _rate_limit_response(
    trace_id: str,
    retry_after: int,
    error_type: str,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    payload: Dict[str, Any] = {
        "error": {
            "type": error_type,
            "message": error_type.replace("_", " "),
            "trace_id": trace_id,
        },
        "trace_id": trace_id,
    }
    if extra:
        payload.update(extra)
    headers = {"X-Trace-Id": trace_id}
    if retry_after > 0:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(status_code=429, content=payload, headers=headers)


# Module-level singleton
quota_enforcer = QuotaEnforcer()
