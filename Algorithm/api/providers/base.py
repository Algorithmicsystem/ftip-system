from __future__ import annotations

import os
from typing import Optional


class ProviderError(Exception):
    """Raised when a provider health check fails unexpectedly."""


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value


def has_env(name: str) -> bool:
    return get_env(name) is not None


class BaseProvider:
    name = "base"

    def enabled(self) -> bool:
        raise NotImplementedError

    def health_check(self):
        raise NotImplementedError
