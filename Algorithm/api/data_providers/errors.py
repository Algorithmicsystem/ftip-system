from __future__ import annotations

from typing import Any, Dict, Optional


class ProviderError(Exception):
    def __init__(
        self,
        reason_code: str,
        reason_detail: str,
        *,
        provider_name: Optional[str] = None,
        source_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(reason_detail)
        self.reason_code = reason_code
        self.reason_detail = reason_detail
        self.provider_name = provider_name
        self.source_type = source_type
        self.metadata = metadata or {}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "reason_detail": self.reason_detail,
            "provider_name": self.provider_name,
            "source_type": self.source_type,
            "metadata": self.metadata,
        }


class ProviderUnavailable(ProviderError):
    pass


class SymbolNoData(ProviderError):
    pass
