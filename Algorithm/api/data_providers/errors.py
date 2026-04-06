from __future__ import annotations


class ProviderError(Exception):
    def __init__(self, reason_code: str, reason_detail: str) -> None:
        super().__init__(reason_detail)
        self.reason_code = reason_code
        self.reason_detail = reason_detail


class ProviderUnavailable(ProviderError):
    pass


class SymbolNoData(ProviderError):
    pass
