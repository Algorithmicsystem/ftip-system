from __future__ import annotations

import re
from typing import Dict

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def canonical_symbol(symbol: str) -> str:
    cleaned = (symbol or "").strip().upper()
    if not cleaned:
        raise ValueError("symbol required")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace(".TSX", ".TO").replace(".TSXV", ".V")
    if not _SYMBOL_RE.match(cleaned):
        raise ValueError("symbol must be alphanumeric up to 10 chars")
    return cleaned


def normalize_symbol(symbol: str) -> str:
    return canonical_symbol(symbol)


def detect_country_exchange(symbol: str) -> Dict[str, str | None]:
    cleaned = canonical_symbol(symbol)
    if cleaned.endswith(".TO"):
        return {"country": "CA", "exchange": "TSX", "currency": "CAD"}
    if cleaned.endswith(".V"):
        return {"country": "CA", "exchange": "TSXV", "currency": "CAD"}
    return {"country": "US", "exchange": "NYSE/NASDAQ", "currency": "USD"}


def provider_symbol(symbol: str, provider: str) -> str:
    cleaned = canonical_symbol(symbol)
    if provider == "stooq":
        if cleaned.endswith(".TO") or cleaned.endswith(".V"):
            return cleaned.lower()
        return f"{cleaned.lower()}.us"
    return cleaned
