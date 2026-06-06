"""Phase 24: AXIOM canonical universe — 30 large-cap symbols."""
from __future__ import annotations

from typing import List

AXIOM_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
    "META", "TSLA", "JPM", "JNJ", "UNH",
    "V", "PG", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "MCD",
    "WMT", "COST", "ACN", "TMO", "DHR",
    "NEE", "XOM", "AVGO", "MA", "GOOGL",
]

_EXTENDED: List[str] = [
    "BAC", "WFC", "GS", "MS", "BRK-B",
    "RTX", "LMT", "BA", "CAT", "DE",
]


def get_universe(include_extended: bool = False) -> List[str]:
    """Return the canonical AXIOM universe symbol list."""
    if include_extended:
        return AXIOM_UNIVERSE + _EXTENDED
    return list(AXIOM_UNIVERSE)
