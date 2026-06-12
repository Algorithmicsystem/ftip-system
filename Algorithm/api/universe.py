"""AXIOM canonical universe — backwards-compatible module with DB-driven registry support."""
from __future__ import annotations

from typing import List, Optional

# ── Canonical 30-symbol list — kept for backwards compatibility ──────────────
# All existing tests, imports, and endpoints that reference AXIOM_UNIVERSE
# continue to work unchanged.
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
    """Return the canonical AXIOM universe symbol list (backwards-compatible)."""
    if include_extended:
        return AXIOM_UNIVERSE + _EXTENDED
    return list(AXIOM_UNIVERSE)


def get_pipeline_universe(tier: Optional[int] = None) -> List[str]:
    """Get the active symbol universe for pipeline execution.

    Reads from the DB registry if available and populated,
    falls back to AXIOM_UNIVERSE when the registry is empty or DB is down.

    tier=None  — all active symbols (default)
    tier=1     — Tier 1 only (~500 large-cap, full scoring)
    tier=2     — Tier 2 only (~2,000 mid-cap, partial scoring)
    tier=3     — Tier 3 only (~7,500 small-cap, macro only)
    """
    try:
        from api.universe_registry import get_symbols_by_tier, get_all_active_symbols
        if tier is not None:
            symbols = get_symbols_by_tier(tier)
        else:
            symbols = get_all_active_symbols()
        if symbols:
            return symbols
    except Exception:
        pass
    return list(AXIOM_UNIVERSE)
