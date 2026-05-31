"""SymbolLinkageGraph — tracks explicit symbol-to-symbol relationships.

Relationship types:
    sector_peer   — same sector in market_symbols (auto-built)
    etf_member    — symbol is a constituent of an ETF
    competitor    — manually-declared direct competitor
    sector_proxy  — ETF/index used as sector benchmark
    benchmark     — broad market benchmark (SPY, QQQ, etc.)

Usage::

    from api.signals.linkage import SymbolLinkageGraph
    graph = SymbolLinkageGraph()
    peers = graph.get_peers("AAPL", link_type="sector_peer")
    n = graph.build_from_sector()   # bootstrap from market_symbols.sector
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from api import db


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SymbolLink(BaseModel):
    symbol: str
    linked_symbol: str
    link_type: str          # sector_peer | etf_member | competitor | sector_proxy | benchmark
    weight: Optional[float] = None
    source: Optional[str] = None
    is_active: bool = True
    meta: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class SymbolLinkageGraph:
    """Thin query layer over the symbol_linkage table."""

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_peers(
        self,
        symbol: str,
        *,
        link_type: Optional[str] = None,
        active_only: bool = True,
        limit: int = 50,
    ) -> List[SymbolLink]:
        """Return symbols linked to *symbol*."""
        if not db.db_read_enabled():
            return []
        symbol = symbol.upper()
        params: list = [symbol]
        type_clause = ""
        if link_type:
            type_clause = "AND link_type = %s"
            params.append(link_type)
        active_clause = "AND is_active = TRUE" if active_only else ""
        params.append(limit)
        rows = db.safe_fetchall(
            f"""
            SELECT symbol, linked_symbol, link_type, weight, source, is_active
            FROM symbol_linkage
            WHERE symbol = %s {type_clause} {active_clause}
            ORDER BY weight DESC NULLS LAST, linked_symbol
            LIMIT %s
            """,
            tuple(params),
        )
        return [
            SymbolLink(
                symbol=row[0],
                linked_symbol=row[1],
                link_type=row[2],
                weight=float(row[3]) if row[3] is not None else None,
                source=row[4],
                is_active=bool(row[5]),
            )
            for row in (rows or [])
        ]

    def get_link_counts(self, symbol: str) -> Dict[str, int]:
        """Return count of active links per link_type for a symbol."""
        if not db.db_read_enabled():
            return {}
        rows = db.safe_fetchall(
            """
            SELECT link_type, COUNT(*) AS n
            FROM symbol_linkage
            WHERE symbol = %s AND is_active = TRUE
            GROUP BY link_type
            ORDER BY n DESC
            """,
            (symbol.upper(),),
        )
        return {str(row[0]): int(row[1]) for row in (rows or [])}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add_link(
        self,
        symbol: str,
        linked_symbol: str,
        link_type: str,
        *,
        weight: Optional[float] = None,
        source: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Upsert a single link. Returns True on success."""
        if not db.db_enabled():
            return False
        try:
            db.safe_execute(
                """
                INSERT INTO symbol_linkage
                    (symbol, linked_symbol, link_type, weight, source, is_active, meta)
                VALUES (%s, %s, %s, %s, %s, TRUE, %s)
                ON CONFLICT (symbol, linked_symbol, link_type)
                DO UPDATE SET
                    weight     = EXCLUDED.weight,
                    source     = EXCLUDED.source,
                    is_active  = TRUE,
                    meta       = EXCLUDED.meta,
                    updated_at = now()
                """,
                (
                    symbol.upper(),
                    linked_symbol.upper(),
                    link_type,
                    weight,
                    source,
                    meta or {},
                ),
            )
            return True
        except Exception:  # pragma: no cover
            return False

    def build_from_sector(self, *, sector: Optional[str] = None) -> int:
        """Bootstrap sector_peer links from market_symbols.sector.

        For each sector, writes bidirectional sector_peer links between all
        active symbols in that sector. Runs in one DB round-trip per sector.
        Returns total link pairs written.
        """
        if not db.db_enabled() or not db.db_read_enabled():
            return 0

        sector_clause = "AND LOWER(sector) = LOWER(%s)" if sector else ""
        params = (sector,) if sector else ()
        rows = db.safe_fetchall(
            f"""
            SELECT symbol, sector
            FROM market_symbols
            WHERE is_active = TRUE AND sector IS NOT NULL AND sector != ''
            {sector_clause}
            ORDER BY sector, symbol
            """,
            params,
        )
        if not rows:
            return 0

        # Group symbols by sector
        by_sector: Dict[str, List[str]] = {}
        for sym, sec in rows:
            key = str(sec).strip().lower()
            by_sector.setdefault(key, []).append(str(sym).upper())

        written = 0
        for sec_name, symbols in by_sector.items():
            for i, sym_a in enumerate(symbols):
                for sym_b in symbols[i + 1:]:
                    # Bidirectional
                    if self.add_link(sym_a, sym_b, "sector_peer", source="sector_auto",
                                     meta={"sector": sec_name}):
                        written += 1
                    if self.add_link(sym_b, sym_a, "sector_peer", source="sector_auto",
                                     meta={"sector": sec_name}):
                        written += 1
        return written


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

graph = SymbolLinkageGraph()
