"""Regression tests for Phase 12 — SymbolLinkageGraph."""

from __future__ import annotations

import datetime as dt
from unittest.mock import call, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_link_row(symbol="AAPL", linked="MSFT", link_type="sector_peer",
                   weight=None, source="sector_auto", active=True):
    return (symbol, linked, link_type, weight, source, active)


# ---------------------------------------------------------------------------
# 1. SymbolLink model
# ---------------------------------------------------------------------------

def test_symbol_link_importable():
    from api.signals.linkage import SymbolLink
    assert SymbolLink


def test_symbol_link_fields():
    from api.signals.linkage import SymbolLink
    link = SymbolLink(
        symbol="AAPL",
        linked_symbol="MSFT",
        link_type="sector_peer",
        weight=0.85,
        source="sector_auto",
    )
    assert link.symbol == "AAPL"
    assert link.linked_symbol == "MSFT"
    assert link.link_type == "sector_peer"
    assert link.weight == 0.85
    assert link.is_active is True


def test_symbol_link_defaults():
    from api.signals.linkage import SymbolLink
    link = SymbolLink(symbol="AAPL", linked_symbol="GOOG", link_type="competitor")
    assert link.weight is None
    assert link.source is None
    assert link.meta == {}


# ---------------------------------------------------------------------------
# 2. SymbolLinkageGraph.get_peers
# ---------------------------------------------------------------------------

def test_get_peers_returns_empty_when_db_disabled():
    from api.signals.linkage import SymbolLinkageGraph
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_read_enabled.return_value = False
        result = g.get_peers("AAPL")
    assert result == []


def test_get_peers_returns_symbol_links():
    from api.signals.linkage import SymbolLinkageGraph, SymbolLink
    rows = [
        _make_link_row("AAPL", "MSFT", "sector_peer"),
        _make_link_row("AAPL", "GOOGL", "sector_peer"),
    ]
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = rows
        result = g.get_peers("AAPL")
    assert len(result) == 2
    for link in result:
        assert isinstance(link, SymbolLink)
    assert result[0].linked_symbol == "MSFT"
    assert result[1].linked_symbol == "GOOGL"


def test_get_peers_filters_by_link_type():
    from api.signals.linkage import SymbolLinkageGraph
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = []
        g.get_peers("AAPL", link_type="etf_member")
    query = mock_db.safe_fetchall.call_args[0][0]
    assert "link_type = %s" in query


def test_get_peers_uppercases_symbol():
    from api.signals.linkage import SymbolLinkageGraph
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = []
        g.get_peers("aapl")
    params = mock_db.safe_fetchall.call_args[0][1]
    assert params[0] == "AAPL"


# ---------------------------------------------------------------------------
# 3. SymbolLinkageGraph.get_link_counts
# ---------------------------------------------------------------------------

def test_get_link_counts_returns_empty_when_db_disabled():
    from api.signals.linkage import SymbolLinkageGraph
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_read_enabled.return_value = False
        result = g.get_link_counts("AAPL")
    assert result == {}


def test_get_link_counts_aggregates_by_type():
    from api.signals.linkage import SymbolLinkageGraph
    rows = [("sector_peer", 12), ("etf_member", 3)]
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = rows
        counts = g.get_link_counts("AAPL")
    assert counts["sector_peer"] == 12
    assert counts["etf_member"] == 3


# ---------------------------------------------------------------------------
# 4. SymbolLinkageGraph.add_link
# ---------------------------------------------------------------------------

def test_add_link_returns_false_when_db_disabled():
    from api.signals.linkage import SymbolLinkageGraph
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_enabled.return_value = False
        result = g.add_link("AAPL", "MSFT", "sector_peer")
    assert result is False


def test_add_link_upserts():
    from api.signals.linkage import SymbolLinkageGraph
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.safe_execute.return_value = None
        result = g.add_link("AAPL", "MSFT", "sector_peer",
                            weight=0.9, source="sector_auto")
    assert result is True
    mock_db.safe_execute.assert_called_once()
    sql = mock_db.safe_execute.call_args[0][0]
    assert "ON CONFLICT" in sql
    assert "DO UPDATE" in sql


def test_add_link_uppercases_symbols():
    from api.signals.linkage import SymbolLinkageGraph
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.safe_execute.return_value = None
        g.add_link("aapl", "msft", "sector_peer")
    params = mock_db.safe_execute.call_args[0][1]
    assert params[0] == "AAPL"
    assert params[1] == "MSFT"


# ---------------------------------------------------------------------------
# 5. SymbolLinkageGraph.build_from_sector
# ---------------------------------------------------------------------------

def test_build_from_sector_returns_zero_when_db_disabled():
    from api.signals.linkage import SymbolLinkageGraph
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_enabled.return_value = False
        mock_db.db_read_enabled.return_value = False
        result = g.build_from_sector()
    assert result == 0


def test_build_from_sector_creates_bidirectional_links():
    """3 symbols in same sector → 3 pairs × 2 directions = 6 calls."""
    from api.signals.linkage import SymbolLinkageGraph
    g = SymbolLinkageGraph()
    sector_rows = [
        ("AAPL", "technology"),
        ("MSFT", "technology"),
        ("GOOGL", "technology"),
    ]
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = sector_rows
        mock_db.safe_execute.return_value = None
        written = g.build_from_sector()

    assert written == 6   # 3 pairs × 2 directions
    assert mock_db.safe_execute.call_count == 6


def test_build_from_sector_filters_by_sector():
    from api.signals.linkage import SymbolLinkageGraph
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = []
        g.build_from_sector(sector="technology")
    query = mock_db.safe_fetchall.call_args[0][0]
    assert "LOWER(sector) = LOWER(%s)" in query


def test_build_from_sector_returns_zero_when_no_rows():
    from api.signals.linkage import SymbolLinkageGraph
    g = SymbolLinkageGraph()
    with patch("api.signals.linkage.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = []
        result = g.build_from_sector()
    assert result == 0


# ---------------------------------------------------------------------------
# 6. Module-level singleton and routes
# ---------------------------------------------------------------------------

def test_linkage_graph_singleton_importable():
    from api.signals.linkage import graph
    from api.signals.linkage import SymbolLinkageGraph
    assert isinstance(graph, SymbolLinkageGraph)


def test_linkage_route_registered():
    from api.signals import routes
    paths = [r.path for r in routes.router.routes]
    assert "/signals/linkage" in paths


def test_build_sector_route_registered():
    from api.signals import routes
    paths = [r.path for r in routes.router.routes]
    assert "/signals/linkage/build-sector" in paths


def test_linkage_endpoint_returns_structure():
    import asyncio
    from api.signals.routes import symbol_linkage
    from api.signals.linkage import SymbolLink
    peers = [
        SymbolLink(symbol="AAPL", linked_symbol="MSFT", link_type="sector_peer",
                   source="sector_auto"),
        SymbolLink(symbol="AAPL", linked_symbol="GOOGL", link_type="sector_peer",
                   source="sector_auto"),
    ]
    with patch("api.signals.routes.db") as mock_db, \
         patch("api.signals.routes.linkage_graph") as mock_graph:
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        mock_graph.get_peers.return_value = peers
        mock_graph.get_link_counts.return_value = {"sector_peer": 2}
        result = asyncio.run(symbol_linkage("AAPL", link_type=None, limit=20))

    assert result["symbol"] == "AAPL"
    assert result["link_counts"] == {"sector_peer": 2}
    assert len(result["links"]) == 2
    assert result["links"][0]["linked_symbol"] == "MSFT"
