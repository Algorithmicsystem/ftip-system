"""Phase 12.4: Intelligence Graph — hidden connections and stress propagation."""
from __future__ import annotations

import datetime as dt
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from api import db
from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)

NODE_TYPES = ["symbol", "sector", "macro_factor", "pe_entity", "executive", "theme"]

EDGE_TYPES = [
    "sector_member",
    "competes_with",
    "influenced_by",
    "supplies_to",
    "shares_executive",
    "correlated_with",
]


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    label: str
    axiom_score: Optional[float] = None
    regime_label: Optional[str] = None
    sri_exposure: float = 0.0
    last_updated: Optional[dt.date] = None


@dataclass
class GraphEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    weight: float = 1.0
    direction: str = "bidirectional"
    last_validated: Optional[dt.date] = None

    def __post_init__(self) -> None:
        self.weight = clamp(self.weight, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_symbol_sector_graph(
    symbols: List[str],
    sector_map: Dict[str, str],
) -> Dict[str, Any]:
    """Build nodes and edges from a symbol list and a symbol→sector mapping."""
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    sector_symbols: Dict[str, List[str]] = {}
    for sym in symbols:
        sector = sector_map.get(sym, "Unknown")
        sector_symbols.setdefault(sector, []).append(sym)

        node_id = f"sym_{sym}"
        nodes[node_id] = GraphNode(
            node_id=node_id,
            node_type="symbol",
            label=sym,
            last_updated=dt.date.today(),
        )

    for sector, syms in sector_symbols.items():
        sec_id = f"sec_{sector.replace(' ', '_')}"
        nodes[sec_id] = GraphNode(
            node_id=sec_id,
            node_type="sector",
            label=sector,
            last_updated=dt.date.today(),
        )
        for sym in syms:
            sym_id = f"sym_{sym}"
            edges.append(GraphEdge(
                edge_id=f"{sym_id}__sector_member__{sec_id}",
                source_node_id=sym_id,
                target_node_id=sec_id,
                edge_type="sector_member",
                weight=1.0,
                direction="unidirectional",
                last_validated=dt.date.today(),
            ))
        # Intra-sector competes_with edges (unique pairs)
        for i, s1 in enumerate(syms):
            for s2 in syms[i + 1:]:
                id1, id2 = f"sym_{s1}", f"sym_{s2}"
                edges.append(GraphEdge(
                    edge_id=f"{id1}__competes_with__{id2}",
                    source_node_id=id1,
                    target_node_id=id2,
                    edge_type="competes_with",
                    weight=0.7,
                    direction="bidirectional",
                    last_validated=dt.date.today(),
                ))

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Stress propagation
# ---------------------------------------------------------------------------

def propagate_stress_through_graph(
    source_node_id: str,
    stress_score: float,
    graph_edges: List[GraphEdge],
    propagation_depth: int = 2,
) -> Dict[str, float]:
    """BFS stress propagation with 0.60 decay per hop.

    Returns {node_id: propagated_stress_score}.
    """
    _DECAY = 0.60

    # Build adjacency: node → [(neighbour, weight)]
    adj: Dict[str, List[tuple]] = {}
    for e in graph_edges:
        adj.setdefault(e.source_node_id, []).append((e.target_node_id, e.weight))
        if e.direction == "bidirectional":
            adj.setdefault(e.target_node_id, []).append((e.source_node_id, e.weight))

    result: Dict[str, float] = {}
    visited: Set[str] = {source_node_id}
    queue: deque = deque([(source_node_id, stress_score, 0)])

    while queue:
        node, current_stress, depth = queue.popleft()
        if depth >= propagation_depth:
            continue
        for neighbour, weight in adj.get(node, []):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            downstream = current_stress * weight * _DECAY
            result[neighbour] = round(downstream, 4)
            queue.append((neighbour, downstream, depth + 1))

    return result


# ---------------------------------------------------------------------------
# Hidden connections
# ---------------------------------------------------------------------------

def find_hidden_connections(
    symbol: str,
    graph_edges: List[GraphEdge],
    max_hops: int = 3,
) -> Dict[str, Any]:
    """BFS to find direct, indirect, and hidden (3-hop) paths from a symbol node."""
    source = f"sym_{symbol}"

    adj: Dict[str, List[str]] = {}
    for e in graph_edges:
        adj.setdefault(e.source_node_id, []).append(e.target_node_id)
        if e.direction == "bidirectional":
            adj.setdefault(e.target_node_id, []).append(e.source_node_id)

    direct: List[str] = []
    indirect: List[str] = []
    hidden: List[str] = []

    visited: Dict[str, int] = {source: 0}
    queue: deque = deque([(source, 0)])

    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbour in adj.get(node, []):
            if neighbour in visited:
                continue
            visited[neighbour] = depth + 1
            if depth + 1 == 1:
                direct.append(neighbour)
            elif depth + 1 == 2:
                indirect.append(neighbour)
            else:
                hidden.append(neighbour)
            queue.append((neighbour, depth + 1))

    return {
        "symbol": symbol,
        "direct_connections": direct,
        "indirect_connections": indirect,
        "hidden_connections": hidden,
        "total_reachable": len(visited) - 1,
    }


# ---------------------------------------------------------------------------
# Network centrality
# ---------------------------------------------------------------------------

def compute_network_centrality(graph_edges: List[GraphEdge]) -> Dict[str, float]:
    """Degree centrality = edges incident to node / (n - 1)."""
    degree: Dict[str, int] = {}
    node_set: Set[str] = set()

    for e in graph_edges:
        node_set.add(e.source_node_id)
        node_set.add(e.target_node_id)
        degree[e.source_node_id] = degree.get(e.source_node_id, 0) + 1
        if e.direction == "bidirectional":
            degree[e.target_node_id] = degree.get(e.target_node_id, 0) + 1

    n = len(node_set)
    if n <= 1:
        return {node: 0.0 for node in node_set}

    return {node: round(degree.get(node, 0) / (n - 1), 4) for node in node_set}


# ---------------------------------------------------------------------------
# DB seeding
# ---------------------------------------------------------------------------

def seed_symbol_graph(symbols: List[str]) -> Dict[str, Any]:
    """Create graph nodes and edges in DB for the given symbols."""
    if not db.db_read_enabled():
        return {"nodes_created": 0, "edges_created": 0}

    try:
        # Pull sector from latest axiom payloads
        sector_map: Dict[str, str] = {}
        for sym in symbols:
            row = db.safe_fetchone(
                """
                SELECT payload->>'sector' FROM axiom_scores_daily
                 WHERE symbol = %s
                 ORDER BY as_of_date DESC LIMIT 1
                """,
                (sym,),
            )
            sector_map[sym] = str(row[0]) if row and row[0] else "Unknown"

        graph = build_symbol_sector_graph(symbols, sector_map)
        nodes: Dict[str, GraphNode] = graph["nodes"]
        edges: List[GraphEdge] = graph["edges"]

        nodes_created = 0
        for n in nodes.values():
            try:
                db.safe_execute(
                    """
                    INSERT INTO intelligence_graph_nodes
                        (node_id, node_type, label, axiom_score, regime_label,
                         sri_exposure, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (node_id) DO UPDATE
                       SET label        = EXCLUDED.label,
                           last_updated = EXCLUDED.last_updated
                    """,
                    (n.node_id, n.node_type, n.label,
                     n.axiom_score, n.regime_label, n.sri_exposure,
                     n.last_updated or dt.date.today()),
                )
                nodes_created += 1
            except Exception:
                pass

        edges_created = 0
        for e in edges:
            try:
                db.safe_execute(
                    """
                    INSERT INTO intelligence_graph_edges
                        (edge_id, source_node_id, target_node_id, edge_type,
                         weight, direction, last_validated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (edge_id) DO NOTHING
                    """,
                    (e.edge_id, e.source_node_id, e.target_node_id,
                     e.edge_type, e.weight, e.direction, e.last_validated),
                )
                edges_created += 1
            except Exception:
                pass

        return {"nodes_created": nodes_created, "edges_created": edges_created}
    except Exception as exc:
        logger.warning("seed_symbol_graph_failed err=%s", exc)
        return {"nodes_created": 0, "edges_created": 0}
