"""Storage layer for the HKG (Hypergraph Knowledge Grounding) subsystem.

Provides an inverted index for efficient neighbor lookup and a
HypergraphStore that persists nodes/edges to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from x_creative.hkg.types import HKGNode, Hyperedge

log = structlog.get_logger(__name__)


class InvertedIndex:
    """Maps node_id -> set of edge_ids for O(1) neighbor lookup.

    This avoids building a full line graph while still allowing the
    traversal algorithm to find all hyperedges touching a given node.
    """

    def __init__(self) -> None:
        self._index: dict[str, set[str]] = {}

    def add(self, node_id: str, edge_id: str) -> None:
        """Register that *edge_id* touches *node_id*."""
        self._index.setdefault(node_id, set()).add(edge_id)

    def get_edges(self, node_id: str) -> set[str]:
        """Return the set of edge_ids touching *node_id* (empty set if unknown)."""
        return set(self._index.get(node_id, set()))

    def remove_edge(self, edge_id: str) -> None:
        """Remove *edge_id* from every node's edge set."""
        for node_id in list(self._index):
            self._index[node_id].discard(edge_id)

    def rebuild(self, edges: dict[str, Hyperedge]) -> None:
        """Replace the entire index from a dict of edge_id -> Hyperedge."""
        self._index.clear()
        for edge in edges.values():
            for node_id in edge.nodes:
                self.add(node_id, edge.edge_id)


class HypergraphStore:
    """In-memory hypergraph with JSON persistence.

    Maintains:
    - ``_nodes``: dict of node_id -> HKGNode
    - ``_edges``: dict of edge_id -> Hyperedge
    - ``_index``: InvertedIndex (node_id -> edge_ids)
    - ``_name_to_node``: lowercase name/alias -> list of node_ids
    """

    def __init__(self) -> None:
        self._nodes: dict[str, HKGNode] = {}
        self._edges: dict[str, Hyperedge] = {}
        self._index: InvertedIndex = InvertedIndex()
        self._name_to_node: dict[str, list[str]] = {}

    # ---------- node operations ----------

    def add_node(self, node: HKGNode) -> None:
        """Add a node to the store and update the name index."""
        self._nodes[node.node_id] = node
        self._register_name(node)
        log.debug("node_added", node_id=node.node_id, name=node.name)

    def get_node(self, node_id: str) -> HKGNode | None:
        """Return the node with *node_id*, or ``None`` if not found."""
        return self._nodes.get(node_id)

    # ---------- edge operations ----------

    def add_edge(self, edge: Hyperedge) -> None:
        """Add a hyperedge and incrementally update the inverted index."""
        self._edges[edge.edge_id] = edge
        for node_id in edge.nodes:
            self._index.add(node_id, edge.edge_id)
        log.debug("edge_added", edge_id=edge.edge_id, nodes=edge.nodes)

    def get_edge(self, edge_id: str) -> Hyperedge | None:
        """Return the edge with *edge_id*, or ``None`` if not found."""
        return self._edges.get(edge_id)

    def get_edges_for_node(self, node_id: str) -> list[Hyperedge]:
        """Return all hyperedges touching *node_id* via the inverted index."""
        edge_ids = self._index.get_edges(node_id)
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    # ---------- name lookup ----------

    def find_nodes_by_name(self, name: str) -> list[str]:
        """Case-insensitive exact + alias lookup. Returns list of node_ids."""
        return list(self._name_to_node.get(name.lower(), []))

    # ---------- introspection ----------

    def stats(self) -> dict[str, int]:
        """Return basic statistics about the store."""
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
        }

    @property
    def all_nodes(self) -> dict[str, HKGNode]:
        """Return the internal node dict."""
        return self._nodes

    @property
    def all_edges(self) -> dict[str, Hyperedge]:
        """Return the internal edge dict."""
        return self._edges

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        """Serialize nodes and edges to a JSON file."""
        data = {
            "nodes": [node.model_dump() for node in self._nodes.values()],
            "edges": [edge.model_dump() for edge in self._edges.values()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        log.info("store_saved", path=str(path), **self.stats())

    @classmethod
    def load(cls, path: Path) -> HypergraphStore:
        """Load a store from a JSON file, rebuilding the inverted index."""
        data = json.loads(path.read_text())
        store = cls()

        for node_data in data.get("nodes", []):
            store.add_node(HKGNode.model_validate(node_data))

        for edge_data in data.get("edges", []):
            edge = Hyperedge.model_validate(edge_data)
            store._edges[edge.edge_id] = edge

        # Rebuild the inverted index from the full edge set
        store._index.rebuild(store._edges)

        log.info("store_loaded", path=str(path), **store.stats())
        return store

    # ---------- private ----------

    def _register_name(self, node: HKGNode) -> None:
        """Add canonical name and aliases to the name-to-node index."""
        keys = [node.name.lower()] + [a.lower() for a in node.aliases]
        for key in keys:
            self._name_to_node.setdefault(key, [])
            if node.node_id not in self._name_to_node[key]:
                self._name_to_node[key].append(node.node_id)
