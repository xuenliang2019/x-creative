"""Tests for HKG store: InvertedIndex and HypergraphStore."""

import json
from pathlib import Path

import pytest


# ---------- helpers ----------
def _make_node(node_id: str, name: str, aliases: list[str] | None = None, type_: str = "entity"):
    """Create an HKGNode for testing."""
    from x_creative.hkg.types import HKGNode

    return HKGNode(node_id=node_id, name=name, aliases=aliases or [], type=type_)


def _make_edge(edge_id: str, nodes: list[str], relation: str = "related"):
    """Create a Hyperedge with a single dummy provenance for testing."""
    from x_creative.hkg.types import Hyperedge, Provenance

    return Hyperedge(
        edge_id=edge_id,
        nodes=nodes,
        relation=relation,
        provenance=[Provenance(doc_id="doc_test", chunk_id="chunk_test")],
    )


# ============================================================
# InvertedIndex
# ============================================================
class TestInvertedIndex:
    """Tests for the InvertedIndex class."""

    def test_add_and_get(self) -> None:
        """Adding a (node_id, edge_id) pair makes it retrievable."""
        from x_creative.hkg.store import InvertedIndex

        idx = InvertedIndex()
        idx.add("n_001", "e_001")
        idx.add("n_001", "e_002")
        idx.add("n_002", "e_001")

        assert idx.get_edges("n_001") == {"e_001", "e_002"}
        assert idx.get_edges("n_002") == {"e_001"}

    def test_remove_edge(self) -> None:
        """Removing an edge_id clears it from every node's set."""
        from x_creative.hkg.store import InvertedIndex

        idx = InvertedIndex()
        idx.add("n_001", "e_001")
        idx.add("n_002", "e_001")
        idx.add("n_001", "e_002")

        idx.remove_edge("e_001")

        assert idx.get_edges("n_001") == {"e_002"}
        assert idx.get_edges("n_002") == set()

    def test_get_unknown_node(self) -> None:
        """Querying an unknown node_id returns an empty set (not KeyError)."""
        from x_creative.hkg.store import InvertedIndex

        idx = InvertedIndex()

        assert idx.get_edges("nonexistent") == set()

    def test_rebuild(self) -> None:
        """rebuild() replaces the entire index from a dict of edges."""
        from x_creative.hkg.store import InvertedIndex

        idx = InvertedIndex()
        # Stale data that should be replaced
        idx.add("n_999", "e_999")

        edges = {
            "e_001": _make_edge("e_001", ["n_001", "n_002"]),
            "e_002": _make_edge("e_002", ["n_002", "n_003"]),
        }

        idx.rebuild(edges)

        assert idx.get_edges("n_001") == {"e_001"}
        assert idx.get_edges("n_002") == {"e_001", "e_002"}
        assert idx.get_edges("n_003") == {"e_002"}
        # Old stale data gone
        assert idx.get_edges("n_999") == set()


# ============================================================
# HypergraphStore
# ============================================================
class TestHypergraphStore:
    """Tests for the HypergraphStore class."""

    def test_add_and_get_node(self) -> None:
        """Adding a node makes it retrievable by node_id."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()
        node = _make_node("n_001", "Entropy")
        store.add_node(node)

        result = store.get_node("n_001")
        assert result is not None
        assert result.name == "Entropy"

    def test_get_node_not_found(self) -> None:
        """get_node returns None for unknown node_id."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()

        assert store.get_node("nonexistent") is None

    def test_add_and_get_edge(self) -> None:
        """Adding an edge makes it retrievable by edge_id."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()
        edge = _make_edge("e_001", ["n_001", "n_002"], relation="causes")
        store.add_edge(edge)

        result = store.get_edge("e_001")
        assert result is not None
        assert result.relation == "causes"

    def test_get_edge_not_found(self) -> None:
        """get_edge returns None for unknown edge_id."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()

        assert store.get_edge("nonexistent") is None

    def test_get_edges_for_node(self) -> None:
        """get_edges_for_node returns all edges touching a given node."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()
        store.add_node(_make_node("n_001", "A"))
        store.add_node(_make_node("n_002", "B"))
        store.add_node(_make_node("n_003", "C"))

        e1 = _make_edge("e_001", ["n_001", "n_002"])
        e2 = _make_edge("e_002", ["n_002", "n_003"])
        e3 = _make_edge("e_003", ["n_001", "n_003"])
        store.add_edge(e1)
        store.add_edge(e2)
        store.add_edge(e3)

        edges = store.get_edges_for_node("n_002")
        edge_ids = {e.edge_id for e in edges}

        assert edge_ids == {"e_001", "e_002"}

    def test_get_edges_for_node_unknown(self) -> None:
        """get_edges_for_node returns empty list for unknown node."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()

        assert store.get_edges_for_node("nonexistent") == []

    def test_find_nodes_by_name_exact(self) -> None:
        """find_nodes_by_name matches the canonical name (case-insensitive)."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()
        store.add_node(_make_node("n_001", "Shannon Entropy"))

        # Exact match (case-insensitive)
        result = store.find_nodes_by_name("shannon entropy")
        assert result == ["n_001"]

        result_upper = store.find_nodes_by_name("SHANNON ENTROPY")
        assert result_upper == ["n_001"]

    def test_find_nodes_by_alias(self) -> None:
        """find_nodes_by_name matches aliases (case-insensitive)."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()
        store.add_node(
            _make_node("n_001", "Shannon Entropy", aliases=["information entropy", "H(X)"])
        )

        result = store.find_nodes_by_name("Information Entropy")
        assert result == ["n_001"]

        result_hx = store.find_nodes_by_name("h(x)")
        assert result_hx == ["n_001"]

    def test_find_nodes_not_found(self) -> None:
        """find_nodes_by_name returns empty list when no match."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()
        store.add_node(_make_node("n_001", "Entropy"))

        assert store.find_nodes_by_name("nonexistent") == []

    def test_stats(self) -> None:
        """stats() returns correct node_count and edge_count."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()
        store.add_node(_make_node("n_001", "A"))
        store.add_node(_make_node("n_002", "B"))
        store.add_edge(_make_edge("e_001", ["n_001", "n_002"]))

        s = store.stats()

        assert s["node_count"] == 2
        assert s["edge_count"] == 1

    def test_all_nodes_property(self) -> None:
        """all_nodes returns the internal node dict."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()
        store.add_node(_make_node("n_001", "A"))
        store.add_node(_make_node("n_002", "B"))

        assert set(store.all_nodes.keys()) == {"n_001", "n_002"}

    def test_all_edges_property(self) -> None:
        """all_edges returns the internal edge dict."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()
        store.add_edge(_make_edge("e_001", ["n_001", "n_002"]))

        assert set(store.all_edges.keys()) == {"e_001"}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """save() then load() produces an equivalent store with a rebuilt index."""
        from x_creative.hkg.store import HypergraphStore

        store = HypergraphStore()
        store.add_node(_make_node("n_001", "Entropy", aliases=["H(X)"]))
        store.add_node(_make_node("n_002", "Temperature"))
        store.add_edge(_make_edge("e_001", ["n_001", "n_002"], relation="causes"))

        file = tmp_path / "hkg.json"
        store.save(file)

        # Verify the file is valid JSON
        data = json.loads(file.read_text())
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

        # Load and verify
        loaded = HypergraphStore.load(file)

        assert loaded.stats() == store.stats()
        assert loaded.get_node("n_001") is not None
        assert loaded.get_node("n_001").name == "Entropy"
        assert loaded.get_edge("e_001") is not None
        assert loaded.get_edge("e_001").relation == "causes"

        # Index should be rebuilt
        edges = loaded.get_edges_for_node("n_001")
        assert len(edges) == 1
        assert edges[0].edge_id == "e_001"

        # Name-to-node index should be rebuilt
        assert loaded.find_nodes_by_name("h(x)") == ["n_001"]
        assert loaded.find_nodes_by_name("entropy") == ["n_001"]
