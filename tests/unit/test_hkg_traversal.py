"""Tests for HKG traversal: TraversalCache and k_shortest_hyperpaths."""

import pytest

from x_creative.hkg.types import Hyperedge, Hyperpath, Provenance


# ---------- helpers ----------
def _make_edge(edge_id: str, nodes: list[str], relation: str = "related") -> Hyperedge:
    """Create a Hyperedge with a single dummy provenance for testing."""
    return Hyperedge(
        edge_id=edge_id,
        nodes=nodes,
        relation=relation,
        provenance=[Provenance(doc_id="doc_test", chunk_id="chunk_test")],
    )


def _build_toy_store():
    """Build the toy hypergraph from the spec.

    e1: {n1, n2, n3}  -- shares n2,n3 with e2
    e2: {n2, n3, n4}  -- shares n4 with e3
    e3: {n4, n5}      -- reaches n5
    e4: {n1, n5}      -- direct shortcut
    e5: {n6, n7}      -- isolated
    """
    from x_creative.hkg.store import HypergraphStore

    store = HypergraphStore()
    store.add_edge(_make_edge("e1", ["n1", "n2", "n3"]))
    store.add_edge(_make_edge("e2", ["n2", "n3", "n4"]))
    store.add_edge(_make_edge("e3", ["n4", "n5"]))
    store.add_edge(_make_edge("e4", ["n1", "n5"]))
    store.add_edge(_make_edge("e5", ["n6", "n7"]))
    return store


# ============================================================
# TraversalCache
# ============================================================
class TestTraversalCache:
    """Tests for the TraversalCache class."""

    def test_miss_returns_none(self) -> None:
        """A cache miss returns None."""
        from x_creative.hkg.cache import TraversalCache

        cache = TraversalCache()
        key = (frozenset(["n1"]), frozenset(["n5"]), 3, 1, 6)
        assert cache.get(key) is None

    def test_hit_returns_stored_value(self) -> None:
        """After put(), get() returns the stored value."""
        from x_creative.hkg.cache import TraversalCache

        cache = TraversalCache()
        key = (frozenset(["n1"]), frozenset(["n5"]), 3, 1, 6)
        paths = [Hyperpath(edges=["e4"], intermediate_nodes=[])]

        cache.put(key, paths)
        result = cache.get(key)

        assert result is not None
        assert len(result) == 1
        assert result[0].edges == ["e4"]

    def test_invalidate_clears_cache(self) -> None:
        """invalidate() clears all cached entries."""
        from x_creative.hkg.cache import TraversalCache

        cache = TraversalCache()
        key = (frozenset(["n1"]), frozenset(["n5"]), 3, 1, 6)
        paths = [Hyperpath(edges=["e4"], intermediate_nodes=[])]

        cache.put(key, paths)
        assert cache.get(key) is not None

        cache.invalidate()
        assert cache.get(key) is None

    def test_lru_eviction(self) -> None:
        """When max_size is exceeded, the least-recently-used entry is evicted."""
        from x_creative.hkg.cache import TraversalCache

        cache = TraversalCache(max_size=2)
        key1 = (frozenset(["a"]), frozenset(["b"]), 1, 1, 6)
        key2 = (frozenset(["c"]), frozenset(["d"]), 1, 1, 6)
        key3 = (frozenset(["e"]), frozenset(["f"]), 1, 1, 6)

        paths = [Hyperpath(edges=["e1"], intermediate_nodes=[])]
        cache.put(key1, paths)
        cache.put(key2, paths)

        # Access key1 to make it recently used
        cache.get(key1)

        # Adding key3 should evict key2 (least recently used)
        cache.put(key3, paths)

        assert cache.get(key1) is not None
        assert cache.get(key2) is None
        assert cache.get(key3) is not None


# ============================================================
# k_shortest_hyperpaths
# ============================================================
class TestKShortestHyperpaths:
    """Tests for the k_shortest_hyperpaths function."""

    def test_shortest_path_direct(self) -> None:
        """n1->n5 should find e4 (length 1) as the shortest path."""
        from x_creative.hkg.traversal import k_shortest_hyperpaths

        store = _build_toy_store()
        paths = k_shortest_hyperpaths(store, ["n1"], ["n5"], K=3, IS=1, max_len=6)

        assert len(paths) >= 1
        # The shortest path is e4 (single edge containing both n1 and n5)
        shortest = paths[0]
        assert shortest.edges == ["e4"]
        assert shortest.length == 1

    def test_no_path_to_isolated(self) -> None:
        """n1->n6 returns empty because n6 is in isolated component e5."""
        from x_creative.hkg.traversal import k_shortest_hyperpaths

        store = _build_toy_store()
        paths = k_shortest_hyperpaths(store, ["n1"], ["n6"], K=3, IS=1, max_len=6)

        assert paths == []

    def test_is_constraint_filters(self) -> None:
        """IS=2 should block e2->e3 transition (overlap only {n4} = 1 node).

        With IS=2, e1->e2 is allowed (overlap {n2, n3} = 2 nodes),
        but e2->e3 is blocked (overlap {n4} = 1 node).
        Only e4 (direct n1->n5) should remain.
        """
        from x_creative.hkg.traversal import k_shortest_hyperpaths

        store = _build_toy_store()
        paths = k_shortest_hyperpaths(store, ["n1"], ["n5"], K=3, IS=2, max_len=6)

        # e4 is still reachable (single edge, no transition needed)
        assert len(paths) >= 1
        # All paths should be e4 only; multi-edge path via e1->e2->e3 blocked
        for p in paths:
            assert "e3" not in p.edges, "e2->e3 should be blocked by IS=2"

    def test_max_len_limits(self) -> None:
        """max_len=1 only finds single-edge paths."""
        from x_creative.hkg.traversal import k_shortest_hyperpaths

        store = _build_toy_store()
        paths = k_shortest_hyperpaths(store, ["n1"], ["n5"], K=3, IS=1, max_len=1)

        # Only single-edge paths: e4 covers n1->n5
        assert len(paths) >= 1
        for p in paths:
            assert p.length == 1

    def test_k_limits_results(self) -> None:
        """K=1 returns at most 1 path."""
        from x_creative.hkg.traversal import k_shortest_hyperpaths

        store = _build_toy_store()
        paths = k_shortest_hyperpaths(store, ["n1"], ["n5"], K=1, IS=1, max_len=6)

        assert len(paths) <= 1

    def test_intermediate_nodes_populated(self) -> None:
        """Multi-edge path n1->n5 via e1->e2->e3 has non-empty intermediate_nodes.

        Remove e4 (direct shortcut) to force the multi-edge path.
        """
        from x_creative.hkg.store import HypergraphStore
        from x_creative.hkg.traversal import k_shortest_hyperpaths

        # Build store without the e4 shortcut
        store = HypergraphStore()
        store.add_edge(_make_edge("e1", ["n1", "n2", "n3"]))
        store.add_edge(_make_edge("e2", ["n2", "n3", "n4"]))
        store.add_edge(_make_edge("e3", ["n4", "n5"]))

        paths = k_shortest_hyperpaths(store, ["n1"], ["n5"], K=3, IS=1, max_len=6)

        assert len(paths) >= 1
        # Should find path e1->e2->e3
        multi_edge = [p for p in paths if p.length > 1]
        assert len(multi_edge) >= 1

        path = multi_edge[0]
        assert path.edges == ["e1", "e2", "e3"]
        assert len(path.intermediate_nodes) > 0
        # Intermediate nodes should NOT include start (n1) or end (n5)
        assert "n1" not in path.intermediate_nodes
        assert "n5" not in path.intermediate_nodes

    def test_empty_start_or_end(self) -> None:
        """Empty start or end node list returns empty results."""
        from x_creative.hkg.traversal import k_shortest_hyperpaths

        store = _build_toy_store()

        assert k_shortest_hyperpaths(store, [], ["n5"]) == []
        assert k_shortest_hyperpaths(store, ["n1"], []) == []

    def test_same_edge_start_and_end(self) -> None:
        """When start and end nodes are in the same edge, a single-edge path is returned."""
        from x_creative.hkg.traversal import k_shortest_hyperpaths

        store = _build_toy_store()
        # n1 and n2 are both in e1
        paths = k_shortest_hyperpaths(store, ["n1"], ["n2"], K=3, IS=1, max_len=6)

        assert len(paths) >= 1
        assert paths[0].length == 1

    def test_paths_include_provenance_refs(self) -> None:
        """Traversal output should expose per-edge provenance refs directly."""
        from x_creative.hkg.traversal import k_shortest_hyperpaths

        store = _build_toy_store()
        paths = k_shortest_hyperpaths(store, ["n1"], ["n5"], K=1, IS=1, max_len=6)

        assert len(paths) == 1
        path = paths[0]
        assert len(path.provenance_refs) == path.length
        assert all(len(refs) >= 1 for refs in path.provenance_refs)
