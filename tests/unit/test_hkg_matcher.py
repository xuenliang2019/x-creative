"""Tests for HKG NodeMatcher with exact/alias/embedding fallback."""

from __future__ import annotations

import pytest

from x_creative.hkg.types import HKGNode


# ---------- helpers ----------
def _make_store():
    """Build a HypergraphStore with two nodes for testing."""
    from x_creative.hkg.store import HypergraphStore

    store = HypergraphStore()
    store.add_node(
        HKGNode(node_id="n1", name="entropy", aliases=["熵", "information entropy"], type="concept")
    )
    store.add_node(
        HKGNode(node_id="n2", name="pressure", aliases=["压力"], type="concept")
    )
    store.add_node(
        HKGNode(
            node_id="n3",
            name="feedback_loop",
            aliases=["feedback loop", "positive feedback"],
            type="schema",
            motif_id="feedback_loop",
        )
    )
    return store


# ============================================================
# NodeMatcher — exact / alias / none
# ============================================================
class TestNodeMatcher:
    """Tests for the three-layer fallback matcher."""

    async def test_exact_match(self) -> None:
        """'entropy' matches n1 exactly by canonical name."""
        from x_creative.hkg.matcher import NodeMatcher

        store = _make_store()
        matcher = NodeMatcher(store=store)

        results = await matcher.match(["entropy"])

        assert len(results) == 1
        r = results[0]
        assert r.term == "entropy"
        assert "n1" in r.matched_node_ids
        assert r.match_type == "exact"
        assert r.confidence == 1.0
        assert r.chosen_id == "n1"
        assert len(r.candidates) >= 1

    async def test_alias_match(self) -> None:
        """'熵' matches n1 via alias."""
        from x_creative.hkg.matcher import NodeMatcher

        store = _make_store()
        matcher = NodeMatcher(store=store)

        results = await matcher.match(["熵"])

        assert len(results) == 1
        r = results[0]
        assert r.term == "熵"
        assert "n1" in r.matched_node_ids
        assert r.match_type == "alias"
        assert r.confidence >= 0.9

    async def test_no_match(self) -> None:
        """'nonexistent' yields empty matched_node_ids with match_type='none'."""
        from x_creative.hkg.matcher import NodeMatcher

        store = _make_store()
        matcher = NodeMatcher(store=store)

        results = await matcher.match(["nonexistent"])

        assert len(results) == 1
        r = results[0]
        assert r.term == "nonexistent"
        assert r.matched_node_ids == []
        assert r.match_type == "none"
        assert r.confidence == 0.0

    async def test_multiple_terms(self) -> None:
        """['entropy', 'pressure'] returns 2 results, both exact matches."""
        from x_creative.hkg.matcher import NodeMatcher

        store = _make_store()
        matcher = NodeMatcher(store=store)

        results = await matcher.match(["entropy", "pressure"])

        assert len(results) == 2
        by_term = {r.term: r for r in results}

        assert "n1" in by_term["entropy"].matched_node_ids
        assert by_term["entropy"].match_type == "exact"

        assert "n2" in by_term["pressure"].matched_node_ids
        assert by_term["pressure"].match_type == "exact"

    async def test_case_insensitive(self) -> None:
        """'ENTROPY' matches n1 — the store does case-insensitive lookup."""
        from x_creative.hkg.matcher import NodeMatcher

        store = _make_store()
        matcher = NodeMatcher(store=store)

        results = await matcher.match(["ENTROPY"])

        assert len(results) == 1
        r = results[0]
        assert "n1" in r.matched_node_ids
        assert r.match_type == "exact"
        assert r.confidence == 1.0

    async def test_schema_match_prefers_schema_candidate(self) -> None:
        """Schema phrase should select schema node with rationale chain."""
        from x_creative.hkg.matcher import NodeMatcher

        store = _make_store()
        matcher = NodeMatcher(store=store)

        results = await matcher.match(
            ["feedback loop"],
            context="need a causal feedback mechanism",
            mechanism_hint="feedback threshold transition",
        )

        assert len(results) == 1
        r = results[0]
        assert r.chosen_id == "n3"
        assert r.candidates[0].type == "schema"
        assert "recall" in r.rationale


# ============================================================
# EmbeddingClient & NodeEmbeddingIndex — unit tests
# ============================================================
class TestCosineHelper:
    """Test the _cosine_similarity helper."""

    def test_identical_vectors(self) -> None:
        from x_creative.hkg.embeddings import _cosine_similarity

        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        from x_creative.hkg.embeddings import _cosine_similarity

        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        from x_creative.hkg.embeddings import _cosine_similarity

        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        """Zero vector should return 0.0 without raising."""
        from x_creative.hkg.embeddings import _cosine_similarity

        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


class TestNodeEmbeddingIndex:
    """Tests for NodeEmbeddingIndex.find_nearest."""

    def test_find_nearest_basic(self) -> None:
        """find_nearest returns the closest node within threshold."""
        from x_creative.hkg.embeddings import NodeEmbeddingIndex

        idx = NodeEmbeddingIndex()
        # Manually set up internal data (skip async build)
        idx._node_ids = ["n1", "n2", "n3"]
        idx._embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        # Query close to n1
        results = idx.find_nearest([0.9, 0.1, 0.0], top_k=2, threshold=0.5)

        assert len(results) >= 1
        # n1 should be the closest
        assert results[0][0] == "n1"
        assert results[0][1] > 0.9

    def test_find_nearest_threshold_filter(self) -> None:
        """Nodes below threshold are excluded."""
        from x_creative.hkg.embeddings import NodeEmbeddingIndex

        idx = NodeEmbeddingIndex()
        idx._node_ids = ["n1", "n2"]
        idx._embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        # Query orthogonal to both with high threshold
        results = idx.find_nearest([0.0, 0.0, 1.0], top_k=5, threshold=0.5)
        assert results == []

    def test_save_and_load_roundtrip(self, tmp_path) -> None:  # noqa: ANN001
        """Embedding index should persist and reload node ids/vectors."""
        from x_creative.hkg.embeddings import NodeEmbeddingIndex

        idx = NodeEmbeddingIndex()
        idx._node_ids = ["n1", "n2"]
        idx._embeddings = [[1.0, 0.0], [0.0, 1.0]]

        path = tmp_path / "index.json"
        idx.save(path)

        loaded = NodeEmbeddingIndex.load(path)
        assert loaded._node_ids == ["n1", "n2"]
        assert loaded._embeddings == [[1.0, 0.0], [0.0, 1.0]]


class TestNodeMatcherWithEmbedding:
    """Test embedding fallback in NodeMatcher."""

    async def test_embedding_fallback(self) -> None:
        """When exact match fails and embedding client is available, use embedding."""
        from unittest.mock import AsyncMock

        from x_creative.hkg.embeddings import EmbeddingClient, NodeEmbeddingIndex
        from x_creative.hkg.matcher import NodeMatcher

        store = _make_store()

        # Mock the embedding client
        mock_client = AsyncMock(spec=EmbeddingClient)
        mock_client.embed_cached.return_value = [0.9, 0.1, 0.0]

        # Set up a pre-built embedding index
        emb_index = NodeEmbeddingIndex()
        emb_index._node_ids = ["n1", "n2"]
        emb_index._embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        matcher = NodeMatcher(
            store=store, embedding_client=mock_client, embedding_index=emb_index
        )

        results = await matcher.match(["randomXYZ"])

        assert len(results) == 1
        r = results[0]
        assert r.match_type == "embedding"
        assert r.confidence > 0.5
        assert "n1" in r.matched_node_ids

    async def test_lazy_build_embedding_index_when_missing(self) -> None:
        """Embedding index should be lazily built when client exists but index is absent."""
        from unittest.mock import AsyncMock

        from x_creative.hkg.matcher import NodeMatcher

        store = _make_store()
        mock_client = AsyncMock()
        # Build-time embeddings for node names ["entropy", "pressure"].
        mock_client.embed = AsyncMock(return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # Query embedding for random term (close to n1).
        mock_client.embed_cached = AsyncMock(return_value=[0.9, 0.1, 0.0])

        matcher = NodeMatcher(store=store, embedding_client=mock_client, embedding_index=None)
        results = await matcher.match(["randomXYZ"], mode="embedding")

        assert len(results) == 1
        assert results[0].match_type == "embedding"
        assert "n1" in results[0].matched_node_ids
        assert matcher._embedding_index is not None
        assert mock_client.embed.await_count == 1


class TestNodeMatcherCache:
    """#9: NodeMatcher should cache term->node_candidates results."""

    async def test_matcher_node_candidates_cached(self) -> None:
        """Same term matched twice should hit cache on second call."""
        from x_creative.hkg.matcher import NodeMatcher
        store = _make_store()
        matcher = NodeMatcher(store=store)
        results1 = await matcher.match(["entropy"])
        results2 = await matcher.match(["entropy"])
        assert results1[0].matched_node_ids == results2[0].matched_node_ids
        assert hasattr(matcher, "_match_cache")
        assert any(
            cache_key[0] == "entropy" and cache_key[1] == "auto"
            for cache_key in matcher._match_cache
        )

    async def test_matcher_different_modes_separate_cache(self) -> None:
        """Different modes should have separate cache entries."""
        from x_creative.hkg.matcher import NodeMatcher
        store = _make_store()
        matcher = NodeMatcher(store=store)
        await matcher.match(["entropy"], mode="auto")
        await matcher.match(["entropy"], mode="exact")
        assert any(
            cache_key[0] == "entropy" and cache_key[1] == "auto"
            for cache_key in matcher._match_cache
        )
        assert any(
            cache_key[0] == "entropy" and cache_key[1] == "exact"
            for cache_key in matcher._match_cache
        )
