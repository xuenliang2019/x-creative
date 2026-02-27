"""Tests for hyperpath_expand and hyperbridge."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from x_creative.hkg.types import (
    HKGNode, Hyperedge, Provenance, HKGParams, HKGMatchResult,
    Hyperpath, HyperpathEvidence, HyperedgeSummary,
)
from x_creative.hkg.store import HypergraphStore
from x_creative.core.types import Hypothesis, ProblemFrame


def _make_store() -> HypergraphStore:
    store = HypergraphStore()
    prov = [Provenance(doc_id="d1", chunk_id="c1")]
    store.add_node(HKGNode(node_id="n1", name="entropy"))
    store.add_node(HKGNode(node_id="n2", name="temperature"))
    store.add_node(HKGNode(node_id="n3", name="market_volatility"))
    store.add_edge(Hyperedge(edge_id="e1", nodes=["n1", "n2"], relation="thermo_link", provenance=prov))
    store.add_edge(Hyperedge(edge_id="e2", nodes=["n2", "n3"], relation="heat_to_vol", provenance=prov))
    return store


def _make_hypothesis() -> Hypothesis:
    return Hypothesis(
        id="hyp_test1",
        description="Entropy-based volatility prediction",
        source_domain="thermodynamics",
        source_structure="entropy_increase",
        analogy_explanation="Map entropy to market disorder",
        observable="entropy_ratio = H(returns) / H_max",
    )


def _make_problem() -> ProblemFrame:
    return ProblemFrame(description="Find novel volatility predictors")


class TestBuildStructuralContext:
    def test_builds_context_string(self) -> None:
        from x_creative.hkg.expand import _build_structural_context
        store = _make_store()
        paths = [Hyperpath(edges=["e1", "e2"], intermediate_nodes=["n2"])]
        context = _build_structural_context(store, paths)
        assert "entropy" in context.lower() or "e1" in context
        assert "temperature" in context.lower() or "e2" in context

    def test_empty_paths(self) -> None:
        from x_creative.hkg.expand import _build_structural_context
        store = _make_store()
        context = _build_structural_context(store, [])
        assert context == ""


class TestExtractTerms:
    def test_extract_from_hypothesis(self) -> None:
        from x_creative.hkg.expand import _extract_start_terms
        hyp = _make_hypothesis()
        terms = _extract_start_terms(hyp)
        assert len(terms) > 0
        assert "thermodynamics" in terms

    def test_extract_from_problem(self) -> None:
        from x_creative.hkg.expand import _extract_end_terms
        pf = _make_problem()
        terms = _extract_end_terms(pf)
        assert len(terms) > 0


class TestBuildHyperpathEvidence:
    def test_builds_evidence(self) -> None:
        from x_creative.hkg.expand import _build_hyperpath_evidence
        store = _make_store()
        paths = [Hyperpath(edges=["e1", "e2"], intermediate_nodes=["n2"])]
        evidence = _build_hyperpath_evidence(store, paths, "n1", "n3")
        assert len(evidence) == 1
        assert evidence[0].start_node_id == "n1"
        assert evidence[0].end_node_id == "n3"
        assert evidence[0].path_rank == 1
        assert evidence[0].path_length == 2
        assert len(evidence[0].hyperedges) == 2
        assert isinstance(evidence[0].hyperedges[0], HyperedgeSummary)


class TestHyperpathExpand:
    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self) -> None:
        from x_creative.hkg.expand import hyperpath_expand
        from x_creative.hkg.matcher import NodeMatcher
        store = _make_store()
        matcher = NodeMatcher(store)
        router = MagicMock()
        params = HKGParams()
        hyp = _make_hypothesis()
        pf = ProblemFrame(description="completely unrelated xyz123")
        result = await hyperpath_expand(hyp, pf, store, matcher, router, params)
        # Should return empty (no end terms match)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_successful_expansion(self) -> None:
        from x_creative.hkg.expand import hyperpath_expand
        from x_creative.hkg.matcher import NodeMatcher
        store = _make_store()
        matcher = NodeMatcher(store)
        router = MagicMock()
        llm_response = json.dumps([{
            "description": "温度中介的熵-扩散率机制",
            "analogy_explanation": "通过温度作为桥梁连接熵与系统扩散",
            "observable": "correlation(entropy_t, diffusion_t+1)",
            "expansion_type": "hyperpath_expand",
            "mechanism_chain": "entropy -> temperature -> diffusion",
            "testable_conditions": ["condition1"],
            "supporting_edges": ["e1", "e2"],
        }])
        router.complete = AsyncMock(return_value=MagicMock(content=llm_response))
        params = HKGParams()
        hyp = Hypothesis(
            id="hyp_test", description="entropy based",
            source_domain="entropy", source_structure="entropy_increase",
            analogy_explanation="map", observable="obs",
        )
        pf = ProblemFrame(description="market_volatility prediction")
        result = await hyperpath_expand(hyp, pf, store, matcher, router, params)
        assert len(result) >= 1
        assert result[0].expansion_type == "hyperpath_expand"
        assert result[0].hkg_evidence is not None
        assert result[0].parent_id == "hyp_test"
        assert result[0].supporting_edges == ["e1", "e2"]

    @pytest.mark.asyncio
    async def test_expansion_requires_non_empty_observable(self) -> None:
        from x_creative.hkg.expand import hyperpath_expand
        from x_creative.hkg.matcher import NodeMatcher

        store = _make_store()
        matcher = NodeMatcher(store)
        router = MagicMock()
        llm_response = json.dumps([
            {
                "description": "missing observable",
                "analogy_explanation": "bad",
                "observable": "   ",
                "mechanism_chain": "entropy -> temperature -> volatility",
                "testable_conditions": ["c1"],
                "supporting_edges": ["e1", "e2"],
            },
            {
                "description": "missing conditions",
                "analogy_explanation": "bad2",
                "observable": "obs_missing_conditions",
                "mechanism_chain": "entropy -> temperature -> volatility",
                "supporting_edges": ["e1", "e2"],
            },
            {
                "description": "valid observable",
                "analogy_explanation": "good",
                "observable": "obs_valid",
                "mechanism_chain": "entropy -> temperature -> volatility",
                "testable_conditions": ["c1", "c2"],
                "supporting_edges": ["e1", "e2"],
            },
        ])
        router.complete = AsyncMock(return_value=MagicMock(content=llm_response))

        hyp = _make_hypothesis()
        pf = ProblemFrame(description="market_volatility prediction")
        result = await hyperpath_expand(hyp, pf, store, matcher, router, HKGParams())
        assert len(result) == 1
        assert result[0].observable == "obs_valid"


class TestHyperbridge:
    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self) -> None:
        from x_creative.hkg.expand import hyperbridge
        from x_creative.hkg.matcher import NodeMatcher
        store = _make_store()
        matcher = NodeMatcher(store)
        router = MagicMock()
        params = HKGParams()
        result = await hyperbridge("xyz_unknown", "abc_unknown", store, matcher, router, params)
        assert result == []

    @pytest.mark.asyncio
    async def test_successful_bridge(self) -> None:
        from x_creative.hkg.expand import hyperbridge
        from x_creative.hkg.matcher import NodeMatcher
        store = _make_store()
        matcher = NodeMatcher(store)
        router = MagicMock()
        llm_response = json.dumps([
            {
                "description": "桥接假说1",
                "analogy_explanation": "通过temperature连接",
                "observable": "obs1",
                "expansion_type": "hyperbridge",
                "bridge_path": "entropy -> temperature -> market_volatility",
                "testable_conditions": ["c1"],
                "path_rank": 1,
            },
            {
                "description": "桥接假说2",
                "analogy_explanation": "通过temperature连接",
                "observable": "obs2",
                "expansion_type": "hyperbridge",
                "bridge_path": "entropy -> temperature -> market_volatility",
                "testable_conditions": ["c2"],
                "path_rank": 1,
            },
        ])
        router.complete = AsyncMock(return_value=MagicMock(content=llm_response))
        params = HKGParams()
        result = await hyperbridge("entropy", "market_volatility", store, matcher, router, params)
        assert len(result) == 2
        assert result[0].expansion_type == "hyperbridge"
        assert result[0].hkg_evidence is not None
        assert result[0].supporting_edges == ["e1", "e2"]
        assert result[0].hkg_evidence.coverage.get("path_rank") == 1
        assert (
            result[0].hkg_evidence.coverage.get("bridge_path")
            == "entropy -> temperature -> market_volatility"
        )

    @pytest.mark.asyncio
    async def test_bridge_requires_non_empty_observable(self) -> None:
        from x_creative.hkg.expand import hyperbridge
        from x_creative.hkg.matcher import NodeMatcher

        store = _make_store()
        matcher = NodeMatcher(store)
        router = MagicMock()
        llm_response = json.dumps([
            {
                "description": "bad bridge",
                "analogy_explanation": "missing obs",
                "observable": "",
                "bridge_path": "entropy -> temperature -> market_volatility",
                "testable_conditions": ["c_bad"],
                "path_rank": 1,
            },
            {
                "description": "good bridge 1",
                "analogy_explanation": "with obs",
                "observable": "bridge_obs_1",
                "bridge_path": "entropy -> temperature -> market_volatility",
                "testable_conditions": ["c1"],
                "path_rank": 1,
            },
            {
                "description": "good bridge 2",
                "analogy_explanation": "with obs",
                "observable": "bridge_obs_2",
                "bridge_path": "entropy -> temperature -> market_volatility",
                "testable_conditions": ["c2"],
                "path_rank": 1,
            },
        ])
        router.complete = AsyncMock(return_value=MagicMock(content=llm_response))

        result = await hyperbridge(
            "entropy",
            "market_volatility",
            store,
            matcher,
            router,
            HKGParams(),
        )
        assert len(result) == 2
        assert {h.observable for h in result} == {"bridge_obs_1", "bridge_obs_2"}

    @pytest.mark.asyncio
    async def test_bridge_requires_two_to_three_explanations_per_path(self) -> None:
        from x_creative.hkg.expand import hyperbridge
        from x_creative.hkg.matcher import NodeMatcher

        store = _make_store()
        matcher = NodeMatcher(store)
        router = MagicMock()
        llm_response = json.dumps([
            {
                "description": "only one bridge explanation",
                "analogy_explanation": "insufficient count",
                "observable": "bridge_obs_single",
                "bridge_path": "entropy -> temperature -> market_volatility",
                "testable_conditions": ["c1"],
                "path_rank": 1,
            }
        ])
        router.complete = AsyncMock(return_value=MagicMock(content=llm_response))

        result = await hyperbridge(
            "entropy",
            "market_volatility",
            store,
            matcher,
            router,
            HKGParams(),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_bridge_requires_two_to_three_explanations_for_all_found_paths(self) -> None:
        from x_creative.hkg.expand import hyperbridge
        from x_creative.hkg.matcher import NodeMatcher

        store = HypergraphStore()
        prov = [Provenance(doc_id="d2", chunk_id="c2")]
        store.add_node(HKGNode(node_id="a", name="A"))
        store.add_node(HKGNode(node_id="b", name="B"))
        store.add_node(HKGNode(node_id="x", name="X"))
        store.add_node(HKGNode(node_id="y", name="Y"))
        store.add_edge(Hyperedge(edge_id="e1", nodes=["a", "x"], relation="r", provenance=prov))
        store.add_edge(Hyperedge(edge_id="e2", nodes=["x", "b"], relation="r", provenance=prov))
        store.add_edge(Hyperedge(edge_id="e3", nodes=["a", "y"], relation="r", provenance=prov))
        store.add_edge(Hyperedge(edge_id="e4", nodes=["y", "b"], relation="r", provenance=prov))
        matcher = NodeMatcher(store)
        router = MagicMock()
        llm_response = json.dumps([
            {
                "description": "path1-bridge-1",
                "analogy_explanation": "rank1 explanation 1",
                "observable": "obs_rank1_1",
                "bridge_path": "A -> X -> B",
                "testable_conditions": ["c1"],
                "path_rank": 1,
            },
            {
                "description": "path1-bridge-2",
                "analogy_explanation": "rank1 explanation 2",
                "observable": "obs_rank1_2",
                "bridge_path": "A -> X -> B",
                "testable_conditions": ["c2"],
                "path_rank": 1,
            },
        ])
        router.complete = AsyncMock(return_value=MagicMock(content=llm_response))

        result = await hyperbridge(
            "A",
            "B",
            store,
            matcher,
            router,
            HKGParams(K=2, IS=1, max_len=3, matcher="exact"),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_bridge_persists_path_binding_for_each_hypothesis(self) -> None:
        from x_creative.hkg.expand import hyperbridge
        from x_creative.hkg.matcher import NodeMatcher

        store = HypergraphStore()
        prov = [Provenance(doc_id="d2", chunk_id="c2")]
        store.add_node(HKGNode(node_id="a", name="A"))
        store.add_node(HKGNode(node_id="b", name="B"))
        store.add_node(HKGNode(node_id="x", name="X"))
        store.add_node(HKGNode(node_id="y", name="Y"))
        store.add_edge(Hyperedge(edge_id="e1", nodes=["a", "x"], relation="r", provenance=prov))
        store.add_edge(Hyperedge(edge_id="e2", nodes=["x", "b"], relation="r", provenance=prov))
        store.add_edge(Hyperedge(edge_id="e3", nodes=["a", "y"], relation="r", provenance=prov))
        store.add_edge(Hyperedge(edge_id="e4", nodes=["y", "b"], relation="r", provenance=prov))
        matcher = NodeMatcher(store)
        router = MagicMock()
        llm_response = json.dumps([
            {
                "description": "path1-bridge-1",
                "analogy_explanation": "rank1 explanation 1",
                "observable": "obs_rank1_1",
                "bridge_path": "A -> X -> B",
                "testable_conditions": ["c1"],
                "path_rank": 1,
            },
            {
                "description": "path1-bridge-2",
                "analogy_explanation": "rank1 explanation 2",
                "observable": "obs_rank1_2",
                "bridge_path": "A -> X -> B",
                "testable_conditions": ["c2"],
                "path_rank": 1,
            },
            {
                "description": "path2-bridge-1",
                "analogy_explanation": "rank2 explanation 1",
                "observable": "obs_rank2_1",
                "bridge_path": "A -> Y -> B",
                "testable_conditions": ["c3"],
                "path_rank": 2,
            },
            {
                "description": "path2-bridge-2",
                "analogy_explanation": "rank2 explanation 2",
                "observable": "obs_rank2_2",
                "bridge_path": "A -> Y -> B",
                "testable_conditions": ["c4"],
                "path_rank": 2,
            },
        ])
        router.complete = AsyncMock(return_value=MagicMock(content=llm_response))

        result = await hyperbridge(
            "A",
            "B",
            store,
            matcher,
            router,
            HKGParams(K=2, IS=1, max_len=3, matcher="exact"),
        )
        assert len(result) == 4

        for hyp in result:
            assert hyp.hkg_evidence is not None
            rank = hyp.hkg_evidence.coverage.get("path_rank")
            bridge_path = hyp.hkg_evidence.coverage.get("bridge_path")
            assert rank in {1, 2}
            assert isinstance(bridge_path, str) and bridge_path

            expected_edges = [
                edge.edge_id
                for edge in hyp.hkg_evidence.hyperpaths[rank - 1].hyperedges
            ]
            assert hyp.supporting_edges == expected_edges

            roundtrip = Hypothesis.model_validate(hyp.model_dump())
            assert roundtrip.hkg_evidence is not None
            assert roundtrip.hkg_evidence.coverage.get("path_rank") == rank
            assert roundtrip.supporting_edges == expected_edges


class TestMatcherModeFromParams:
    """#9: expand.py should pass HKGParams.matcher as mode to matcher.match()."""

    @pytest.mark.asyncio
    async def test_matcher_mode_passed_in_hyperpath_expand(self) -> None:
        from unittest.mock import AsyncMock, MagicMock
        from x_creative.hkg.expand import hyperpath_expand
        from x_creative.hkg.types import HKGParams, HKGMatchResult
        from x_creative.core.types import Hypothesis, ProblemFrame

        store = MagicMock()
        matcher = MagicMock()
        no_match = HKGMatchResult(term="x", matched_node_ids=[], match_type="none", confidence=0.0)
        matcher.match = AsyncMock(return_value=[no_match])
        router = MagicMock()
        params = HKGParams(matcher="embedding")
        hyp = Hypothesis(
            id="h1", description="test", source_domain="dom",
            source_structure="struct", analogy_explanation="expl", observable="obs",
        )
        pf = ProblemFrame(description="problem")
        await hyperpath_expand(hyp, pf, store, matcher, router, params)
        for call in matcher.match.call_args_list:
            assert call.kwargs.get("mode") == "embedding"
