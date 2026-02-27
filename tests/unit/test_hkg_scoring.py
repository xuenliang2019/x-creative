"""Tests for HKG structural grounding score."""

from x_creative.hkg.types import (
    HKGEvidence,
    HKGParams,
    HyperedgeSummary,
    HyperpathEvidence,
)


def _make_path(
    num_edges: int,
    nodes_per_edge: int = 2,
    provenance_refs_per_edge: int = 1,
) -> HyperpathEvidence:
    """Helper to build a HyperpathEvidence with controllable parameters."""
    hyperedges = []
    for i in range(num_edges):
        node_ids = [f"n_{i}_{j}" for j in range(nodes_per_edge)]
        prov_refs = [f"prov_{i}_{k}" for k in range(provenance_refs_per_edge)]
        hyperedges.append(
            HyperedgeSummary(
                edge_id=f"e_{i}",
                nodes=node_ids,
                relation="relates",
                provenance_refs=prov_refs,
            )
        )
    intermediate = [f"n_inter_{i}" for i in range(max(0, num_edges - 1))]
    return HyperpathEvidence(
        start_node_id="n_start",
        end_node_id="n_end",
        path_rank=1,
        path_length=num_edges,
        hyperedges=hyperedges,
        intermediate_nodes=intermediate,
    )


class TestStructuralGroundingScore:
    """Tests for structural_grounding_score."""

    def test_no_evidence_returns_none(self) -> None:
        """None input -> None."""
        from x_creative.hkg.scoring import structural_grounding_score

        assert structural_grounding_score(None) is None

    def test_empty_hyperpaths_returns_none(self) -> None:
        """HKGEvidence with no hyperpaths -> None."""
        from x_creative.hkg.scoring import structural_grounding_score

        evidence = HKGEvidence(hyperpaths=[])
        assert structural_grounding_score(evidence) is None

    def test_valid_evidence_returns_score(self) -> None:
        """Single path with 2 edges -> score in [0, 10]."""
        from x_creative.hkg.scoring import structural_grounding_score

        path = _make_path(num_edges=2, nodes_per_edge=3, provenance_refs_per_edge=2)
        evidence = HKGEvidence(hyperpaths=[path])

        score = structural_grounding_score(evidence)

        assert score is not None
        assert 0.0 <= score <= 10.0

    def test_score_in_range(self) -> None:
        """Short path (1 edge) -> score in [0, 10]."""
        from x_creative.hkg.scoring import structural_grounding_score

        path = _make_path(num_edges=1)
        evidence = HKGEvidence(hyperpaths=[path])

        score = structural_grounding_score(evidence)

        assert score is not None
        assert 0.0 <= score <= 10.0

    def test_shorter_paths_score_higher(self) -> None:
        """A 1-edge path should score higher than a 3-edge path (all else equal)."""
        from x_creative.hkg.scoring import structural_grounding_score

        short_path = _make_path(num_edges=1, nodes_per_edge=2, provenance_refs_per_edge=1)
        long_path = _make_path(num_edges=3, nodes_per_edge=2, provenance_refs_per_edge=1)

        short_evidence = HKGEvidence(hyperpaths=[short_path])
        long_evidence = HKGEvidence(hyperpaths=[long_path])

        short_score = structural_grounding_score(short_evidence)
        long_score = structural_grounding_score(long_evidence)

        assert short_score is not None
        assert long_score is not None
        assert short_score > long_score

    def test_larger_intersections_score_higher(self) -> None:
        """avg_intersection_size should improve score when overlap is larger."""
        from x_creative.hkg.scoring import structural_grounding_score

        low_overlap = HyperpathEvidence(
            start_node_id="s",
            end_node_id="t",
            path_rank=1,
            path_length=3,
            hyperedges=[
                HyperedgeSummary(edge_id="e1", nodes=["a", "b", "c", "d"], relation="r", provenance_refs=["p1"]),
                HyperedgeSummary(edge_id="e2", nodes=["d", "e", "f", "g"], relation="r", provenance_refs=["p2"]),
                HyperedgeSummary(edge_id="e3", nodes=["g", "h", "i", "j"], relation="r", provenance_refs=["p3"]),
            ],
            intermediate_nodes=["d", "g"],
        )
        high_overlap = HyperpathEvidence(
            start_node_id="s",
            end_node_id="t",
            path_rank=1,
            path_length=3,
            hyperedges=[
                HyperedgeSummary(edge_id="e1", nodes=["a", "b", "c", "d"], relation="r", provenance_refs=["p1"]),
                HyperedgeSummary(edge_id="e2", nodes=["a", "b", "e", "f"], relation="r", provenance_refs=["p2"]),
                HyperedgeSummary(edge_id="e3", nodes=["e", "f", "g", "h"], relation="r", provenance_refs=["p3"]),
            ],
            intermediate_nodes=["a", "b", "e", "f"],
        )

        low_score = structural_grounding_score(
            HKGEvidence(hyperpaths=[low_overlap], hkg_params=HKGParams(IS=2))
        )
        high_score = structural_grounding_score(
            HKGEvidence(hyperpaths=[high_overlap], hkg_params=HKGParams(IS=2))
        )
        assert low_score is not None
        assert high_score is not None
        assert high_score > low_score

    def test_super_hub_paths_are_penalized(self) -> None:
        """hub_ratio should penalize paths that over-rely on one node."""
        from x_creative.hkg.scoring import structural_grounding_score

        super_hub = HyperpathEvidence(
            start_node_id="s",
            end_node_id="t",
            path_rank=1,
            path_length=3,
            hyperedges=[
                HyperedgeSummary(edge_id="e1", nodes=["hub", "a"], relation="r", provenance_refs=["p1"]),
                HyperedgeSummary(edge_id="e2", nodes=["hub", "b"], relation="r", provenance_refs=["p2"]),
                HyperedgeSummary(edge_id="e3", nodes=["hub", "c"], relation="r", provenance_refs=["p3"]),
            ],
            intermediate_nodes=["hub"],
        )
        less_hub = HyperpathEvidence(
            start_node_id="s",
            end_node_id="t",
            path_rank=1,
            path_length=3,
            hyperedges=[
                HyperedgeSummary(edge_id="e1", nodes=["a", "b"], relation="r", provenance_refs=["p1"]),
                HyperedgeSummary(edge_id="e2", nodes=["b", "c"], relation="r", provenance_refs=["p2"]),
                HyperedgeSummary(edge_id="e3", nodes=["c", "d"], relation="r", provenance_refs=["p3"]),
            ],
            intermediate_nodes=["b", "c"],
        )

        score_super_hub = structural_grounding_score(
            HKGEvidence(hyperpaths=[super_hub], hkg_params=HKGParams(IS=1))
        )
        score_less_hub = structural_grounding_score(
            HKGEvidence(hyperpaths=[less_hub], hkg_params=HKGParams(IS=1))
        )
        assert score_super_hub is not None
        assert score_less_hub is not None
        assert score_less_hub > score_super_hub

    def test_low_match_confidence_lowers_structural_score(self) -> None:
        """Low start/end match confidence should gate structural score."""
        from x_creative.hkg.scoring import structural_grounding_score

        path = HyperpathEvidence(
            start_node_id="s",
            end_node_id="t",
            path_rank=1,
            path_length=2,
            hyperedges=[
                HyperedgeSummary(edge_id="e1", nodes=["a", "b"], relation="r", provenance_refs=["p1"]),
                HyperedgeSummary(edge_id="e2", nodes=["b", "c"], relation="r", provenance_refs=["p2"]),
            ],
            intermediate_nodes=["b"],
        )

        high_conf = HKGEvidence(
            hyperpaths=[path],
            hkg_params=HKGParams(IS=1),
            coverage={
                "start_match": {"confidence": 0.95},
                "end_match": {"confidence": 0.9},
            },
        )
        low_conf = HKGEvidence(
            hyperpaths=[path],
            hkg_params=HKGParams(IS=1),
            coverage={
                "start_match": {"confidence": 0.2},
                "end_match": {"confidence": 0.1},
            },
        )

        high_score = structural_grounding_score(high_conf)
        low_score = structural_grounding_score(low_conf)
        assert high_score is not None
        assert low_score is not None
        assert high_score > low_score
