"""Tests for HKG data types."""

import pytest
from pydantic import ValidationError


class TestSpan:
    """Tests for the Span model."""

    def test_create_valid(self) -> None:
        """Test creating a valid Span."""
        from x_creative.hkg.types import Span

        span = Span(start=10, end=50)

        assert span.start == 10
        assert span.end == 50


class TestProvenance:
    """Tests for Provenance model."""

    def test_create_valid(self) -> None:
        """Test creating a valid Provenance with required fields only."""
        from x_creative.hkg.types import Provenance

        prov = Provenance(doc_id="doc_001", chunk_id="chunk_042")

        assert prov.doc_id == "doc_001"
        assert prov.chunk_id == "chunk_042"
        assert prov.span is None
        assert prov.quote_hash is None
        assert prov.source_uri is None

    def test_create_with_span(self) -> None:
        """Test creating a Provenance with a Span model and optional fields."""
        from x_creative.hkg.types import Provenance, Span

        prov = Provenance(
            doc_id="doc_001",
            chunk_id="chunk_042",
            span=Span(start=10, end=50),
            quote_hash="abc123",
            source_uri="https://example.com/paper.pdf",
        )

        assert prov.span is not None
        assert prov.span.start == 10
        assert prov.span.end == 50
        assert prov.quote_hash == "abc123"
        assert prov.source_uri == "https://example.com/paper.pdf"

    def test_doc_id_must_be_non_empty(self) -> None:
        """Provenance doc_id should reject blank values."""
        from x_creative.hkg.types import Provenance

        with pytest.raises(ValidationError):
            Provenance(doc_id="   ", chunk_id="chunk_001")

    def test_chunk_id_must_be_non_empty(self) -> None:
        """Provenance chunk_id should reject blank values."""
        from x_creative.hkg.types import Provenance

        with pytest.raises(ValidationError):
            Provenance(doc_id="doc_001", chunk_id="")


class TestHKGNode:
    """Tests for HKGNode model."""

    def test_create_minimal(self) -> None:
        """Test creating an HKGNode with required fields only."""
        from x_creative.hkg.types import HKGNode

        node = HKGNode(node_id="n_001", name="entropy")

        assert node.node_id == "n_001"
        assert node.name == "entropy"
        assert node.aliases == []
        assert node.type == "entity"

    def test_create_with_aliases(self) -> None:
        """Test creating an HKGNode with aliases and custom type."""
        from x_creative.hkg.types import HKGNode

        node = HKGNode(
            node_id="n_002",
            name="Shannon Entropy",
            aliases=["information entropy", "H(X)"],
            type="concept",
        )

        assert node.node_id == "n_002"
        assert node.name == "Shannon Entropy"
        assert len(node.aliases) == 2
        assert "H(X)" in node.aliases
        assert node.type == "concept"

    def test_schema_node_with_motif(self) -> None:
        """Schema nodes should support motif_id for pattern-level alignment."""
        from x_creative.hkg.types import HKGNode

        node = HKGNode(
            node_id="n_schema_1",
            name="feedback_loop",
            type="schema",
            motif_id="feedback_loop",
        )
        assert node.type == "schema"
        assert node.motif_id == "feedback_loop"


class TestHyperedgeSummary:
    """Tests for HyperedgeSummary model."""

    def test_create_minimal(self) -> None:
        """Test creating a HyperedgeSummary with required fields only."""
        from x_creative.hkg.types import HyperedgeSummary

        summary = HyperedgeSummary(edge_id="e_001", nodes=["n_001", "n_002"])

        assert summary.edge_id == "e_001"
        assert summary.nodes == ["n_001", "n_002"]
        assert summary.relation == ""
        assert summary.provenance_refs == []

    def test_create_full(self) -> None:
        """Test creating a HyperedgeSummary with all fields."""
        from x_creative.hkg.types import HyperedgeSummary

        summary = HyperedgeSummary(
            edge_id="e_001",
            nodes=["n_001", "n_002"],
            relation="causes",
            provenance_refs=["prov_001", "prov_002"],
        )

        assert summary.relation == "causes"
        assert len(summary.provenance_refs) == 2


class TestHyperedge:
    """Tests for Hyperedge model."""

    def test_create_valid(self) -> None:
        """Test creating a valid Hyperedge."""
        from x_creative.hkg.types import Hyperedge, Provenance

        edge = Hyperedge(
            edge_id="e_001",
            nodes=["n_001", "n_002", "n_003"],
            relation="causes",
            provenance=[
                Provenance(doc_id="doc_001", chunk_id="chunk_001"),
            ],
        )

        assert edge.edge_id == "e_001"
        assert len(edge.nodes) == 3
        assert edge.relation == "causes"
        assert len(edge.provenance) == 1

    def test_requires_at_least_one_provenance(self) -> None:
        """Test that Hyperedge requires at least 1 provenance entry."""
        from x_creative.hkg.types import Hyperedge

        with pytest.raises(ValidationError) as exc_info:
            Hyperedge(
                edge_id="e_bad",
                nodes=["n_001", "n_002"],
                relation="related",
                provenance=[],  # Empty - should fail
            )

        assert "provenance" in str(exc_info.value).lower()

    def test_requires_at_least_two_nodes(self) -> None:
        """Test that Hyperedge requires at least 2 nodes."""
        from x_creative.hkg.types import Hyperedge, Provenance

        with pytest.raises(ValidationError) as exc_info:
            Hyperedge(
                edge_id="e_bad",
                nodes=["n_001"],  # Only 1 node - should fail
                relation="self",
                provenance=[
                    Provenance(doc_id="doc_001", chunk_id="chunk_001"),
                ],
            )

        assert "nodes" in str(exc_info.value).lower()


class TestHyperpath:
    """Tests for Hyperpath model."""

    def test_create_valid(self) -> None:
        """Test creating a valid Hyperpath with computed length."""
        from x_creative.hkg.types import Hyperpath

        path = Hyperpath(
            edges=["e_001", "e_002", "e_003"],
            intermediate_nodes=["n_002", "n_003"],
        )

        assert path.edges == ["e_001", "e_002", "e_003"]
        assert path.intermediate_nodes == ["n_002", "n_003"]
        assert path.length == 3  # computed from len(edges)

    def test_length_is_computed(self) -> None:
        """Test that length is always derived from edges, not stored."""
        from x_creative.hkg.types import Hyperpath

        path = Hyperpath(
            edges=["e_001"],
            intermediate_nodes=[],
        )

        assert path.length == 1

    def test_length_in_serialization(self) -> None:
        """Test that computed length appears in model_dump output."""
        from x_creative.hkg.types import Hyperpath

        path = Hyperpath(
            edges=["e_001", "e_002"],
            intermediate_nodes=["n_002"],
        )

        data = path.model_dump()
        assert data["length"] == 2


class TestHKGParams:
    """Tests for HKGParams model."""

    def test_defaults(self) -> None:
        """Test that HKGParams has correct default values."""
        from x_creative.hkg.types import HKGParams

        params = HKGParams()

        assert params.K == 3
        assert params.IS == 1
        assert params.max_len == 6
        assert params.matcher == "auto"
        assert params.top_n_hypotheses == 5

    def test_custom_values(self) -> None:
        """Test HKGParams with custom values."""
        from x_creative.hkg.types import HKGParams

        params = HKGParams(K=5, IS=2, max_len=10, matcher="embedding", top_n_hypotheses=10)

        assert params.K == 5
        assert params.IS == 2
        assert params.max_len == 10
        assert params.matcher == "embedding"
        assert params.top_n_hypotheses == 10

    def test_docstring_mentions_paper(self) -> None:
        """Test that HKGParams docstring references the paper notation."""
        from x_creative.hkg.types import HKGParams

        assert "arXiv:2601.04878" in (HKGParams.__doc__ or "")


class TestHyperpathEvidence:
    """Tests for HyperpathEvidence model."""

    def test_create(self) -> None:
        """Test creating a HyperpathEvidence instance with HyperedgeSummary."""
        from x_creative.hkg.types import HyperedgeSummary, HyperpathEvidence

        evidence = HyperpathEvidence(
            start_node_id="n_001",
            end_node_id="n_005",
            path_rank=1,
            path_length=3,
            hyperedges=[
                HyperedgeSummary(
                    edge_id="e_001", nodes=["n_001", "n_002"], relation="causes"
                ),
                HyperedgeSummary(
                    edge_id="e_002", nodes=["n_002", "n_003"], relation="implies"
                ),
                HyperedgeSummary(
                    edge_id="e_003", nodes=["n_003", "n_005"], relation="leads_to"
                ),
            ],
            intermediate_nodes=["n_002", "n_003"],
        )

        assert evidence.start_node_id == "n_001"
        assert evidence.end_node_id == "n_005"
        assert evidence.path_rank == 1
        assert evidence.path_length == 3
        assert len(evidence.hyperedges) == 3
        assert len(evidence.intermediate_nodes) == 2
        assert evidence.hyperedges[0].edge_id == "e_001"
        assert evidence.hyperedges[0].relation == "causes"


class TestHKGMatchResult:
    """Tests for HKGMatchResult model."""

    def test_create_exact_match(self) -> None:
        """Test creating an exact match result."""
        from x_creative.hkg.types import HKGMatchResult

        result = HKGMatchResult(
            term="entropy",
            matched_node_ids=["n_001"],
            match_type="exact",
            confidence=1.0,
        )

        assert result.term == "entropy"
        assert result.matched_node_ids == ["n_001"]
        assert result.match_type == "exact"
        assert result.confidence == 1.0

    def test_create_no_match(self) -> None:
        """Test creating a no-match result."""
        from x_creative.hkg.types import HKGMatchResult

        result = HKGMatchResult(
            term="nonexistent",
            matched_node_ids=[],
            match_type="none",
            confidence=0.0,
        )
        assert result.matched_node_ids == []
        assert result.match_type == "none"
        assert result.confidence == 0.0

    def test_create_with_candidate_chain(self) -> None:
        """HKGMatchResult should carry rerank candidates and rationale chain."""
        from x_creative.hkg.types import HKGMatchCandidate, HKGMatchResult

        result = HKGMatchResult(
            term="feedback",
            matched_node_ids=["n_schema_1", "n_2"],
            match_type="exact",
            confidence=0.92,
            chosen_id="n_schema_1",
            rationale="recall=exact:1.00, schema_boost",
            candidates=[
                HKGMatchCandidate(
                    node_id="n_schema_1",
                    name="feedback_loop",
                    type="schema",
                    method="exact",
                    score=0.92,
                    rationale="recall=exact:1.00, schema_boost",
                )
            ],
        )
        assert result.chosen_id == "n_schema_1"
        assert result.candidates[0].type == "schema"

    def test_invalid_match_type_rejected(self) -> None:
        """Test that an invalid match_type literal is rejected."""
        from x_creative.hkg.types import HKGMatchResult

        with pytest.raises(ValidationError) as exc_info:
            HKGMatchResult(
                term="entropy",
                matched_node_ids=["n_001"],
                match_type="fuzzy",  # not in Literal
                confidence=0.5,
            )

        assert "match_type" in str(exc_info.value).lower()

    def test_confidence_out_of_range_rejected(self) -> None:
        """Test that confidence outside [0, 1] is rejected."""
        from x_creative.hkg.types import HKGMatchResult

        with pytest.raises(ValidationError):
            HKGMatchResult(
                term="entropy",
                matched_node_ids=["n_001"],
                match_type="exact",
                confidence=1.5,  # > 1.0
            )

        with pytest.raises(ValidationError):
            HKGMatchResult(
                term="entropy",
                matched_node_ids=["n_001"],
                match_type="exact",
                confidence=-0.1,  # < 0.0
            )


class TestHKGEvidence:
    """Tests for HKGEvidence model."""

    def test_empty(self) -> None:
        """Test creating an empty HKGEvidence with defaults."""
        from x_creative.hkg.types import HKGEvidence

        evidence = HKGEvidence()

        assert evidence.hyperpaths == []
        assert evidence.hkg_params is None
        assert evidence.coverage == {}

    def test_with_data(self) -> None:
        """Test creating HKGEvidence with actual data."""
        from x_creative.hkg.types import (
            HKGEvidence,
            HKGParams,
            HyperedgeSummary,
            HyperpathEvidence,
        )

        evidence = HKGEvidence(
            hyperpaths=[
                HyperpathEvidence(
                    start_node_id="n_001",
                    end_node_id="n_005",
                    path_rank=1,
                    path_length=2,
                    hyperedges=[
                        HyperedgeSummary(edge_id="e_001", nodes=["n_001", "n_003"]),
                        HyperedgeSummary(edge_id="e_002", nodes=["n_003", "n_005"]),
                    ],
                    intermediate_nodes=["n_003"],
                ),
            ],
            hkg_params=HKGParams(K=5),
            coverage={"node_coverage": 0.8, "edge_coverage": 0.6},
        )

        assert len(evidence.hyperpaths) == 1
        assert evidence.hkg_params is not None
        assert evidence.hkg_params.K == 5
        assert evidence.coverage["node_coverage"] == 0.8
