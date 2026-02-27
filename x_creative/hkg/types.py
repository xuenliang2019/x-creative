"""Data types for the HKG (Hypergraph Knowledge Grounding) subsystem."""

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator

# ---------- match_type literal ----------
MatchType = Literal["exact", "alias", "embedding", "none"]
NodeType = Literal[
    "entity",
    "schema",
    "method",
    "metric",
    "property",
    "concept",
    "variable",
    "domain",
    "other",
]


# ---------- Span (replaces dict[str, int]) ----------
class Span(BaseModel):
    """Character-level span within a document chunk."""

    start: int = Field(..., description="Start character offset (inclusive)")
    end: int = Field(..., description="End character offset (exclusive)")


class Provenance(BaseModel):
    """Source provenance for a hyperedge, linking back to document chunks."""

    doc_id: str = Field(..., description="Document identifier")
    chunk_id: str = Field(..., description="Chunk identifier within the document")
    span: Span | None = Field(default=None, description="Character span within the chunk")
    quote_hash: str | None = Field(
        default=None, description="Hash of the quoted text for dedup"
    )
    source_uri: str | None = Field(
        default=None, description="URI of the original source document"
    )

    @field_validator("doc_id", "chunk_id")
    @classmethod
    def _non_empty_refs(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class HKGNode(BaseModel):
    """A node in the hypergraph knowledge graph."""

    node_id: str = Field(..., description="Unique node identifier")
    name: str = Field(..., description="Canonical name of the node")
    aliases: list[str] = Field(
        default_factory=list, description="Alternative names or symbols"
    )
    type: NodeType = Field(
        default="entity",
        description="Node type (entity/schema/method/metric/property/concept/variable/domain/other)",
    )
    motif_id: str | None = Field(
        default=None,
        description="Optional schema motif identifier (e.g. feedback_loop, phase_transition)",
    )


# ---------- HyperedgeSummary (replaces list[dict[str, Any]]) ----------
class HyperedgeSummary(BaseModel):
    """Lightweight summary of a hyperedge used inside HyperpathEvidence."""

    edge_id: str = Field(..., description="Hyperedge identifier")
    nodes: list[str] = Field(..., description="Node IDs participating in this edge")
    relation: str = Field(default="", description="Relation label")
    provenance_refs: list[str] = Field(
        default_factory=list, description="Provenance reference IDs"
    )


class Hyperedge(BaseModel):
    """A hyperedge connecting two or more nodes, with provenance."""

    edge_id: str = Field(..., description="Unique hyperedge identifier")
    nodes: list[str] = Field(
        ..., min_length=2, description="Node IDs connected by this edge (>= 2)"
    )
    relation: str = Field(default="", description="Relation label for this edge")
    provenance: list[Provenance] = Field(
        ..., min_length=1, description="Provenance entries (>= 1 required)"
    )


class Hyperpath(BaseModel):
    """An ordered sequence of hyperedges forming a path through the graph."""

    edges: list[str] = Field(..., description="Ordered hyperedge IDs in the path")
    intermediate_nodes: list[str] = Field(
        ..., description="Intersection nodes between consecutive edges"
    )
    provenance_refs: list[list[str]] = Field(
        default_factory=list,
        description=(
            "Per-edge provenance references aligned with `edges`; "
            "each inner list contains refs like 'doc_id/chunk_id'"
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def length(self) -> int:
        """Number of edges in the path."""
        return len(self.edges)


class HyperpathEvidence(BaseModel):
    """Evidence from a single hyperpath traversal."""

    start_node_id: str = Field(..., description="Starting node ID of the path")
    end_node_id: str = Field(..., description="Ending node ID of the path")
    path_rank: int = Field(..., description="Rank of this path among candidates")
    path_length: int = Field(..., description="Number of edges in the path")
    hyperedges: list[HyperedgeSummary] = Field(
        ..., description="Summaries of hyperedges along the path"
    )
    intermediate_nodes: list[str] = Field(
        ..., description="Intermediate node IDs along the path"
    )


class HKGMatchResult(BaseModel):
    """Result of matching a term to nodes in the hypergraph."""

    term: str = Field(..., description="The query term that was matched")
    matched_node_ids: list[str] = Field(
        ..., description="IDs of nodes matching the term"
    )
    match_type: MatchType = Field(
        ..., description="How the match was found (exact/alias/embedding/none)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Match confidence (0-1)"
    )
    candidates: list["HKGMatchCandidate"] = Field(
        default_factory=list,
        description="Top-k candidates after recall+rereank with confidence chain",
    )
    chosen_id: str | None = Field(
        default=None,
        description="Final chosen node id after reranking (if any)",
    )
    rationale: str = Field(
        default="",
        description="Short rationale for the chosen candidate",
    )


class HKGMatchCandidate(BaseModel):
    """Candidate node produced by matcher recall+rereank."""

    node_id: str = Field(..., description="Candidate node id")
    name: str = Field(..., description="Candidate node canonical name")
    type: NodeType = Field(..., description="Candidate node type")
    method: MatchType = Field(..., description="Candidate retrieval method")
    score: float = Field(..., ge=0.0, le=1.0, description="Candidate confidence score")
    rationale: str = Field(default="", description="Why this candidate was ranked here")


class HKGParams(BaseModel):
    """Parameters for HKG hyperpath search.

    ``K`` and ``IS`` follow the notation from arXiv:2601.04878:
    - K  -- number of top-ranked hyperpaths to return per query pair.
    - IS -- intersection size threshold for consecutive hyperedges.
    """

    K: int = Field(default=3, description="Number of top-ranked hyperpaths to return")
    IS: int = Field(
        default=1, description="Intersection size threshold for consecutive edges"
    )
    max_len: int = Field(default=6, description="Maximum hyperpath length (in edges)")
    matcher: str = Field(
        default="auto", description="Matcher strategy (auto/exact/embedding)"
    )
    top_n_hypotheses: int = Field(
        default=5, description="Number of hypotheses to surface from HKG evidence"
    )


class HKGEvidence(BaseModel):
    """Aggregated evidence from HKG hyperpath search."""

    hyperpaths: list[HyperpathEvidence] = Field(
        default_factory=list, description="Collected hyperpath evidence items"
    )
    hkg_params: HKGParams | None = Field(
        default=None, description="Parameters used for the search"
    )
    coverage: dict[str, Any] = Field(
        default_factory=dict, description="Coverage statistics (node/edge ratios)"
    )
