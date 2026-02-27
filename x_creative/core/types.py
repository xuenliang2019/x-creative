"""Core data types for X-Creative."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from x_creative.hkg.types import HKGEvidence, HKGParams, HyperpathEvidence

from pydantic import AliasChoices, BaseModel, Field, model_validator


class DomainStructure(BaseModel):
    """A transferable structure from a source domain."""

    id: str = Field(..., description="Unique identifier for the structure")
    name: str = Field(..., description="Display name of the structure")
    description: str = Field(..., description="Description of the structure")
    key_variables: list[str] = Field(
        default_factory=list, description="Key variables in this structure"
    )
    dynamics: str = Field(..., description="Description of the dynamics")


class TargetMapping(BaseModel):
    """Mapping from a source domain structure to a target domain concept."""

    structure: str = Field(..., description="ID of the source structure")
    target: str = Field(..., description="Target domain concept")
    observable: str = Field(..., description="Observable proxy variables or formula")


class Domain(BaseModel):
    """A source domain for bisociation."""

    id: str = Field(..., description="Unique identifier for the domain")
    name: str = Field(..., description="Display name in Chinese")
    name_en: str | None = Field(default=None, description="Display name in English")
    description: str = Field(..., description="Description of the domain")
    structures: list[DomainStructure] = Field(
        default_factory=list, description="List of transferable structures"
    )
    target_mappings: list[TargetMapping] = Field(
        default_factory=list, description="Mappings to target domain concepts"
    )

    def get_structure(self, structure_id: str) -> DomainStructure | None:
        """Get a structure by its ID."""
        for structure in self.structures:
            if structure.id == structure_id:
                return structure
        return None


class MappingItem(BaseModel):
    """A single row in the structural mapping table.

    Maps a concept/relation from the source domain to the target domain,
    with systematicity grouping for evaluating mapping quality.
    """

    source_concept: str = Field(..., description="Source domain concept")
    target_concept: str = Field(..., description="Target domain concept")
    source_relation: str = Field(..., description="Source domain relation/mechanism")
    target_relation: str = Field(..., description="Target domain relation/mechanism")
    mapping_type: Literal["entity", "relation", "constraint", "process"] = Field(
        ..., description="Type of mapping"
    )
    systematicity_group_id: str = Field(
        ..., description="ID of the relation system this mapping belongs to"
    )
    observable_link: str | None = Field(
        default=None, description="Which observable this mapping corresponds to"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Mapping confidence (0-1)"
    )
    evidence_refs: list[str] = Field(
        default_factory=list, description="References to supporting evidence"
    )


class FailureMode(BaseModel):
    """A failure mode describing when a structural mapping breaks down."""

    scenario: str = Field(..., description="Under what conditions the mapping fails")
    why_breaks: str = Field(..., description="Why it fails (which mapping item)")
    detectable_signal: str = Field(
        ..., description="Observable signal for detecting this failure"
    )


class HypothesisScores(BaseModel):
    """Scores for evaluating a hypothesis."""

    divergence: float = Field(
        ..., ge=0.0, le=10.0, description="Semantic divergence score (0-10)"
    )
    testability: float = Field(
        ..., ge=0.0, le=10.0, description="Testability score (0-10)"
    )
    rationale: float = Field(
        ..., ge=0.0, le=10.0, description="Domain rationale score (0-10)"
    )
    robustness: float = Field(
        ..., ge=0.0, le=10.0, description="Robustness prior score (0-10)"
    )
    feasibility: float = Field(
        ..., ge=0.0, le=10.0, description="Data feasibility score (0-10)"
    )

    divergence_reason: str | None = Field(
        default=None, description="Reason for divergence score"
    )
    testability_reason: str | None = Field(
        default=None, description="Reason for testability score"
    )
    rationale_reason: str | None = Field(
        default=None, description="Reason for rationale score"
    )
    robustness_reason: str | None = Field(
        default=None, description="Reason for robustness score"
    )
    feasibility_reason: str | None = Field(
        default=None, description="Reason for feasibility score"
    )

    def composite(
        self,
        w_divergence: float = 0.21,
        w_testability: float = 0.26,
        w_rationale: float = 0.21,
        w_robustness: float = 0.17,
        w_feasibility: float = 0.15,
    ) -> float:
        """Calculate composite score with given weights."""
        return (
            w_divergence * self.divergence
            + w_testability * self.testability
            + w_rationale * self.rationale
            + w_robustness * self.robustness
            + w_feasibility * self.feasibility
        )


class HypothesisEvidence(BaseModel):
    """Unified evidence container on a hypothesis node."""

    hyperpaths: list[HyperpathEvidence] = Field(
        default_factory=list,
        description="Structured hyperpath evidence from HKG traversal",
    )
    hkg_params: dict[str, Any] | None = Field(
        default=None,
        description="HKG traversal parameters (K/IS/max_len/matcher)",
    )
    coverage: dict[str, Any] = Field(
        default_factory=dict,
        description="HKG match coverage metadata",
    )
    blend_network: "BlendNetwork | None" = Field(
        default=None,
        description="Blend network from blend_expand operator",
    )
    space_transform_diff: "SpaceTransformDiff | None" = Field(
        default=None,
        description="Concept-space transform diff from transform_space operator",
    )

    @classmethod
    def from_flat_fields(
        cls,
        hkg_evidence: HKGEvidence | None,
        blend_network: "BlendNetwork | None",
        space_transform_diff: "SpaceTransformDiff | None",
    ) -> "HypothesisEvidence":
        hkg_params: dict[str, Any] | None = None
        coverage: dict[str, Any] = {}
        hyperpaths: list[HyperpathEvidence] = []
        if hkg_evidence is not None:
            hyperpaths = list(hkg_evidence.hyperpaths)
            coverage = dict(hkg_evidence.coverage)
            if hkg_evidence.hkg_params is not None:
                hkg_params = hkg_evidence.hkg_params.model_dump()
        return cls(
            hyperpaths=hyperpaths,
            hkg_params=hkg_params,
            coverage=coverage,
            blend_network=blend_network,
            space_transform_diff=space_transform_diff,
        )


class Hypothesis(BaseModel):
    """A generated hypothesis from the creativity engine."""

    id: str = Field(..., description="Unique identifier for the hypothesis")
    description: str = Field(
        ...,
        validation_alias=AliasChoices("description", "text"),
        serialization_alias="text",
        description="Description text of the hypothesis",
    )
    source_domain: str = Field(..., description="Source domain ID")
    source_structure: str = Field(..., description="Source structure ID")
    analogy_explanation: str = Field(
        ..., description="Explanation of the analogy mapping"
    )
    observable: str = Field(..., description="Observable proxy variables/formula")

    # Optional fields for enhanced hypotheses
    formula: str | None = Field(default=None, description="Factor formula if generated")
    parent_id: str | None = Field(
        default=None, description="Parent hypothesis ID if derived"
    )
    generation: int = Field(
        default=0,
        validation_alias=AliasChoices("generation", "generation_depth"),
        serialization_alias="generation_depth",
        description="Generation depth in the search tree",
    )
    expansion_type: str | None = Field(
        default=None, description="How this was derived (refine/variant/combine/oppose)"
    )

    scores: HypothesisScores | None = Field(default=None, description="Evaluation scores")
    final_score: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Cross-model verification score when available",
    )
    logic_passed: bool | None = Field(
        default=None,
        description="Logic verifier hard-gate result when dual verification is available",
    )
    verify_status: "VerifyStatus | None" = Field(
        default=None,
        description="VERIFY status: passed/failed/escalated/abstained",
    )
    judge_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="VERIFY confidence from multi-sample logic judging",
    )
    position_consistency: bool | None = Field(
        default=None,
        description="Whether logic judging remained consistent across samples",
    )
    injection_detected: bool | None = Field(
        default=None,
        description="Whether prompt-injection patterns were detected",
    )
    novelty_score: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Novelty verifier score when available",
    )
    structural_grounding_score: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Optional HKG structural grounding score in VERIFY stage",
    )
    hkg_evidence: HKGEvidence | None = Field(default=None, description="HKG structural evidence")
    supporting_edges: list[str] | None = Field(
        default=None,
        description="Edge IDs that directly support this hypothesis when available",
    )
    quick_score: float | None = Field(
        default=None,
        description="Lightweight pre-score for SEARCH selection (0-10)",
    )
    mapping_table: list[MappingItem] = Field(
        default_factory=list,
        description="Structural mapping table (source→target relation pairs)",
    )
    failure_modes: list[FailureMode] = Field(
        default_factory=list,
        description="Known failure modes for this hypothesis",
    )
    mapping_quality: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Mapping quality score from MappingScorer (0-10)",
    )
    evidence: HypothesisEvidence | None = Field(
        default=None,
        description="Unified evidence payload (hyperpaths/hkg_params/blend/transform)",
    )

    # Conceptual Blending
    blend_network: BlendNetwork | None = Field(
        default=None, description="Blend network from blend_expand operator"
    )
    # Transformational Creativity
    space_transform_diff: SpaceTransformDiff | None = Field(
        default=None, description="Space transform diff from transform_space operator"
    )
    # QD Behavior Descriptor
    behavior_descriptor: BehaviorDescriptor | None = Field(
        default=None, description="Behavior descriptor for MAP-Elites"
    )

    @model_validator(mode="after")
    def _sync_evidence_fields(self) -> "Hypothesis":
        """Keep legacy flat evidence fields and nested evidence in sync."""
        if self.evidence is None:
            if (
                self.hkg_evidence is None
                and self.blend_network is None
                and self.space_transform_diff is None
            ):
                return self
            self.evidence = HypothesisEvidence.from_flat_fields(
                hkg_evidence=self.hkg_evidence,
                blend_network=self.blend_network,
                space_transform_diff=self.space_transform_diff,
            )
            return self

        if self.blend_network is None and self.evidence.blend_network is not None:
            self.blend_network = self.evidence.blend_network
        if (
            self.space_transform_diff is None
            and self.evidence.space_transform_diff is not None
        ):
            self.space_transform_diff = self.evidence.space_transform_diff
        if (
            self.hkg_evidence is None
            and (
                self.evidence.hyperpaths
                or self.evidence.hkg_params is not None
                or self.evidence.coverage
            )
        ):
            params: HKGParams | None = None
            if self.evidence.hkg_params is not None:
                try:
                    params = HKGParams(**self.evidence.hkg_params)
                except Exception:
                    params = None
            self.hkg_evidence = HKGEvidence(
                hyperpaths=list(self.evidence.hyperpaths),
                hkg_params=params,
                coverage=dict(self.evidence.coverage),
            )

        # Ensure nested view always mirrors any already-populated flat fields.
        if self.evidence.blend_network is None and self.blend_network is not None:
            self.evidence.blend_network = self.blend_network
        if (
            self.evidence.space_transform_diff is None
            and self.space_transform_diff is not None
        ):
            self.evidence.space_transform_diff = self.space_transform_diff
        if self.hkg_evidence is not None:
            self.evidence.hyperpaths = list(self.hkg_evidence.hyperpaths)
            self.evidence.coverage = dict(self.hkg_evidence.coverage)
            if self.hkg_evidence.hkg_params is not None:
                self.evidence.hkg_params = self.hkg_evidence.hkg_params.model_dump()

        return self

    def composite_score(
        self,
        w_divergence: float = 0.21,
        w_testability: float = 0.26,
        w_rationale: float = 0.21,
        w_robustness: float = 0.17,
        w_feasibility: float = 0.15,
    ) -> float:
        """Get composite score, or 0.0 if not scored."""
        if self.scores is None:
            return 0.0
        return self.scores.composite(
            w_divergence, w_testability, w_rationale, w_robustness, w_feasibility
        )

    def novelty_axis(self) -> float | None:
        """Divergence score as novelty axis for Pareto selection."""
        if self.scores is None:
            return None
        return self.scores.divergence

    def feasibility_axis(self) -> float | None:
        """Average of testability+rationale+robustness+feasibility as feasibility axis."""
        if self.scores is None:
            return None
        return (
            self.scores.testability
            + self.scores.rationale
            + self.scores.robustness
            + self.scores.feasibility
        ) / 4.0


class VerifyStatus(str, Enum):
    """Verification outcome status."""

    PASSED = "passed"
    FAILED = "failed"
    ESCALATED = "escalated"
    ABSTAINED = "abstained"


class ConstraintSpec(BaseModel):
    """Structured constraint with priority and metadata."""

    text: str = Field(..., description="Constraint description")
    priority: Literal["critical", "high", "medium", "low"] = Field(
        default="medium", description="Constraint priority"
    )
    type: Literal["hard", "soft"] = Field(
        default="soft", description="Hard (reject if violated) or soft (penalize)"
    )
    origin: Literal["user", "domain_plugin", "risk_refinement"] = Field(
        default="user", description="Source of the constraint"
    )
    evidence: str | None = Field(default=None, description="Supporting evidence")
    weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Constraint importance weight"
    )


class ProblemFrame(BaseModel):
    """Framing of a research problem with constraints."""

    description: str = Field(..., description="Natural language problem description")

    # General fields
    target_domain: str = Field(
        default="open_source_development",
        description="Target domain plugin ID",
    )
    constraints: list[str] = Field(
        default_factory=list, description="Additional constraints"
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific context (free-form)",
    )

    # --- Extended fields (optional, for answer engine) ---
    objective: str | None = Field(
        default=None, description="What the user wants to achieve"
    )
    scope: dict[str, Any] | None = Field(
        default=None, description="In-scope and out-of-scope boundaries"
    )
    definitions: dict[str, str] | None = Field(
        default=None, description="Key term definitions"
    )
    success_criteria: list[str] = Field(
        default_factory=list, description="How to evaluate success"
    )
    open_questions: list[str] = Field(
        default_factory=list, description="Unresolved ambiguities"
    )
    structured_constraints: list[ConstraintSpec] = Field(
        default_factory=list,
        description="Structured constraints with priority and metadata",
    )
    domain_hint: dict[str, Any] | None = Field(
        default=None, description="LLM-inferred domain hint with confidence"
    )


class SearchConfig(BaseModel):
    """Configuration for the Graph of Thoughts search."""

    num_hypotheses: int = Field(
        default=50, ge=1, description="Target number of hypotheses to generate"
    )
    search_depth: int = Field(
        default=3, ge=0, le=10, description="Depth of the search tree"
    )
    search_breadth: int = Field(
        default=5, ge=1, le=20, description="Breadth of expansion at each node"
    )
    prune_threshold: float = Field(
        default=5.0, ge=0.0, le=10.0, description="Minimum score to avoid pruning"
    )
    enable_combination: bool = Field(
        default=True, description="Enable combining hypotheses"
    )
    enable_opposition: bool = Field(
        default=True, description="Enable generating opposing hypotheses"
    )
    enable_extreme: bool = Field(
        default=True, description="Enable extreme-variant hypothesis expansion"
    )
    enable_blending: bool = Field(
        default=False, description="Enable blend_expand operator"
    )
    enable_transform_space: bool = Field(
        default=False, description="Enable transform_space operator"
    )
    max_blend_pairs: int = Field(
        default=3, ge=1, le=10, description="Max blend pairs per round"
    )
    max_transform_hypotheses: int = Field(
        default=2, ge=1, le=5, description="Max hypotheses for transform_space per round"
    )
    runtime_profile: Literal["interactive", "research"] = Field(
        default="research",
        description="Runtime profile controlling heavy-operator policies",
    )
    blend_expand_budget_per_round: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Per-round budget for blend_expand heavy operator",
    )
    transform_space_budget_per_round: int = Field(
        default=2,
        ge=0,
        le=20,
        description="Per-round budget for transform_space heavy operator",
    )
    hyperpath_expand_topN: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Per-round top-N hypotheses eligible for hyperpath_expand",
    )


class LogicVerdict(BaseModel):
    """Result of logic verification."""

    passed: bool = Field(..., description="Whether the hypothesis passed logic verification")
    analogy_validity: float = Field(
        ..., ge=0.0, le=10.0, description="Score for analogy mapping validity"
    )
    internal_consistency: float = Field(
        ..., ge=0.0, le=10.0, description="Score for internal consistency"
    )
    causal_rigor: float = Field(
        ..., ge=0.0, le=10.0, description="Score for causal reasoning rigor"
    )
    reasoning: str = Field(..., description="Explanation of the verdict")
    issues: list[str] = Field(
        default_factory=list, description="Specific issues found"
    )
    judge_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Judge confidence estimate"
    )
    score_std: float | None = Field(
        default=None, ge=0.0, description="Score standard deviation across samples"
    )
    position_consistency: bool = Field(
        default=True, description="Whether scores are consistent across position swaps"
    )
    position_bias_flag: bool = Field(
        default=False, description="Whether position bias was detected"
    )
    injection_detected: bool = Field(
        default=False,
        description="Whether prompt injection was detected in hypothesis content",
    )


class SimilarWork(BaseModel):
    """A similar work found during novelty search."""

    title: str = Field(..., description="Title of the work")
    url: str = Field(..., description="URL to the work")
    source: str = Field(..., description="Source type: arxiv, ssrn, blog, etc.")
    similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity score (0-1)"
    )
    difference_summary: str = Field(
        ..., description="How this work differs from the hypothesis"
    )


class NoveltyVerdict(BaseModel):
    """Result of novelty verification."""

    score: float = Field(..., ge=0.0, le=10.0, description="Novelty score (0-10)")
    searched: bool = Field(..., description="Whether web search was performed")
    similar_works: list[SimilarWork] = Field(
        default_factory=list, description="Similar works found"
    )
    novelty_analysis: str = Field(..., description="Analysis of novelty")
    error: bool = Field(default=False, description="Whether verification failed due to model error")


class VerifiedHypothesis(BaseModel):
    """A hypothesis that has been verified by both logic and novelty checks."""

    # Basic info (same as Hypothesis)
    id: str = Field(..., description="Unique identifier")
    description: str = Field(
        ...,
        validation_alias=AliasChoices("description", "text"),
        serialization_alias="text",
        description="Description text of the hypothesis",
    )
    source_domain: str = Field(..., description="Source domain ID")
    source_structure: str = Field(..., description="Source structure ID")
    analogy_explanation: str = Field(..., description="Explanation of the analogy")
    observable: str = Field(..., description="Observable proxy variables/formula")

    # Optional from Hypothesis
    formula: str | None = Field(default=None, description="Factor formula if generated")
    parent_id: str | None = Field(default=None, description="Parent hypothesis ID")
    generation: int = Field(
        default=0,
        validation_alias=AliasChoices("generation", "generation_depth"),
        serialization_alias="generation_depth",
        description="Generation depth in search tree",
    )
    expansion_type: str | None = Field(default=None, description="Expansion type")

    # Mapping fields
    mapping_table: list["MappingItem"] = Field(
        default_factory=list, description="Structural mapping table"
    )
    failure_modes: list["FailureMode"] = Field(
        default_factory=list, description="Known failure modes"
    )
    mapping_quality: float | None = Field(
        default=None, ge=0.0, le=10.0, description="Mapping quality score"
    )
    evidence: HypothesisEvidence | None = Field(
        default=None,
        description="Unified evidence payload (hyperpaths/hkg_params/blend/transform)",
    )
    hkg_evidence: HKGEvidence | None = Field(
        default=None,
        description="HKG structural evidence copied from hypothesis when available",
    )

    # Conceptual Blending
    blend_network: BlendNetwork | None = Field(
        default=None, description="Blend network from blend_expand operator"
    )
    # Transformational Creativity
    space_transform_diff: SpaceTransformDiff | None = Field(
        default=None, description="Space transform diff from transform_space operator"
    )
    # QD Behavior Descriptor
    behavior_descriptor: BehaviorDescriptor | None = Field(
        default=None, description="Behavior descriptor for MAP-Elites"
    )

    # Verification results
    logic_verdict: LogicVerdict = Field(..., description="Logic verification result")
    novelty_verdict: NoveltyVerdict = Field(..., description="Novelty verification result")
    verify_status: VerifyStatus = Field(
        default=VerifyStatus.FAILED,
        description="Final VERIFY status after confidence gating",
    )
    judge_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Confidence copied from logic verifier",
    )
    position_consistency: bool = Field(
        default=True,
        description="Position consistency copied from logic verifier",
    )
    injection_detected: bool = Field(
        default=False,
        description="Prompt-injection flag copied from logic verifier",
    )
    final_score: float = Field(..., ge=0.0, le=10.0, description="Final composite score")
    structural_grounding_score: float | None = Field(
        default=None, ge=0.0, le=10.0,
        description="Structural grounding score from HKG evidence",
    )

    @classmethod
    def from_hypothesis(
        cls,
        hypothesis: "Hypothesis",
        logic_verdict: LogicVerdict,
        novelty_verdict: NoveltyVerdict,
        final_score: float,
        structural_grounding_score: float | None = None,
        verify_status: VerifyStatus = VerifyStatus.FAILED,
    ) -> "VerifiedHypothesis":
        """Create a VerifiedHypothesis from a Hypothesis and verdicts."""
        return cls(
            id=hypothesis.id,
            description=hypothesis.description,
            source_domain=hypothesis.source_domain,
            source_structure=hypothesis.source_structure,
            analogy_explanation=hypothesis.analogy_explanation,
            observable=hypothesis.observable,
            formula=hypothesis.formula,
            parent_id=hypothesis.parent_id,
            generation=hypothesis.generation,
            expansion_type=hypothesis.expansion_type,
            mapping_table=hypothesis.mapping_table,
            failure_modes=hypothesis.failure_modes,
            mapping_quality=hypothesis.mapping_quality,
            evidence=hypothesis.evidence,
            hkg_evidence=hypothesis.hkg_evidence,
            blend_network=hypothesis.blend_network,
            space_transform_diff=hypothesis.space_transform_diff,
            behavior_descriptor=hypothesis.behavior_descriptor,
            logic_verdict=logic_verdict,
            novelty_verdict=novelty_verdict,
            verify_status=verify_status,
            judge_confidence=logic_verdict.judge_confidence,
            position_consistency=logic_verdict.position_consistency,
            injection_detected=logic_verdict.injection_detected,
            final_score=final_score,
            structural_grounding_score=structural_grounding_score,
        )


# Deferred imports to avoid circular dependency (transform_types imports FailureMode).
# Must come after all class definitions so that FailureMode is available when
# transform_types is first imported.
from x_creative.core.blend_types import BlendNetwork as BlendNetwork  # noqa: E402, F811
from x_creative.core.transform_types import SpaceTransformDiff as SpaceTransformDiff  # noqa: E402, F811
from x_creative.creativity.qd_types import BehaviorDescriptor as BehaviorDescriptor  # noqa: E402, F811

HypothesisEvidence.model_rebuild()
Hypothesis.model_rebuild()
VerifiedHypothesis.model_rebuild()
