"""ConceptSpace data types for Boden's transformational creativity."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ConceptSpacePrimitive(BaseModel):
    """A primitive object/variable type in the concept space."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Description")
    type: Literal["variable", "entity", "process", "relation"] = Field(
        ..., description="Primitive type"
    )


class ConceptSpaceRelation(BaseModel):
    """An allowed relation/mechanism pattern in the concept space."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Description")
    connects: list[str] = Field(
        default_factory=list, description="IDs of primitives this relation connects"
    )


class ConceptSpaceConstraint(BaseModel):
    """A constraint in the concept space (hard = constitutive rule)."""

    id: str = Field(..., description="Unique identifier")
    text: str = Field(..., description="Constraint description")
    constraint_type: Literal["hard", "soft"] = Field(
        ..., description="Hard (constitutive) or soft (preference)"
    )
    rationale: str = Field(..., description="Why this is hard/soft")
    examples: list[str] = Field(default_factory=list, description="Examples")
    counterexamples: list[str] = Field(default_factory=list, description="Counterexamples")


class ConceptSpaceAssumption(BaseModel):
    """An assumption (fixed or mutable) in the concept space."""

    id: str = Field(..., description="Unique identifier")
    text: str = Field(..., description="Assumption description")
    mutable: bool = Field(..., description="True if this assumption can be transformed")
    rationale: str = Field(..., description="Rationale for mutability classification")


class AllowedTransformOp(BaseModel):
    """An allowed space transformation operation."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="What this operation does")
    op_type: Literal[
        "drop_constraint",
        "relax_threshold",
        "change_representation",
        "swap_causal_direction",
        "add_new_primitive",
        "merge_primitives",
        "split_process",
        "negate_assumption",
    ] = Field(..., description="Operation type")
    target_type: Literal["constraint", "assumption", "primitive", "relation"] = Field(
        ..., description="What type of element this operation targets"
    )


class ConceptSpace(BaseModel):
    """A versionable, diffable, replayable concept space."""

    version: str = Field(..., description="Semantic version")
    domain_id: str = Field(..., description="Target domain this belongs to")
    provenance: Literal["yaml", "plugin_derived", "llm_inferred"] = Field(
        ..., description="Source of this concept space"
    )
    primitives: list[ConceptSpacePrimitive] = Field(default_factory=list)
    relations: list[ConceptSpaceRelation] = Field(default_factory=list)
    hard_constraints: list[ConceptSpaceConstraint] = Field(default_factory=list)
    soft_preferences: list[ConceptSpaceConstraint] = Field(default_factory=list)
    assumptions_fixed: list[ConceptSpaceAssumption] = Field(default_factory=list)
    assumptions_mutable: list[ConceptSpaceAssumption] = Field(default_factory=list)
    allowed_ops: list[AllowedTransformOp] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)
