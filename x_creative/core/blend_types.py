"""Data types for Conceptual Blending (Fauconnier-Turner four-space model)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BlendSpaceMapping(BaseModel):
    """A single projection from an input space to the blend."""

    source_element: str = Field(..., description="Element in the source input space")
    blend_element: str = Field(..., description="Projected element in blend space")
    projection_type: Literal["identity", "compression", "elaboration", "completion"] = Field(
        ..., description="Type of projection"
    )
    conflict_resolution: str | None = Field(
        default=None, description="How conflicts between input spaces are resolved"
    )


class EmergentStructure(BaseModel):
    """A structure that emerges in the blend but does not exist in any input space."""

    description: str = Field(..., description="Description of the emergent structure")
    emergence_type: Literal["composition", "completion", "elaboration"] = Field(
        ..., description="Type of emergence"
    )
    observable_link: str | None = Field(
        default=None, description="Link to an observable variable"
    )
    testable_prediction: str | None = Field(
        default=None, description="A testable prediction from this structure"
    )


class BlendNetwork(BaseModel):
    """Fauconnier-Turner four-space conceptual integration network."""

    input1_summary: str = Field(..., description="Summary of input space 1")
    input2_summary: str = Field(..., description="Summary of input space 2")
    generic_space: str = Field(..., description="Shared abstract structure")
    blend_description: str = Field(..., description="Description of the blend space")
    cross_space_mappings: list[BlendSpaceMapping] = Field(
        default_factory=list, description="Mappings across input spaces into blend"
    )
    emergent_structures: list[EmergentStructure] = Field(
        default_factory=list, description="Structures emerging in the blend"
    )
    blend_consistency_score: float | None = Field(
        default=None, ge=0.0, le=10.0, description="Consistency score from VERIFY"
    )
