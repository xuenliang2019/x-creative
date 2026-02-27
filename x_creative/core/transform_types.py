"""Data types for transform_space operator output (SpaceTransformDiff)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from x_creative.core.types import FailureMode


class TransformAction(BaseModel):
    """A single space transformation action."""

    op_id: str = Field(..., description="Reference to AllowedTransformOp.id")
    op_type: str = Field(..., description="Operation type performed")
    target_id: str = Field(..., description="ID of the transformed element")
    before_state: str = Field(..., description="State before transformation")
    after_state: str = Field(..., description="State after transformation")
    rationale: str = Field(..., description="Why this transformation was applied")


class TransformStatus(str, Enum):
    """Lifecycle status for transform_space outputs."""

    PROPOSED = "PROPOSED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class SpaceTransformDiff(BaseModel):
    """Output of the transform_space operator: concept space transformation diff."""

    concept_space_version: str = Field(
        ..., description="Version of ConceptSpace this is based on"
    )
    actions: list[TransformAction] = Field(
        default_factory=list, description="Transformation actions performed"
    )
    new_failure_modes: list[FailureMode] = Field(
        default_factory=list, description="New failure modes introduced"
    )
    new_detectable_signals: list[str] = Field(
        default_factory=list, description="New detectable signals"
    )
    new_observables: list[str] = Field(
        default_factory=list, description="New observable variables produced"
    )
    transform_status: TransformStatus = Field(
        default=TransformStatus.PROPOSED,
        description="Transform proposal status: PROPOSED/ACCEPTED/REJECTED",
    )
    validation_notes: list[str] = Field(
        default_factory=list,
        description="Validation notes produced by transform gate",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="Reason when transform_status=REJECTED",
    )
