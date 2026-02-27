"""Data types for Quality-Diversity / MAP-Elites archive."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class GridConfig(BaseModel):
    """Configuration for a single grid dimension."""

    name: str = Field(..., description="Dimension name")
    dim_type: Literal["categorical", "continuous"] = Field(
        ..., description="Dimension type"
    )
    labels: list[str] | None = Field(
        default=None, description="Category labels (for categorical dims)"
    )
    num_bins: int = Field(default=5, ge=1, description="Number of bins (continuous)")
    bin_edges: list[float] | None = Field(
        default=None,
        description="Custom bin edges — left edge of each bin. "
        "If N+1 boundary points are supplied they are converted to N left edges.",
    )

    @field_validator("bin_edges", mode="before")
    @classmethod
    def _boundaries_to_left_edges(cls, v: list[float] | None) -> list[float] | None:
        """Convert N+1 boundary points to N left-edge values (drop the last)."""
        if v is not None and len(v) >= 2:
            return v[:-1]
        return v


class BDSchema(BaseModel):
    """Behavior Descriptor schema definition (versionable)."""

    version: str = Field(..., description="Schema version")
    grid_dimensions: list[GridConfig] = Field(
        default_factory=list, description="Dimensions forming the MAP-Elites grid"
    )
    raw_dimensions: list[str] = Field(
        default_factory=list, description="Extra dimensions for novelty distance only"
    )


class BehaviorDescriptor(BaseModel):
    """Behavior descriptor for a hypothesis in MAP-Elites."""

    grid_dims: dict[str, int] = Field(
        ..., description="Grid coordinate: dimension_name -> bin_index"
    )
    raw: dict[str, float | str] = Field(
        default_factory=dict, description="Raw values for novelty distance"
    )
    version: str = Field(..., description="BD schema version")
    bd_version: str | None = Field(
        default=None,
        description="Explicit BD schema version tag for replay stability",
    )
    extraction_method: Literal["rule", "llm", "hybrid"] = Field(
        ..., description="How this BD was extracted"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )
    evidence_path: list[str] = Field(
        default_factory=list,
        description="Evidence path used by mixed extraction (for replay/audit)",
    )
