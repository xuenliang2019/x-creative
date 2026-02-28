"""Core data models for SAGA shared cognition state.

These Pydantic models are referenced by all other SAGA modules:
- GenerationMetrics: Fast Agent generation statistics
- CognitionAlert: Slow Agent anomaly detections
- EvaluationAdjustments: Dynamic evaluation parameter adjustments
- SharedCognitionState: Shared state between Fast and Slow Agents
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from x_creative.core.plugin import TargetDomainPlugin

import structlog

logger = structlog.get_logger()


class GenerationMetrics(BaseModel):
    """Fast Agent generation metrics.

    Tracks diversity, score distribution, and quality indicators
    across the hypothesis generation pipeline.
    """

    # Diversity metrics
    source_domain_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Number of hypotheses per source domain",
    )
    unique_structure_count: int = Field(
        default=0,
        description="Number of distinct source structures used",
    )
    expansion_type_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of expansion types (refine/variant/combine/oppose)",
    )

    # Score distribution metrics
    score_mean: float = Field(default=0.0, description="Mean composite score")
    score_std: float = Field(default=0.0, description="Standard deviation of scores")
    score_min: float = Field(default=0.0, description="Minimum composite score")
    score_max: float = Field(default=0.0, description="Maximum composite score")
    dimension_correlations: dict[str, float] = Field(
        default_factory=dict,
        description="Pairwise correlation coefficients between scoring dimensions",
    )

    # Quality metrics
    generation_depth_vs_quality: list[tuple[int, float]] = Field(
        default_factory=list,
        description="(depth, avg_score) pairs showing quality trend across search depth",
    )
    deduplication_ratio: float = Field(
        default=0.0,
        description="Ratio of duplicates removed (high = many duplicate generations)",
    )


class CognitionAlert(BaseModel):
    """Anomaly detected by Slow Agent.

    Represents a cognitive alert raised when the Slow Agent detects
    problematic patterns in Fast Agent's generation process.
    """

    alert_type: str = Field(
        ...,
        description="Alert category: reward_hacking, diversity_decay, score_drift, etc.",
    )
    severity: str = Field(
        ...,
        description="Severity level: info, warning, critical",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the anomaly",
    )
    evidence: dict = Field(
        default_factory=dict,
        description="Supporting evidence (statistics, examples)",
    )
    suggested_action: str = Field(
        default="",
        description="Recommended intervention action",
    )
    timestamp: float = Field(
        default=0.0,
        description="Unix timestamp when the alert was raised",
    )


class EvaluationAdjustments(BaseModel):
    """Dynamic evaluation parameter adjustments by Slow Agent.

    Controls how hypotheses are scored and filtered, allowing the
    Slow Agent to modify evaluation criteria in response to detected
    anomalies or domain-specific requirements.
    """

    # Weight overrides
    weight_overrides: dict[str, float] | None = Field(
        default=None,
        description="Override default scoring weights, e.g. {'divergence': 0.30}",
    )

    # Threshold overrides
    prune_threshold_override: float | None = Field(
        default=None,
        description="Override the default pruning threshold",
    )

    # Extra scoring dimensions
    extra_dimensions: list[dict] | None = Field(
        default=None,
        description="Additional scoring dimensions with name, weight, and prompt",
    )

    # Hard constraint gates
    hard_gates: list[str] | None = Field(
        default=None,
        description="Constraints that must pass (e.g. no_lookahead_bias). Failure = immediate reject.",
    )

    # Audit trail
    adjustment_reasons: list[str] = Field(
        default_factory=list,
        description="Log of why adjustments were made",
    )


class SharedCognitionState(BaseModel):
    """Shared cognition state between Fast and Slow Agents.

    This is the communication hub inspired by the Talker-Reasoner
    architecture's shared belief state, extended with metacognitive
    monitoring dimensions.
    """

    # === Fast Agent write zone ===
    current_stage: str = Field(
        default="idle",
        description="Current pipeline stage: idle, biso, search, verify, completed",
    )
    hypotheses_pool: list = Field(
        default_factory=list,
        description="Current hypothesis pool (list of dicts to avoid circular imports)",
    )
    generation_metrics: GenerationMetrics | None = Field(
        default=None,
        description="Real-time generation metrics",
    )

    # === Slow Agent write zone ===
    active_alerts: list[CognitionAlert] = Field(
        default_factory=list,
        description="Active anomaly alerts from Slow Agent detectors",
    )
    evaluation_adjustments: EvaluationAdjustments = Field(
        default_factory=EvaluationAdjustments,
        description="Current evaluation parameter adjustments",
    )
    intervention_count: int = Field(
        default=0,
        description="Total number of Slow Agent interventions",
    )

    # === Shared knowledge zone ===
    target_domain_id: str | None = Field(
        default=None,
        description="Target domain plugin ID (loaded from plugin.py)",
    )
    target_plugin: TargetDomainPlugin | None = Field(
        default=None,
        description="Resolved target domain plugin (for fresh/ephemeral domains not in YAML)",
    )
    historical_hypothesis_hashes: set[str] = Field(
        default_factory=set,
        description="Semantic hashes of hypotheses from previous sessions for cross-session dedup",
    )
    adversarial_challenges: list[dict] = Field(
        default_factory=list,
        description="Red-team challenges constructed by Slow Agent for Fast Agent to respond to",
    )
