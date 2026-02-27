"""Cognitive budget management for SAGA Slow Agent.

Controls the depth of Slow Agent's thinking, preventing cost explosion
by making the agent selectively apply deep review — like human System 2
choosing when to engage in effortful reasoning.

Budget unit: approximately one standard LLM API call in token cost.
"""

from enum import Enum

from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger()


class AllocationStrategy(str, Enum):
    """Cognitive budget allocation strategies."""

    UNIFORM = "uniform"
    """Uniform: every hypothesis gets the same review depth."""

    ATTENTION_WEIGHTED = "attention_weighted"
    """Attention-weighted: high-scoring hypotheses get more budget."""

    NOVELTY_GATED = "novelty_gated"
    """Novelty-gated: only high-novelty hypotheses get deep review."""

    ANOMALY_DRIVEN = "anomaly_driven"
    """Anomaly-driven: concentrate budget when anomalies are detected."""


class CognitiveBudget(BaseModel):
    """Slow Agent cognitive budget tracker.

    Manages a finite pool of review budget, allocating it across
    pipeline stages and controlling when deep review is warranted.
    """

    total_budget: float = Field(
        default=100.0,
        description="Total cognitive budget units (1 unit ~ 1 standard LLM call)",
    )
    spent: float = Field(
        default=0.0,
        description="Budget consumed so far",
    )
    strategy: AllocationStrategy = Field(
        default=AllocationStrategy.ANOMALY_DRIVEN,
        description="Budget allocation strategy",
    )
    stage_allocation: dict[str, float] = Field(
        default_factory=lambda: {
            "domain_audit": 0.10,
            "biso_monitoring": 0.15,
            "search_monitoring": 0.15,
            "verify_monitoring": 0.20,
            "adversarial": 0.25,
            "global_review": 0.15,
        },
        description="Budget allocation ratios per stage (should sum to 1.0)",
    )
    reserve_ratio: float = Field(
        default=0.10,
        description="Fraction of budget reserved for emergency detections",
    )
    stage_spent: dict[str, float] = Field(
        default_factory=dict,
        description="Consumed budget by stage bucket",
    )
    reserve_spent: float = Field(
        default=0.0,
        description="Consumed emergency reserve budget",
    )

    @property
    def remaining(self) -> float:
        """Remaining budget."""
        return self.total_budget - self.spent

    @property
    def available_for_stage(self) -> dict[str, float]:
        """Available budget per stage, after accounting for reserve."""
        effective = self.total_budget * (1 - self.reserve_ratio)
        return {
            stage: effective * ratio
            for stage, ratio in self.stage_allocation.items()
        }

    @property
    def reserve_budget(self) -> float:
        """Emergency reserve budget size."""
        return self.total_budget * self.reserve_ratio

    @property
    def reserve_remaining(self) -> float:
        """Remaining emergency reserve."""
        return max(0.0, self.reserve_budget - self.reserve_spent)

    def stage_limit(self, stage: str) -> float:
        """Configured budget ceiling for a stage bucket."""
        return float(self.available_for_stage.get(stage, 0.0))

    def stage_remaining(self, stage: str) -> float:
        """Remaining budget in a stage bucket (excluding reserve)."""
        return max(0.0, self.stage_limit(stage) - float(self.stage_spent.get(stage, 0.0)))

    def can_afford(self, cost: float) -> bool:
        """Check if there is enough budget for a given cost.

        Args:
            cost: The cost to check against remaining budget.

        Returns:
            True if remaining budget >= cost.
        """
        return self.remaining >= cost

    def spend(self, cost: float, reason: str) -> bool:
        """Consume budget.

        Args:
            cost: Amount of budget to consume.
            reason: Why this budget was spent (for audit trail).

        Returns:
            True if budget was consumed, False if insufficient.
        """
        if not self.can_afford(cost):
            logger.warning(
                "Insufficient cognitive budget",
                requested=cost,
                remaining=self.remaining,
                reason=reason,
            )
            return False
        self.spent += cost
        logger.debug(
            "Cognitive budget spent",
            cost=cost,
            remaining=self.remaining,
            reason=reason,
        )
        return True

    def can_afford_stage(
        self,
        stage: str,
        cost: float,
        allow_reserve: bool = False,
    ) -> bool:
        """Check budget affordability with stage constraints."""
        if not self.can_afford(cost):
            return False
        stage_rem = self.stage_remaining(stage)
        if stage_rem >= cost:
            return True
        if not allow_reserve:
            return False
        return (stage_rem + self.reserve_remaining) >= cost

    def spend_stage(
        self,
        cost: float,
        stage: str,
        reason: str,
        allow_reserve: bool = False,
    ) -> bool:
        """Spend budget while enforcing per-stage allocation constraints."""
        if not self.can_afford_stage(stage, cost, allow_reserve=allow_reserve):
            logger.warning(
                "Insufficient stage cognitive budget",
                stage=stage,
                requested=cost,
                stage_remaining=self.stage_remaining(stage),
                reserve_remaining=self.reserve_remaining,
                total_remaining=self.remaining,
                reason=reason,
            )
            return False

        stage_rem = self.stage_remaining(stage)
        from_stage = min(cost, stage_rem)
        from_reserve = cost - from_stage

        if from_stage > 0:
            self.stage_spent[stage] = float(self.stage_spent.get(stage, 0.0)) + from_stage
        if from_reserve > 0:
            self.reserve_spent += from_reserve

        self.spent += cost
        logger.debug(
            "Cognitive budget spent with stage allocation",
            stage=stage,
            cost=cost,
            from_stage=from_stage,
            from_reserve=from_reserve,
            stage_remaining=self.stage_remaining(stage),
            reserve_remaining=self.reserve_remaining,
            total_remaining=self.remaining,
            reason=reason,
        )
        return True


class BudgetPolicy:
    """Budget allocation decision logic.

    Decides whether to allocate budget for deep review of a hypothesis
    based on the current strategy and state.
    """

    def should_deep_review(
        self,
        hypothesis_score: float,
        novelty_score: float,
        anomaly_detected: bool,
        budget: CognitiveBudget,
    ) -> tuple[bool, float]:
        """Decide whether a hypothesis warrants deep review.

        Args:
            hypothesis_score: Composite score of the hypothesis (0-10).
            novelty_score: Novelty score of the hypothesis (0-10).
            anomaly_detected: Whether an anomaly was detected for this hypothesis.
            budget: Current cognitive budget state.

        Returns:
            Tuple of (should_review, allocated_budget_cost).
        """
        strategy = budget.strategy

        if strategy == AllocationStrategy.ANOMALY_DRIVEN:
            if anomaly_detected:
                cost = 3.0
                return budget.can_afford(cost), cost
            if hypothesis_score >= 7.0:
                cost = 1.0
                return budget.can_afford(cost), cost
            return False, 0.0

        elif strategy == AllocationStrategy.ATTENTION_WEIGHTED:
            if hypothesis_score >= 8.0:
                cost = 3.0
            elif hypothesis_score >= 6.0:
                cost = 1.5
            else:
                return False, 0.0
            return budget.can_afford(cost), cost

        elif strategy == AllocationStrategy.NOVELTY_GATED:
            if novelty_score >= 7.0:
                cost = 2.0
                return budget.can_afford(cost), cost
            return False, 0.0

        elif strategy == AllocationStrategy.UNIFORM:
            cost = 1.0
            return budget.can_afford(cost), cost

        return False, 0.0
