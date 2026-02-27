"""Tests for SAGA cognitive budget management."""

import pytest

from x_creative.saga.budget import (
    AllocationStrategy,
    BudgetPolicy,
    CognitiveBudget,
)


class TestAllocationStrategy:
    """Tests for AllocationStrategy enum."""

    def test_all_strategies_exist(self) -> None:
        """Test that all expected strategies are defined."""
        assert AllocationStrategy.UNIFORM == "uniform"
        assert AllocationStrategy.ATTENTION_WEIGHTED == "attention_weighted"
        assert AllocationStrategy.NOVELTY_GATED == "novelty_gated"
        assert AllocationStrategy.ANOMALY_DRIVEN == "anomaly_driven"


class TestCognitiveBudget:
    """Tests for CognitiveBudget model."""

    def test_default_creation(self) -> None:
        """Test creating CognitiveBudget with defaults."""
        budget = CognitiveBudget()
        assert budget.total_budget == 100.0
        assert budget.spent == 0.0
        assert budget.strategy == AllocationStrategy.ANOMALY_DRIVEN
        assert budget.reserve_ratio == 0.10

    def test_remaining_property(self) -> None:
        """Test remaining budget calculation."""
        budget = CognitiveBudget(total_budget=100.0, spent=30.0)
        assert budget.remaining == 70.0

    def test_can_afford(self) -> None:
        """Test can_afford with sufficient budget."""
        budget = CognitiveBudget(total_budget=100.0, spent=50.0)
        assert budget.can_afford(50.0) is True
        assert budget.can_afford(51.0) is False
        assert budget.can_afford(0.0) is True

    def test_spend_success(self) -> None:
        """Test successful budget spending."""
        budget = CognitiveBudget(total_budget=100.0)
        result = budget.spend(10.0, "test audit")
        assert result is True
        assert budget.spent == 10.0
        assert budget.remaining == 90.0

    def test_spend_failure(self) -> None:
        """Test spending beyond budget."""
        budget = CognitiveBudget(total_budget=10.0, spent=8.0)
        result = budget.spend(5.0, "too expensive")
        assert result is False
        assert budget.spent == 8.0  # Unchanged

    def test_spend_cumulative(self) -> None:
        """Test cumulative spending."""
        budget = CognitiveBudget(total_budget=100.0)
        budget.spend(20.0, "audit 1")
        budget.spend(30.0, "audit 2")
        assert budget.spent == 50.0
        assert budget.remaining == 50.0

    def test_spend_stage_within_allocation(self) -> None:
        """Stage spend should succeed within stage budget ceiling."""
        budget = CognitiveBudget(total_budget=100.0, reserve_ratio=0.10)
        ok = budget.spend_stage(
            cost=8.0,
            stage="domain_audit",
            reason="domain audit batch",
        )
        assert ok is True
        assert budget.spent == 8.0
        assert budget.stage_spent["domain_audit"] == 8.0
        assert budget.reserve_spent == 0.0

    def test_spend_stage_rejects_over_allocation_without_reserve(self) -> None:
        """Without reserve override, stage over-spend should be rejected."""
        budget = CognitiveBudget(total_budget=100.0, reserve_ratio=0.10)
        assert budget.spend_stage(9.0, "domain_audit", "fill stage") is True
        assert budget.spend_stage(2.0, "domain_audit", "overflow", allow_reserve=False) is False
        assert budget.spent == 9.0

    def test_spend_stage_allows_reserve_override(self) -> None:
        """Reserve can be used only when explicitly allowed."""
        budget = CognitiveBudget(total_budget=100.0, reserve_ratio=0.10)
        assert budget.spend_stage(9.0, "domain_audit", "fill stage") is True
        assert budget.spend_stage(2.0, "domain_audit", "use reserve", allow_reserve=True) is True
        assert budget.stage_spent["domain_audit"] == pytest.approx(9.0)
        assert budget.reserve_spent == pytest.approx(2.0)
        assert budget.spent == pytest.approx(11.0)

    def test_available_for_stage(self) -> None:
        """Test per-stage budget calculation."""
        budget = CognitiveBudget(
            total_budget=100.0,
            reserve_ratio=0.10,
        )
        stage_budgets = budget.available_for_stage

        # Total effective = 100 * 0.9 = 90
        assert "domain_audit" in stage_budgets
        assert "adversarial" in stage_budgets

        # domain_audit = 90 * 0.10 = 9.0
        assert abs(stage_budgets["domain_audit"] - 9.0) < 0.01

        # adversarial = 90 * 0.25 = 22.5
        assert abs(stage_budgets["adversarial"] - 22.5) < 0.01

    def test_stage_allocation_sum(self) -> None:
        """Test that default stage allocation ratios sum to ~1.0."""
        budget = CognitiveBudget()
        total_ratio = sum(budget.stage_allocation.values())
        assert abs(total_ratio - 1.0) < 0.01

    def test_custom_budget(self) -> None:
        """Test creating budget with custom values."""
        budget = CognitiveBudget(
            total_budget=200.0,
            strategy=AllocationStrategy.UNIFORM,
            reserve_ratio=0.20,
        )
        assert budget.total_budget == 200.0
        assert budget.strategy == AllocationStrategy.UNIFORM
        assert budget.reserve_ratio == 0.20


class TestBudgetPolicy:
    """Tests for BudgetPolicy decision logic."""

    def test_anomaly_driven_with_anomaly(self) -> None:
        """Test ANOMALY_DRIVEN strategy when anomaly is detected."""
        policy = BudgetPolicy()
        budget = CognitiveBudget(strategy=AllocationStrategy.ANOMALY_DRIVEN)

        should_review, cost = policy.should_deep_review(
            hypothesis_score=5.0,
            novelty_score=5.0,
            anomaly_detected=True,
            budget=budget,
        )
        assert should_review is True
        assert cost == 3.0

    def test_anomaly_driven_high_score(self) -> None:
        """Test ANOMALY_DRIVEN with high score but no anomaly."""
        policy = BudgetPolicy()
        budget = CognitiveBudget(strategy=AllocationStrategy.ANOMALY_DRIVEN)

        should_review, cost = policy.should_deep_review(
            hypothesis_score=7.5,
            novelty_score=5.0,
            anomaly_detected=False,
            budget=budget,
        )
        assert should_review is True
        assert cost == 1.0

    def test_anomaly_driven_low_score_no_anomaly(self) -> None:
        """Test ANOMALY_DRIVEN with low score and no anomaly."""
        policy = BudgetPolicy()
        budget = CognitiveBudget(strategy=AllocationStrategy.ANOMALY_DRIVEN)

        should_review, cost = policy.should_deep_review(
            hypothesis_score=4.0,
            novelty_score=5.0,
            anomaly_detected=False,
            budget=budget,
        )
        assert should_review is False
        assert cost == 0.0

    def test_attention_weighted_high_score(self) -> None:
        """Test ATTENTION_WEIGHTED with high score hypothesis."""
        policy = BudgetPolicy()
        budget = CognitiveBudget(strategy=AllocationStrategy.ATTENTION_WEIGHTED)

        should_review, cost = policy.should_deep_review(
            hypothesis_score=8.5,
            novelty_score=5.0,
            anomaly_detected=False,
            budget=budget,
        )
        assert should_review is True
        assert cost == 3.0

    def test_attention_weighted_medium_score(self) -> None:
        """Test ATTENTION_WEIGHTED with medium score hypothesis."""
        policy = BudgetPolicy()
        budget = CognitiveBudget(strategy=AllocationStrategy.ATTENTION_WEIGHTED)

        should_review, cost = policy.should_deep_review(
            hypothesis_score=6.5,
            novelty_score=5.0,
            anomaly_detected=False,
            budget=budget,
        )
        assert should_review is True
        assert cost == 1.5

    def test_attention_weighted_low_score(self) -> None:
        """Test ATTENTION_WEIGHTED with low score hypothesis."""
        policy = BudgetPolicy()
        budget = CognitiveBudget(strategy=AllocationStrategy.ATTENTION_WEIGHTED)

        should_review, cost = policy.should_deep_review(
            hypothesis_score=4.0,
            novelty_score=5.0,
            anomaly_detected=False,
            budget=budget,
        )
        assert should_review is False

    def test_novelty_gated_high_novelty(self) -> None:
        """Test NOVELTY_GATED with high novelty score."""
        policy = BudgetPolicy()
        budget = CognitiveBudget(strategy=AllocationStrategy.NOVELTY_GATED)

        should_review, cost = policy.should_deep_review(
            hypothesis_score=5.0,
            novelty_score=8.0,
            anomaly_detected=False,
            budget=budget,
        )
        assert should_review is True
        assert cost == 2.0

    def test_novelty_gated_low_novelty(self) -> None:
        """Test NOVELTY_GATED with low novelty score."""
        policy = BudgetPolicy()
        budget = CognitiveBudget(strategy=AllocationStrategy.NOVELTY_GATED)

        should_review, cost = policy.should_deep_review(
            hypothesis_score=9.0,
            novelty_score=3.0,
            anomaly_detected=True,
            budget=budget,
        )
        assert should_review is False

    def test_uniform_always_reviews(self) -> None:
        """Test UNIFORM strategy always allocates budget."""
        policy = BudgetPolicy()
        budget = CognitiveBudget(strategy=AllocationStrategy.UNIFORM)

        should_review, cost = policy.should_deep_review(
            hypothesis_score=3.0,
            novelty_score=3.0,
            anomaly_detected=False,
            budget=budget,
        )
        assert should_review is True
        assert cost == 1.0

    def test_insufficient_budget_rejects(self) -> None:
        """Test that insufficient budget prevents review."""
        policy = BudgetPolicy()
        budget = CognitiveBudget(
            total_budget=5.0,
            spent=4.5,
            strategy=AllocationStrategy.ANOMALY_DRIVEN,
        )

        should_review, cost = policy.should_deep_review(
            hypothesis_score=5.0,
            novelty_score=5.0,
            anomaly_detected=True,
            budget=budget,
        )
        # remaining = 0.5, cost = 3.0 -> can't afford
        assert should_review is False
