"""Tests for ConstraintConflictChecker."""

import pytest

from x_creative.core.types import ConstraintSpec


class TestConstraintBudget:
    def test_budget_limits_constraints(self) -> None:
        from x_creative.saga.constraint_checker import apply_constraint_budget

        constraints = [
            ConstraintSpec(
                text=f"Constraint {i}",
                origin="risk_refinement",
                weight=0.5 + i * 0.02,
            )
            for i in range(20)
        ]

        result = apply_constraint_budget(constraints, max_constraints=10)
        assert len(result) <= 10

    def test_budget_preserves_critical(self) -> None:
        from x_creative.saga.constraint_checker import apply_constraint_budget

        constraints = [
            ConstraintSpec(
                text="Critical one",
                origin="risk_refinement",
                priority="critical",
                type="hard",
                weight=0.9,
            ),
            *[
                ConstraintSpec(
                    text=f"Low {i}",
                    origin="risk_refinement",
                    priority="low",
                    weight=0.1,
                )
                for i in range(15)
            ],
        ]

        result = apply_constraint_budget(constraints, max_constraints=5)
        assert any(c.text == "Critical one" for c in result)

    def test_budget_preserves_all_user_constraints(self) -> None:
        from x_creative.saga.constraint_checker import apply_constraint_budget

        constraints = [
            ConstraintSpec(
                text=f"User {i}",
                origin="user",
                priority="critical",
                type="hard",
                weight=1.0,
            )
            for i in range(20)
        ]
        result = apply_constraint_budget(constraints, max_constraints=5)
        assert len(result) == 20


class TestDuplicateRiskDetection:
    def test_detects_repeated_risks(self) -> None:
        from x_creative.saga.constraint_checker import detect_repeated_risks

        history = [
            [{"risk": "缺乏市场微观结构验证", "severity": "high"}],
            [{"risk": "没有微观结构的验证步骤", "severity": "high"}],
        ]

        result = detect_repeated_risks(history, similarity_threshold=0.5)
        assert result.has_repeats

    def test_no_repeats_for_different_risks(self) -> None:
        from x_creative.saga.constraint_checker import detect_repeated_risks

        history = [
            [{"risk": "缺乏市场微观结构验证", "severity": "high"}],
            [{"risk": "因子可能在样本外失效", "severity": "high"}],
        ]

        result = detect_repeated_risks(history, similarity_threshold=0.8)
        assert not result.has_repeats


class TestConstraintConflictDetection:
    def test_detects_conflicting_constraints(self) -> None:
        from x_creative.saga.constraint_checker import detect_conflicting_constraints

        constraints = [
            "must use daily frequency data",
            "do not use daily frequency data",
            "avoid lookahead bias",
        ]

        result = detect_conflicting_constraints(constraints)
        assert result.has_conflicts
        assert len(result.conflict_pairs) >= 1

    def test_no_conflicts_for_consistent_constraints(self) -> None:
        from x_creative.saga.constraint_checker import detect_conflicting_constraints

        constraints = [
            "must use daily frequency data",
            "avoid lookahead bias",
            "use robust out-of-sample validation",
        ]

        result = detect_conflicting_constraints(constraints)
        assert not result.has_conflicts

    def test_conflict_threshold_is_configurable(self) -> None:
        from x_creative.saga.constraint_checker import detect_conflicting_constraints

        constraints = [
            "must use daily frequency data for modeling",
            "do not use daily frequency data",
        ]

        loose = detect_conflicting_constraints(constraints, similarity_threshold=0.4)
        strict = detect_conflicting_constraints(constraints, similarity_threshold=0.95)
        assert loose.has_conflicts is True
        assert strict.has_conflicts is False

    def test_conflict_resolution_preserves_user_constraints(self) -> None:
        from x_creative.saga.constraint_checker import (
            detect_conflicting_constraints,
            resolve_conflicting_constraints,
        )

        constraints = [
            ConstraintSpec(
                text="must use daily frequency data",
                origin="user",
                priority="critical",
                type="hard",
                weight=1.0,
            ),
            ConstraintSpec(
                text="do not use daily frequency data",
                origin="risk_refinement",
                priority="critical",
                type="hard",
                weight=1.0,
            ),
        ]
        conflicts = detect_conflicting_constraints([c.text for c in constraints])
        assert conflicts.has_conflicts is True

        kept = resolve_conflicting_constraints(constraints, conflicts.conflict_pairs)
        assert any(c.origin == "user" for c in kept)
        assert not any(
            c.origin != "user" and "do not use daily frequency data" in c.text
            for c in kept
        )


class TestConstraintCompiler:
    def test_compile_constraint_activation(self) -> None:
        from x_creative.saga.constraint_checker import compile_constraint_activation

        constraints = [
            ConstraintSpec(
                text="must avoid lookahead bias",
                priority="critical",
                type="hard",
                weight=0.9,
            ),
            ConstraintSpec(
                text="prefer out-of-sample validation",
                priority="high",
                type="soft",
                weight=0.8,
            ),
            ConstraintSpec(
                text="prefer microstructure diagnostics",
                priority="medium",
                type="soft",
                weight=0.6,
            ),
            ConstraintSpec(
                text="prefer turnover constraints",
                priority="low",
                type="soft",
                weight=0.4,
            ),
        ]

        compiled = compile_constraint_activation(
            constraints=constraints,
            reference_texts=[
                "out-of-sample performance and microstructure stability",
            ],
            active_soft_min=2,
            active_soft_max=3,
        )
        assert len(compiled.hard_core) == 1
        assert compiled.hard_core[0].text == "must avoid lookahead bias"
        assert 2 <= len(compiled.active_soft_set) <= 3

    def test_prompt_block_repeats_hardcore(self) -> None:
        from x_creative.saga.constraint_checker import (
            compile_constraint_activation,
            format_constraint_prompt_block,
        )

        constraints = [
            ConstraintSpec(
                text="must avoid lookahead bias",
                priority="critical",
                type="hard",
            ),
            ConstraintSpec(
                text="prefer robust risk controls",
                priority="high",
                type="soft",
            ),
        ]
        compiled = compile_constraint_activation(
            constraints=constraints,
            reference_texts=["risk controls"],
            active_soft_min=1,
            active_soft_max=1,
        )
        prompt_block = format_constraint_prompt_block(compiled)
        assert prompt_block.count("must avoid lookahead bias") == 2


class TestConstraintCoverageAudit:
    def test_audit_detects_uncovered_risks(self) -> None:
        from x_creative.saga.constraint_checker import audit_constraint_coverage

        constraints = [
            ConstraintSpec(text="must avoid lookahead bias", type="hard", priority="critical"),
        ]
        risks = [
            {"risk": "lookahead bias may leak future info", "severity": "high"},
            {"risk": "microstructure validation is missing", "severity": "high"},
        ]
        audit = audit_constraint_coverage(constraints, risks, similarity_threshold=0.55)
        assert audit.has_violations is True
        assert any("microstructure" in risk for risk in audit.uncovered_risks)
