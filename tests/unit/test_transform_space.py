# tests/unit/test_transform_space.py
"""Tests for transform_space operator."""

import pytest
from unittest.mock import AsyncMock

from x_creative.core.types import Hypothesis
from x_creative.core.transform_types import TransformStatus
from x_creative.core.concept_space import (
    AllowedTransformOp, ConceptSpace, ConceptSpaceAssumption, ConceptSpaceConstraint,
)
from x_creative.creativity.transform import transform_space


def _concept_space() -> ConceptSpace:
    return ConceptSpace(
        version="1.0.0", domain_id="test", provenance="yaml",
        primitives=[], relations=[],
        hard_constraints=[
            ConceptSpaceConstraint(id="c1", text="must be linear", constraint_type="hard", rationale="r")
        ],
        soft_preferences=[], assumptions_fixed=[],
        assumptions_mutable=[
            ConceptSpaceAssumption(id="a1", text="linear model", mutable=True, rationale="r")
        ],
        allowed_ops=[
            AllowedTransformOp(id="negate", name="negate", description="d", op_type="negate_assumption", target_type="assumption")
        ],
        evaluation_criteria=[],
    )


def _hyp() -> Hypothesis:
    return Hypothesis(
        id="h1", description="test hyp", source_domain="thermo",
        source_structure="s", analogy_explanation="a", observable="obs",
    )


class TestTransformSpaceRouting:
    @pytest.mark.asyncio
    async def test_transform_space_uses_task_routing(self) -> None:
        """transform_space should call router.complete with task='transform_space'."""
        mock_router = AsyncMock()
        mock_router.complete.return_value = AsyncMock(
            content='[{"target_id": "a1", "op_id": "negate", "op_type": "negate_assumption", '
            '"before_state": "old", "after_state": "new", "rationale": "why", '
            '"description": "transformed", "observable": "x", "failure_modes": []}]'
        )

        await transform_space(_hyp(), _concept_space(), mock_router)

        _, kwargs = mock_router.complete.call_args
        assert kwargs["task"] == "transform_space"
        assert isinstance(kwargs["messages"], list)
        assert kwargs["messages"][0]["role"] == "user"


class TestTransformSpace:
    @pytest.mark.asyncio
    async def test_returns_hypothesis_with_diff(self) -> None:
        mock_router = AsyncMock()
        mock_router.complete.return_value = AsyncMock(
            content='[{"target_id": "a1", "op_id": "negate", "op_type": "negate_assumption", "before_state": "linear model", "after_state": "nonlinear model", "rationale": "explore nonlinearity", "description": "nonlinear hypothesis", "observable": "nl_obs", "failure_modes": [{"scenario": "overfit", "why_breaks": "too many params", "detectable_signal": "train-val gap"}]}]'
        )

        results = await transform_space(
            hypothesis=_hyp(),
            concept_space=_concept_space(),
            router=mock_router,
        )

        assert len(results) >= 1
        assert results[0].space_transform_diff is not None
        assert results[0].space_transform_diff.transform_status == TransformStatus.PROPOSED
        assert results[0].expansion_type == "transform_space"
        assert len(results[0].space_transform_diff.actions) >= 1

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure(self) -> None:
        mock_router = AsyncMock()
        mock_router.complete.return_value = AsyncMock(content="error")

        results = await transform_space(
            hypothesis=_hyp(),
            concept_space=_concept_space(),
            router=mock_router,
        )
        assert results == []
