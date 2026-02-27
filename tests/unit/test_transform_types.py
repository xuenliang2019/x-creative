"""Tests for SpaceTransformDiff data types."""

from x_creative.core.types import FailureMode
from x_creative.core.transform_types import SpaceTransformDiff, TransformAction


class TestTransformAction:
    def test_create(self) -> None:
        a = TransformAction(
            op_id="negate",
            op_type="negate_assumption",
            target_id="linear",
            before_state="收益率可用线性因子模型近似",
            after_state="收益率需要非线性模型",
            rationale="线性假设可能丢失非线性交互",
        )
        assert a.op_type == "negate_assumption"


class TestSpaceTransformDiff:
    def test_create_full(self) -> None:
        diff = SpaceTransformDiff(
            concept_space_version="1.0.0",
            actions=[
                TransformAction(
                    op_id="negate",
                    op_type="negate_assumption",
                    target_id="linear",
                    before_state="线性",
                    after_state="非线性",
                    rationale="探索非线性",
                ),
            ],
            new_failure_modes=[
                FailureMode(
                    scenario="过拟合风险增加",
                    why_breaks="非线性模型参数更多",
                    detectable_signal="训练集和验证集性能差距",
                ),
            ],
            new_detectable_signals=["train-val gap > 0.3"],
            new_observables=["nonlinear_alpha = f(features)"],
        )
        assert diff.concept_space_version == "1.0.0"
        assert len(diff.actions) == 1
        assert len(diff.new_failure_modes) == 1
        assert len(diff.new_observables) == 1

    def test_empty_diff(self) -> None:
        diff = SpaceTransformDiff(
            concept_space_version="1.0.0",
            actions=[],
            new_failure_modes=[],
            new_detectable_signals=[],
            new_observables=[],
        )
        assert len(diff.actions) == 0
