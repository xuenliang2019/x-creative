"""Tests for ConceptSpace data types."""

from x_creative.core.concept_space import (
    AllowedTransformOp,
    ConceptSpace,
    ConceptSpaceAssumption,
    ConceptSpaceConstraint,
    ConceptSpacePrimitive,
    ConceptSpaceRelation,
)


class TestConceptSpacePrimitive:
    def test_create(self) -> None:
        p = ConceptSpacePrimitive(
            id="price", name="价格", description="资产市场价格", type="variable"
        )
        assert p.type == "variable"


class TestConceptSpaceConstraint:
    def test_hard_constraint(self) -> None:
        c = ConceptSpaceConstraint(
            id="no_lookahead",
            text="不使用未来信息",
            constraint_type="hard",
            rationale="回测一致性",
            examples=["使用t时刻之前的数据"],
            counterexamples=["使用t+1收盘价"],
        )
        assert c.constraint_type == "hard"
        assert len(c.examples) == 1

    def test_soft_constraint(self) -> None:
        c = ConceptSpaceConstraint(
            id="prefer_simple",
            text="优先简单因子",
            constraint_type="soft",
            rationale="可解释性",
        )
        assert c.examples == []


class TestConceptSpaceAssumption:
    def test_mutable(self) -> None:
        a = ConceptSpaceAssumption(
            id="linear_model",
            text="线性因子模型",
            mutable=True,
            rationale="常见近似",
        )
        assert a.mutable is True

    def test_fixed(self) -> None:
        a = ConceptSpaceAssumption(
            id="exchange_rules",
            text="价格时间优先",
            mutable=False,
            rationale="基本规则",
        )
        assert a.mutable is False


class TestAllowedTransformOp:
    def test_create(self) -> None:
        op = AllowedTransformOp(
            id="drop_hard",
            name="放弃硬约束",
            description="移除一条硬约束",
            op_type="drop_constraint",
            target_type="constraint",
        )
        assert op.op_type == "drop_constraint"


class TestConceptSpace:
    def test_create_full(self) -> None:
        cs = ConceptSpace(
            version="1.0.0",
            domain_id="open_source_development",
            provenance="yaml",
            primitives=[
                ConceptSpacePrimitive(
                    id="price", name="价格", description="市场价格", type="variable"
                ),
            ],
            relations=[
                ConceptSpaceRelation(
                    id="momentum",
                    name="动量",
                    description="价格趋势持续",
                    connects=["price"],
                ),
            ],
            hard_constraints=[
                ConceptSpaceConstraint(
                    id="no_lookahead",
                    text="不使用未来信息",
                    constraint_type="hard",
                    rationale="基本要求",
                ),
            ],
            soft_preferences=[],
            assumptions_fixed=[],
            assumptions_mutable=[
                ConceptSpaceAssumption(
                    id="linear",
                    text="线性假设",
                    mutable=True,
                    rationale="常见近似",
                ),
            ],
            allowed_ops=[
                AllowedTransformOp(
                    id="negate",
                    name="否定假设",
                    description="取反",
                    op_type="negate_assumption",
                    target_type="assumption",
                ),
            ],
            evaluation_criteria=["User Growth Rate"],
        )
        assert cs.version == "1.0.0"
        assert cs.provenance == "yaml"
        assert len(cs.primitives) == 1
        assert len(cs.hard_constraints) == 1
        assert len(cs.assumptions_mutable) == 1
        assert len(cs.allowed_ops) == 1

    def test_minimal(self) -> None:
        cs = ConceptSpace(
            version="0.1.0",
            domain_id="test",
            provenance="llm_inferred",
            primitives=[],
            relations=[],
            hard_constraints=[],
            soft_preferences=[],
            assumptions_fixed=[],
            assumptions_mutable=[],
            allowed_ops=[],
            evaluation_criteria=[],
        )
        assert cs.provenance == "llm_inferred"
