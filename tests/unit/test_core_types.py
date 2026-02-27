"""Tests for core data types."""

import pytest
from pydantic import ValidationError


class TestDomainStructure:
    """Tests for DomainStructure model."""

    def test_create_valid_structure(self) -> None:
        """Test creating a valid domain structure."""
        from x_creative.core.types import DomainStructure

        structure = DomainStructure(
            id="entropy_increase",
            name="熵增定律",
            description="孤立系统趋向最大熵状态",
            key_variables=["entropy", "energy", "temperature"],
            dynamics="单向增加，不可逆",
        )

        assert structure.id == "entropy_increase"
        assert structure.name == "熵增定律"
        assert len(structure.key_variables) == 3

    def test_structure_requires_id(self) -> None:
        """Test that structure requires an id."""
        from x_creative.core.types import DomainStructure

        with pytest.raises(ValidationError):
            DomainStructure(
                name="熵增定律",
                description="test",
                key_variables=[],
                dynamics="test",
            )  # type: ignore


class TestTargetMapping:
    """Tests for TargetMapping model."""

    def test_create_target_mapping(self) -> None:
        """Test creating a valid target mapping."""
        from x_creative.core.types import TargetMapping

        mapping = TargetMapping(
            structure="entropy_increase",
            target="市场混乱度指标",
            observable="收益率分布熵、行业集中度变化",
        )

        assert mapping.structure == "entropy_increase"
        assert mapping.target == "市场混乱度指标"


class TestDomain:
    """Tests for Domain model."""

    def test_create_valid_domain(self) -> None:
        """Test creating a valid domain."""
        from x_creative.core.types import Domain, DomainStructure, TargetMapping

        domain = Domain(
            id="thermodynamics",
            name="热力学系统",
            name_en="Thermodynamics",
            description="研究能量转换、熵增、相变等物理过程",
            structures=[
                DomainStructure(
                    id="entropy_increase",
                    name="熵增定律",
                    description="孤立系统趋向最大熵状态",
                    key_variables=["entropy", "energy"],
                    dynamics="单向增加",
                )
            ],
            target_mappings=[
                TargetMapping(
                    structure="entropy_increase",
                    target="市场混乱度",
                    observable="收益率分布熵",
                )
            ],
        )

        assert domain.id == "thermodynamics"
        assert len(domain.structures) == 1
        assert len(domain.target_mappings) == 1

    def test_create_with_target_mappings(self) -> None:
        """Test creating a domain with target_mappings."""
        from x_creative.core.types import Domain, DomainStructure, TargetMapping

        domain = Domain(
            id="thermodynamics",
            name="热力学系统",
            description="研究能量转换、熵增、相变等物理过程",
            structures=[
                DomainStructure(
                    id="entropy_increase",
                    name="熵增定律",
                    description="孤立系统趋向最大熵状态",
                    key_variables=["entropy"],
                    dynamics="单向增加",
                )
            ],
            target_mappings=[
                TargetMapping(
                    structure="entropy_increase",
                    target="A股收益率分布熵",
                    observable="rolling_entropy",
                )
            ],
        )

        assert len(domain.target_mappings) == 1
        assert domain.target_mappings[0].target == "A股收益率分布熵"

    def test_domain_from_dict_with_target_mappings(self) -> None:
        """Test Domain creation from dict data."""
        from x_creative.core.types import Domain

        data = {
            "id": "test",
            "name": "Test",
            "description": "test",
            "structures": [],
            "target_mappings": [
                {"structure": "s1", "target": "t1", "observable": "o1"}
            ],
        }
        domain = Domain(**data)

        assert len(domain.target_mappings) == 1
        assert domain.target_mappings[0].target == "t1"

    def test_domain_empty_mappings(self) -> None:
        """Test Domain with no mappings at all."""
        from x_creative.core.types import Domain

        domain = Domain(
            id="test",
            name="Test",
            description="test",
            structures=[],
        )

        assert domain.target_mappings == []

    def test_domain_get_structure(self) -> None:
        """Test getting a structure by id."""
        from x_creative.core.types import Domain, DomainStructure

        domain = Domain(
            id="test",
            name="Test",
            description="test",
            structures=[
                DomainStructure(
                    id="s1",
                    name="Structure 1",
                    description="desc",
                    key_variables=["x"],
                    dynamics="d1",
                ),
                DomainStructure(
                    id="s2",
                    name="Structure 2",
                    description="desc",
                    key_variables=["y"],
                    dynamics="d2",
                ),
            ],
        )

        s1 = domain.get_structure("s1")
        assert s1 is not None
        assert s1.name == "Structure 1"

        s3 = domain.get_structure("s3")
        assert s3 is None


class TestHypothesisScores:
    """Tests for HypothesisScores model."""

    def test_create_valid_scores(self) -> None:
        """Test creating valid scores."""
        from x_creative.core.types import HypothesisScores

        scores = HypothesisScores(
            divergence=8.0,
            testability=9.0,
            rationale=7.5,
            robustness=8.5,
            feasibility=7.0,
        )

        assert scores.divergence == 8.0
        assert scores.testability == 9.0
        assert scores.feasibility == 7.0

    def test_scores_validation_range(self) -> None:
        """Test that scores must be in valid range (0-10)."""
        from x_creative.core.types import HypothesisScores

        with pytest.raises(ValidationError):
            HypothesisScores(
                divergence=11.0,  # Invalid: > 10
                testability=9.0,
                rationale=7.5,
                robustness=8.5,
                feasibility=7.0,
            )

        with pytest.raises(ValidationError):
            HypothesisScores(
                divergence=8.0,
                testability=-1.0,  # Invalid: < 0
                rationale=7.5,
                robustness=8.5,
                feasibility=7.0,
            )

    def test_composite_score_default_weights(self) -> None:
        """Test composite score calculation with default weights."""
        from x_creative.core.types import HypothesisScores

        scores = HypothesisScores(
            divergence=8.0,
            testability=8.0,
            rationale=8.0,
            robustness=8.0,
            feasibility=8.0,
        )

        # All scores equal, composite should be 8.0
        composite = scores.composite()
        assert composite == 8.0

    def test_composite_score_custom_weights(self) -> None:
        """Test composite score with custom weights."""
        from x_creative.core.types import HypothesisScores

        scores = HypothesisScores(
            divergence=10.0,
            testability=0.0,
            rationale=0.0,
            robustness=0.0,
            feasibility=0.0,
        )

        # Only divergence weighted
        composite = scores.composite(
            w_divergence=1.0, w_testability=0.0, w_rationale=0.0, w_robustness=0.0, w_feasibility=0.0
        )
        assert composite == 10.0


class TestHypothesis:
    """Tests for Hypothesis model."""

    def test_create_valid_hypothesis(self) -> None:
        """Test creating a valid hypothesis."""
        from x_creative.core.types import Hypothesis, HypothesisScores

        hypothesis = Hypothesis(
            id="hyp_001",
            description="Use epidemic spreading models to design viral adoption",
            source_domain="queueing_theory",
            source_structure="queue_dynamics",
            analogy_explanation="Issue queue dynamics map to contributor onboarding flow",
            observable="open_issues_rate / avg_contributor_response_time",
            scores=HypothesisScores(
                divergence=8.0,
                testability=9.0,
                rationale=8.5,
                robustness=7.5,
                feasibility=8.0,
            ),
        )

        assert hypothesis.id == "hyp_001"
        assert hypothesis.source_domain == "queueing_theory"

    def test_hypothesis_composite_score(self) -> None:
        """Test hypothesis composite score calculation."""
        from x_creative.core.types import Hypothesis, HypothesisScores

        hypothesis = Hypothesis(
            id="test",
            description="Test hypothesis",
            source_domain="test",
            source_structure="test",
            analogy_explanation="test",
            observable="test",
            scores=HypothesisScores(
                divergence=10.0,
                testability=10.0,
                rationale=10.0,
                robustness=10.0,
                feasibility=10.0,
            ),
        )

        assert hypothesis.composite_score() == 10.0

    def test_hypothesis_accepts_theory_alias_fields(self) -> None:
        from x_creative.core.types import Hypothesis

        hypothesis = Hypothesis(
            id="hyp_alias",
            text="alias description",
            source_domain="test",
            source_structure="test",
            analogy_explanation="alias",
            observable="obs",
            generation_depth=2,
        )

        assert hypothesis.description == "alias description"
        assert hypothesis.generation == 2


class TestProblemFrame:
    """Tests for ProblemFrame model."""

    def test_create_valid_problem_frame(self) -> None:
        """Test creating a valid problem frame."""
        from x_creative.core.types import ProblemFrame

        frame = ProblemFrame(
            description="设计一个能实现病毒式传播的开源命令行工具",
            target_domain="open_source_development",
        )

        assert frame.target_domain == "open_source_development"
        assert frame.description == "设计一个能实现病毒式传播的开源命令行工具"

    def test_problem_frame_defaults(self) -> None:
        """Test problem frame default values."""
        from x_creative.core.types import ProblemFrame

        frame = ProblemFrame(description="test problem")

        assert frame.target_domain == "open_source_development"
        assert frame.constraints == []


class TestSearchConfig:
    """Tests for SearchConfig model."""

    def test_create_valid_search_config(self) -> None:
        """Test creating a valid search config."""
        from x_creative.core.types import SearchConfig

        config = SearchConfig(
            num_hypotheses=50,
            search_depth=3,
            search_breadth=5,
            prune_threshold=5.0,
        )

        assert config.num_hypotheses == 50
        assert config.search_depth == 3

    def test_search_config_defaults(self) -> None:
        """Test search config default values."""
        from x_creative.core.types import SearchConfig

        config = SearchConfig()

        assert config.num_hypotheses == 50
        assert config.search_depth == 3
        assert config.search_breadth == 5
        assert config.prune_threshold == 5.0


class TestGeneralProblemFrame:
    """Tests for generalized ProblemFrame."""

    def test_create_with_target_domain(self):
        from x_creative.core.types import ProblemFrame

        problem = ProblemFrame(
            description="发现能预测分子活性的结构特征",
            target_domain="drug_discovery",
        )

        assert problem.target_domain == "drug_discovery"
        assert problem.context == {}

    def test_create_with_context(self):
        from x_creative.core.types import ProblemFrame

        problem = ProblemFrame(
            description="设计一个能实现病毒式传播的开源命令行工具",
            target_domain="open_source_development",
            context={
                "language": "python",
                "license": "MIT",
                "target_audience": {"primary": "developers", "secondary": "devops"},
            },
        )

        assert problem.context["language"] == "python"
        assert problem.context["target_audience"]["primary"] == "developers"


class TestLogicVerdict:
    """Tests for LogicVerdict."""

    def test_create(self):
        from x_creative.core.types import LogicVerdict

        verdict = LogicVerdict(
            passed=True,
            analogy_validity=8.5,
            internal_consistency=9.0,
            causal_rigor=7.5,
            reasoning="类比映射逻辑清晰，因果链条完整",
        )

        assert verdict.passed is True
        assert verdict.analogy_validity == 8.5
        assert verdict.issues == []

    def test_with_issues(self):
        from x_creative.core.types import LogicVerdict

        verdict = LogicVerdict(
            passed=False,
            analogy_validity=4.0,
            internal_consistency=5.0,
            causal_rigor=3.0,
            reasoning="存在逻辑问题",
            issues=["类比映射存在跳跃", "因果关系不明确"],
        )

        assert verdict.passed is False
        assert len(verdict.issues) == 2


class TestSimilarWork:
    """Tests for SimilarWork."""

    def test_create(self):
        from x_creative.core.types import SimilarWork

        work = SimilarWork(
            title="Order Imbalance and Stock Returns",
            url="https://ssrn.com/abstract=123456",
            source="ssrn",
            similarity=0.65,
            difference_summary="该文献关注线性关系，本假说引入非线性效应",
        )

        assert work.source == "ssrn"
        assert 0 <= work.similarity <= 1


class TestNoveltyVerdict:
    """Tests for NoveltyVerdict."""

    def test_create_without_search(self):
        from x_creative.core.types import NoveltyVerdict

        verdict = NoveltyVerdict(
            score=5.5,
            searched=False,
            novelty_analysis="LLM 初步判断创新性中等",
        )

        assert verdict.searched is False
        assert verdict.similar_works == []

    def test_create_with_search(self):
        from x_creative.core.types import NoveltyVerdict, SimilarWork

        verdict = NoveltyVerdict(
            score=7.8,
            searched=True,
            similar_works=[
                SimilarWork(
                    title="Test Paper",
                    url="https://example.com",
                    source="arxiv",
                    similarity=0.4,
                    difference_summary="核心概念不同",
                ),
            ],
            novelty_analysis="经搜索验证，该假说具有较高创新性",
        )

        assert verdict.searched is True
        assert len(verdict.similar_works) == 1


class TestVerifiedHypothesis:
    """Tests for VerifiedHypothesis."""

    def test_create(self):
        from x_creative.core.types import (
            LogicVerdict,
            NoveltyVerdict,
            VerifiedHypothesis,
        )

        hyp = VerifiedHypothesis(
            id="hyp_test123",
            description="测试假说",
            source_domain="thermodynamics",
            source_structure="entropy_increase",
            analogy_explanation="熵增与市场分散度的类比",
            observable="entropy = -sum(p_i * log(p_i))",
            logic_verdict=LogicVerdict(
                passed=True,
                analogy_validity=8.0,
                internal_consistency=8.5,
                causal_rigor=7.5,
                reasoning="逻辑清晰",
            ),
            novelty_verdict=NoveltyVerdict(
                score=7.0,
                searched=False,
                novelty_analysis="概念新颖",
            ),
            final_score=7.5,
        )

        assert hyp.logic_verdict.passed is True
        assert hyp.final_score == 7.5

    def test_from_hypothesis(self):
        from x_creative.core.types import (
            Hypothesis,
            LogicVerdict,
            NoveltyVerdict,
            VerifiedHypothesis,
        )

        hypothesis = Hypothesis(
            id="hyp_original",
            description="原始假说",
            source_domain="thermodynamics",
            source_structure="entropy_increase",
            analogy_explanation="熵增与市场分散度的类比",
            observable="entropy = -sum(p_i * log(p_i))",
            formula="entropy_factor",
            parent_id="hyp_parent",
            generation=1,
            expansion_type="refine",
        )

        logic_verdict = LogicVerdict(
            passed=True,
            analogy_validity=8.0,
            internal_consistency=8.5,
            causal_rigor=7.5,
            reasoning="逻辑清晰",
        )

        novelty_verdict = NoveltyVerdict(
            score=7.0,
            searched=False,
            novelty_analysis="概念新颖",
        )

        verified = VerifiedHypothesis.from_hypothesis(
            hypothesis=hypothesis,
            logic_verdict=logic_verdict,
            novelty_verdict=novelty_verdict,
            final_score=7.5,
        )

        # Verify all fields are correctly transferred
        assert verified.id == hypothesis.id
        assert verified.description == hypothesis.description
        assert verified.formula == hypothesis.formula
        assert verified.parent_id == hypothesis.parent_id
        assert verified.generation == hypothesis.generation
        assert verified.expansion_type == hypothesis.expansion_type
        assert verified.logic_verdict == logic_verdict
        assert verified.novelty_verdict == novelty_verdict
        assert verified.final_score == 7.5


class TestProblemFrameExtendedFields:
    """Tests for ProblemFrame extended fields (answer engine)."""

    def test_backward_compatible_creation(self) -> None:
        """Create ProblemFrame without new fields; verify new fields are None/empty."""
        from x_creative.core.types import ProblemFrame

        frame = ProblemFrame(description="test problem")

        # Optional fields default to None
        assert frame.objective is None
        assert frame.scope is None
        assert frame.definitions is None
        assert frame.domain_hint is None

        # List fields default to empty list
        assert frame.success_criteria == []
        assert frame.open_questions == []

    def test_full_creation_with_new_fields(self) -> None:
        """Create ProblemFrame with all new fields; verify values."""
        from x_creative.core.types import ProblemFrame

        frame = ProblemFrame(
            description="设计一个能实现病毒式传播的开源命令行工具",
            objective="Design a viral open-source CLI tool",
            scope={"in_scope": ["CLI tools", "developer workflow"], "out_of_scope": ["GUI apps"]},
            definitions={"viral": "self-propagating adoption through usage"},
            success_criteria=["GitHub stars > 1000 in 3 months", "weekly active users > 500"],
            open_questions=["Which programming language to target?"],
            domain_hint={"domain": "open_source_development", "confidence": 0.95},
        )

        assert frame.objective == "Design a viral open-source CLI tool"
        assert frame.scope == {"in_scope": ["CLI tools", "developer workflow"], "out_of_scope": ["GUI apps"]}
        assert frame.definitions == {"viral": "self-propagating adoption through usage"}
        assert frame.success_criteria == ["GitHub stars > 1000 in 3 months", "weekly active users > 500"]
        assert frame.open_questions == ["Which programming language to target?"]
        assert frame.domain_hint == {"domain": "open_source_development", "confidence": 0.95}

    def test_serialization_roundtrip(self) -> None:
        """model_dump() -> model_validate() preserves new fields."""
        from x_creative.core.types import ProblemFrame

        original = ProblemFrame(
            description="test roundtrip",
            objective="test objective",
            scope={"in": ["a"], "out": ["b"]},
            definitions={"term1": "def1"},
            success_criteria=["criteria1", "criteria2"],
            open_questions=["q1"],
            domain_hint={"domain": "test", "confidence": 0.8},
        )

        dumped = original.model_dump()
        restored = ProblemFrame.model_validate(dumped)

        assert restored.objective == original.objective
        assert restored.scope == original.scope
        assert restored.definitions == original.definitions
        assert restored.success_criteria == original.success_criteria
        assert restored.open_questions == original.open_questions
        assert restored.domain_hint == original.domain_hint

        # Also check the existing fields survived
        assert restored.description == original.description
        assert restored.target_domain == original.target_domain


class TestMappingItem:
    """Tests for MappingItem model."""

    def test_create_valid_mapping_item(self) -> None:
        from x_creative.core.types import MappingItem

        item = MappingItem(
            source_concept="熵",
            target_concept="订单簿信息混乱度",
            source_relation="熵随孤立系统演化单调递增",
            target_relation="订单簿信息熵在无新信息注入时趋于增大",
            mapping_type="relation",
            systematicity_group_id="entropy_dynamics",
            observable_link="orderbook_entropy = -Σ(p_i × log(p_i))",
            confidence=0.8,
        )

        assert item.source_concept == "熵"
        assert item.mapping_type == "relation"
        assert item.confidence == 0.8

    def test_mapping_item_defaults(self) -> None:
        from x_creative.core.types import MappingItem

        item = MappingItem(
            source_concept="A",
            target_concept="B",
            source_relation="R1",
            target_relation="R2",
            mapping_type="entity",
            systematicity_group_id="g1",
        )

        assert item.observable_link is None
        assert item.confidence == 0.5
        assert item.evidence_refs == []

    def test_mapping_item_invalid_type(self) -> None:
        from x_creative.core.types import MappingItem
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MappingItem(
                source_concept="A",
                target_concept="B",
                source_relation="R1",
                target_relation="R2",
                mapping_type="invalid_type",
                systematicity_group_id="g1",
            )

    def test_mapping_item_confidence_range(self) -> None:
        from x_creative.core.types import MappingItem
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MappingItem(
                source_concept="A",
                target_concept="B",
                source_relation="R1",
                target_relation="R2",
                mapping_type="entity",
                systematicity_group_id="g1",
                confidence=1.5,
            )


class TestFailureMode:
    """Tests for FailureMode model."""

    def test_create_valid_failure_mode(self) -> None:
        from x_creative.core.types import FailureMode

        fm = FailureMode(
            scenario="当市场处于极端流动性危机时",
            why_breaks="熵增假设依赖正常市场微观结构，危机中做市商撤离导致订单簿结构崩塌",
            detectable_signal="bid-ask spread > 历史99分位",
        )

        assert "极端流动性" in fm.scenario
        assert "做市商" in fm.why_breaks


class TestHypothesisMappingFields:
    """Tests for new mapping fields on Hypothesis."""

    def test_hypothesis_with_mapping_table(self) -> None:
        from x_creative.core.types import Hypothesis, MappingItem, FailureMode

        h = Hypothesis(
            id="hyp_test",
            description="Test",
            source_domain="thermodynamics",
            source_structure="entropy",
            analogy_explanation="test",
            observable="test",
            mapping_table=[
                MappingItem(
                    source_concept="熵",
                    target_concept="市场混乱度",
                    source_relation="熵增",
                    target_relation="混乱度增加",
                    mapping_type="relation",
                    systematicity_group_id="g1",
                ),
            ],
            failure_modes=[
                FailureMode(
                    scenario="极端行情",
                    why_breaks="微观结构崩塌",
                    detectable_signal="spread > 99%ile",
                ),
            ],
            mapping_quality=7.5,
        )

        assert len(h.mapping_table) == 1
        assert len(h.failure_modes) == 1
        assert h.mapping_quality == 7.5

    def test_hypothesis_mapping_defaults_empty(self) -> None:
        from x_creative.core.types import Hypothesis

        h = Hypothesis(
            id="hyp_test",
            description="Test",
            source_domain="test",
            source_structure="test",
            analogy_explanation="test",
            observable="test",
        )

        assert h.mapping_table == []
        assert h.failure_modes == []
        assert h.mapping_quality is None

    def test_verified_hypothesis_from_hypothesis_preserves_mapping(self) -> None:
        from x_creative.core.types import (
            Hypothesis, MappingItem, FailureMode,
            LogicVerdict, NoveltyVerdict, VerifiedHypothesis,
        )

        h = Hypothesis(
            id="hyp_test",
            description="Test",
            source_domain="test",
            source_structure="test",
            analogy_explanation="test",
            observable="test",
            mapping_table=[
                MappingItem(
                    source_concept="A",
                    target_concept="B",
                    source_relation="R1",
                    target_relation="R2",
                    mapping_type="entity",
                    systematicity_group_id="g1",
                ),
            ],
            failure_modes=[
                FailureMode(
                    scenario="S",
                    why_breaks="W",
                    detectable_signal="D",
                ),
            ],
            mapping_quality=8.0,
        )

        logic = LogicVerdict(
            passed=True,
            analogy_validity=8.0,
            internal_consistency=8.5,
            causal_rigor=7.5,
            reasoning="OK",
        )
        novelty = NoveltyVerdict(
            score=7.0,
            searched=False,
            novelty_analysis="Novel",
        )

        verified = VerifiedHypothesis.from_hypothesis(
            hypothesis=h,
            logic_verdict=logic,
            novelty_verdict=novelty,
            final_score=7.5,
        )

        assert len(verified.mapping_table) == 1
        assert len(verified.failure_modes) == 1
        assert verified.mapping_quality == 8.0


class TestHypothesisParetoAxes:
    """Tests for Pareto axis helpers on Hypothesis."""

    def test_novelty_axis_returns_divergence(self) -> None:
        from x_creative.core.types import Hypothesis, HypothesisScores

        h = Hypothesis(
            id="hyp_pareto",
            description="Test",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=HypothesisScores(
                divergence=8.5,
                testability=7.0,
                rationale=6.0,
                robustness=5.0,
                feasibility=9.0,
            ),
        )

        assert h.novelty_axis() == 8.5

    def test_feasibility_axis_returns_four_dim_average(self) -> None:
        from x_creative.core.types import Hypothesis, HypothesisScores

        h = Hypothesis(
            id="hyp_pareto",
            description="Test",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=HypothesisScores(
                divergence=8.5,
                testability=7.0,
                rationale=6.0,
                robustness=5.0,
                feasibility=9.0,
            ),
        )

        # (7.0 + 6.0 + 5.0 + 9.0) / 4.0 = 6.75
        assert h.feasibility_axis() == pytest.approx(6.75)

    def test_axes_return_none_when_unscored(self) -> None:
        from x_creative.core.types import Hypothesis

        h = Hypothesis(
            id="hyp_unscored",
            description="Test",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
        )

        assert h.novelty_axis() is None
        assert h.feasibility_axis() is None


class TestVerifyStatus:
    def test_verify_status_values(self) -> None:
        from x_creative.core.types import VerifyStatus

        assert VerifyStatus.PASSED == "passed"
        assert VerifyStatus.FAILED == "failed"
        assert VerifyStatus.ESCALATED == "escalated"
        assert VerifyStatus.ABSTAINED == "abstained"


class TestConstraintSpec:
    def test_create_valid_constraint(self) -> None:
        from x_creative.core.types import ConstraintSpec

        c = ConstraintSpec(
            text="不使用未来信息",
            priority="critical",
            type="hard",
            origin="user",
        )
        assert c.priority == "critical"
        assert c.weight == 0.5  # default

    def test_constraint_defaults(self) -> None:
        from x_creative.core.types import ConstraintSpec

        c = ConstraintSpec(text="soft constraint")
        assert c.priority == "medium"
        assert c.type == "soft"
        assert c.origin == "user"

    def test_problem_frame_structured_constraints(self) -> None:
        from x_creative.core.types import ConstraintSpec, ProblemFrame

        frame = ProblemFrame(
            description="test",
            structured_constraints=[
                ConstraintSpec(text="C1", priority="critical", type="hard"),
                ConstraintSpec(text="C2", priority="low", type="soft"),
            ],
        )
        assert len(frame.structured_constraints) == 2
        assert frame.structured_constraints[0].priority == "critical"

    def test_problem_frame_backward_compat(self) -> None:
        """Old-style constraints (list[str]) should still work."""
        from x_creative.core.types import ProblemFrame

        frame = ProblemFrame(
            description="test",
            constraints=["no lookahead", "daily frequency"],
        )
        assert len(frame.constraints) == 2
        assert frame.structured_constraints == []


class TestHypothesisNewFields:
    def test_blend_network_field(self) -> None:
        from x_creative.core.blend_types import BlendNetwork
        from x_creative.core.types import Hypothesis

        h = Hypothesis(
            id="hyp_test",
            description="test",
            source_domain="thermo",
            source_structure="entropy",
            analogy_explanation="test",
            observable="test",
            blend_network=BlendNetwork(
                input1_summary="A",
                input2_summary="B",
                generic_space="C",
                blend_description="D",
                cross_space_mappings=[],
                emergent_structures=[],
            ),
        )
        assert h.blend_network is not None
        assert h.blend_network.input1_summary == "A"

    def test_space_transform_diff_field(self) -> None:
        from x_creative.core.transform_types import SpaceTransformDiff
        from x_creative.core.types import Hypothesis

        h = Hypothesis(
            id="hyp_test",
            description="test",
            source_domain="thermo",
            source_structure="entropy",
            analogy_explanation="test",
            observable="test",
            space_transform_diff=SpaceTransformDiff(
                concept_space_version="1.0.0",
                actions=[],
                new_failure_modes=[],
                new_detectable_signals=[],
                new_observables=[],
            ),
        )
        assert h.space_transform_diff is not None

    def test_behavior_descriptor_field(self) -> None:
        from x_creative.creativity.qd_types import BehaviorDescriptor
        from x_creative.core.types import Hypothesis

        h = Hypothesis(
            id="hyp_test",
            description="test",
            source_domain="thermo",
            source_structure="entropy",
            analogy_explanation="test",
            observable="test",
            behavior_descriptor=BehaviorDescriptor(
                grid_dims={"mechanism_family": 0},
                raw={},
                version="1.0.0",
                extraction_method="rule",
            ),
        )
        assert h.behavior_descriptor is not None

    def test_new_fields_default_none(self) -> None:
        from x_creative.core.types import Hypothesis

        h = Hypothesis(
            id="hyp_test",
            description="test",
            source_domain="thermo",
            source_structure="entropy",
            analogy_explanation="test",
            observable="test",
        )
        assert h.blend_network is None
        assert h.space_transform_diff is None
        assert h.behavior_descriptor is None

    def test_unified_evidence_is_built_from_flat_fields(self) -> None:
        from x_creative.core.blend_types import BlendNetwork
        from x_creative.core.types import Hypothesis
        from x_creative.hkg.types import HKGEvidence

        h = Hypothesis(
            id="hyp_with_evidence",
            description="test",
            source_domain="thermo",
            source_structure="entropy",
            analogy_explanation="test",
            observable="test",
            blend_network=BlendNetwork(
                input1_summary="A",
                input2_summary="B",
                generic_space="C",
                blend_description="D",
                cross_space_mappings=[],
                emergent_structures=[],
            ),
            hkg_evidence=HKGEvidence(),
        )

        assert h.evidence is not None
        assert h.evidence.blend_network is not None
        assert h.evidence.hyperpaths == []

    def test_unified_evidence_can_populate_flat_fields(self) -> None:
        from x_creative.core.blend_types import BlendNetwork
        from x_creative.core.types import Hypothesis, HypothesisEvidence
        from x_creative.hkg.types import HyperedgeSummary, HyperpathEvidence

        h = Hypothesis(
            id="hyp_from_nested",
            description="test",
            source_domain="thermo",
            source_structure="entropy",
            analogy_explanation="test",
            observable="test",
            evidence=HypothesisEvidence(
                hyperpaths=[
                    HyperpathEvidence(
                        start_node_id="n1",
                        end_node_id="n2",
                        path_rank=1,
                        path_length=1,
                        hyperedges=[
                            HyperedgeSummary(
                                edge_id="e1",
                                nodes=["n1", "n2"],
                                relation="rel",
                                provenance_refs=[],
                            )
                        ],
                        intermediate_nodes=[],
                    )
                ],
                hkg_params={"K": 3, "IS": 1, "max_len": 6, "matcher": "auto"},
                coverage={"start_match": 1.0, "end_match": 1.0},
                blend_network=BlendNetwork(
                    input1_summary="A",
                    input2_summary="B",
                    generic_space="C",
                    blend_description="D",
                    cross_space_mappings=[],
                    emergent_structures=[],
                ),
            ),
        )

        assert h.hkg_evidence is not None
        assert len(h.hkg_evidence.hyperpaths) == 1
        assert h.blend_network is not None


class TestLogicVerdictConfidence:
    def test_logic_verdict_with_confidence(self) -> None:
        from x_creative.core.types import LogicVerdict

        verdict = LogicVerdict(
            passed=True,
            analogy_validity=8.0,
            internal_consistency=8.5,
            causal_rigor=7.5,
            reasoning="Test",
            judge_confidence=0.85,
            score_std=0.3,
            position_consistency=True,
            position_bias_flag=False,
        )

        assert verdict.judge_confidence == 0.85
        assert verdict.score_std == 0.3
        assert verdict.position_consistency is True
        assert verdict.position_bias_flag is False

    def test_logic_verdict_confidence_defaults(self) -> None:
        from x_creative.core.types import LogicVerdict

        verdict = LogicVerdict(
            passed=True,
            analogy_validity=8.0,
            internal_consistency=8.5,
            causal_rigor=7.5,
            reasoning="Test",
        )

        assert verdict.judge_confidence is None
        assert verdict.position_consistency is True  # default
        assert verdict.position_bias_flag is False  # default
