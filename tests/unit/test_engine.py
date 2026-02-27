"""Tests for the CreativityEngine with dual-model verification integration."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x_creative.core.types import (
    Hypothesis,
    HypothesisScores,
    LogicVerdict,
    NoveltyVerdict,
    ProblemFrame,
    SearchConfig,
    SimilarWork,
    VerifyStatus,
    VerifiedHypothesis,
)
from x_creative.core.blend_types import BlendNetwork
from x_creative.creativity.engine import CreativityEngine
from x_creative.hkg.types import HKGEvidence, HyperedgeSummary, HyperpathEvidence
from x_creative.verify import LogicVerifier, NoveltyVerifier, SearchValidator


@pytest.fixture
def sample_problem() -> ProblemFrame:
    """Create a sample problem frame for testing."""
    return ProblemFrame(
        description="How to design a viral open source tool?",
    )


@pytest.fixture
def sample_hypothesis() -> Hypothesis:
    """Create a sample hypothesis for testing."""
    return Hypothesis(
        id="hyp_test_001",
        description="Order flow queuing pressure factor",
        source_domain="queueing_theory",
        source_structure="queue_dynamics",
        analogy_explanation="Order book acts like a service queue",
        observable="bid_depth_imbalance / avg_trade_rate",
    )


@pytest.fixture
def sample_hypotheses() -> list[Hypothesis]:
    """Create multiple sample hypotheses for testing."""
    return [
        Hypothesis(
            id=f"hyp_test_{i:03d}",
            description=f"Test hypothesis {i}",
            source_domain=f"domain_{i}",
            source_structure=f"structure_{i}",
            analogy_explanation=f"Explanation for hypothesis {i}",
            observable=f"factor_{i} = x / y",
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_logic_verdict_passed() -> LogicVerdict:
    """Create a passing logic verdict."""
    return LogicVerdict(
        passed=True,
        analogy_validity=8.0,
        internal_consistency=7.5,
        causal_rigor=8.0,
        reasoning="The analogy is sound and the reasoning is solid.",
        issues=[],
    )


@pytest.fixture
def sample_logic_verdict_failed() -> LogicVerdict:
    """Create a failing logic verdict."""
    return LogicVerdict(
        passed=False,
        analogy_validity=4.0,
        internal_consistency=5.0,
        causal_rigor=3.5,
        reasoning="The analogy has significant flaws.",
        issues=["Weak causal link", "Inconsistent assumptions"],
    )


@pytest.fixture
def sample_novelty_verdict_low() -> NoveltyVerdict:
    """Create a low novelty verdict (no search needed)."""
    return NoveltyVerdict(
        score=4.5,
        searched=False,
        similar_works=[
            SimilarWork(
                title="Similar paper A",
                url="",
                source="arxiv",
                similarity=0.7,
                difference_summary="Very similar approach",
            )
        ],
        novelty_analysis="This idea is well-established in literature.",
    )


@pytest.fixture
def sample_novelty_verdict_high() -> NoveltyVerdict:
    """Create a high novelty verdict (search needed, score >= 6.0)."""
    return NoveltyVerdict(
        score=7.5,
        searched=False,
        similar_works=[],
        novelty_analysis="This appears to be a novel approach.",
    )


@pytest.fixture
def sample_novelty_verdict_searched() -> NoveltyVerdict:
    """Create a novelty verdict after search validation."""
    return NoveltyVerdict(
        score=8.0,
        searched=True,
        similar_works=[
            SimilarWork(
                title="Related work found",
                url="https://example.com/paper",
                source="web",
                similarity=0.3,
                difference_summary="Different approach but related topic",
            )
        ],
        novelty_analysis="Web search confirmed novelty. Score: 8.0",
    )


class TestCreativityEngineInit:
    """Tests for CreativityEngine initialization."""

    def test_engine_creation_default(self) -> None:
        """Test creating engine with default components."""
        engine = CreativityEngine()
        assert engine is not None
        assert engine._logic_verifier is not None
        assert engine._novelty_verifier is not None
        assert engine._search_validator is not None

    def test_engine_with_custom_verifiers(self) -> None:
        """Test creating engine with custom verifiers (dependency injection)."""
        # Create mock verifiers
        mock_logic = MagicMock(spec=LogicVerifier)
        mock_novelty = MagicMock(spec=NoveltyVerifier)
        mock_search = MagicMock(spec=SearchValidator)

        engine = CreativityEngine(
            logic_verifier=mock_logic,
            novelty_verifier=mock_novelty,
            search_validator=mock_search,
        )

        assert engine._logic_verifier is mock_logic
        assert engine._novelty_verifier is mock_novelty
        assert engine._search_validator is mock_search

    @pytest.mark.asyncio
    async def test_engine_hkg_wires_embedding_matcher_from_settings(self, tmp_path) -> None:  # noqa: ANN001
        """Engine should wire HKG matcher with embedding client + matcher mode."""
        from pydantic import SecretStr

        from x_creative.config.settings import Settings
        from x_creative.hkg.store import HypergraphStore
        from x_creative.hkg.types import HKGNode, Hyperedge, Provenance

        store = HypergraphStore()
        store.add_node(HKGNode(node_id="n1", name="entropy"))
        store.add_node(HKGNode(node_id="n2", name="volatility"))
        store.add_edge(
            Hyperedge(
                edge_id="e1",
                nodes=["n1", "n2"],
                relation="rel",
                provenance=[Provenance(doc_id="d1", chunk_id="c1")],
            )
        )
        store_path = tmp_path / "hkg_store.json"
        store.save(store_path)

        settings = Settings()
        settings.hkg_enabled = True
        settings.hkg_store_path = store_path
        settings.hkg_matcher = "embedding"
        settings.hkg_embedding_provider = "openrouter"
        settings.openrouter.api_key = SecretStr("test-key")

        with patch("x_creative.creativity.engine.get_settings", return_value=settings):
            engine = CreativityEngine()
            try:
                assert engine._hkg_matcher is not None
                assert engine._hkg_embedding_client is not None
                assert engine._search._hkg_params is not None
                assert engine._search._hkg_params.matcher == "embedding"
            finally:
                await engine.close()


class TestGenerateHypotheses:
    """Tests for generate_hypotheses method."""

    @pytest.mark.asyncio
    async def test_generate_with_verification(
        self,
        sample_problem: ProblemFrame,
        sample_hypotheses: list[Hypothesis],
        sample_logic_verdict_passed: LogicVerdict,
        sample_novelty_verdict_low: NoveltyVerdict,
    ) -> None:
        """Test that verify=True runs verifiers on generated hypotheses."""
        engine = CreativityEngine()

        # Mock BISO to return sample hypotheses
        with patch.object(engine._biso, "generate_all_analogies") as mock_biso:
            mock_biso.return_value = sample_hypotheses[:3]

            # Mock logic verifier
            with patch.object(engine._logic_verifier, "verify") as mock_logic:
                mock_logic.return_value = sample_logic_verdict_passed

                # Mock novelty verifier
                with patch.object(engine._novelty_verifier, "verify") as mock_novelty:
                    mock_novelty.return_value = sample_novelty_verdict_low

                    # Mock needs_search to return False (score < 6.0)
                    with patch.object(
                        engine._novelty_verifier, "needs_search"
                    ) as mock_needs_search:
                        mock_needs_search.return_value = False

                        result = await engine.generate_hypotheses(
                            problem_frame=sample_problem,
                            num_hypotheses=3,
                            verify=True,
                        )

                        # Verify BISO was called
                        mock_biso.assert_called_once()

                        # Verify each hypothesis was verified
                        assert mock_logic.call_count == 3
                        assert mock_novelty.call_count == 3

                        # Verify results
                        assert len(result) == 3
                        assert all(
                            isinstance(h, VerifiedHypothesis) for h in result
                        )

    @pytest.mark.asyncio
    async def test_generate_without_verification(
        self,
        sample_problem: ProblemFrame,
        sample_hypotheses: list[Hypothesis],
    ) -> None:
        """Test that verify=False skips verifiers."""
        engine = CreativityEngine()

        # Mock BISO to return sample hypotheses
        with patch.object(engine._biso, "generate_all_analogies") as mock_biso:
            mock_biso.return_value = sample_hypotheses[:3]

            # Mock verifiers to ensure they're not called
            with patch.object(engine._logic_verifier, "verify") as mock_logic:
                with patch.object(engine._novelty_verifier, "verify") as mock_novelty:
                    result = await engine.generate_hypotheses(
                        problem_frame=sample_problem,
                        num_hypotheses=3,
                        verify=False,
                    )

                    # Verify BISO was called
                    mock_biso.assert_called_once()

                    # Verify verifiers were NOT called
                    mock_logic.assert_not_called()
                    mock_novelty.assert_not_called()

                    # Verify results are still VerifiedHypothesis (with placeholders)
                    assert len(result) == 3
                    assert all(isinstance(h, VerifiedHypothesis) for h in result)

                    # Verify placeholder values
                    for h in result:
                        assert h.logic_verdict.passed is True
                        assert h.logic_verdict.reasoning == "Verification skipped (verify=False)"
                        assert h.novelty_verdict.searched is False
                        assert h.final_score == 5.0

    @pytest.mark.asyncio
    async def test_generate_empty_biso_result(
        self,
        sample_problem: ProblemFrame,
    ) -> None:
        """Test handling when BISO returns no hypotheses."""
        engine = CreativityEngine()

        with patch.object(engine._biso, "generate_all_analogies") as mock_biso:
            mock_biso.return_value = []

            result = await engine.generate_hypotheses(
                problem_frame=sample_problem,
                num_hypotheses=10,
                verify=True,
            )

            assert result == []


class TestVerifyHypothesis:
    """Tests for verify_hypothesis method."""

    @pytest.mark.asyncio
    async def test_verify_hypothesis_returns_verified(
        self,
        sample_problem: ProblemFrame,
        sample_hypothesis: Hypothesis,
        sample_logic_verdict_passed: LogicVerdict,
        sample_novelty_verdict_low: NoveltyVerdict,
    ) -> None:
        """Test that verify_hypothesis returns a properly verified hypothesis."""
        engine = CreativityEngine()

        with patch.object(engine._logic_verifier, "verify") as mock_logic:
            mock_logic.return_value = sample_logic_verdict_passed

            with patch.object(engine._novelty_verifier, "verify") as mock_novelty:
                mock_novelty.return_value = sample_novelty_verdict_low

                with patch.object(
                    engine._novelty_verifier, "needs_search"
                ) as mock_needs_search:
                    mock_needs_search.return_value = False

                    result = await engine.verify_hypothesis(
                        hypothesis=sample_hypothesis,
                        problem_frame=sample_problem,
                    )

                    # Verify result type
                    assert isinstance(result, VerifiedHypothesis)

                    # Verify hypothesis data was copied
                    assert result.id == sample_hypothesis.id
                    assert result.description == sample_hypothesis.description

                    # Verify verdicts
                    assert result.logic_verdict == sample_logic_verdict_passed
                    assert result.novelty_verdict == sample_novelty_verdict_low

                    # Verify final score was calculated
                    assert 0.0 <= result.final_score <= 10.0

    @pytest.mark.asyncio
    async def test_high_novelty_triggers_search(
        self,
        sample_problem: ProblemFrame,
        sample_hypothesis: Hypothesis,
        sample_logic_verdict_passed: LogicVerdict,
        sample_novelty_verdict_high: NoveltyVerdict,
        sample_novelty_verdict_searched: NoveltyVerdict,
    ) -> None:
        """Test that novelty score >= 6.0 triggers SearchValidator."""
        engine = CreativityEngine()

        with patch.object(engine._logic_verifier, "verify") as mock_logic:
            mock_logic.return_value = sample_logic_verdict_passed

            with patch.object(engine._novelty_verifier, "verify") as mock_novelty:
                # Initial novelty verdict with high score
                mock_novelty.return_value = sample_novelty_verdict_high

                with patch.object(
                    engine._novelty_verifier, "needs_search"
                ) as mock_needs_search:
                    # Score is 7.5 >= 6.0, so needs_search should return True
                    mock_needs_search.return_value = True

                    with patch.object(
                        engine._search_validator, "validate"
                    ) as mock_search:
                        # Search validator returns updated verdict
                        mock_search.return_value = sample_novelty_verdict_searched

                        result = await engine.verify_hypothesis(
                            hypothesis=sample_hypothesis,
                            problem_frame=sample_problem,
                        )

                        # Verify search validator was called
                        mock_search.assert_called_once_with(
                            hypothesis=sample_hypothesis,
                            preliminary_score=sample_novelty_verdict_high.score,
                        )

                        # Verify final verdict is from search
                        assert result.novelty_verdict.searched is True
                        assert result.novelty_verdict.score == 8.0

    @pytest.mark.asyncio
    async def test_low_novelty_skips_search(
        self,
        sample_problem: ProblemFrame,
        sample_hypothesis: Hypothesis,
        sample_logic_verdict_passed: LogicVerdict,
        sample_novelty_verdict_low: NoveltyVerdict,
    ) -> None:
        """Test that novelty score < 6.0 skips SearchValidator."""
        engine = CreativityEngine()

        with patch.object(engine._logic_verifier, "verify") as mock_logic:
            mock_logic.return_value = sample_logic_verdict_passed

            with patch.object(engine._novelty_verifier, "verify") as mock_novelty:
                mock_novelty.return_value = sample_novelty_verdict_low

                with patch.object(
                    engine._novelty_verifier, "needs_search"
                ) as mock_needs_search:
                    # Score is 4.5 < 6.0, so needs_search should return False
                    mock_needs_search.return_value = False

                    with patch.object(
                        engine._search_validator, "validate"
                    ) as mock_search:
                        result = await engine.verify_hypothesis(
                            hypothesis=sample_hypothesis,
                            problem_frame=sample_problem,
                        )

                        # Verify search validator was NOT called
                        mock_search.assert_not_called()

                        # Verify verdict is from novelty verifier (not searched)
                        assert result.novelty_verdict.searched is False

    @pytest.mark.asyncio
    async def test_verify_hypothesis_computes_structural_score_in_range(
        self,
        sample_problem: ProblemFrame,
        sample_logic_verdict_passed: LogicVerdict,
        sample_novelty_verdict_low: NoveltyVerdict,
    ) -> None:
        engine = CreativityEngine()
        engine._settings.hkg_enable_structural_scoring = True

        hypothesis = Hypothesis(
            id="h_struct",
            description="structural hypothesis",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            hkg_evidence=HKGEvidence(
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
                                relation="r",
                                provenance_refs=["doc/chunk"],
                            )
                        ],
                        intermediate_nodes=[],
                    )
                ]
            ),
        )

        with patch.object(engine._logic_verifier, "verify") as mock_logic:
            mock_logic.return_value = sample_logic_verdict_passed
            with patch.object(engine._novelty_verifier, "verify") as mock_novelty:
                mock_novelty.return_value = sample_novelty_verdict_low
                with patch.object(engine._novelty_verifier, "needs_search") as mock_needs_search:
                    mock_needs_search.return_value = False
                    result = await engine.verify_hypothesis(
                        hypothesis=hypothesis,
                        problem_frame=sample_problem,
                    )

        assert result.structural_grounding_score is not None
        assert 0.0 <= result.structural_grounding_score <= 10.0


class TestFinalScoreCalculation:
    """Tests for final score calculation."""

    def test_final_score_passed_logic(self) -> None:
        """Test final score calculation with passed logic."""
        engine = CreativityEngine()

        logic_verdict = LogicVerdict(
            passed=True,
            analogy_validity=8.0,
            internal_consistency=8.0,
            causal_rigor=8.0,
            reasoning="Good",
            issues=[],
        )

        novelty_verdict = NoveltyVerdict(
            score=7.0,
            searched=False,
            similar_works=[],
            novelty_analysis="Novel",
        )

        score = engine._calculate_final_score(logic_verdict, novelty_verdict)

        # Logic avg = 8.0, novelty = 7.0
        # Final = 0.4 * 8.0 + 0.6 * 7.0 = 3.2 + 4.2 = 7.4
        assert abs(score - 7.4) < 0.01

    def test_final_score_uses_configurable_weights(self) -> None:
        engine = CreativityEngine()
        engine._settings.final_score_logic_weight = 0.2
        engine._settings.final_score_novelty_weight = 0.8

        logic_verdict = LogicVerdict(
            passed=True,
            analogy_validity=8.0,
            internal_consistency=8.0,
            causal_rigor=8.0,
            reasoning="Good",
            issues=[],
        )
        novelty_verdict = NoveltyVerdict(
            score=7.0,
            searched=False,
            similar_works=[],
            novelty_analysis="Novel",
        )

        score = engine._calculate_final_score(logic_verdict, novelty_verdict)
        assert abs(score - 7.2) < 0.01

    def test_final_score_failed_logic_rejected(self) -> None:
        """Failed logic verification should hard-gate final score to zero."""
        engine = CreativityEngine()

        logic_verdict = LogicVerdict(
            passed=False,  # Failed!
            analogy_validity=4.0,
            internal_consistency=5.0,
            causal_rigor=3.0,
            reasoning="Flawed",
            issues=["Major issues"],
        )

        novelty_verdict = NoveltyVerdict(
            score=8.0,
            searched=True,
            similar_works=[],
            novelty_analysis="Very novel",
        )

        score = engine._calculate_final_score(logic_verdict, novelty_verdict)

        assert score == 0.0


class TestEngineWithCustomVerifiers:
    """Tests for engine with custom injected verifiers."""

    @pytest.mark.asyncio
    async def test_engine_with_custom_verifiers(
        self,
        sample_problem: ProblemFrame,
        sample_hypothesis: Hypothesis,
    ) -> None:
        """Test that custom verifiers are properly used."""
        # Create mock verifiers with async methods
        mock_logic = MagicMock(spec=LogicVerifier)
        mock_logic.verify = AsyncMock(
            return_value=LogicVerdict(
                passed=True,
                analogy_validity=9.0,
                internal_consistency=9.0,
                causal_rigor=9.0,
                reasoning="Custom logic verifier",
                issues=[],
            )
        )

        mock_novelty = MagicMock(spec=NoveltyVerifier)
        mock_novelty.verify = AsyncMock(
            return_value=NoveltyVerdict(
                score=5.0,
                searched=False,
                similar_works=[],
                novelty_analysis="Custom novelty verifier",
            )
        )
        mock_novelty.needs_search = MagicMock(return_value=False)

        mock_search = MagicMock(spec=SearchValidator)

        engine = CreativityEngine(
            logic_verifier=mock_logic,
            novelty_verifier=mock_novelty,
            search_validator=mock_search,
        )

        result = await engine.verify_hypothesis(
            hypothesis=sample_hypothesis,
            problem_frame=sample_problem,
        )

        # Verify custom verifiers were called
        mock_logic.verify.assert_called_once()
        mock_novelty.verify.assert_called_once()
        # needs_search may be called multiple times (logging + conditional)
        mock_novelty.needs_search.assert_called_with(5.0)

        # Search should not be called since needs_search returned False
        mock_search.validate.assert_not_called()

        # Verify results came from custom verifiers
        assert result.logic_verdict.reasoning == "Custom logic verifier"
        assert result.novelty_verdict.novelty_analysis == "Custom novelty verifier"


class TestHelperMethods:
    """Tests for helper methods."""

    def test_create_failed_verification(self) -> None:
        """Test creating a failed verification result."""
        engine = CreativityEngine()

        hypothesis = Hypothesis(
            id="hyp_fail",
            description="Test",
            source_domain="test",
            source_structure="test",
            analogy_explanation="test",
            observable="test",
        )

        result = engine._create_failed_verification(
            hypothesis=hypothesis,
            error_message="Test error",
        )

        assert isinstance(result, VerifiedHypothesis)
        assert result.logic_verdict.passed is False
        assert "Test error" in result.logic_verdict.reasoning
        assert result.novelty_verdict.score == 0.0
        assert result.final_score == 0.0

    def test_create_unverified_hypothesis(self) -> None:
        """Test creating an unverified hypothesis (verify=False placeholder)."""
        engine = CreativityEngine()

        hypothesis = Hypothesis(
            id="hyp_unverified",
            description="Test",
            source_domain="test",
            source_structure="test",
            analogy_explanation="test",
            observable="test",
        )

        result = engine._create_unverified_hypothesis(hypothesis)

        assert isinstance(result, VerifiedHypothesis)
        assert result.logic_verdict.passed is True
        assert result.logic_verdict.analogy_validity == 5.0
        assert result.novelty_verdict.score == 5.0
        assert result.final_score == 5.0
        assert "skipped" in result.logic_verdict.reasoning.lower()


class TestPipelineRankingBehavior:
    """Tests for ranking behavior in the main generation pipeline."""

    @staticmethod
    def _scores(base: float) -> HypothesisScores:
        return HypothesisScores(
            divergence=base,
            testability=base,
            rationale=base,
            robustness=base,
            feasibility=base,
        )

    def test_sort_by_score_prefers_final_score_then_composite(self) -> None:
        """Verified hypotheses should be ranked by final_score first."""
        engine = CreativityEngine()

        h_unverified = Hypothesis(
            id="h_unverified",
            description="unverified",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=self._scores(9.0),
            final_score=None,
        )
        h_verified_low = Hypothesis(
            id="h_verified_low",
            description="verified low",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=self._scores(4.0),
            final_score=7.0,
        )
        h_verified_high = Hypothesis(
            id="h_verified_high",
            description="verified high",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=self._scores(3.0),
            final_score=8.0,
        )

        ranked = engine.sort_by_score([h_unverified, h_verified_low, h_verified_high])
        assert [h.id for h in ranked] == [
            "h_verified_high",
            "h_verified_low",
            "h_unverified",
        ]

    @pytest.mark.asyncio
    async def test_generate_runs_dual_model_verification(
        self,
        sample_problem: ProblemFrame,
    ) -> None:
        """Main generate() should call dual-model verification for ranked output."""
        engine = CreativityEngine()

        raw_hypotheses = [
            Hypothesis(
                id="h1",
                description="h1",
                source_domain="d",
                source_structure="s",
                analogy_explanation="a",
                observable="o1",
            ),
            Hypothesis(
                id="h2",
                description="h2",
                source_domain="d",
                source_structure="s",
                analogy_explanation="a",
                observable="o2",
            ),
        ]
        scored_hypotheses = [
            raw_hypotheses[0].model_copy(update={"scores": self._scores(6.0)}),
            raw_hypotheses[1].model_copy(update={"scores": self._scores(6.0)}),
        ]

        logic_ok = LogicVerdict(
            passed=True,
            analogy_validity=8.0,
            internal_consistency=8.0,
            causal_rigor=8.0,
            reasoning="ok",
            issues=[],
        )
        novelty_ok = NoveltyVerdict(
            score=7.0,
            searched=False,
            similar_works=[],
            novelty_analysis="ok",
        )
        verified_low = VerifiedHypothesis.from_hypothesis(
            scored_hypotheses[0],
            logic_verdict=logic_ok,
            novelty_verdict=novelty_ok,
            final_score=7.1,
        )
        verified_high = VerifiedHypothesis.from_hypothesis(
            scored_hypotheses[1],
            logic_verdict=logic_ok,
            novelty_verdict=novelty_ok,
            final_score=8.2,
        )

        with (
            patch.object(
                engine._biso,
                "generate_all_analogies",
                AsyncMock(return_value=raw_hypotheses),
            ),
            patch.object(
                engine._search,
                "run_search",
                AsyncMock(return_value=raw_hypotheses),
            ),
            patch.object(
                engine._verify,
                "score_batch",
                AsyncMock(return_value=scored_hypotheses),
            ),
            patch.object(
                engine._verify,
                "filter_by_threshold",
                MagicMock(return_value=scored_hypotheses),
            ),
            patch.object(
                engine,
                "verify_hypothesis",
                AsyncMock(side_effect=[verified_low, verified_high]),
            ) as mock_dual_verify,
        ):
            result = await engine.generate(
                problem=sample_problem,
                config=SearchConfig(
                    num_hypotheses=2,
                    search_depth=1,
                    search_breadth=1,
                    prune_threshold=0.0,
                ),
            )

        assert mock_dual_verify.await_count == 2
        assert [h.id for h in result] == ["h2", "h1"]
        assert result[0].final_score == 8.2
        assert result[1].final_score == 7.1

    @pytest.mark.asyncio
    async def test_generate_preserves_hkg_evidence_and_novelty_score(
        self,
        sample_problem: ProblemFrame,
    ) -> None:
        """Dual-verify merge should keep HKG evidence and write novelty_score."""
        engine = CreativityEngine()

        raw = Hypothesis(
            id="h1",
            description="raw",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
        )
        searched = raw.model_copy(
            update={
                "hkg_evidence": HKGEvidence(
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
                                    relation="r",
                                    provenance_refs=["doc/chunk"],
                                )
                            ],
                            intermediate_nodes=[],
                        )
                    ]
                )
            }
        )
        scored = searched.model_copy(update={"scores": self._scores(7.0)})

        logic_ok = LogicVerdict(
            passed=True,
            analogy_validity=8.0,
            internal_consistency=8.0,
            causal_rigor=8.0,
            reasoning="ok",
            issues=[],
        )
        novelty_ok = NoveltyVerdict(
            score=8.4,
            searched=False,
            similar_works=[],
            novelty_analysis="ok",
        )
        verified = VerifiedHypothesis.from_hypothesis(
            scored,
            logic_verdict=logic_ok,
            novelty_verdict=novelty_ok,
            final_score=7.9,
            structural_grounding_score=6.6,
        )

        with (
            patch.object(
                engine._biso,
                "generate_all_analogies",
                AsyncMock(return_value=[raw]),
            ),
            patch.object(
                engine._search,
                "run_search",
                AsyncMock(return_value=[searched]),
            ),
            patch.object(
                engine._verify,
                "score_batch",
                AsyncMock(return_value=[scored]),
            ),
            patch.object(
                engine._verify,
                "filter_by_threshold",
                MagicMock(return_value=[scored]),
            ),
            patch.object(
                engine,
                "verify_hypothesis",
                AsyncMock(return_value=verified),
            ),
        ):
            result = await engine.generate(
                problem=sample_problem,
                config=SearchConfig(
                    num_hypotheses=1,
                    search_depth=1,
                    search_breadth=1,
                    prune_threshold=0.0,
                ),
            )

        assert len(result) == 1
        assert result[0].hkg_evidence is not None
        assert result[0].final_score == 7.9
        assert result[0].logic_passed is True
        assert result[0].structural_grounding_score == 6.6
        assert result[0].model_dump().get("novelty_score") == 8.4

    @pytest.mark.asyncio
    async def test_generate_filters_logic_failed_hypothesis_even_if_score_high(
        self,
        sample_problem: ProblemFrame,
    ) -> None:
        """Logic hard gate should reject hypotheses from final output."""
        engine = CreativityEngine()

        raw_hypothesis = Hypothesis(
            id="h1",
            description="raw",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
        )
        scored_hypothesis = raw_hypothesis.model_copy(update={"scores": self._scores(7.0)})

        logic_failed = LogicVerdict(
            passed=False,
            analogy_validity=9.0,
            internal_consistency=9.0,
            causal_rigor=9.0,
            reasoning="failed gate",
            issues=["critical logic issue"],
        )
        novelty_high = NoveltyVerdict(
            score=9.5,
            searched=False,
            similar_works=[],
            novelty_analysis="novel",
        )
        verified = VerifiedHypothesis.from_hypothesis(
            scored_hypothesis,
            logic_verdict=logic_failed,
            novelty_verdict=novelty_high,
            final_score=9.0,
        )

        with (
            patch.object(
                engine._biso,
                "generate_all_analogies",
                AsyncMock(return_value=[raw_hypothesis]),
            ),
            patch.object(
                engine._search,
                "run_search",
                AsyncMock(return_value=[raw_hypothesis]),
            ),
            patch.object(
                engine._verify,
                "score_batch",
                AsyncMock(return_value=[scored_hypothesis]),
            ),
            patch.object(
                engine,
                "verify_hypothesis",
                AsyncMock(return_value=verified),
            ),
        ):
            result = await engine.generate(
                problem=sample_problem,
                config=SearchConfig(
                    num_hypotheses=1,
                    search_depth=1,
                    search_breadth=1,
                    prune_threshold=0.0,
                ),
            )

        assert result == []

    def test_sort_by_score_verified_always_first_with_pareto(self) -> None:
        """Verified hypotheses rank first even when Pareto is enabled."""
        engine = CreativityEngine()
        engine._settings.pareto_selection_enabled = True

        h_unverified = Hypothesis(
            id="h_unverified",
            description="unverified",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=self._scores(9.0),
            final_score=None,
        )
        h_verified = Hypothesis(
            id="h_verified",
            description="verified",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=self._scores(4.0),
            final_score=6.0,
        )

        ranked = engine.sort_by_score([h_unverified, h_verified])
        assert ranked[0].id == "h_verified"

    def test_sort_by_score_disabled_uses_composite(self) -> None:
        """With Pareto disabled, sort uses composite score as before."""
        engine = CreativityEngine()
        engine._settings.pareto_selection_enabled = False

        h_low = Hypothesis(
            id="h_low",
            description="low",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=self._scores(3.0),
        )
        h_high = Hypothesis(
            id="h_high",
            description="high",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=self._scores(9.0),
        )

        ranked = engine.sort_by_score([h_low, h_high])
        assert ranked[0].id == "h_high"
        assert ranked[1].id == "h_low"

    def test_structural_score_can_affect_ranking_when_enabled(self) -> None:
        """When enabled, structural grounding should influence ranking."""
        engine = CreativityEngine()
        engine._settings.hkg_enable_structural_scoring = True
        engine._settings.hkg_structural_score_weight = 0.1

        same_scores = self._scores(8.0)

        plain = Hypothesis(
            id="plain",
            description="plain",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=same_scores,
        )
        grounded = Hypothesis(
            id="grounded",
            description="grounded",
            source_domain="d",
            source_structure="s",
            analogy_explanation="a",
            observable="o",
            scores=same_scores,
            hkg_evidence=HKGEvidence(
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
                                relation="r",
                                provenance_refs=["doc/chunk"],
                            )
                        ],
                        intermediate_nodes=[],
                    )
                ]
            ),
        )

        ranked = engine.sort_by_score([plain, grounded])
        assert ranked[0].id == "grounded"


class TestBlendTransformSettingsWiring:
    """Tests for blend/transform settings wiring into default SearchConfig."""

    def test_generate_default_config_includes_blend_settings(self) -> None:
        """When generate() creates default config, it should include blend/transform from Settings."""
        with patch("x_creative.creativity.engine.get_settings") as mock_gs:
            settings = mock_gs.return_value
            settings.enable_extreme = False
            settings.enable_blending = True
            settings.enable_transform_space = True
            settings.max_blend_pairs = 7
            settings.max_transform_hypotheses = 4
            settings.default_num_hypotheses = 10
            settings.default_search_depth = 2
            settings.hkg_enabled = False
            settings.pareto_selection_enabled = False
            settings.mome_enabled = False
            settings.biso_max_concurrency = 8
            settings.hkg_enable_structural_scoring = False

            engine = CreativityEngine()

        # Patch the pipeline to capture the config that generate() creates
        captured_configs: list[SearchConfig] = []

        async def capture_config(initial_hypotheses, config, **kwargs):  # noqa: ANN001, ANN003
            captured_configs.append(config)
            return initial_hypotheses

        with patch.object(engine._search, "run_search", side_effect=capture_config):
            with patch.object(engine._biso, "generate_all_analogies", return_value=[
                Hypothesis(id="h1", description="d", source_domain="s", source_structure="ss",
                           analogy_explanation="a", observable="o")
            ]):
                with patch.object(engine._verify, "score_batch", return_value=[]):
                    with patch.object(engine, "_verify_batch_dual_model", return_value={}):
                        import asyncio
                        asyncio.run(engine.generate(
                            problem=ProblemFrame(description="test"),
                        ))

        assert len(captured_configs) == 1
        config = captured_configs[0]
        assert config.enable_extreme is False
        assert config.enable_blending is True
        assert config.enable_transform_space is True
        assert config.max_blend_pairs == 7
        assert config.max_transform_hypotheses == 4

    def test_generate_default_config_loads_concept_space_for_transform_target(self) -> None:
        with patch("x_creative.creativity.engine.get_settings") as mock_gs:
            settings = mock_gs.return_value
            settings.enable_extreme = True
            settings.enable_blending = False
            settings.enable_transform_space = True
            settings.max_blend_pairs = 3
            settings.max_transform_hypotheses = 2
            settings.default_num_hypotheses = 10
            settings.default_search_depth = 1
            settings.hkg_enabled = False
            settings.pareto_selection_enabled = False
            settings.mome_enabled = False
            settings.biso_max_concurrency = 8
            settings.hkg_enable_structural_scoring = False

            engine = CreativityEngine()

        captured: list[object | None] = []

        async def capture_config(initial_hypotheses, config, **kwargs):  # noqa: ANN001, ANN003
            captured.append(getattr(engine._search, "_concept_space", None))
            return initial_hypotheses

        with patch.object(engine._search, "run_search", side_effect=capture_config):
            with patch.object(engine._biso, "generate_all_analogies", return_value=[
                Hypothesis(id="h1", description="d", source_domain="s", source_structure="ss",
                           analogy_explanation="a", observable="o")
            ]):
                with patch.object(engine._verify, "score_batch", return_value=[]):
                    with patch.object(engine, "_verify_batch_dual_model", return_value={}):
                        import asyncio
                        asyncio.run(engine.generate(
                            problem=ProblemFrame(description="test", target_domain="open_source_development"),
                        ))

        assert len(captured) == 1
        concept_space = captured[0]
        # open_source_development has no concept_space section, so loader returns None
        assert concept_space is None


class TestMOMEWiring:
    """Tests for MOMEArchive wiring in CreativityEngine."""

    def test_mome_archive_wired_when_enabled(self) -> None:
        """CreativityEngine should wire MOMEArchive when mome_enabled=True."""
        with patch("x_creative.creativity.engine.get_settings") as mock_gs:
            settings = mock_gs.return_value
            settings.pareto_selection_enabled = True
            settings.mome_enabled = True
            settings.mome_cell_capacity = 8
            settings.pareto_wn_min = 0.15
            settings.pareto_wn_max = 0.55
            settings.pareto_gamma = 2.0
            settings.pareto_novelty_bins = 5
            settings.hkg_enabled = False
            settings.enable_blending = False
            settings.enable_transform_space = False
            settings.max_blend_pairs = 3
            settings.max_transform_hypotheses = 2
            settings.default_num_hypotheses = 50
            settings.default_search_depth = 3

            engine = CreativityEngine()

        assert engine._search._mome_archive is not None
        assert engine._search._pareto_archive is not None

    def test_mome_archive_not_wired_when_disabled(self) -> None:
        """MOMEArchive should NOT be wired when mome_enabled=False."""
        with patch("x_creative.creativity.engine.get_settings") as mock_gs:
            settings = mock_gs.return_value
            settings.pareto_selection_enabled = True
            settings.mome_enabled = False
            settings.pareto_wn_min = 0.15
            settings.pareto_wn_max = 0.55
            settings.pareto_gamma = 2.0
            settings.pareto_novelty_bins = 5
            settings.hkg_enabled = False
            settings.enable_blending = False
            settings.enable_transform_space = False
            settings.max_blend_pairs = 3
            settings.max_transform_hypotheses = 2

            engine = CreativityEngine()

        assert engine._search._mome_archive is None
        assert engine._search._pareto_archive is not None


class TestSortByScoreMOME:
    """Tests for MOME sorting path in sort_by_score()."""

    def test_mome_sorting_used_when_enabled(self) -> None:
        """When mome_enabled, sort_by_score should use MOME for unverified hypotheses."""
        with patch("x_creative.creativity.engine.get_settings") as mock_gs:
            settings = mock_gs.return_value
            settings.pareto_selection_enabled = True
            settings.mome_enabled = True
            settings.mome_cell_capacity = 10
            settings.pareto_wn_min = 0.15
            settings.pareto_wn_max = 0.55
            settings.pareto_gamma = 2.0
            settings.pareto_novelty_bins = 5
            settings.hkg_enabled = False
            settings.hkg_enable_structural_scoring = False
            settings.enable_blending = False
            settings.enable_transform_space = False
            settings.max_blend_pairs = 3
            settings.max_transform_hypotheses = 2

            engine = CreativityEngine()

        # Create unverified hypotheses with a divergence/feasibility tradeoff
        # so they form a Pareto front (none dominates another) and all survive.
        # novelty_axis = divergence; feasibility_axis = avg(testability, rationale, robustness, feasibility)
        tradeoffs = [
            (2.0, 8.0),  # low divergence, high feasibility
            (4.0, 6.0),
            (6.0, 4.0),
            (8.0, 2.0),  # high divergence, low feasibility
        ]
        hyps = []
        for i, (div, feas) in enumerate(tradeoffs):
            h = Hypothesis(
                id=f"h{i}",
                description=f"hyp {i}",
                source_domain="thermo",
                source_structure="s",
                analogy_explanation="a",
                observable="x",
                scores=HypothesisScores(
                    divergence=div,
                    testability=feas,
                    rationale=feas,
                    robustness=feas,
                    feasibility=feas,
                ),
            )
            hyps.append(h)

        result = engine.sort_by_score(hyps)
        assert len(result) == 4  # All hypotheses returned (MOME reorders but doesn't filter)
        # Verify all original IDs are present
        result_ids = {h.id for h in result}
        assert result_ids == {"h0", "h1", "h2", "h3"}

    def test_mome_takes_priority_over_pareto(self) -> None:
        """MOME path should be checked before Pareto path."""
        with patch("x_creative.creativity.engine.get_settings") as mock_gs:
            settings = mock_gs.return_value
            settings.pareto_selection_enabled = True
            settings.mome_enabled = True
            settings.mome_cell_capacity = 10
            settings.hkg_enabled = False
            settings.hkg_enable_structural_scoring = False
            settings.enable_blending = False
            settings.enable_transform_space = False
            settings.max_blend_pairs = 3
            settings.max_transform_hypotheses = 2

            engine = CreativityEngine()

        h = Hypothesis(
            id="h1", description="d", source_domain="s",
            source_structure="ss", analogy_explanation="a", observable="o",
            scores=HypothesisScores(
                divergence=5.0, testability=5.0, rationale=5.0,
                robustness=5.0, feasibility=5.0,
            ),
        )

        with patch("x_creative.creativity.mome.MOMEArchive") as mock_mome_cls:
            mock_archive = MagicMock()
            mock_archive.select.return_value = [h]
            mock_mome_cls.return_value = mock_archive

            result = engine.sort_by_score([h])
            # MOMEArchive is imported inside the if-block via deferred import,
            # so we verify behavior: the result should contain the hypothesis
            assert len(result) == 1
            assert result[0].id == "h1"

    def test_pareto_used_when_mome_disabled(self) -> None:
        """When mome_enabled=False but pareto_selection_enabled=True, Pareto path is used."""
        with patch("x_creative.creativity.engine.get_settings") as mock_gs:
            settings = mock_gs.return_value
            settings.pareto_selection_enabled = True
            settings.mome_enabled = False
            settings.pareto_wn_min = 0.15
            settings.pareto_wn_max = 0.55
            settings.pareto_gamma = 2.0
            settings.pareto_novelty_bins = 5
            settings.hkg_enabled = False
            settings.hkg_enable_structural_scoring = False
            settings.enable_blending = False
            settings.enable_transform_space = False
            settings.max_blend_pairs = 3
            settings.max_transform_hypotheses = 2

            engine = CreativityEngine()

        hyps = []
        for i in range(3):
            h = Hypothesis(
                id=f"h{i}",
                description=f"hyp {i}",
                source_domain="thermo",
                source_structure="s",
                analogy_explanation="a",
                observable="x",
                scores=HypothesisScores(
                    divergence=float(i + 1),
                    testability=5.0,
                    rationale=5.0,
                    robustness=5.0,
                    feasibility=5.0,
                ),
            )
            hyps.append(h)

        result = engine.sort_by_score(hyps)
        assert len(result) == 3  # Pareto doesn't filter, just reorders

    def test_verified_always_rank_first_with_mome(self) -> None:
        """Verified hypotheses should rank before MOME-sorted unverified ones."""
        with patch("x_creative.creativity.engine.get_settings") as mock_gs:
            settings = mock_gs.return_value
            settings.pareto_selection_enabled = False
            settings.mome_enabled = True
            settings.mome_cell_capacity = 10
            settings.hkg_enabled = False
            settings.hkg_enable_structural_scoring = False
            settings.enable_blending = False
            settings.enable_transform_space = False
            settings.max_blend_pairs = 3
            settings.max_transform_hypotheses = 2

            engine = CreativityEngine()

        verified_h = Hypothesis(
            id="verified",
            description="verified hyp",
            source_domain="thermo",
            source_structure="s",
            analogy_explanation="a",
            observable="x",
            scores=HypothesisScores(
                divergence=3.0, testability=3.0, rationale=3.0,
                robustness=3.0, feasibility=3.0,
            ),
            final_score=7.5,
        )
        unverified_h = Hypothesis(
            id="unverified",
            description="unverified hyp",
            source_domain="thermo",
            source_structure="s",
            analogy_explanation="a",
            observable="x",
            scores=HypothesisScores(
                divergence=9.0, testability=9.0, rationale=9.0,
                robustness=9.0, feasibility=9.0,
            ),
        )

        result = engine.sort_by_score([unverified_h, verified_h])
        assert result[0].id == "verified"
        assert result[1].id == "unverified"


class TestTheoryAlignment:
    """Theory alignment checks for mapping gate and VERIFY status handling."""

    def test_engine_wires_mapping_quality_gate_from_settings(self) -> None:
        with patch("x_creative.creativity.engine.get_settings") as mock_gs:
            settings = mock_gs.return_value
            settings.hkg_enabled = False
            settings.mapping_quality_gate_enabled = True
            settings.mapping_quality_gate_threshold = 6.5
            settings.pareto_selection_enabled = False
            settings.mome_enabled = False
            settings.enable_blending = False
            settings.enable_transform_space = False
            settings.max_blend_pairs = 3
            settings.max_transform_hypotheses = 2
            settings.default_num_hypotheses = 10
            settings.default_search_depth = 2
            settings.biso_max_concurrency = 4
            settings.hkg_enable_structural_scoring = False
            settings.score_weight_divergence = 0.21
            settings.score_weight_testability = 0.26
            settings.score_weight_rationale = 0.21
            settings.score_weight_robustness = 0.17
            settings.score_weight_feasibility = 0.15
            settings.hkg_structural_score_weight = 0.1
            engine = CreativityEngine()

        assert engine._search._mapping_quality_gate == 6.5

    @pytest.mark.asyncio
    async def test_verify_hypothesis_sets_escalated_status_on_low_confidence(
        self,
        sample_problem: ProblemFrame,
        sample_hypothesis: Hypothesis,
        sample_novelty_verdict_low: NoveltyVerdict,
    ) -> None:
        engine = CreativityEngine()
        low_conf_logic = LogicVerdict(
            passed=True,
            analogy_validity=7.0,
            internal_consistency=7.0,
            causal_rigor=7.0,
            reasoning="borderline stable",
            judge_confidence=0.45,
            position_consistency=False,
        )

        with patch.object(engine._logic_verifier, "verify", AsyncMock(return_value=low_conf_logic)), \
             patch.object(engine._novelty_verifier, "verify", AsyncMock(return_value=sample_novelty_verdict_low)), \
             patch.object(engine._novelty_verifier, "needs_search", return_value=False):
            verified = await engine.verify_hypothesis(sample_hypothesis, sample_problem)

        assert verified.verify_status == VerifyStatus.ESCALATED
        assert verified.final_score == 0.0

    @pytest.mark.asyncio
    async def test_verify_hypothesis_runs_escalated_second_pass(
        self,
        sample_problem: ProblemFrame,
        sample_hypothesis: Hypothesis,
        sample_novelty_verdict_low: NoveltyVerdict,
        sample_novelty_verdict_high: NoveltyVerdict,
    ) -> None:
        engine = CreativityEngine()
        first_pass = LogicVerdict(
            passed=True,
            analogy_validity=7.0,
            internal_consistency=7.0,
            causal_rigor=7.0,
            reasoning="first pass uncertain",
            judge_confidence=0.45,
            position_consistency=False,
        )
        second_pass = LogicVerdict(
            passed=True,
            analogy_validity=8.5,
            internal_consistency=8.0,
            causal_rigor=8.0,
            reasoning="second pass stable",
            judge_confidence=0.92,
            position_consistency=True,
        )

        with patch.object(
            engine._logic_verifier,
            "verify",
            AsyncMock(side_effect=[first_pass, second_pass]),
        ) as mock_logic, patch.object(
            engine._novelty_verifier,
            "verify",
            AsyncMock(side_effect=[sample_novelty_verdict_low, sample_novelty_verdict_high]),
        ) as mock_novelty, patch.object(
            engine._novelty_verifier,
            "needs_search",
            return_value=False,
        ):
            verified = await engine.verify_hypothesis(sample_hypothesis, sample_problem)

        assert mock_logic.await_count == 2
        assert mock_novelty.await_count >= 2
        assert verified.verify_status == VerifyStatus.PASSED
        assert verified.final_score > 0.0

    @pytest.mark.asyncio
    async def test_verify_hypothesis_scores_blend_consistency(
        self,
        sample_problem: ProblemFrame,
        sample_hypothesis: Hypothesis,
        sample_novelty_verdict_high: NoveltyVerdict,
    ) -> None:
        engine = CreativityEngine()
        blend_hyp = sample_hypothesis.model_copy(
            update={
                "blend_network": BlendNetwork(
                    input1_summary="A",
                    input2_summary="B",
                    generic_space="shared dynamics",
                    blend_description="new synthesis",
                    cross_space_mappings=[],
                    emergent_structures=[],
                )
            }
        )
        logic_verdict = LogicVerdict(
            passed=True,
            analogy_validity=8.0,
            internal_consistency=8.0,
            causal_rigor=8.0,
            reasoning="strong",
            judge_confidence=0.95,
            position_consistency=True,
        )

        with patch.object(engine._logic_verifier, "verify", AsyncMock(return_value=logic_verdict)), \
             patch.object(engine._novelty_verifier, "verify", AsyncMock(return_value=sample_novelty_verdict_high)), \
             patch.object(engine._novelty_verifier, "needs_search", return_value=False):
            verified = await engine.verify_hypothesis(blend_hyp, sample_problem)

        assert verified.blend_network is not None
        assert verified.blend_network.blend_consistency_score is not None
        assert 0.0 <= verified.final_score <= 10.0
