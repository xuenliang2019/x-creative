"""Integration tests for the full creativity pipeline.

Note: These tests require valid API keys to run against real LLMs.
They are marked with @pytest.mark.integration and skipped by default.
Run with: pytest -m integration --run-integration
"""

import os

import pytest

from x_creative.core.types import ProblemFrame, SearchConfig


# Skip integration tests by default
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete creativity pipeline."""

    @pytest.fixture
    def sample_problem(self) -> ProblemFrame:
        """Create a sample research problem."""
        return ProblemFrame(
            description="设计一个能实现病毒式传播的开源命令行工具",
        )

    @pytest.mark.asyncio
    async def test_quick_generation(self, sample_problem: ProblemFrame) -> None:
        """Test quick generation with minimal settings."""
        from x_creative.creativity.engine import CreativityEngine

        engine = CreativityEngine()

        try:
            result = await engine.generate_quick(
                problem=sample_problem,
                num_hypotheses=3,
            )

            # Should generate some hypotheses
            assert len(result) > 0

            # Each should have scores
            for hyp in result:
                assert hyp.scores is not None
                assert hyp.composite_score() > 0

        finally:
            await engine.close()

    @pytest.mark.asyncio
    async def test_biso_single_domain(self, sample_problem: ProblemFrame) -> None:
        """Test BISO module with a single domain."""
        from x_creative.core.domain_loader import DomainLibrary
        from x_creative.creativity.biso import BISOModule

        library = DomainLibrary.from_target_domain("open_source_development")
        domain = library.get("queueing_theory")
        assert domain is not None

        biso = BISOModule()

        try:
            analogies = await biso.generate_analogies(
                domain=domain,
                problem=sample_problem,
                num_analogies=2,
            )

            # Should generate analogies
            assert len(analogies) > 0

            # Each should have required fields
            for hyp in analogies:
                assert hyp.source_domain == "queueing_theory"
                assert hyp.observable

        finally:
            await biso._router.close()

    @pytest.mark.asyncio
    async def test_verify_scoring(self, sample_problem: ProblemFrame) -> None:
        """Test the verify module scoring."""
        from x_creative.core.types import Hypothesis
        from x_creative.creativity.verify import VerifyModule

        verify = VerifyModule()

        test_hypothesis = Hypothesis(
            id="test_hyp",
            description="基于订单流不平衡的短期反转因子",
            source_domain="queueing_theory",
            source_structure="queue_dynamics",
            analogy_explanation="订单簿中买卖单的到达率差异类似于排队系统中的服务压力",
            observable="bid_volume_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)",
        )

        try:
            scored = await verify.score_hypothesis(test_hypothesis)

            assert scored.scores is not None
            assert 0 <= scored.scores.divergence <= 10
            assert 0 <= scored.scores.testability <= 10
            assert 0 <= scored.scores.rationale <= 10
            assert 0 <= scored.scores.robustness <= 10

        finally:
            await verify._router.close()


@pytest.mark.integration
class TestDomainLibrary:
    """Integration tests for domain library."""

    def test_load_all_domains(self) -> None:
        """Test that all domains load correctly."""
        from x_creative.core.domain_loader import DomainLibrary

        library = DomainLibrary.from_target_domain("open_source_development")

        # Should have many domains
        assert len(library) >= 10

        # Check specific domains exist
        assert library.get("thermodynamics") is not None
        assert library.get("queueing_theory") is not None
        assert library.get("ecology") is not None

    def test_domain_structures_valid(self) -> None:
        """Test that domain structures are valid."""
        from x_creative.core.domain_loader import DomainLibrary

        library = DomainLibrary.from_target_domain("open_source_development")

        for domain in library:
            # Each domain should have structures
            assert len(domain.structures) > 0

            # Each structure should have required fields
            for structure in domain.structures:
                assert structure.id
                assert structure.name
                assert structure.description
                assert len(structure.key_variables) > 0


@pytest.mark.integration
class TestModelRouting:
    """Integration tests for model routing."""

    @pytest.mark.asyncio
    async def test_model_routing_basic(self) -> None:
        """Test basic model routing."""
        from x_creative.llm.router import ModelRouter

        router = ModelRouter()

        try:
            result = await router.complete(
                task="hypothesis_scoring",
                messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            )

            assert result.content
            assert result.prompt_tokens > 0

        finally:
            await router.close()
