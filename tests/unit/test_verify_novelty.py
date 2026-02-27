"""Tests for NoveltyVerifier."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x_creative.core.types import Hypothesis, NoveltyVerdict, ProblemFrame, SimilarWork
from x_creative.llm.client import CompletionResult
from x_creative.verify.novelty import NoveltyVerifier


@pytest.fixture
def sample_hypothesis() -> Hypothesis:
    """Create a sample hypothesis for testing."""
    return Hypothesis(
        id="hyp-001",
        description="Use epidemic spreading models to design viral adoption",
        source_domain="thermodynamics",
        source_structure="entropy-disorder",
        analogy_explanation="Information diffusion maps to thermodynamic entropy",
        observable="adoption_rate / expected_adoption",
    )


@pytest.fixture
def sample_problem_frame() -> ProblemFrame:
    """Create a sample problem frame for testing."""
    return ProblemFrame(
        description="How to design a viral open source tool?",
        target_domain="open_source_development",
        constraints=["Must be CLI-based", "No vendor lock-in"],
    )


@pytest.fixture
def mock_router() -> MagicMock:
    """Create a mock ModelRouter."""
    return MagicMock()


class TestNoveltyVerifier:
    """Tests for NoveltyVerifier."""

    def test_verifier_creation(self) -> None:
        """Test creating a NoveltyVerifier."""
        verifier = NoveltyVerifier()
        assert verifier is not None

    def test_verifier_with_custom_router(self, mock_router: MagicMock) -> None:
        """Test creating a verifier with custom router."""
        verifier = NoveltyVerifier(router=mock_router)
        assert verifier._router is mock_router

    @pytest.mark.asyncio
    async def test_verify_returns_novelty_verdict(
        self,
        sample_hypothesis: Hypothesis,
        sample_problem_frame: ProblemFrame,
    ) -> None:
        """Test that verify returns a NoveltyVerdict."""
        verifier = NoveltyVerifier()

        # Mock a successful response with high novelty
        mock_response = CompletionResult(
            content=json.dumps({
                "novelty_score": 8.5,
                "known_similar": [],
                "reasoning": "This approach is novel and not commonly seen.",
            }),
            model="google/gemini-3-pro",
            prompt_tokens=500,
            completion_tokens=200,
        )

        with patch.object(verifier, "_router") as mock_router:
            mock_router.complete = AsyncMock(return_value=mock_response)

            result = await verifier.verify(sample_hypothesis, sample_problem_frame)

            assert isinstance(result, NoveltyVerdict)
            assert result.score >= 0.0
            assert result.score <= 10.0
            assert len(result.novelty_analysis) > 0

    @pytest.mark.asyncio
    async def test_high_score_needs_search(
        self,
        sample_hypothesis: Hypothesis,
        sample_problem_frame: ProblemFrame,
    ) -> None:
        """Test that score >= 6.0 sets searched=False (needs web search later)."""
        verifier = NoveltyVerifier()

        # Score >= 6.0 means idea seems novel, needs web search verification
        mock_response = CompletionResult(
            content=json.dumps({
                "novelty_score": 8.0,
                "known_similar": [],
                "reasoning": "Novel approach not seen in training data.",
            }),
            model="google/gemini-3-pro",
            prompt_tokens=500,
            completion_tokens=200,
        )

        with patch.object(verifier, "_router") as mock_router:
            mock_router.complete = AsyncMock(return_value=mock_response)

            result = await verifier.verify(sample_hypothesis, sample_problem_frame)

            # High score means novel - needs search verification
            assert result.score == 8.0
            # searched=False because web search hasn't happened yet
            assert result.searched is False

    @pytest.mark.asyncio
    async def test_low_score_no_search(
        self,
        sample_hypothesis: Hypothesis,
        sample_problem_frame: ProblemFrame,
    ) -> None:
        """Test that score < 6.0 means already known, no search needed."""
        verifier = NoveltyVerifier()

        # Score < 6.0 means idea is already known from training data
        mock_response = CompletionResult(
            content=json.dumps({
                "novelty_score": 4.0,
                "known_similar": [
                    {
                        "title": "Shannon Entropy for Adoption Prediction",
                        "source": "arxiv",
                        "similarity_explanation": "Same entropy-based approach",
                    }
                ],
                "reasoning": "This approach has been explored in prior work.",
            }),
            model="google/gemini-3-pro",
            prompt_tokens=500,
            completion_tokens=200,
        )

        with patch.object(verifier, "_router") as mock_router:
            mock_router.complete = AsyncMock(return_value=mock_response)

            result = await verifier.verify(sample_hypothesis, sample_problem_frame)

            # Low score means not novel - skip search
            assert result.score == 4.0
            # searched=False, but this is because search isn't needed
            assert result.searched is False
            assert len(result.similar_works) > 0

    @pytest.mark.asyncio
    async def test_known_similar_parsed(
        self,
        sample_hypothesis: Hypothesis,
        sample_problem_frame: ProblemFrame,
    ) -> None:
        """Test that SimilarWork objects are created from response."""
        verifier = NoveltyVerifier()

        mock_response = CompletionResult(
            content=json.dumps({
                "novelty_score": 5.0,
                "known_similar": [
                    {
                        "title": "Entropy-Based Adoption Model",
                        "source": "ssrn",
                        "similarity_explanation": "Uses entropy for adoption prediction",
                    },
                    {
                        "title": "Information Theory in Software Ecosystems",
                        "source": "blog",
                        "similarity_explanation": "Discusses entropy applications",
                    },
                ],
                "reasoning": "Similar approaches exist in literature.",
            }),
            model="google/gemini-3-pro",
            prompt_tokens=500,
            completion_tokens=200,
        )

        with patch.object(verifier, "_router") as mock_router:
            mock_router.complete = AsyncMock(return_value=mock_response)

            result = await verifier.verify(sample_hypothesis, sample_problem_frame)

            assert len(result.similar_works) == 2
            assert all(isinstance(w, SimilarWork) for w in result.similar_works)
            assert result.similar_works[0].title == "Entropy-Based Adoption Model"
            assert result.similar_works[0].source == "ssrn"
            assert result.similar_works[1].title == "Information Theory in Software Ecosystems"

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(
        self,
        sample_hypothesis: Hypothesis,
        sample_problem_frame: ProblemFrame,
    ) -> None:
        """Test that errors return a failed verdict."""
        verifier = NoveltyVerifier()

        with patch.object(verifier, "_router") as mock_router:
            mock_router.complete = AsyncMock(side_effect=Exception("API Error"))

            result = await verifier.verify(sample_hypothesis, sample_problem_frame)

            assert result.score == 0.0
            assert result.searched is False
            assert "error" in result.novelty_analysis.lower() or "Error" in result.novelty_analysis

    @pytest.mark.asyncio
    async def test_handles_invalid_json_gracefully(
        self,
        sample_hypothesis: Hypothesis,
        sample_problem_frame: ProblemFrame,
    ) -> None:
        """Test that invalid JSON response returns a failed verdict."""
        verifier = NoveltyVerifier()

        mock_response = CompletionResult(
            content="This is not valid JSON",
            model="google/gemini-3-pro",
            prompt_tokens=500,
            completion_tokens=200,
        )

        with patch.object(verifier, "_router") as mock_router:
            mock_router.complete = AsyncMock(return_value=mock_response)

            result = await verifier.verify(sample_hypothesis, sample_problem_frame)

            assert result.score == 0.0
            assert "parse" in result.novelty_analysis.lower() or "json" in result.novelty_analysis.lower()

    @pytest.mark.asyncio
    async def test_prompt_includes_context(
        self,
        sample_hypothesis: Hypothesis,
        sample_problem_frame: ProblemFrame,
    ) -> None:
        """Test that the prompt includes hypothesis and problem context."""
        verifier = NoveltyVerifier()

        mock_response = CompletionResult(
            content=json.dumps({
                "novelty_score": 7.0,
                "known_similar": [],
                "reasoning": "Novel approach.",
            }),
            model="google/gemini-3-pro",
            prompt_tokens=500,
            completion_tokens=200,
        )

        with patch.object(verifier, "_router") as mock_router:
            mock_router.complete = AsyncMock(return_value=mock_response)

            await verifier.verify(sample_hypothesis, sample_problem_frame)

            # Check the call was made with correct content
            call_args = mock_router.complete.call_args
            messages = call_args.kwargs.get("messages") or call_args.args[1]

            # Find the user message content
            user_message = next(
                (m for m in messages if m.get("role") == "user"),
                None,
            )
            assert user_message is not None
            content = user_message["content"]

            # Verify hypothesis details are in the prompt
            assert sample_hypothesis.description in content
            assert sample_hypothesis.source_domain in content
            assert sample_problem_frame.target_domain in content

    @pytest.mark.asyncio
    async def test_verify_uses_novelty_verification_task(
        self,
        sample_hypothesis: Hypothesis,
        sample_problem_frame: ProblemFrame,
    ) -> None:
        """Test that verify uses the novelty_verification task."""
        verifier = NoveltyVerifier()

        mock_response = CompletionResult(
            content=json.dumps({
                "novelty_score": 7.0,
                "known_similar": [],
                "reasoning": "Novel approach.",
            }),
            model="google/gemini-3-pro",
            prompt_tokens=500,
            completion_tokens=200,
        )

        with patch.object(verifier, "_router") as mock_router:
            mock_router.complete = AsyncMock(return_value=mock_response)

            await verifier.verify(sample_hypothesis, sample_problem_frame)

            # Check that the correct task was used
            call_args = mock_router.complete.call_args
            task = call_args.kwargs.get("task") or call_args.args[0]
            assert task == "novelty_verification"

    @pytest.mark.asyncio
    async def test_verify_with_custom_threshold(
        self,
        sample_hypothesis: Hypothesis,
        sample_problem_frame: ProblemFrame,
    ) -> None:
        """Test verify with a custom search threshold."""
        verifier = NoveltyVerifier(search_threshold=8.0)

        mock_response = CompletionResult(
            content=json.dumps({
                "novelty_score": 7.5,
                "known_similar": [],
                "reasoning": "Fairly novel approach.",
            }),
            model="google/gemini-3-pro",
            prompt_tokens=500,
            completion_tokens=200,
        )

        with patch.object(verifier, "_router") as mock_router:
            mock_router.complete = AsyncMock(return_value=mock_response)

            result = await verifier.verify(sample_hypothesis, sample_problem_frame)

            # Score 7.5 with threshold 8.0 means below threshold
            assert result.score == 7.5
            # Does not need search since below custom threshold
            assert verifier.needs_search(result.score) is False

    def test_needs_search_threshold(self) -> None:
        """Test the needs_search method with default threshold."""
        verifier = NoveltyVerifier()

        # Default threshold is 6.0
        assert verifier.needs_search(6.0) is True
        assert verifier.needs_search(7.0) is True
        assert verifier.needs_search(5.9) is False
        assert verifier.needs_search(0.0) is False


class TestNoveltyVerdictIntegration:
    """Integration tests for NoveltyVerdict with NoveltyVerifier."""

    def test_novelty_verdict_fields_match_verifier_output(self) -> None:
        """Ensure NoveltyVerdict fields match what verifier produces."""
        verdict = NoveltyVerdict(
            score=8.0,
            searched=False,
            similar_works=[],
            novelty_analysis="This approach is novel.",
        )

        assert verdict.score == 8.0
        assert verdict.searched is False
        assert verdict.similar_works == []
        assert verdict.novelty_analysis == "This approach is novel."

    def test_novelty_verdict_with_similar_works(self) -> None:
        """Test NoveltyVerdict with similar works populated."""
        similar = SimilarWork(
            title="Entropy in Software Ecosystems",
            url="",  # LLM doesn't provide URLs, web search will
            source="arxiv",
            similarity=0.7,
            difference_summary="Uses different entropy measure",
        )
        verdict = NoveltyVerdict(
            score=5.0,
            searched=False,
            similar_works=[similar],
            novelty_analysis="Similar work exists.",
        )

        assert len(verdict.similar_works) == 1
        assert verdict.similar_works[0].title == "Entropy in Software Ecosystems"
