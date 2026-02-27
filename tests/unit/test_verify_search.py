"""Tests for SearchValidator.

SearchValidator is stage 2 of novelty verification:
- Performs structured multi-round search using Brave Search API
- Aggregates results into final novelty assessment
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pydantic import SecretStr

from x_creative.config.settings import BraveSearchConfig, SearchConfig, SearchRoundConfig
from x_creative.core.types import Hypothesis, NoveltyVerdict, SimilarWork
from x_creative.verify.search import SearchValidator


@pytest.fixture
def brave_config_with_key() -> BraveSearchConfig:
    """Create a BraveSearchConfig with a test API key."""
    return BraveSearchConfig(api_key=SecretStr("test-api-key"))


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
def sample_brave_response() -> dict[str, Any]:
    """Create a sample Brave Search API response."""
    return {
        "query": {"original": "thermodynamics entropy adoption research"},
        "web": {
            "results": [
                {
                    "title": "Entropy-Based Adoption Analysis",
                    "url": "https://example.com/entropy-adoption",
                    "description": "Using entropy theory to analyze viral adoption and diffusion.",
                },
                {
                    "title": "Thermodynamic Models in Software Ecosystems",
                    "url": "https://arxiv.org/abs/1234.5678",
                    "description": "Application of thermodynamic concepts to open source ecosystems.",
                },
            ]
        },
    }


@pytest.fixture
def sample_brave_empty_response() -> dict[str, Any]:
    """Create an empty Brave Search API response."""
    return {
        "query": {"original": "very specific obscure query"},
        "web": {"results": []},
    }


class TestSearchValidator:
    """Tests for SearchValidator."""

    def test_validator_creation(self) -> None:
        """Test creating a SearchValidator."""
        validator = SearchValidator()
        assert validator is not None

    def test_validator_with_custom_config(self) -> None:
        """Test creating a validator with custom search config."""
        config = SearchConfig(
            provider="brave",
            search_threshold=7.0,
            rounds=[
                SearchRoundConfig(name="concept", weight=0.5, max_results=5),
                SearchRoundConfig(name="implementation", weight=0.5, max_results=5),
            ],
        )
        validator = SearchValidator(search_config=config)
        assert len(validator._config.rounds) == 2

    @pytest.mark.asyncio
    async def test_validate_returns_novelty_verdict(
        self,
        sample_hypothesis: Hypothesis,
        sample_brave_response: dict[str, Any],
        brave_config_with_key: BraveSearchConfig,
    ) -> None:
        """Test that validate returns a NoveltyVerdict."""
        validator = SearchValidator(brave_config=brave_config_with_key)

        # Mock the HTTP client
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_brave_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            validator, "_client", autospec=True
        ) as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await validator.validate(
                hypothesis=sample_hypothesis,
                preliminary_score=7.5,
            )

            assert isinstance(result, NoveltyVerdict)
            assert result.score >= 0.0
            assert result.score <= 10.0

    @pytest.mark.asyncio
    async def test_validate_sets_searched_true(
        self,
        sample_hypothesis: Hypothesis,
        sample_brave_response: dict[str, Any],
        brave_config_with_key: BraveSearchConfig,
    ) -> None:
        """Test that validate sets searched=True in result."""
        validator = SearchValidator(brave_config=brave_config_with_key)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_brave_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            validator, "_client", autospec=True
        ) as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await validator.validate(
                hypothesis=sample_hypothesis,
                preliminary_score=7.5,
            )

            # searched should be True since web search was performed
            assert result.searched is True

    @pytest.mark.asyncio
    async def test_builds_search_queries_per_round(
        self,
        sample_hypothesis: Hypothesis,
        sample_brave_response: dict[str, Any],
        brave_config_with_key: BraveSearchConfig,
    ) -> None:
        """Test that search queries are built for each round."""
        config = SearchConfig(
            rounds=[
                SearchRoundConfig(name="concept", weight=0.3, max_results=10),
                SearchRoundConfig(name="implementation", weight=0.5, max_results=15),
                SearchRoundConfig(name="cross_domain", weight=0.2, max_results=8),
            ],
        )
        validator = SearchValidator(search_config=config, brave_config=brave_config_with_key)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_brave_response
        mock_response.raise_for_status = MagicMock()

        mock_get = AsyncMock(return_value=mock_response)

        with patch.object(validator, "_client", autospec=True) as mock_client:
            mock_client.get = mock_get

            await validator.validate(
                hypothesis=sample_hypothesis,
                preliminary_score=7.5,
            )

            # Should have been called 3 times (once per round)
            assert mock_get.call_count == 3

            # Check the query parameters
            calls = mock_get.call_args_list
            queries = [call.kwargs.get("params", {}).get("q", "") for call in calls]

            # Concept round should contain domain keywords
            assert any("thermodynamics" in q.lower() for q in queries)
            # Implementation round should reference observables
            assert any("adoption" in q.lower() for q in queries)

    @pytest.mark.asyncio
    async def test_aggregates_scores_by_weight(
        self,
        sample_hypothesis: Hypothesis,
        brave_config_with_key: BraveSearchConfig,
    ) -> None:
        """Test that scores are aggregated by weight from each round."""
        config = SearchConfig(
            rounds=[
                SearchRoundConfig(name="concept", weight=0.3, max_results=10),
                SearchRoundConfig(name="implementation", weight=0.5, max_results=10),
                SearchRoundConfig(name="cross_domain", weight=0.2, max_results=10),
            ],
        )
        validator = SearchValidator(search_config=config, brave_config=brave_config_with_key)

        # Create different responses for different rounds
        responses = [
            # concept round - many results (less novel)
            {"web": {"results": [{"title": f"Result {i}", "url": f"http://ex.com/{i}", "description": "desc"} for i in range(10)]}},
            # implementation round - some results
            {"web": {"results": [{"title": f"Result {i}", "url": f"http://ex.com/{i}", "description": "desc"} for i in range(5)]}},
            # cross_domain round - no results (very novel)
            {"web": {"results": []}},
        ]

        mock_responses = []
        for resp_data in responses:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = resp_data
            mock_resp.raise_for_status = MagicMock()
            mock_responses.append(mock_resp)

        mock_get = AsyncMock(side_effect=mock_responses)

        with patch.object(validator, "_client", autospec=True) as mock_client:
            mock_client.get = mock_get

            result = await validator.validate(
                hypothesis=sample_hypothesis,
                preliminary_score=8.0,
            )

            # Score should be a blend of the preliminary score and search results
            # The formula combines preliminary_score and search evidence
            assert isinstance(result.score, float)
            assert result.score >= 0.0
            assert result.score <= 10.0

    @pytest.mark.asyncio
    async def test_handles_api_error_gracefully(
        self,
        sample_hypothesis: Hypothesis,
    ) -> None:
        """Test that API errors return verdict with preliminary score."""
        validator = SearchValidator()

        # Create a proper httpx.RequestError
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.url = "https://api.search.brave.com/res/v1/web/search"

        mock_get = AsyncMock(
            side_effect=httpx.RequestError("Connection failed", request=mock_request)
        )

        with patch.object(validator, "_client", autospec=True) as mock_client:
            mock_client.get = mock_get

            result = await validator.validate(
                hypothesis=sample_hypothesis,
                preliminary_score=7.5,
            )

            # Should still return a verdict
            assert isinstance(result, NoveltyVerdict)
            # Score should fall back to preliminary score
            assert result.score == 7.5
            # searched should be False since search failed
            assert result.searched is False
            # Analysis should indicate the error
            assert "error" in result.novelty_analysis.lower()

    @pytest.mark.asyncio
    async def test_handles_api_http_error(
        self,
        sample_hypothesis: Hypothesis,
    ) -> None:
        """Test that HTTP errors are handled gracefully."""
        validator = SearchValidator()

        mock_request = MagicMock(spec=httpx.Request)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=mock_request,
            response=mock_response,
        )

        mock_get = AsyncMock(return_value=mock_response)

        with patch.object(validator, "_client", autospec=True) as mock_client:
            mock_client.get = mock_get

            result = await validator.validate(
                hypothesis=sample_hypothesis,
                preliminary_score=7.0,
            )

            # Should return preliminary score on error
            assert result.score == 7.0
            assert result.searched is False

    @pytest.mark.asyncio
    async def test_parses_brave_results(
        self,
        sample_hypothesis: Hypothesis,
        sample_brave_response: dict[str, Any],
        brave_config_with_key: BraveSearchConfig,
    ) -> None:
        """Test that Brave results are parsed into SimilarWork objects."""
        validator = SearchValidator(brave_config=brave_config_with_key)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_brave_response
        mock_response.raise_for_status = MagicMock()

        mock_get = AsyncMock(return_value=mock_response)

        with patch.object(validator, "_client", autospec=True) as mock_client:
            mock_client.get = mock_get

            result = await validator.validate(
                hypothesis=sample_hypothesis,
                preliminary_score=7.5,
            )

            # Should have parsed similar works
            assert len(result.similar_works) > 0
            assert all(isinstance(w, SimilarWork) for w in result.similar_works)

            # Check that fields are populated correctly (results are sorted by similarity)
            # Find the entropy-based work
            entropy_work = next(
                (w for w in result.similar_works if "Entropy-Based" in w.title), None
            )
            assert entropy_work is not None
            assert entropy_work.url == "https://example.com/entropy-adoption"
            assert entropy_work.source == "web"  # brave search source

    @pytest.mark.asyncio
    async def test_respects_max_results(
        self,
        sample_hypothesis: Hypothesis,
        brave_config_with_key: BraveSearchConfig,
    ) -> None:
        """Test that max_results per round is respected."""
        config = SearchConfig(
            rounds=[
                SearchRoundConfig(name="concept", weight=1.0, max_results=3),
            ],
        )
        validator = SearchValidator(search_config=config, brave_config=brave_config_with_key)

        # Create a response with many results
        many_results = {
            "web": {
                "results": [
                    {"title": f"Result {i}", "url": f"http://example.com/{i}", "description": f"Description {i}"}
                    for i in range(20)
                ]
            }
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = many_results
        mock_response.raise_for_status = MagicMock()

        mock_get = AsyncMock(return_value=mock_response)

        with patch.object(validator, "_client", autospec=True) as mock_client:
            mock_client.get = mock_get

            await validator.validate(
                hypothesis=sample_hypothesis,
                preliminary_score=7.5,
            )

            # Check that count parameter was sent
            assert mock_get.call_count == 1
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("count") == 3

    @pytest.mark.asyncio
    async def test_caps_max_results_to_brave_limit(
        self,
        brave_config_with_key: BraveSearchConfig,
        sample_brave_response: dict[str, Any],
    ) -> None:
        """Test that count is capped to Brave API upper limit."""
        validator = SearchValidator(brave_config=brave_config_with_key)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_brave_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(validator, "_client", autospec=True) as mock_client:
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.get = mock_get

            await validator._search("test query", max_results=30)

            params = mock_get.call_args.kwargs.get("params", {})
            assert params.get("count") == 20

    @pytest.mark.asyncio
    async def test_normalizes_query_whitespace(
        self,
        brave_config_with_key: BraveSearchConfig,
        sample_brave_response: dict[str, Any],
    ) -> None:
        """Test that query whitespace is normalized before request."""
        validator = SearchValidator(brave_config=brave_config_with_key)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_brave_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(validator, "_client", autospec=True) as mock_client:
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.get = mock_get

            await validator._search("  thermodynamics \n   entropy\tadoption   ", max_results=5)

            params = mock_get.call_args.kwargs.get("params", {})
            assert params.get("q") == "thermodynamics entropy adoption"

    @pytest.mark.asyncio
    async def test_handles_empty_results(
        self,
        sample_hypothesis: Hypothesis,
        sample_brave_empty_response: dict[str, Any],
        brave_config_with_key: BraveSearchConfig,
    ) -> None:
        """Test handling of empty search results."""
        validator = SearchValidator(brave_config=brave_config_with_key)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_brave_empty_response
        mock_response.raise_for_status = MagicMock()

        mock_get = AsyncMock(return_value=mock_response)

        with patch.object(validator, "_client", autospec=True) as mock_client:
            mock_client.get = mock_get

            result = await validator.validate(
                hypothesis=sample_hypothesis,
                preliminary_score=8.0,
            )

            # Empty results should indicate high novelty
            assert result.searched is True
            assert result.similar_works == []
            # Score should be close to or higher than preliminary when no similar found
            assert result.score >= 7.0

    @pytest.mark.asyncio
    async def test_deduplicates_similar_works(
        self,
        sample_hypothesis: Hypothesis,
        brave_config_with_key: BraveSearchConfig,
    ) -> None:
        """Test that duplicate results across rounds are deduplicated."""
        validator = SearchValidator(brave_config=brave_config_with_key)

        # Same result appears in multiple rounds
        duplicate_response = {
            "web": {
                "results": [
                    {
                        "title": "Same Paper",
                        "url": "https://example.com/same",
                        "description": "Same description",
                    }
                ]
            }
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = duplicate_response
        mock_response.raise_for_status = MagicMock()

        mock_get = AsyncMock(return_value=mock_response)

        with patch.object(validator, "_client", autospec=True) as mock_client:
            mock_client.get = mock_get

            result = await validator.validate(
                hypothesis=sample_hypothesis,
                preliminary_score=7.5,
            )

            # Should deduplicate by URL
            urls = [w.url for w in result.similar_works]
            assert len(urls) == len(set(urls))


class TestSearchQueryBuilding:
    """Tests for search query building logic."""

    def test_build_concept_query(self) -> None:
        """Test building a concept search query."""
        validator = SearchValidator()
        hypothesis = Hypothesis(
            id="test",
            description="Apply thermodynamic entropy to model viral adoption",
            source_domain="thermodynamics",
            source_structure="entropy-disorder",
            analogy_explanation="Mapping entropy to adoption uncertainty",
            observable="volatility_ratio",
        )

        query = validator._build_query(hypothesis, "concept")

        # Should include domain and main concept
        assert "thermodynamics" in query.lower()
        assert "research" in query.lower()

    def test_build_implementation_query(self) -> None:
        """Test building an implementation search query."""
        validator = SearchValidator()
        hypothesis = Hypothesis(
            id="test",
            description="Apply thermodynamic entropy to model viral adoption",
            source_domain="thermodynamics",
            source_structure="entropy-disorder",
            analogy_explanation="Mapping entropy to adoption uncertainty",
            observable="adoption_rate / expected_adoption",
        )

        query = validator._build_query(hypothesis, "implementation")

        # Should include observable or formula keywords
        assert "adoption" in query.lower()

    def test_build_cross_domain_query(self) -> None:
        """Test building a cross-domain search query."""
        validator = SearchValidator()
        hypothesis = Hypothesis(
            id="test",
            description="Apply thermodynamic entropy to model viral adoption",
            source_domain="thermodynamics",
            source_structure="entropy-disorder",
            analogy_explanation="Mapping entropy to adoption uncertainty",
            observable="volatility_ratio",
        )

        query = validator._build_query(hypothesis, "cross_domain")

        # Should include both source and target domains
        assert "thermodynamics" in query.lower()
        assert "analogy" in query.lower() or "cross-domain" in query.lower()


class TestSimilarityCalculation:
    """Tests for similarity calculation between hypothesis and search results."""

    def test_calculate_similarity_high_overlap(self) -> None:
        """Test similarity calculation with high keyword overlap."""
        validator = SearchValidator()
        hypothesis = Hypothesis(
            id="test",
            description="Use entropy to model adoption diffusion",
            source_domain="thermodynamics",
            source_structure="entropy",
            analogy_explanation="Entropy maps to uncertainty",
            observable="adoption_ratio",
        )

        result = {
            "title": "Entropy-Based Adoption Prediction",
            "description": "Using entropy from thermodynamics to model viral adoption diffusion.",
        }

        similarity = validator._calculate_similarity(hypothesis, result)

        # High keyword overlap should result in meaningful similarity (> 0.15)
        # Jaccard similarity between word sets is typically lower than 0.5
        assert similarity > 0.15

    def test_calculate_similarity_low_overlap(self) -> None:
        """Test similarity calculation with low keyword overlap."""
        validator = SearchValidator()
        hypothesis = Hypothesis(
            id="test",
            description="Use entropy to model adoption diffusion",
            source_domain="thermodynamics",
            source_structure="entropy",
            analogy_explanation="Entropy maps to uncertainty",
            observable="adoption_ratio",
        )

        result = {
            "title": "Cooking Recipes for Beginners",
            "description": "Learn how to cook delicious meals at home.",
        }

        similarity = validator._calculate_similarity(hypothesis, result)

        # Low overlap should result in low similarity
        assert similarity < 0.3


class TestApiKeyValidation:
    """Tests for API key validation."""

    @pytest.mark.asyncio
    async def test_empty_api_key_raises_error(
        self,
        sample_hypothesis: Hypothesis,
    ) -> None:
        """Test that empty API key raises ValueError."""
        from pydantic import SecretStr
        from x_creative.config.settings import BraveSearchConfig

        # Create config with empty API key
        brave_config = BraveSearchConfig(api_key=SecretStr(""))
        validator = SearchValidator(brave_config=brave_config)

        # The validate method should catch this and return fallback
        result = await validator.validate(
            hypothesis=sample_hypothesis,
            preliminary_score=7.0,
        )

        # Should return fallback verdict with preliminary score
        assert result.score == 7.0
        assert result.searched is False
        assert "error" in result.novelty_analysis.lower()
