"""Search-based novelty validation using Brave Search API.

This module implements stage 2 of novelty verification:
- Performs structured multi-round web search
- Aggregates results into final novelty assessment
- Works with results from NoveltyVerifier (stage 1)
"""

import asyncio
import re
from typing import Any

import httpx
import structlog

from x_creative.config.settings import BraveSearchConfig, SearchConfig, SearchRoundConfig, get_settings
from x_creative.core.types import Hypothesis, NoveltyVerdict, SimilarWork

logger = structlog.get_logger()


class SearchValidator:
    """Validates hypothesis novelty using Brave Search API.

    This is stage 2 of novelty verification:
    - Performs structured multi-round search
    - Round 1: concept (weight 0.3) - Core concept search
    - Round 2: implementation (weight 0.5) - Implementation details
    - Round 3: cross_domain (weight 0.2) - Cross-domain applications
    - Aggregates results into final novelty assessment
    """

    BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
    BRAVE_MAX_COUNT = 20
    BRAVE_MIN_COUNT = 1

    def __init__(
        self,
        search_config: SearchConfig | None = None,
        brave_config: BraveSearchConfig | None = None,
    ) -> None:
        """Initialize the SearchValidator.

        Args:
            search_config: Search configuration with rounds and weights.
            brave_config: Brave Search API configuration.
        """
        settings = get_settings()
        self._config = search_config or settings.search
        self._brave_config = brave_config or settings.brave_search
        self._client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self) -> "SearchValidator":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client.

        Call this method when done using the SearchValidator to properly
        release resources. Alternatively, use the async context manager pattern.
        """
        await self._client.aclose()

    async def validate(
        self,
        hypothesis: Hypothesis,
        preliminary_score: float,
    ) -> NoveltyVerdict:
        """Validate novelty via web search.

        Performs multi-round search using Brave Search API and aggregates
        results to assess novelty. Each round uses different query strategies:
        - concept: Core concept research
        - implementation: Implementation details
        - cross_domain: Cross-domain applications

        Args:
            hypothesis: The hypothesis to validate.
            preliminary_score: Score from NoveltyVerifier (LLM assessment).

        Returns:
            NoveltyVerdict with searched=True and updated score.
        """
        try:
            all_results: list[dict[str, Any]] = []
            round_scores: list[tuple[float, float]] = []  # (score, weight)
            successful_rounds = 0  # Track rounds that completed without request errors

            async def _search_round(
                round_config: SearchRoundConfig,
            ) -> tuple[float, float, list[dict[str, Any]], bool]:
                """Execute a single search round."""
                try:
                    query = self._build_query(hypothesis, round_config.name)
                    results = await self._search(query, round_config.max_results)
                    round_score = self._calculate_round_score(
                        hypothesis, results, preliminary_score
                    )
                    return (round_score, round_config.weight, results, True)
                except Exception as e:
                    hint = None
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 422:
                        hint = (
                            "Brave returned 422: request parameters were rejected "
                            "(often count/query format)."
                        )
                    logger.warning(
                        "Search round failed",
                        round=round_config.name,
                        error=str(e),
                        hint=hint,
                    )
                    return (preliminary_score, round_config.weight, [], False)

            round_results = await asyncio.gather(
                *[_search_round(rc) for rc in self._config.rounds]
            )
            for score, weight, results, ok in round_results:
                round_scores.append((score, weight))
                all_results.extend(results)
                if ok:
                    successful_rounds += 1

            # If every round failed at request level, fallback to LLM-only verdict.
            if successful_rounds == 0:
                return self._create_fallback_verdict(
                    preliminary_score,
                    "All search rounds failed",
                )

            # Aggregate scores across rounds
            final_score = self._aggregate_scores(round_scores)

            # Parse results into SimilarWork objects
            similar_works = self._parse_results(hypothesis, all_results)

            # Generate novelty analysis
            analysis = self._generate_analysis(
                preliminary_score, final_score, len(similar_works)
            )

            return NoveltyVerdict(
                score=final_score,
                searched=True,
                similar_works=similar_works,
                novelty_analysis=analysis,
            )

        except Exception as e:
            logger.error(
                "Search validation failed",
                hypothesis_id=hypothesis.id,
                error=str(e),
            )
            return self._create_fallback_verdict(
                preliminary_score,
                f"Search validation error: {str(e)}",
            )

    async def _search(
        self,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Perform a search using Brave Search API.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            List of search result dictionaries.

        Raises:
            httpx.RequestError: If the request fails.
            httpx.HTTPStatusError: If the response indicates an error.
        """
        api_key = self._brave_config.api_key.get_secret_value()
        if not api_key:
            raise ValueError("Brave Search API key is not configured")

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }

        params = {
            "q": self._normalize_query(query),
            "count": self._normalize_count(max_results),
        }

        response = await self._client.get(
            self.BRAVE_SEARCH_ENDPOINT,
            headers=headers,
            params=params,
        )
        response.raise_for_status()

        data = response.json()

        # Extract results from response
        web_results = data.get("web", {}).get("results", [])
        logger.debug(
            "Search completed",
            query=query,
            results_count=len(web_results),
        )
        return web_results

    def _normalize_count(self, count: int) -> int:
        """Normalize requested result count to Brave API accepted range."""
        normalized = max(self.BRAVE_MIN_COUNT, min(count, self.BRAVE_MAX_COUNT))
        if normalized != count:
            logger.warning(
                "Search result count adjusted to Brave API limit",
                requested=count,
                adjusted=normalized,
            )
        return normalized

    def _normalize_query(self, query: str) -> str:
        """Normalize query string before sending to Brave."""
        return re.sub(r"\s+", " ", query).strip()

    def _build_query(self, hypothesis: Hypothesis, round_type: str) -> str:
        """Build a search query based on hypothesis and round type.

        Args:
            hypothesis: The hypothesis to search for.
            round_type: Type of search round (concept, implementation, cross_domain).

        Returns:
            Search query string.
        """
        if round_type == "concept":
            # Core concept search: "{domain} {main_concept} research"
            # Extract main concepts from description
            main_concept = self._extract_main_concept(hypothesis.description)
            return f"{hypothesis.source_domain} {main_concept} research"

        elif round_type == "implementation":
            # Implementation search: "{observable} {formula_keywords}"
            formula_keywords = self._extract_formula_keywords(hypothesis.observable)
            return f"{formula_keywords} implementation method"

        elif round_type == "cross_domain":
            # Cross-domain search: "{source_domain} analogy {target_domain}"
            # Assume target domain is finance for now (most common case)
            return f"{hypothesis.source_domain} analogy cross-domain application"

        else:
            # Default to concept-style query
            return f"{hypothesis.source_domain} {hypothesis.source_structure} research"

    def _extract_main_concept(self, description: str) -> str:
        """Extract main concept keywords from description.

        Args:
            description: Hypothesis description.

        Returns:
            Main concept keywords.
        """
        # Remove common stop words and extract key terms
        stop_words = {
            "the", "a", "an", "to", "for", "of", "in", "on", "and", "or",
            "use", "using", "apply", "predict", "analyze", "with", "that",
            "this", "is", "are", "be", "been", "has", "have", "will",
        }

        words = description.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Take first few meaningful words
        return " ".join(keywords[:4])

    def _extract_formula_keywords(self, observable: str) -> str:
        """Extract keywords from observable/formula.

        Args:
            observable: Observable proxy variables or formula.

        Returns:
            Formula keywords for search.
        """
        # Split on common formula operators and clean up
        parts = re.split(r"[/+\-*()=,]", observable)
        keywords = []

        for part in parts:
            # Clean and extract meaningful terms
            cleaned = part.strip().replace("_", " ")
            if cleaned and len(cleaned) > 2:
                keywords.append(cleaned)

        return " ".join(keywords[:3])

    def _calculate_round_score(
        self,
        hypothesis: Hypothesis,
        results: list[dict[str, Any]],
        preliminary_score: float,
    ) -> float:
        """Calculate novelty score for a search round.

        More similar results found = less novel = lower score.

        Args:
            hypothesis: The hypothesis being evaluated.
            results: Search results from this round.
            preliminary_score: LLM preliminary score.

        Returns:
            Round novelty score (0-10).
        """
        if not results:
            # No results = very novel in this dimension
            return min(preliminary_score + 1.0, 10.0)

        # Calculate average similarity across results
        similarities = [
            self._calculate_similarity(hypothesis, result)
            for result in results
        ]

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Convert similarity to novelty score
        # High similarity = low novelty, Low similarity = high novelty
        # Blend with preliminary score
        search_novelty = 10.0 * (1.0 - avg_similarity)

        # Weight: 60% search evidence, 40% preliminary score
        blended_score = 0.6 * search_novelty + 0.4 * preliminary_score

        return max(0.0, min(10.0, blended_score))

    def _calculate_similarity(
        self,
        hypothesis: Hypothesis,
        result: dict[str, Any],
    ) -> float:
        """Calculate similarity between hypothesis and a search result.

        Uses keyword overlap as a simple similarity measure.

        Args:
            hypothesis: The hypothesis.
            result: A single search result.

        Returns:
            Similarity score (0.0-1.0).
        """
        # Collect hypothesis keywords
        hyp_text = " ".join([
            hypothesis.description,
            hypothesis.source_domain,
            hypothesis.source_structure,
            hypothesis.analogy_explanation,
            hypothesis.observable,
        ]).lower()

        # Collect result keywords
        result_text = " ".join([
            str(result.get("title", "")),
            str(result.get("description", "")),
        ]).lower()

        # Tokenize
        hyp_words = set(re.findall(r'\b\w{3,}\b', hyp_text))
        result_words = set(re.findall(r'\b\w{3,}\b', result_text))

        # Remove common stop words
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "had", "her", "was", "one", "our", "out", "has",
            "have", "been", "this", "that", "with", "they", "from",
            "will", "what", "about", "which", "when", "there", "their",
        }
        hyp_words -= stop_words
        result_words -= stop_words

        if not hyp_words or not result_words:
            return 0.0

        # Jaccard similarity
        intersection = len(hyp_words & result_words)
        union = len(hyp_words | result_words)

        return intersection / union if union > 0 else 0.0

    def _aggregate_scores(
        self,
        round_scores: list[tuple[float, float]],
    ) -> float:
        """Aggregate scores from all rounds using weights.

        Args:
            round_scores: List of (score, weight) tuples.

        Returns:
            Weighted average score.
        """
        if not round_scores:
            return 0.0

        total_weight = sum(weight for _, weight in round_scores)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(score * weight for score, weight in round_scores)
        return weighted_sum / total_weight

    def _parse_results(
        self,
        hypothesis: Hypothesis,
        results: list[dict[str, Any]],
    ) -> list[SimilarWork]:
        """Parse search results into SimilarWork objects.

        Args:
            hypothesis: The hypothesis (for similarity calculation).
            results: All search results.

        Returns:
            Deduplicated list of SimilarWork objects.
        """
        seen_urls: set[str] = set()
        similar_works: list[SimilarWork] = []

        for result in results:
            url = str(result.get("url", ""))

            # Skip duplicates
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = str(result.get("title", "Unknown"))
            description = str(result.get("description", ""))

            # Calculate similarity
            similarity = self._calculate_similarity(hypothesis, result)

            similar_work = SimilarWork(
                title=title,
                url=url,
                source="web",  # Brave search results
                similarity=similarity,
                difference_summary=description[:200] if description else "No description available.",
            )
            similar_works.append(similar_work)

        # Sort by similarity (highest first)
        similar_works.sort(key=lambda w: w.similarity, reverse=True)

        return similar_works

    def _generate_analysis(
        self,
        preliminary_score: float,
        final_score: float,
        num_similar: int,
    ) -> str:
        """Generate novelty analysis text.

        Args:
            preliminary_score: LLM preliminary score.
            final_score: Final aggregated score.
            num_similar: Number of similar works found.

        Returns:
            Novelty analysis string.
        """
        if num_similar == 0:
            return (
                f"Web search found no similar works. "
                f"LLM assessment: {preliminary_score:.1f}, "
                f"Final score: {final_score:.1f}. "
                f"This hypothesis appears to be highly novel."
            )
        elif num_similar <= 3:
            return (
                f"Web search found {num_similar} potentially related works. "
                f"LLM assessment: {preliminary_score:.1f}, "
                f"Final score: {final_score:.1f}. "
                f"The hypothesis shows moderate novelty with some related prior art."
            )
        else:
            return (
                f"Web search found {num_similar} related works. "
                f"LLM assessment: {preliminary_score:.1f}, "
                f"Final score: {final_score:.1f}. "
                f"Similar concepts exist in the literature."
            )

    def _create_fallback_verdict(
        self,
        preliminary_score: float,
        error_message: str,
    ) -> NoveltyVerdict:
        """Create a fallback verdict when search fails.

        Args:
            preliminary_score: LLM preliminary score to use.
            error_message: Description of what failed.

        Returns:
            NoveltyVerdict with searched=False.
        """
        return NoveltyVerdict(
            score=preliminary_score,
            searched=False,
            similar_works=[],
            novelty_analysis=f"Search validation error: {error_message}",
        )
