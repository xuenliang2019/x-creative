"""Novelty verification for hypotheses using LLM.

This module provides preliminary novelty assessment using an LLM (Gemini 3 Pro).
This is stage 1 of novelty verification:
- LLM evaluates novelty based on its training data
- If score >= threshold (6.0), triggers web search (stage 2)
- If score < threshold, considered already known (skip search)
"""

import json
from typing import Any

import structlog

from x_creative.core.types import Hypothesis, NoveltyVerdict, ProblemFrame, SimilarWork
from x_creative.llm.router import ModelRouter
from x_creative.verify.utils import (
    DEFAULT_LLM_SIMILARITY,
    safe_float,
    strip_markdown_code_block,
)

logger = structlog.get_logger()


class NoveltyVerifier:
    """Preliminary novelty assessment using LLM (Gemini 3 Pro).

    This is stage 1 of novelty verification:
    - LLM evaluates novelty based on its training data
    - If score >= threshold (6.0), triggers web search (stage 2)
    - If score < threshold, skipped for search (but still recorded)
    """

    SEARCH_THRESHOLD = 6.0  # Default threshold for triggering web search

    def __init__(
        self,
        router: ModelRouter | None = None,
        search_threshold: float | None = None,
    ) -> None:
        """Initialize the NoveltyVerifier.

        Args:
            router: Model router to use for LLM calls. If None, creates one.
            search_threshold: Minimum score to trigger web search. Defaults to 6.0.
        """
        self._router = router or ModelRouter()
        self._search_threshold = search_threshold or self.SEARCH_THRESHOLD

    def needs_search(self, score: float) -> bool:
        """Check if a score requires web search verification.

        Args:
            score: The novelty score from LLM assessment.

        Returns:
            True if score >= threshold and web search should be performed.
        """
        return score >= self._search_threshold

    async def verify(
        self,
        hypothesis: Hypothesis,
        problem_frame: ProblemFrame,
    ) -> NoveltyVerdict:
        """Preliminary novelty verification.

        Assesses novelty based on LLM's training data knowledge.
        High scores indicate the hypothesis seems novel and should undergo
        web search verification. Low scores indicate the idea is already
        known in the literature.

        Args:
            hypothesis: The hypothesis to verify.
            problem_frame: The problem context.

        Returns:
            NoveltyVerdict with:
            - score (0-10): LLM-assigned novelty score
            - searched: Always False (web search not performed here)
            - similar_works: Similar works the LLM recalls
            - novelty_analysis: Explanation of the assessment
        """
        try:
            # Build the verification prompt
            messages = self._build_prompt(hypothesis, problem_frame)

            # Call the LLM
            result = await self._router.complete(
                task="novelty_verification",
                messages=messages,
            )

            # Parse the response
            verdict = self._parse_response(result.content)
            return verdict

        except Exception as e:
            logger.error(
                "Novelty verification failed",
                hypothesis_id=hypothesis.id,
                error=str(e),
            )
            return self._create_error_verdict(str(e))

    def _build_prompt(
        self,
        hypothesis: Hypothesis,
        problem_frame: ProblemFrame,
    ) -> list[dict[str, Any]]:
        """Build the verification prompt.

        Args:
            hypothesis: The hypothesis to verify.
            problem_frame: The problem context.

        Returns:
            List of message dictionaries for the LLM.
        """
        system_prompt = """You are an expert at evaluating the novelty of research ideas and hypotheses.

Your task is to assess whether a given hypothesis represents a novel contribution based on your knowledge of existing literature, research, and common approaches.

Evaluate the hypothesis and provide:
1. **Novelty Score** (0-10):
   - 0-3: Very common idea, well-established in literature
   - 4-5: Somewhat known, has been explored with minor variations
   - 6-7: Relatively novel, combines known concepts in new ways
   - 8-10: Highly novel, represents a genuinely new approach

2. **Known Similar Works**: List any similar works, papers, or approaches you are aware of from your training data. For each, provide:
   - title: Name or description of the work
   - source: Where it's from (arxiv, ssrn, blog, paper, etc.)
   - similarity_explanation: How it relates to the hypothesis

3. **Reasoning**: Explain your novelty assessment.

Respond ONLY with a JSON object in this exact format:
{
    "novelty_score": <score 0-10>,
    "known_similar": [
        {
            "title": "<work title or description>",
            "source": "<arxiv|ssrn|blog|paper|other>",
            "similarity_explanation": "<how it relates>"
        }
    ],
    "reasoning": "<explanation of novelty assessment>"
}

Be thorough in recalling similar works, but also fair - a hypothesis that applies known concepts to a new domain or combines them in a novel way deserves credit for novelty."""

        user_prompt = f"""Please assess the novelty of the following hypothesis:

## Hypothesis
**Description:** {hypothesis.description}

**Source Domain:** {hypothesis.source_domain}
**Source Structure:** {hypothesis.source_structure}
**Analogy Explanation:** {hypothesis.analogy_explanation}

**Observable/Formula:** {hypothesis.observable}

## Problem Context
**Problem Description:** {problem_frame.description}
**Target Domain:** {problem_frame.target_domain}
**Constraints:** {', '.join(problem_frame.constraints) if problem_frame.constraints else 'None specified'}

Please evaluate this hypothesis for novelty and provide your JSON response."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response(self, content: str) -> NoveltyVerdict:
        """Parse the LLM response into a NoveltyVerdict.

        Args:
            content: Raw LLM response content.

        Returns:
            NoveltyVerdict parsed from the response.
        """
        # Try to extract JSON from the response
        try:
            # Handle potential markdown code blocks
            clean_content = strip_markdown_code_block(content)
            data = json.loads(clean_content)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON response", error=str(e), content=content[:200])
            return NoveltyVerdict(
                score=0.0,
                searched=False,
                similar_works=[],
                novelty_analysis=f"Failed to parse LLM response as JSON: {str(e)}",
            )

        # Extract score with safe conversion
        score = safe_float(data.get("novelty_score"), 0.0)
        # Clamp score to valid range
        score = max(0.0, min(10.0, score))

        # Extract and parse similar works
        similar_works = self._parse_similar_works(data.get("known_similar", []))

        # Extract reasoning
        reasoning = data.get("reasoning", "No reasoning provided.")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        return NoveltyVerdict(
            score=score,
            searched=False,  # Web search not performed in this stage
            similar_works=similar_works,
            novelty_analysis=reasoning,
        )

    def _parse_similar_works(self, raw_works: Any) -> list[SimilarWork]:
        """Parse similar works from LLM response.

        Args:
            raw_works: Raw list of similar works from LLM.

        Returns:
            List of SimilarWork objects.
        """
        if not isinstance(raw_works, list):
            return []

        similar_works = []
        for work in raw_works:
            if not isinstance(work, dict):
                continue

            try:
                title = str(work.get("title", "Unknown"))
                source = str(work.get("source", "unknown"))
                similarity_explanation = str(work.get("similarity_explanation", ""))

                # Create SimilarWork with placeholder values for fields
                # that will be populated by web search
                similar_work = SimilarWork(
                    title=title,
                    url="",  # LLM doesn't provide URLs, web search will
                    source=source,
                    similarity=DEFAULT_LLM_SIMILARITY,  # Web search will refine
                    difference_summary=similarity_explanation,
                )
                similar_works.append(similar_work)
            except Exception as e:
                logger.warning("Failed to parse similar work", error=str(e), work=work)
                continue

        return similar_works

    def _create_error_verdict(self, error_message: str) -> NoveltyVerdict:
        """Create a failed verdict for error cases.

        Args:
            error_message: Description of the error.

        Returns:
            NoveltyVerdict indicating failure due to error.
        """
        return NoveltyVerdict(
            score=0.0,
            searched=False,
            similar_works=[],
            novelty_analysis=f"Verification error: {error_message}",
            error=True,
        )
