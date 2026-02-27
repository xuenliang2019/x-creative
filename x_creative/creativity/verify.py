"""VERIFY module for scoring and evaluating hypotheses."""

import json
import inspect
from collections.abc import Awaitable, Callable

import structlog

from x_creative.core.types import Hypothesis, HypothesisScores
from x_creative.creativity.prompts import VERIFY_SCORE_PROMPT
from x_creative.creativity.utils import safe_json_loads
from x_creative.llm.router import ModelRouter

logger = structlog.get_logger()


class VerifyModule:
    """Module for verifying and scoring hypotheses.

    Evaluates hypotheses on multiple dimensions:
    - Divergence: How novel is this compared to known factors?
    - Testability: Can it be converted to a concrete test?
    - Rationale: Is there a sound mechanism?
    - Robustness: How likely to avoid overfitting?
    """

    def __init__(self, router: ModelRouter | None = None) -> None:
        """Initialize the verify module.

        Args:
            router: Model router for LLM calls.
        """
        self._router = router or ModelRouter()
        self._problem_frame = None

    async def score_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        """Score a single hypothesis.

        Args:
            hypothesis: The hypothesis to score.

        Returns:
            The hypothesis with scores attached.
        """
        target_domain_id = "unknown"
        target_domain_name = "Unknown"
        target_domain_description = ""
        domain_constraints = "None"
        domain_evaluation_criteria = "None"
        domain_anti_patterns = "None"

        problem_frame = getattr(self, "_problem_frame", None)
        if problem_frame is not None:
            target_domain_id = str(getattr(problem_frame, "target_domain", "") or "").strip() or "unknown"

        try:
            from x_creative.core.plugin import load_target_domain

            plugin = load_target_domain(target_domain_id) if target_domain_id else None
            if plugin is not None:
                target_domain_name = plugin.name
                target_domain_description = plugin.description
                if plugin.constraints:
                    domain_constraints = "\n".join(
                        f"- {c.name} ({c.severity}): {c.description}" for c in plugin.constraints
                    )
                if plugin.evaluation_criteria:
                    domain_evaluation_criteria = "\n".join(f"- {c}" for c in plugin.evaluation_criteria)
                if plugin.anti_patterns:
                    domain_anti_patterns = "\n".join(f"- {p}" for p in plugin.anti_patterns)
            else:
                target_domain_name = target_domain_id
        except Exception:
            target_domain_name = target_domain_id

        prompt = VERIFY_SCORE_PROMPT.format(
            target_domain_id=target_domain_id,
            target_domain_name=target_domain_name,
            target_domain_description=target_domain_description,
            domain_constraints=domain_constraints,
            domain_evaluation_criteria=domain_evaluation_criteria,
            domain_anti_patterns=domain_anti_patterns,
            hypothesis_id=hypothesis.id,
            hypothesis_description=hypothesis.description,
            source_domain=hypothesis.source_domain,
            source_structure=hypothesis.source_structure,
            analogy_explanation=hypothesis.analogy_explanation,
            observable=hypothesis.observable,
        )

        result = await self._router.complete(
            task="hypothesis_scoring",
            messages=[{"role": "user", "content": prompt}],
        )

        scores = self._parse_scores(result.content)

        # Preserve all hypothesis fields (including HKG evidence / metadata),
        # only updating the scoring payload.
        return hypothesis.model_copy(update={"scores": scores})

    def _parse_scores(self, content: str) -> HypothesisScores:
        """Parse score response into HypothesisScores.

        Args:
            content: Raw LLM response.

        Returns:
            Parsed scores.
        """
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = safe_json_loads(json_str)

                return HypothesisScores(
                    divergence=float(data.get("divergence", 5.0)),
                    testability=float(data.get("testability", 5.0)),
                    rationale=float(data.get("rationale", 5.0)),
                    robustness=float(data.get("robustness", 5.0)),
                    feasibility=float(data.get("feasibility", 5.0)),
                    divergence_reason=data.get("divergence_reason"),
                    testability_reason=data.get("testability_reason"),
                    rationale_reason=data.get("rationale_reason"),
                    robustness_reason=data.get("robustness_reason"),
                    feasibility_reason=data.get("feasibility_reason"),
                )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                "Failed to parse scores, using defaults",
                error=str(e),
                content_preview=content[:200],
            )

        # Return default scores
        return HypothesisScores(
            divergence=5.0,
            testability=5.0,
            rationale=5.0,
            robustness=5.0,
            feasibility=5.0,
        )

    async def score_batch(
        self,
        hypotheses: list[Hypothesis],
        concurrency: int = 5,
        on_progress: Callable[[int, int, str], Awaitable[None] | None] | None = None,
    ) -> list[Hypothesis]:
        """Score multiple hypotheses.

        Args:
            hypotheses: List of hypotheses to score.
            concurrency: Number of concurrent scoring tasks.
            on_progress: Optional callback invoked after each hypothesis is scored:
                (completed_count, total_count, hypothesis_id).

        Returns:
            List of hypotheses with scores.
        """
        import asyncio

        scored: list[Hypothesis] = []
        semaphore = asyncio.Semaphore(concurrency)
        total = len(hypotheses)
        completed = 0
        completed_lock = asyncio.Lock()

        async def _report(hypothesis_id: str) -> None:
            nonlocal completed
            if on_progress is None:
                return
            async with completed_lock:
                completed += 1
                current = completed
            try:
                maybe_awaitable = on_progress(current, total, hypothesis_id)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            except Exception as exc:
                logger.debug(
                    "on_progress callback failed",
                    hypothesis_id=hypothesis_id,
                    error=str(exc),
                )

        async def score_with_semaphore(hyp: Hypothesis) -> Hypothesis:
            async with semaphore:
                try:
                    scored_hyp = await self.score_hypothesis(hyp)
                except Exception as e:
                    logger.warning(
                        "Failed to score hypothesis",
                        hypothesis_id=hyp.id,
                        error=str(e),
                    )
                    # Preserve all fields when falling back to default scores.
                    scored_hyp = hyp.model_copy(
                        update={
                            "scores": HypothesisScores(
                            divergence=5.0,
                            testability=5.0,
                            rationale=5.0,
                            robustness=5.0,
                            feasibility=5.0,
                            )
                        }
                    )
            await _report(hyp.id)
            return scored_hyp

        logger.info(
            "Scoring hypotheses",
            count=len(hypotheses),
            concurrency=concurrency,
        )

        tasks = [score_with_semaphore(h) for h in hypotheses]
        scored = await asyncio.gather(*tasks)

        logger.info(
            "Scoring complete",
            count=len(scored),
        )

        return list(scored)

    def filter_by_threshold(
        self,
        hypotheses: list[Hypothesis],
        threshold: float = 5.0,
    ) -> list[Hypothesis]:
        """Filter hypotheses by minimum composite score.

        Args:
            hypotheses: Hypotheses to filter.
            threshold: Minimum composite score.

        Returns:
            Filtered list.
        """
        return [h for h in hypotheses if h.composite_score() >= threshold]
