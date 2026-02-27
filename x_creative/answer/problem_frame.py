"""ProblemFrameBuilder - converts natural language question to structured ProblemFrame."""

from __future__ import annotations

import json
import logging
from typing import Any

from x_creative.answer.prompts import PROBLEM_FRAME_PROMPT
from x_creative.answer.types import FrameBuildResult
from x_creative.core.plugin import list_target_domains
from x_creative.core.types import ProblemFrame
from x_creative.llm.router import ModelRouter

logger = logging.getLogger(__name__)

CLARIFICATION_THRESHOLD = 0.3


class ProblemFrameBuilder:
    """Builds a structured ProblemFrame from a natural language question."""

    def __init__(self, router: ModelRouter | None = None):
        self._router = router
        self.available_domains = list_target_domains()

    async def build(self, question: str) -> FrameBuildResult:
        """Build a ProblemFrame from a natural language question.

        Args:
            question: The user's natural language question.

        Returns:
            FrameBuildResult with either a complete frame or a clarification request.
        """
        router = self._router or ModelRouter()
        owns_router = self._router is None
        try:
            return await self._build(question, router)
        finally:
            if owns_router:
                await router.close()

    async def _build(self, question: str, router: ModelRouter) -> FrameBuildResult:
        prompt = PROBLEM_FRAME_PROMPT.format(
            available_domains=", ".join(self.available_domains),
            question=question,
        )
        result = await router.complete(
            task="creativity",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        parsed = self._parse_response(result.content)
        return self._to_frame_result(question, parsed)

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response content as JSON, stripping markdown fences if present."""
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
        return json.loads(text)

    def _to_frame_result(self, question: str, parsed: dict[str, Any]) -> FrameBuildResult:
        """Convert parsed LLM output into a FrameBuildResult."""
        domain_hint = parsed.get("domain_hint", {})
        confidence = domain_hint.get("confidence", 0.5)

        constraints_raw = parsed.get("constraints", [])
        constraints = [c["text"] for c in constraints_raw if isinstance(c, dict) and "text" in c]

        frame = ProblemFrame(
            description=question,
            target_domain=domain_hint.get("domain_id", "general"),
            constraints=constraints,
            context=parsed.get("context", {}),
            objective=parsed.get("objective"),
            scope=parsed.get("scope"),
            definitions=parsed.get("definitions"),
            success_criteria=parsed.get("success_criteria", []),
            open_questions=parsed.get("open_questions", []),
            domain_hint=domain_hint,
        )

        if confidence < CLARIFICATION_THRESHOLD:
            return FrameBuildResult(
                frame=None,
                needs_clarification=True,
                clarification_question=self._make_clarification_question(parsed),
                partial_frame=frame,
                confidence=confidence,
            )

        return FrameBuildResult(frame=frame, confidence=confidence)

    def _make_clarification_question(self, parsed: dict[str, Any]) -> str:
        """Generate a clarification question from parsed output."""
        open_qs = parsed.get("open_questions", [])
        if open_qs:
            return open_qs[0]
        return "Could you specify which domain or field this question relates to?"
