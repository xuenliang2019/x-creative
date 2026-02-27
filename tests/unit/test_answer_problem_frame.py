"""Tests for ProblemFrameBuilder."""

from unittest.mock import AsyncMock, patch

import pytest


class TestProblemFrameBuilder:
    @pytest.mark.asyncio
    async def test_build_returns_frame(self):
        """Normal question returns a complete ProblemFrame."""
        from x_creative.answer.problem_frame import ProblemFrameBuilder

        llm_response = """{
            "objective": "Design a CLI tool that achieves viral adoption",
            "constraints": [{"text": "Must be cross-platform", "source": "inferred", "confidence": 0.95}],
            "scope": {"in_scope": ["CLI tools", "developer experience"], "out_of_scope": ["GUI apps"]},
            "definitions": {"viral adoption": "organic growth driven by user recommendations"},
            "success_criteria": ["GitHub stars > 1000 in first month"],
            "domain_hint": {"domain_id": "open_source_development", "confidence": 0.85},
            "open_questions": [],
            "context": {"ecosystem": "open source"}
        }"""

        with patch("x_creative.answer.problem_frame.ModelRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.complete = AsyncMock(
                return_value=type("R", (), {"content": llm_response})()
            )
            mock_router.close = AsyncMock()

            builder = ProblemFrameBuilder(router=mock_router)
            result = await builder.build("How to design a viral open source tool?")

        assert result.needs_clarification is False
        assert result.frame is not None
        assert result.frame.objective == "Design a CLI tool that achieves viral adoption"
        assert result.frame.domain_hint["domain_id"] == "open_source_development"
        assert result.frame.domain_hint["confidence"] == 0.85
        assert result.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_build_low_confidence_triggers_clarification(self):
        """Very low confidence domain hint triggers smart interrupt."""
        from x_creative.answer.problem_frame import ProblemFrameBuilder

        llm_response = """{
            "objective": "Something vague",
            "constraints": [],
            "scope": {"in_scope": [], "out_of_scope": []},
            "definitions": {},
            "success_criteria": [],
            "domain_hint": {"domain_id": "unknown", "confidence": 0.15},
            "open_questions": ["Cannot determine domain"],
            "context": {}
        }"""

        with patch("x_creative.answer.problem_frame.ModelRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.complete = AsyncMock(
                return_value=type("R", (), {"content": llm_response})()
            )
            mock_router.close = AsyncMock()

            builder = ProblemFrameBuilder(router=mock_router)
            result = await builder.build("Do the thing")

        assert result.needs_clarification is True
        assert result.clarification_question is not None
        assert result.partial_frame is not None

    @pytest.mark.asyncio
    async def test_build_medium_confidence_continues(self):
        """Medium confidence (0.3-0.7) continues without interrupt."""
        from x_creative.answer.problem_frame import ProblemFrameBuilder

        llm_response = """{
            "objective": "Improve developer tooling",
            "constraints": [],
            "scope": {"in_scope": ["developer tools"], "out_of_scope": []},
            "definitions": {},
            "success_criteria": [],
            "domain_hint": {"domain_id": "open_source_development", "confidence": 0.5},
            "open_questions": ["Exact tool category unclear"],
            "context": {}
        }"""

        with patch("x_creative.answer.problem_frame.ModelRouter") as MockRouter:
            mock_router = MockRouter.return_value
            mock_router.complete = AsyncMock(
                return_value=type("R", (), {"content": llm_response})()
            )
            mock_router.close = AsyncMock()

            builder = ProblemFrameBuilder(router=mock_router)
            result = await builder.build("Improve my developer tooling")

        assert result.needs_clarification is False
        assert result.frame is not None
        assert len(result.frame.open_questions) > 0

    @pytest.mark.asyncio
    async def test_build_with_known_domains(self):
        """Builder is aware of available target domain IDs."""
        from x_creative.answer.problem_frame import ProblemFrameBuilder

        builder = ProblemFrameBuilder()
        assert "open_source_development" in builder.available_domains
